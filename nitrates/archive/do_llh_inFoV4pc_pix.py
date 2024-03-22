import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import time
import pandas as pd

from ..analysis_seeds.bkg_rate_estimation import rate_obj_from_sqltab
from ..lib.sqlite_funcs import (
    get_conn,
    write_result,
    write_results,
    timeID2time_dur,
    write_results_fromSigImg,
    update_square_stat,
    write_square_res_line,
    write_square_results,
)
from ..lib.dbread_funcs import (
    get_rate_fits_tab,
    guess_dbfname,
    get_seeds_tab,
    get_info_tab,
    get_files_tab,
    get_square_tab,
    get_full_sqlite_table_as_df,
)
from ..config import EBINS0, EBINS1, solid_angle_dpi_fname, fp_dir, rt_dir
from ..models.flux_models import Plaw_Flux, Cutoff_Plaw_Flux
from ..llh_analysis.minimizers import (
    NLLH_ScipyMinimize_Wjacob,
    imxy_grid_miner,
    NLLH_ScipyMinimize,
)

# from ..lib.drm_funcs import DRMs
from ..response.ray_trace_funcs import RayTraces, FootPrints
from ..llh_analysis.LLH import LLH_webins

# from do_InFoV_scan3 import Swift_Mask_Interactions, Source_Model_InFoV, Bkg_Model_wFlatA,\
#                             CompoundModel, Point_Source_Model_Binned_Rates,\
#                             theta_phi2imxy, bldmask2batxys, imxy2theta_phi,\
#                             get_fixture_struct, LLH_webins
from ..models.models import (
    CompoundModel,
    Point_Source_Model_Binned_Rates,
    Bkg_Model_wFlatA,
    Source_Model_InFoV,
    Source_Model_InOutFoV,
)
from ..lib.coord_conv_funcs import theta_phi2imxy, imxy2theta_phi
from ..lib.gti_funcs import mk_gti_bl, union_gtis


# need to read rate fits from DB
# and read twinds
# and read/get event, dmask, and ebins
# then get bkg_llh_obj and a minimizer
# then loop over all time windows
# minimizing nllh and recording bf params


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--job_id", type=int, help="ID to tell it what seeds to do", default=-1
    )
    parser.add_argument(
        "--square_id", type=int, help="squareID to do if job_id is < 0", default=-1
    )
    parser.add_argument(
        "--Njobs", type=int, help="Total number of jobs submitted", default=64
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--rt_dir", type=str, help="Directory with ray traces", default=None
    )
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default="pc_2.img"
    )
    parser.add_argument(
        "--job_fname",
        type=str,
        help="File name for table with what imx/y square for each job",
        default="rate_seeds.csv",
    )
    parser.add_argument(
        "--rate_fname",
        type=str,
        help="Rate results file name",
        default="rate_seeds.csv",
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--pix_fname",
        type=str,
        help="Name of the file with good imx/y coordinates",
        default="good_pix2scan.npy",
    )
    parser.add_argument(
        "--log_fname",
        type=str,
        help="Name for the log file",
        default="llh_analysis_from_rate_seeds",
    )
    parser.add_argument(
        "--min_pc", type=float, help="Min partical coding fraction to use", default=0.01
    )
    args = parser.parse_args()
    return args


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot(imx0 - imx1, imy0 - imy1)


def parse_bkg_csv(bkg_fname, solid_angle_dpi, ebins0, ebins1, bl_dmask, rt_dir):
    bkg_df = pd.read_csv(bkg_fname)
    col_names = bkg_df.columns
    nebins = len(ebins0)

    PSnames = []
    for name in col_names:
        if "_imx" in name:
            PSnames.append(name.split("_")[0])
    print(PSnames)
    Nsrcs = len(PSnames)
    if Nsrcs > 0:
        bkg_name = "Background_"
    else:
        bkg_name = ""

    bkg_mod = Bkg_Model_wFlatA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)

    ps_mods = []

    if Nsrcs > 0:
        rt_obj = RayTraces(rt_dir)
        for i in range(Nsrcs):
            name = PSnames[i]
            imx = bkg_df[name + "_imx"][0]
            imy = bkg_df[name + "_imy"][0]
            mod = Point_Source_Model_Binned_Rates(
                imx,
                imy,
                0.1,
                [ebins0, ebins1],
                rt_obj,
                bl_dmask,
                use_deriv=True,
                name=name,
            )
            ps_mods.append(mod)

    return bkg_df, bkg_name, PSnames, bkg_mod, ps_mods


def find_peaks2scan(
    res_df,
    max_dv=10.0,
    min_sep=9e-3,
    max_Npeaks=6,
    min_Npeaks=2,
    minTS=6.0,
    nllh_name="nllh",
):
    tgrps = res_df.groupby("timeID")

    peak_dfs = []

    for timeID, df_ in tgrps:
        if np.nanmax(df_["TS"]) < minTS:
            continue

        df = df_.sort_values(nllh_name)
        vals = df[nllh_name]
        #         ind_sort = np.argsort(vals)
        min_val = np.nanmin(df[nllh_name])

        peak_dict = {
            "timeID": int(timeID),
            "time": np.nanmean(df["time"]),
            "dur": np.nanmean(df["dur"]),
        }

        imxs_ = np.empty(0)
        imys_ = np.empty_like(imxs_)
        As_ = np.empty_like(imxs_)
        Eps_ = np.empty_like(imxs_)
        Gs_ = np.empty_like(imxs_)
        bkg_nllhs = np.empty_like(imxs_)
        nllhs = np.empty_like(imxs_)
        TSs = np.empty_like(imxs_)

        for row_ind, row in df.iterrows():
            if row[nllh_name] > (min_val + max_dv) and len(imxs_) >= min_Npeaks:
                break
            if len(imxs_) >= max_Npeaks:
                break

            if len(imxs_) > 0:
                imdist = np.min(im_dist(row["imx"], row["imy"], imxs_, imys_))
                if imdist <= min_sep:
                    continue

            imxs_ = np.append(imxs_, [row["imx"]])
            imys_ = np.append(imys_, [row["imy"]])
            As_ = np.append(As_, [row["A"]])
            Eps_ = np.append(Eps_, [row["Epeak"]])
            Gs_ = np.append(Gs_, [row["gamma"]])
            bkg_nllhs = np.append(bkg_nllhs, [row["bkg_nllh"]])
            nllhs = np.append(nllhs, [row["nllh"]])
            TSs = np.append(TSs, [row["TS"]])

        peak_dict["imx"] = imxs_
        peak_dict["imy"] = imys_
        peak_dict["A"] = As_
        peak_dict["Epeak"] = Eps_
        peak_dict["gamma"] = Gs_
        peak_dict["nllh"] = nllhs
        peak_dict["bkg_nllh"] = bkg_nllhs
        peak_dict["TS"] = TSs
        peak_dfs.append(pd.DataFrame(peak_dict))

    peaks_df = pd.concat(peak_dfs, ignore_index=True)

    return peaks_df


def do_scan_around_peak(
    peak_row,
    bkg_bf_params,
    bkg_name,
    sig_miner,
    sig_llh_obj,
    sig_mod,
    imstep=2e-3,
    dimx=2e-3,
    dimy=2e-3,
    dgamma=0.2,
    dlog10Ep=0.2,
    gam_steps=3,
    Ep_steps=3,
):
    flux_params = {"A": 1.0, "Epeak": 150.0, "gamma": 0.5}

    t1 = peak_row["time"] + peak_row["dur"]
    sig_llh_obj.set_time(peak_row["time"], t1)

    parss = {}
    for pname, val in bkg_bf_params.items():
        # pars_['Background_'+pname] = val
        parss[bkg_name + "_" + pname] = val
    sig_miner.set_fixed_params(list(parss.keys()), values=list(parss.values()))

    imxax = np.arange(-dimx, dimx + (imstep / 2.0), imstep) + peak_row["imx"]
    imyax = np.arange(-dimy, dimy + (imstep / 2.0), imstep) + peak_row["imy"]

    imxg, imyg = np.meshgrid(imxax, imyax)
    imxs = imxg.ravel()
    imys = imyg.ravel()
    thetas, phis = imxy2theta_phi(imxs, imys)

    N_impnts = len(imxs)

    logging.info("N_impnts: ", N_impnts)

    Epeak_ax = np.logspace(
        np.log10(peak_row["Epeak"]) - dlog10Ep,
        np.log10(peak_row["Epeak"]) + dlog10Ep,
        Ep_steps,
    )
    gamma_ax = np.linspace(
        peak_row["gamma"] - dgamma, peak_row["gamma"] + dgamma, gam_steps
    )

    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    Nspec_pnts = len(Epeaks)

    logging.info("Nspec_pnts: ", Nspec_pnts)

    logging.info("Epeak_ax: ")
    logging.info(Epeak_ax)
    logging.info("gammas_ax: ")
    logging.info(gamma_ax)

    res_dfs = []

    for ii in range(N_impnts):
        print(imxs[ii], imys[ii])
        print(thetas[ii], phis[ii])
        sig_miner.set_fixed_params(
            ["Signal_theta", "Signal_phi"], values=[thetas[ii], phis[ii]]
        )

        res_dict = {}
        res_dict["imx"] = imxs[ii]
        res_dict["imy"] = imys[ii]
        res_dict["theta"] = thetas[ii]
        res_dict["phi"] = phis[ii]
        res_dict["Epeak"] = Epeaks
        res_dict["gamma"] = gammas
        res_dict["time"] = peak_row["time"]
        res_dict["dur"] = peak_row["dur"]
        res_dict["timeID"] = peak_row["timeID"]
        res_dict["bkg_nllh"] = peak_row["bkg_nllh"]

        nllhs = np.zeros(Nspec_pnts)
        As = np.zeros(Nspec_pnts)

        for jj in range(Nspec_pnts):
            flux_params["gamma"] = gammas[jj]
            flux_params["Epeak"] = Epeaks[jj]
            sig_mod.set_flux_params(flux_params)

            try:
                pars, nllh, res = sig_miner.minimize()
                As[jj] = pars[0][0]
                nllhs[jj] = nllh[0]
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Failed to minimize seed: ")
                logging.error((imxs[ii], imys[ii]))
                nllhs[jj] = np.nan

        res_dict["nllh"] = nllhs
        res_dict["A"] = As
        res_dict["TS"] = np.sqrt(2 * (res_dict["bkg_nllh"] - nllhs))

        res_dfs.append(pd.DataFrame(res_dict))

    res_df = pd.concat(res_dfs, ignore_index=True)
    res_df["TS"][np.isnan(res_df["TS"])] = 0.0

    return res_df


def analysis_for_imxy_square(
    imx0,
    imx1,
    imy0,
    imy1,
    bkg_bf_params_list,
    bkg_mod,
    flux_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
    timeIDs,
    TS2keep=4.5,
    max_frac2keep=0.75,
):
    bl_dmask = bkg_mod.bl_dmask

    # dimxy = 0.0025
    dimxy = np.round(imx1 - imx0, decimals=4)
    imstep = 0.003
    imxstep = 0.004

    # imx_ax = np.arange(imx0, imx1+dimxy/2., dimxy)
    # imy_ax = np.arange(imy0, imy1+dimxy/2., dimxy)
    # imxg,imyg = np.meshgrid(imx_ax, imy_ax)

    # imx_ax = np.arange(imx0, imx1, imxstep)
    # imy_ax = np.arange(imy0, imy1, imstep)
    imx_ax = np.arange(0, dimxy, imxstep)
    imy_ax = np.arange(0, dimxy, imstep)
    imxg, imyg = np.meshgrid(imx_ax, imy_ax)
    bl = np.isclose((imyg * 1e4).astype(np.int64) % int(imstep * 2 * 1e4), 0)
    imxg[bl] += imxstep / 2.0
    imxs = np.ravel(imxg) + imx0
    imys = np.ravel(imyg) + imy0
    Npnts = len(imxs)

    print(Npnts)
    logging.info("%d imxy points to do" % (Npnts))

    thetas, phis = imxy2theta_phi(imxs, imys)

    gamma_ax = np.linspace(-0.4, 1.6, 8 + 1)
    gamma_ax = np.linspace(-0.4, 1.6, 4 + 1)[1:-1]
    # gamma_ax = np.array([0.4, 0.9])
    #     gamma_ax = np.linspace(-0.4, 1.6, 3+1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 10 + 1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 5 + 1)[1:-1]
    Epeak_ax = np.logspace(np.log10(45.0), 3, 4 + 1)[1:-1]
    #     Epeak_ax = np.logspace(np.log10(45.0), 3, 5+1)[3:]
    logging.info("Epeak_ax: ")
    logging.info(Epeak_ax)
    logging.info("gammas_ax: ")
    logging.info(gamma_ax)
    #     Epeak_ax = np.logspace(np.log10(25.0), 3, 3+1)
    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    Nspec_pnts = len(Epeaks)
    ntbins = len(tbins0)

    rt_obj = RayTraces(rt_dir)
    # fp_obj = FootPrints(fp_dir)

    sig_mod = Source_Model_InOutFoV(
        flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True
    )
    sig_mod.set_theta_phi(np.mean(thetas), np.mean(phis))

    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    tmin = np.min(tbins0)
    tmax = np.max(tbins1)
    if (tmax - tmin) > 40.0:
        logging.debug("tmax - tmin > 40.0s, using twinds for tbl")
        gti_dict = {"START": tbins0, "STOP": tbins1}
        gti_twinds = Table(data=gti_dict)
        gtis = union_gtis([gti_twinds])
        tbl = mk_gti_bl(ev_data["TIME"], gtis, time_pad=0.1)
        logging.debug("np.sum(tbl): %d" % (np.sum(tbl)))
    else:
        tbl = (ev_data["TIME"] >= (tmin - 1.0)) & (ev_data["TIME"] < (tmax + 1.0))
    logging.debug("np.sum(tbl): %d" % (np.sum(tbl)))
    sig_llh_obj = LLH_webins(ev_data[tbl], ebins0, ebins1, bl_dmask, has_err=True)

    sig_llh_obj.set_model(comp_mod)

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}

    bkg_name = bkg_mod.name

    pars_ = {}
    pars_["Signal_theta"] = np.mean(thetas)
    pars_["Signal_phi"] = np.mean(phis)
    for pname, val in bkg_bf_params_list[0].items():
        # pars_['Background_'+pname] = val
        pars_[bkg_name + "_" + pname] = val
    for pname, val in flux_params.items():
        pars_["Signal_" + pname] = val

    sig_miner.set_llh(sig_llh_obj)

    fixed_pnames = list(pars_.keys())
    fixed_vals = list(pars_.values())
    trans = [None for i in range(len(fixed_pnames))]
    sig_miner.set_trans(fixed_pnames, trans)
    sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    sig_miner.set_fixed_params(["Signal_A"], fixed=False)

    res_dfs_ = []

    for ii in range(Npnts):
        print(imxs[ii], imys[ii])
        print(thetas[ii], phis[ii])
        sig_miner.set_fixed_params(
            ["Signal_theta", "Signal_phi"], values=[thetas[ii], phis[ii]]
        )

        res_dfs = []

        for j in range(Nspec_pnts):
            flux_params["gamma"] = gammas[j]
            flux_params["Epeak"] = Epeaks[j]
            sig_mod.set_flux_params(flux_params)

            res_dict = {}

            res_dict["Epeak"] = Epeaks[j]
            res_dict["gamma"] = gammas[j]

            nllhs = np.zeros(ntbins)
            As = np.zeros(ntbins)

            for i in range(ntbins):
                parss_ = {}
                for pname, val in bkg_bf_params_list[i].items():
                    # pars_['Background_'+pname] = val
                    parss_[bkg_name + "_" + pname] = val
                sig_miner.set_fixed_params(
                    list(parss_.keys()), values=list(parss_.values())
                )

                t0 = tbins0[i]
                t1 = tbins1[i]
                dt = t1 - t0
                sig_llh_obj.set_time(tbins0[i], tbins1[i])

                try:
                    pars, nllh, res = sig_miner.minimize()
                    As[i] = pars[0][0]
                    nllhs[i] = nllh[0]
                except Exception as E:
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    logging.error("Failed to minimize seed: ")
                    logging.error((imxs[ii], imys[ii]))
                    logging.error((timeIDs[i]))
                    nllhs[i] = np.nan

                # print "res: "
                # print res

            res_dict["nllh"] = nllhs
            res_dict["A"] = As
            res_dict["time"] = np.array(tbins0)
            res_dict["dur"] = np.array(tbins1) - np.array(tbins0)
            res_dict["timeID"] = np.array(timeIDs)

            res_dict["theta"] = thetas[ii]
            res_dict["phi"] = phis[ii]
            res_dict["imx"] = imxs[ii]
            res_dict["imy"] = imys[ii]

            res_dfs.append(pd.DataFrame(res_dict))

            # logging.info("Done with spec %d of %d" %(j+1,Nspec_pnts))

        res_df = pd.concat(res_dfs, ignore_index=True)
        bkg_nllhs = np.zeros(len(res_df))

        bkg_bf_param_dict = {}

        for i in range(ntbins):
            t0 = tbins0[i]
            t1 = tbins1[i]
            dt = t1 - t0
            sig_llh_obj.set_time(tbins0[i], tbins1[i])
            for pname, val in bkg_bf_params_list[i].items():
                pars_[bkg_name + "_" + pname] = val
            bkg_bf_param_dict[timeIDs[i]] = bkg_bf_params_list[i]
            pars_["Signal_theta"] = thetas[ii]
            pars_["Signal_phi"] = phis[ii]
            pars_["Signal_A"] = 1e-10
            bkg_nllh = -sig_llh_obj.get_logprob(pars_)
            bl = np.isclose(res_df["time"] - t0, t0 - t0) & np.isclose(
                res_df["dur"], dt
            )
            bkg_nllhs[bl] = bkg_nllh

        # pars_['Signal_A'] = 1e-10
        # bkg_nllh = -sig_llh_obj.get_logprob(pars_)

        res_df["bkg_nllh"] = bkg_nllhs
        res_df["TS"] = np.sqrt(2.0 * (bkg_nllhs - res_df["nllh"]))
        res_df["TS"][np.isnan(res_df["TS"])] = 0.0

        res_dfs_.append(res_df)

        logging.info("Done with imxy %d of %d" % (ii + 1, Npnts))

    res_df = pd.concat(res_dfs_, ignore_index=True)

    TSbl = res_df["TS"] >= TS2keep
    if np.sum(TSbl) > (len(res_df) / 5.0):
        TSwrite_ = np.nanpercentile(res_df["TS"], max_frac2keep * 100.0)
        TSbl = res_df["TS"] >= TSwrite_
    elif np.sum(TSbl) < 1:
        TSbl = np.isclose(res_df["TS"], np.max(res_df["TS"]))
    else:
        TSbl = res_df["TS"] >= TS2keep
    res_df = res_df[TSbl]

    minTS2scan = 6.0
    if np.max(res_df["TS"]) >= minTS2scan:
        peaks_df = find_peaks2scan(res_df)
        Npeaks2scan = len(peaks_df)
    else:
        Npeaks2scan = 0
    logging.info("%d peaks to scan" % (Npeaks2scan))

    if Npeaks2scan > 0:
        peak_res_dfs = []

        for peak_ind, peak_row in peaks_df.iterrows():
            bkg_bf_params = bkg_bf_param_dict[peak_row["timeID"]]

            logging.info("Starting to scan peak_row")
            logging.info(peak_row)

            df = do_scan_around_peak(
                peak_row, bkg_bf_params, bkg_name, sig_miner, sig_llh_obj, sig_mod
            )

            max_peak_row = df.loc[df["TS"].idxmax()]

            df2 = do_scan_around_peak(
                max_peak_row,
                bkg_bf_params,
                bkg_name,
                sig_miner,
                sig_llh_obj,
                sig_mod,
                imstep=1e-3,
                dimx=1e-3,
                dimy=1e-3,
                dgamma=0.1,
                dlog10Ep=0.1,
            )

            peak_res_dfs.append(df)
            peak_res_dfs.append(df2)

        peak_res_df = pd.concat(peak_res_dfs, ignore_index=True)

        return res_df, peak_res_df
    else:
        return res_df, None


def analysis_for_imxy_square(
    imxs,
    imys,
    bkg_bf_params_list,
    bkg_mod,
    flux_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
    timeIDs,
    TS2keep=None,
    max_frac2keep=0.75,
    do_scan=False,
):
    bl_dmask = bkg_mod.bl_dmask

    Npnts = len(imxs)

    logging.info("%d imxy points to do" % (Npnts))

    thetas, phis = imxy2theta_phi(imxs, imys)

    gamma_ax = np.linspace(-0.4, 1.6, 8 + 1)
    gamma_ax = np.linspace(-0.4, 1.6, 4 + 1)[1:-1]
    # gamma_ax = np.array([0.4, 0.9])
    #     gamma_ax = np.linspace(-0.4, 1.6, 3+1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 10 + 1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 5 + 1)[1:-1]
    Epeak_ax = np.logspace(np.log10(45.0), 3, 4 + 1)[1:-1]
    #     Epeak_ax = np.logspace(np.log10(45.0), 3, 5+1)[3:]
    logging.info("Epeak_ax: ")
    logging.info(Epeak_ax)
    logging.info("gammas_ax: ")
    logging.info(gamma_ax)
    #     Epeak_ax = np.logspace(np.log10(25.0), 3, 3+1)
    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    Nspec_pnts = len(Epeaks)
    ntbins = len(tbins0)

    rt_obj = RayTraces(rt_dir)
    # fp_obj = FootPrints(fp_dir)

    sig_mod = Source_Model_InOutFoV(
        flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True
    )
    sig_mod.set_theta_phi(np.mean(thetas), np.mean(phis))

    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    tmin = np.min(tbins0)
    tmax = np.max(tbins1)
    if (tmax - tmin) > 40.0:
        logging.debug("tmax - tmin > 40.0s, using twinds for tbl")
        gti_dict = {"START": tbins0, "STOP": tbins1}
        gti_twinds = Table(data=gti_dict)
        gtis = union_gtis([gti_twinds])
        tbl = mk_gti_bl(ev_data["TIME"], gtis, time_pad=0.1)
        logging.debug("np.sum(tbl): %d" % (np.sum(tbl)))
    else:
        tbl = (ev_data["TIME"] >= (tmin - 1.0)) & (ev_data["TIME"] < (tmax + 1.0))
    logging.debug("np.sum(tbl): %d" % (np.sum(tbl)))
    sig_llh_obj = LLH_webins(ev_data[tbl], ebins0, ebins1, bl_dmask, has_err=True)

    sig_llh_obj.set_model(comp_mod)

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}

    bkg_name = bkg_mod.name

    pars_ = {}
    pars_["Signal_theta"] = np.mean(thetas)
    pars_["Signal_phi"] = np.mean(phis)
    for pname, val in bkg_bf_params_list[0].items():
        # pars_['Background_'+pname] = val
        pars_[bkg_name + "_" + pname] = val
    for pname, val in flux_params.items():
        pars_["Signal_" + pname] = val

    sig_miner.set_llh(sig_llh_obj)

    fixed_pnames = list(pars_.keys())
    fixed_vals = list(pars_.values())
    trans = [None for i in range(len(fixed_pnames))]
    sig_miner.set_trans(fixed_pnames, trans)
    sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    sig_miner.set_fixed_params(["Signal_A"], fixed=False)

    res_dfs_ = []

    sig_llh_obj.set_time(tbins0[0], tbins1[0])
    parss_ = {}
    for pname, val in bkg_bf_params_list[0].items():
        # pars_['Background_'+pname] = val
        parss_[bkg_name + "_" + pname] = val
    sig_miner.set_fixed_params(list(parss_.keys()), values=list(parss_.values()))

    for ii in range(Npnts):
        sig_miner.set_fixed_params(
            ["Signal_theta", "Signal_phi"], values=[thetas[ii], phis[ii]]
        )

        res_dfs = []

        for j in range(Nspec_pnts):
            flux_params["gamma"] = gammas[j]
            flux_params["Epeak"] = Epeaks[j]
            sig_mod.set_flux_params(flux_params)

            res_dict = {}

            res_dict["Epeak"] = Epeaks[j]
            res_dict["gamma"] = gammas[j]

            nllhs = np.zeros(ntbins)
            As = np.zeros(ntbins)

            for i in range(ntbins):
                t0 = tbins0[i]
                t1 = tbins1[i]
                dt = t1 - t0

                if ntbins > 1:
                    parss_ = {}
                    for pname, val in bkg_bf_params_list[i].items():
                        # pars_['Background_'+pname] = val
                        parss_[bkg_name + "_" + pname] = val
                    sig_miner.set_fixed_params(
                        list(parss_.keys()), values=list(parss_.values())
                    )

                    sig_llh_obj.set_time(tbins0[i], tbins1[i])

                try:
                    pars, nllh, res = sig_miner.minimize()
                    As[i] = pars[0][0]
                    nllhs[i] = nllh[0]
                except Exception as E:
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    logging.error("Failed to minimize seed: ")
                    logging.error((imxs[ii], imys[ii]))
                    logging.error((timeIDs[i]))
                    nllhs[i] = np.nan

                # print "res: "
                # print res

            res_dict["nllh"] = nllhs
            res_dict["A"] = As
            res_dict["time"] = np.array(tbins0)
            res_dict["dur"] = np.array(tbins1) - np.array(tbins0)
            res_dict["timeID"] = np.array(timeIDs)

            res_dict["theta"] = thetas[ii]
            res_dict["phi"] = phis[ii]
            res_dict["imx"] = imxs[ii]
            res_dict["imy"] = imys[ii]

            res_dfs.append(pd.DataFrame(res_dict))

            # logging.info("Done with spec %d of %d" %(j+1,Nspec_pnts))

        res_df = pd.concat(res_dfs, ignore_index=True)
        bkg_nllhs = np.zeros(len(res_df))

        bkg_bf_param_dict = {}

        for i in range(ntbins):
            t0 = tbins0[i]
            t1 = tbins1[i]
            dt = t1 - t0
            sig_llh_obj.set_time(tbins0[i], tbins1[i])
            for pname, val in bkg_bf_params_list[i].items():
                pars_[bkg_name + "_" + pname] = val
            bkg_bf_param_dict[timeIDs[i]] = bkg_bf_params_list[i]
            pars_["Signal_theta"] = thetas[ii]
            pars_["Signal_phi"] = phis[ii]
            pars_["Signal_A"] = 1e-10
            bkg_nllh = -sig_llh_obj.get_logprob(pars_)
            bl = np.isclose(res_df["time"] - t0, t0 - t0) & np.isclose(
                res_df["dur"], dt
            )
            bkg_nllhs[bl] = bkg_nllh

        # pars_['Signal_A'] = 1e-10
        # bkg_nllh = -sig_llh_obj.get_logprob(pars_)

        res_df["bkg_nllh"] = bkg_nllhs
        res_df["TS"] = np.sqrt(2.0 * (bkg_nllhs - res_df["nllh"]))
        res_df["TS"][np.isnan(res_df["TS"])] = 0.0

        res_dfs_.append(res_df)

        logging.info("Done with imxy %d of %d" % (ii + 1, Npnts))

    res_df = pd.concat(res_dfs_, ignore_index=True)

    if TS2keep is not None:
        TSbl = res_df["TS"] >= TS2keep
        if np.sum(TSbl) > (len(res_df) / 5.0):
            TSwrite_ = np.nanpercentile(res_df["TS"], max_frac2keep * 100.0)
            TSbl = res_df["TS"] >= TSwrite_
        elif np.sum(TSbl) < 1:
            TSbl = np.isclose(res_df["TS"], np.max(res_df["TS"]))
        else:
            TSbl = res_df["TS"] >= TS2keep
        res_df = res_df[TSbl]

    minTS2scan = 6.0
    if np.max(res_df["TS"]) >= minTS2scan:
        peaks_df = find_peaks2scan(res_df)
        Npeaks2scan = len(peaks_df)
    else:
        Npeaks2scan = 0
    logging.info("%d peaks to scan" % (Npeaks2scan))

    if Npeaks2scan > 0 and do_scan:
        peak_res_dfs = []

        for peak_ind, peak_row in peaks_df.iterrows():
            bkg_bf_params = bkg_bf_param_dict[peak_row["timeID"]]

            logging.info("Starting to scan peak_row")
            logging.info(peak_row)

            df = do_scan_around_peak(
                peak_row, bkg_bf_params, bkg_name, sig_miner, sig_llh_obj, sig_mod
            )

            max_peak_row = df.loc[df["TS"].idxmax()]

            df2 = do_scan_around_peak(
                max_peak_row,
                bkg_bf_params,
                bkg_name,
                sig_miner,
                sig_llh_obj,
                sig_mod,
                imstep=1e-3,
                dimx=1e-3,
                dimy=1e-3,
                dgamma=0.1,
                dlog10Ep=0.1,
            )

            peak_res_dfs.append(df)
            peak_res_dfs.append(df2)

        peak_res_df = pd.concat(peak_res_dfs, ignore_index=True)

        return res_df, peak_res_df
    else:
        return res_df, None


def do_analysis(
    square_tab,
    ev_data,
    flux_mod,
    rt_dir,
    ebins0,
    ebins1,
    bl_dmask,
    trigger_time,
    work_dir,
    bkg_fname,
    pcfname,
    TSwrite=None,
    pc_min=0.01,
):
    nebins = len(ebins0)

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    job_id = np.min(square_tab["proc_group"])

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    bkg_df, bkg_name, PSnames, bkg_mod, ps_mods = parse_bkg_csv(
        bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir
    )

    bkg_mod.has_deriv = False
    bkg_mod_list = [bkg_mod]
    Nsrcs = len(ps_mods)
    if Nsrcs > 0:
        bkg_mod_list += ps_mods
        for ps_mod in ps_mods:
            ps_mod.has_deriv = False
        bkg_mod = CompoundModel(bkg_mod_list)

    PC = fits.open(pcfname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key="T")

    pcbl = pc >= pc_min
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    df_sq_grps = square_tab.groupby("squareID")

    for squareID, square_df in df_sq_grps:
        logging.info("Starting squareID: %d" % (squareID))

        rt_obj = RayTraces(rt_dir, max_nbytes=2e9)

        imx0 = np.mean(square_df["imx0"])
        imx1 = np.mean(square_df["imx1"])
        imy0 = np.mean(square_df["imy0"])
        imy1 = np.mean(square_df["imy1"])

        im_bl = (
            (pc_imxs >= imx0) & (pc_imxs < imx1) & (pc_imys >= imy0) & (pc_imys < imy1)
        )
        # Npix = np.sum(im_bl)
        # logging.debug("%d Pixels to minimize at" %(Npix))
        #
        # if Npix < 1:
        #     fname = os.path.join(work_dir,\
        #             'res_%d_.csv' %(square_row['squareID']))
        #     logging.info("Nothing to write for squareID %d"\
        #                 %(square_row['squareID']))
        #     f = open(fname, 'w')
        #     f.write('NONE')
        #     f.close()
        #     continue

        imxs = pc_imxs[im_bl]
        imys = pc_imys[im_bl]
        #
        # tab = Table()
        res_dfs2write = []

        # bl = (rate_res_tab['squareID']==square_row['squareID'])

        logging.info("%d timeIDs to do" % (len(square_df)))
        logging.info("%d pix to do" % (len(imxs)))

        t0s = []
        t1s = []
        timeIDs = []
        bkg_params_list = []

        for sq_ind, sq_row in square_df.iterrows():
            t0s.append(sq_row["time"])
            t1s.append(sq_row["time"] + sq_row["dur"])
            timeIDs.append(sq_row["timeID"])
            tmid = sq_row["time"] + (sq_row["dur"] / 2.0)
            bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
            bkg_params = {pname: bkg_row[pname] for pname in bkg_mod.param_names}
            bkg_params_list.append(bkg_params)

        try:
            # res_df, peak_res_df = analysis_for_imxy_square(imx0, imx1, imy0, imy1,\
            #                             bkg_params_list,\
            #                             bkg_mod, flux_mod, ev_data,\
            #                             ebins0, ebins1, t0s, t1s, timeIDs)
            res_df, peak_res_df = analysis_for_imxy_square(
                imxs,
                imys,
                bkg_params_list,
                bkg_mod,
                flux_mod,
                ev_data,
                ebins0,
                ebins1,
                t0s,
                t1s,
                timeIDs,
            )

            res_df["squareID"] = squareID

            fname = os.path.join(work_dir, "res_%d_%d_.csv" % (squareID, job_id))
            fname = "res_%d_%d_.csv" % (squareID, job_id)

            res_df.to_csv(fname)
            logging.info("Saved results to")
            logging.info(fname)

            if peak_res_df is not None:
                peak_res_df["squareID"] = squareID
                fname = os.path.join(
                    work_dir, "peak_res_%d_%d_.csv" % (squareID, job_id)
                )
                fname = "peak_res_%d_%d_.csv" % (squareID, job_id)

                peak_res_df.to_csv(fname)
                logging.info("Saved peak results to")
                logging.info(fname)

        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.warn("Messed up with squareID %d" % (squareID))
            fname = "res_%d_%d_.csv" % (squareID, job_id)

            f = open(fname, "w")
            f.write("NONE")
            f.close()


def main(args):
    # fname = 'llh_analysis_from_rate_seeds_' + str(args.job_id)
    fname = args.log_fname + "_" + str(args.job_id)

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    t_0 = time.time()

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB")
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    trigtime = info_tab["trigtimeMET"][0]

    evfname = files_tab["evfname"][0]
    ev_data = fits.open(evfname)[1].data
    dmask_fname = files_tab["detmask"][0]
    dmask = fits.open(dmask_fname)[0].data
    bl_dmask = dmask == 0.0
    logging.debug("Opened up event and detmask files")

    bkg_fits_df = pd.read_csv(args.bkg_fname)

    # rate_fits_df = get_rate_fits_tab(conn)
    # bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    time_starting = time.time()
    proc_num = args.job_id
    # init classes up here

    # drm_dir = files_tab['drmDir'][0]
    # if args.rt_dir is None:
    #     rt_dir = files_tab['rtDir'][0]
    # else:
    #     rt_dir = args.rt_dir
    # drm_obj = DRMs(drm_dir)
    # rt_obj = RayTraces(rt_dir, max_nbytes=1e10)
    work_dir = files_tab["workDir"][0]
    conn.close()

    # pl_flux = Plaw_Flux()
    flux_mod = Cutoff_Plaw_Flux(E0=100.0)

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    # sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    # try:
    #     good_pix = np.load(args.pix_fname)
    # except Exception as E:
    #     logging.error(E)
    #     logging.warning("No pix2scan file")

    # PC = fits.open(args.pcfname)[0]
    # pc = PC.data
    # w_t = WCS(PC.header, key='T')
    #
    # pcbl = (pc>=args.min_pc)
    # pc_inds = np.where(pcbl)
    # pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)
    # logging.debug("Min pc_imx, pc_imy: %.2f, %.2f" %(np.nanmin(pc_imxs), np.nanmin(pc_imys)))
    # logging.debug("Max pc_imx, pc_imy: %.2f, %.2f" %(np.nanmax(pc_imxs), np.nanmax(pc_imys)))

    # conn = get_conn(db_fname)
    # if proc_num >= 0:
    #     square_tab = get_square_tab(conn, proc_group=proc_num)
    # else:
    #     square_tab = get_square_tab(conn)

    square_tab = pd.read_csv(args.job_fname)
    if proc_num >= 0:
        bl = square_tab["proc_group"] == proc_num
    elif args.square_id >= 0:
        bl = square_tab["squareID"] == args.square_id
    else:
        bl = np.ones(len(square_tab), dtype=bool)
    square_tab = square_tab[bl]

    logging.info("Read in Square Seed Table, now to do analysis")

    do_analysis(
        square_tab,
        ev_data,
        flux_mod,
        rt_dir,
        ebins0,
        ebins1,
        bl_dmask,
        trigtime,
        work_dir,
        args.bkg_fname,
        args.pcfname,
    )
    # do_analysis(square_tab, rate_res_tab, good_pix['imx'], good_pix['imy'], pl_flux,\
    #                 drm_obj, rt_dir,\
    #                 bkg_llh_obj, sig_llh_obj,\
    #                 conn, db_fname, trigtime, work_dir,bkg_fits_df)


if __name__ == "__main__":
    args = cli()

    main(args)
