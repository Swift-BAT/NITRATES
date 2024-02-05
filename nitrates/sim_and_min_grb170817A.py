import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import time
import pandas as pd
from copy import copy, deepcopy


from config import solid_angle_dpi_fname, rt_dir
from lib.sqlite_funcs import get_conn, make_timeIDs
from lib.wcs_funcs import world2val
from lib.event2dpi_funcs import det2dpis, mask_detxy
from models.models import (
    Bkg_Model_wFlatA,
    CompoundModel,
    im_dist,
    Point_Source_Model_Binned_Rates,
    Source_Model_InOutFoV,
)
from models.flux_models import Plaw_Flux, Cutoff_Plaw_Flux
from lib.gti_funcs import add_bti2gti, bti2gti, gti2bti, union_gtis
from response.ray_trace_funcs import RayTraces
from lib.coord_conv_funcs import (
    convert_radec2imxy,
    convert_imxy2radec,
    imxy2theta_phi,
    theta_phi2imxy,
)
from llh_analysis.LLH import LLH_webins
from llh_analysis.minimizers import NLLH_ScipyMinimize_Wjacob, NLLH_ScipyMinimize
from llh_analysis.do_llh_inFoV4realtime2 import (
    do_scan_around_peak,
    find_peaks2scan,
    parse_bkg_csv,
)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evfname", type=str, help="Event data file", default="filter_evdata.fits"
    )
    parser.add_argument(
        "--dmask", type=str, help="Detmask fname", default="detmask.fits"
    )
    parser.add_argument(
        "--job_id", type=int, help="ID to tell it what seeds to do", default=-1
    )
    parser.add_argument(
        "--Njobs", type=int, help="Total number of jobs submitted", default=64
    )
    parser.add_argument(
        "--Ntrials", type=int, help="Number of trials to run", default=8
    )
    parser.add_argument("--Afact", type=float, help="A factor to use", default=1.0)
    parser.add_argument("--theta", type=float, help="Theta to sim at", default=0.0)
    parser.add_argument("--phi", type=float, help="phi to sim at", default=0.0)
    parser.add_argument(
        "--trig_time", type=float, help="trigger time to center sims at", default=0.0
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default="pc_2.img"
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--log_fname", type=str, help="Name for the log file", default="sim_and_min"
    )
    args = parser.parse_args()
    return args


def mk_sim_evdata(sim_mod, sim_params, dur, tstart):
    # first make sim DPIs
    # then make an event for each count
    # so each event will have the DETX, DETY and
    # it can just be given the energy of middle of the ebin
    # then can assign times with just a uniform distribution
    # from tstart to tstop
    # then sort the events and return the Table

    col_names = ["TIME", "DET_ID", "EVENT_FLAGS", "PHA", "DETX", "DETY", "PI", "ENERGY"]
    # only need to assign, TIME, DETX, and DETY
    # make all flags 0, and the rest don't matter
    tab = Table(names=["DETX", "DETY", "ENERGY"], dtype=(np.int64, np.int64, np.float64))

    ebins0 = sim_mod.ebins0
    ebins1 = sim_mod.ebins1

    rate_dpis = sim_mod.get_rate_dpis(sim_params)

    sim_dpis = np.random.poisson(lam=(rate_dpis * dur))

    for ebin, sim_dpi in enumerate(sim_dpis):
        simdpi = np.zeros_like(sim_mod.bl_dmask, dtype=np.int64)
        simdpi[sim_mod.bl_dmask] = sim_dpi
        detys, detxs = np.where(simdpi > 0)
        emid = (ebins1[ebin] + ebins0[ebin]) / 2.0
        for jj in range(len(detys)):
            dety = detys[jj]
            detx = detxs[jj]
            for ii in range(simdpi[dety, detx]):
                row = (detx, dety, emid)
                tab.add_row(row)

    tab["TIME"] = dur * np.random.random(size=len(tab)) + tstart
    tab["DET_ID"] = np.zeros(len(tab), dtype=np.int64)
    tab["PHA"] = np.ones(len(tab), dtype=np.int64)
    tab["EVENT_FLAGS"] = np.zeros(len(tab), dtype=np.int64)
    tab["PI"] = np.rint(tab["ENERGY"] * 10).astype(np.int64)
    tab.sort(keys="TIME")
    return tab


def analysis_for_imxy_square(
    imx0,
    imx1,
    imy0,
    imy1,
    bkg_bf_params_list,
    bkg_mod,
    sig_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
    timeIDs,
    TS2keep=4.5,
    max_frac2keep=0.75,
    minTS2scan=6.0,
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

    #     rt_obj = RayTraces(rt_dir)
    # fp_obj = FootPrints(fp_dir)

    #     sig_mod = Source_Model_InOutFoV(flux_mod, [ebins0,ebins1], bl_dmask,\
    #                                     rt_obj, use_deriv=True)
    #     sig_mod.set_theta_phi(np.mean(thetas), np.mean(phis))

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

    # minTS2scan = 6.0
    if np.max(res_df["TS"]) >= minTS2scan:
        peaks_df = find_peaks2scan(res_df, minTS=minTS2scan)
        Npeaks2scan = len(peaks_df)
    else:
        Npeaks2scan = 0
    logging.info("%d peaks to scan" % (Npeaks2scan))

    print(list(bkg_bf_param_dict.keys()))

    if Npeaks2scan > 0:
        peak_res_dfs = []

        for peak_ind, peak_row in peaks_df.iterrows():
            bkg_bf_params = bkg_bf_param_dict[str(int(peak_row["timeID"]))]

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


def min_sim_llhs(
    ev_data,
    sim_evdata,
    sim_params,
    sim_tstart,
    sim_tstop,
    bkg_fname,
    solid_ang_dpi,
    bl_dmask,
    rt_dir,
    sig_mod,
    ebins0,
    ebins1,
    durs=[0.256, 0.512, 1.024],
):
    trig_time = sim_tstart

    imx_sim, imy_sim = sim_params["imx"], sim_params["imy"]

    imx_cent = np.round(imx_sim + 2e-3 * (np.random.random() - 0.5) * 2, decimals=3)
    imy_cent = np.round(imy_sim + 2e-3 * (np.random.random() - 0.5) * 2, decimals=3)

    dimxy = 0.016
    imx0 = imx_cent - dimxy / 2.0
    imx1 = imx_cent + dimxy / 2.0
    imy0 = imy_cent - dimxy / 2.0
    imy1 = imy_cent + dimxy / 2.0

    evdata_ = ev_data.copy()
    print((type(evdata_)))
    print((type(sim_evdata)))
    evdata = vstack([evdata_, sim_evdata])
    evdata.sort("TIME")

    bkg_df = pd.read_csv(bkg_fname)

    bkg_params_list = []

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

    tbins0_ = []
    tbins1_ = []

    for dur in durs:
        tstep = dur / 4.0
        tbins0 = np.arange(
            np.round(np.min(evdata["TIME"])), np.max(evdata["TIME"]), tstep
        )
        tbins1 = tbins0 + dur

        bl = (tbins0 < (sim_tstop - (tstep))) & (tbins1 > (sim_tstart + (tstep)))
        tbins0 = tbins0[bl]
        tbins1 = tbins1[bl]
        ntbins = np.sum(bl)
        print((ntbins, " tbins to do for dur", dur))

        for i in range(ntbins):
            res_dict = {}
            t0 = tbins0[i]
            t1 = tbins1[i]
            tmid = (t0 + t1) / 2.0
            tbins0_.append(t0)
            tbins1_.append(t1)

            bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
            bkg_params = {pname: bkg_row[pname] for pname in bkg_mod.param_names}
            bkg_params_list.append(bkg_params)

    tbins0 = np.array(tbins0_)
    tbins1 = np.array(tbins1_)
    timeIDs = make_timeIDs(tbins0, tbins1 - tbins0, trig_time)

    res_df, res_peak_df = analysis_for_imxy_square(
        imx0,
        imx1,
        imy0,
        imy1,
        bkg_params_list,
        bkg_mod,
        sig_mod,
        evdata,
        ebins0,
        ebins1,
        tbins0,
        tbins1,
        timeIDs,
    )

    return res_df, res_peak_df

    # imx0, imx1, imy0, imy1, bkg_bf_params_list,\


#                             bkg_mod, sig_mod, ev_data,\
#                             ebins0, ebins1, tbins0, tbins1,\
#                             timeIDs


def sim_and_min(
    trigtime,
    lc_tab,
    sim_mod,
    sim_params,
    Afact,
    ev_data,
    bkg_fname,
    solid_ang_dpi,
    bl_dmask,
    rt_dir,
    sig_mod,
    ebins0,
    ebins1,
    Ntrials=16,
):
    res_list = []
    durs = [0.256, 0.512, 1.024]
    imx_sim, imy_sim = sim_params["imx"], sim_params["imy"]
    sim_params = deepcopy(sim_params)
    sim_params["imx0"] = imx_sim
    sim_params["imy0"] = imy_sim
    for i in range(Ntrials):
        print(("starting trial %d of %d" % (i + 1, Ntrials)))
        tstart = trigtime + (np.random.random() - 0.5) * 20.0

        imx0 = imx_sim + 2e-2 * (np.random.random() - 0.5) * 2
        imy0 = imy_sim + 2e-2 * (np.random.random() - 0.5) * 2
        sim_params["imx"] = imx0
        sim_params["imy"] = imy0
        sim_params["theta"], sim_params["phi"] = imxy2theta_phi(imx0, imy0)

        sim_tab = mk_sim_from_lc(lc_tab, sim_mod, sim_params, tstart, Afact=Afact)
        print((len(sim_tab), "sim events"))
        sim_tstart = tstart + lc_tab["time"][0]
        sim_tstop = sim_tstart + (lc_tab["time"][-1] - lc_tab["time"][0])
        #         print("tstart - trigtime: ", tstart - trigtime)
        #         print("sim_tstart - trigtime: ", sim_tstart - trigtime)
        #         print("sim_tstop - trigtime: ", sim_tstop - trigtime)
        res_df, res_peak_df = min_sim_llhs(
            ev_data,
            sim_tab,
            sim_params,
            sim_tstart,
            sim_tstop,
            bkg_fname,
            solid_ang_dpi,
            bl_dmask,
            rt_dir,
            sig_mod,
            ebins0,
            ebins1,
            durs=durs,
        )

        res_dict = {"imx_sim": sim_params["imx"], "imy_sim": sim_params["imy"]}
        if not res_peak_df is None:
            idx = res_peak_df.TS.argmax()
            max_row = res_peak_df.loc[idx]
            for k, val in max_row.items():
                res_dict[k] = val
            res_dict["dt"] = max_row["time"] - sim_tstart
        else:
            res_dict["TS"] = 0.0
        res_dict["Trial"] = i
        res_list.append(res_dict)

    res_df = pd.DataFrame(res_list)
    #         print(len(res_df), " result rows")
    print(("max TS: ", np.nanmax(res_df["TS"])))
    return res_df


def mk_sim_from_lc(lc_tab, sim_mod, sim_params, tstart, lc_tstep=0.128, Afact=1.0):
    sim_tabs = []
    sim_mod.set_theta_phi(sim_params["theta"], sim_params["phi"])
    sim_mod.set_flux_params(sim_params)
    for row in lc_tab:
        sim_params_ = copy(sim_params)
        sim_params_["A"] = Afact * row["A"]
        sim_tabs.append(
            mk_sim_evdata(sim_mod, sim_params_, lc_tstep, tstart + row["time"])
        )
    sim_tab = vstack(sim_tabs)
    return sim_tab


def get_eflux_from_model(flux_mod, params, E0, E1, esteps=1e4):
    Es = np.linspace(E0, E1, int(esteps))
    dE = Es[1] - Es[0]
    kev2erg = 1.60218e-9
    flux = np.sum(flux_mod.spec(Es, params) * Es) * dE * kev2erg
    return flux


def get_fluxA_from_eflux_model(flux_mod, params, eflux, E0, E1, esteps=1e4):
    params_ = copy(params)
    params_["A"] = 1.0
    flux1 = get_eflux_from_model(flux_mod, params_, E0, E1)
    return eflux / flux1


def main(args):
    fname = (
        args.log_fname
        + "_"
        + str(args.job_id)
        + "_"
        + str(args.Afact)
        + "_"
        + str(args.theta)
    )

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    light_curve = Table.read(
        "/storage/work/jjd330/local/bat_data/GRB170817A_stuff/gbm_lc.txt",
        format="ascii",
        names=["time", "flux"],
    )

    trigger_time = args.trig_time

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)
    print(nebins)

    ev_data = Table.read(args.evfname)
    dmask = fits.open(args.dmask)[0].data

    t_end = trigger_time + 1e3
    t_start = trigger_time - 1e3
    mask_vals = mask_detxy(dmask, ev_data)
    bl_dmask = dmask == 0.0

    bl_ev = (
        (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] <= 500.0)
        & (ev_data["ENERGY"] >= 14.0)
        & (mask_vals == 0.0)
        & (ev_data["TIME"] <= t_end)
        & (ev_data["TIME"] >= t_start)
    )

    print(np.sum(bl_ev))
    ev_data0 = ev_data[bl_ev]

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    theta = args.theta
    phi = args.phi
    imx, imy = theta_phi2imxy(theta, phi)
    print(imx, imy)
    sim_flux_mod = Cutoff_Plaw_Flux(E0=100.0)
    rt_obj = RayTraces(rt_dir)

    sim_mod = Source_Model_InOutFoV(
        sim_flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True
    )

    flux_mod = Cutoff_Plaw_Flux(E0=100.0)
    sig_mod = Source_Model_InOutFoV(
        flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True
    )

    sim_flux_params = {
        "theta": theta,
        "phi": phi,
        "A": 1.4922e-2,
        "gamma": 0.62,
        "Epeak": 185.0,
        "imx": imx,
        "imy": imy,
    }

    sim_params = {pname: val["val"] for pname, val in sim_mod.param_dict.items()}
    for pname, val in sim_flux_params.items():
        sim_params[pname] = val

    sim_mod.set_theta_phi(theta, phi)
    sim_mod.set_flux_params(sim_flux_params)

    sig_mod.set_theta_phi(theta, phi)
    sig_mod.set_flux_params(sim_flux_params)

    BAT_As = []
    for row in light_curve:
        BAT_As.append(
            get_fluxA_from_eflux_model(
                sim_flux_mod, sim_flux_params, row["flux"], 10.0, 300.0
            )
        )
    light_curve["A"] = BAT_As
    ind0 = np.argmin(np.abs(light_curve["time"] - -0.4))
    ind1 = np.argmin(np.abs(light_curve["time"] - 0.5))
    lc_tab = light_curve[ind0:ind1].copy()

    Afact = args.Afact
    Ntrials = args.Ntrials
    jobid = args.job_id

    bkg_fname = "bkg_estimation.csv"

    fname = (
        "/storage/work/jjd330/local/bat_data/GRB170817A_stuff/grb170817A_sim_theta_%.2f_phi_%.2f_Afact_%.3f_jobid_%d_.csv"
        % (theta, phi, Afact, jobid)
    )
    logging.info("save fname")
    logging.info(fname)

    df = sim_and_min(
        trigger_time,
        lc_tab,
        sim_mod,
        sim_params,
        Afact,
        ev_data0,
        bkg_fname,
        solid_ang_dpi,
        bl_dmask,
        rt_dir,
        sig_mod,
        ebins0,
        ebins1,
        Ntrials=Ntrials,
    )

    df["Afact"] = np.ones(len(df)) * Afact
    df["jobID"] = np.ones(len(df), dtype=np.int64) * jobid

    df.to_csv(fname)


if __name__ == "__main__":
    args = cli()

    main(args)
