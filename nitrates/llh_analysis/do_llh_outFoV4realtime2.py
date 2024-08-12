import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import time
import pandas as pd
import gc
from copy import copy, deepcopy

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
from ..llh_analysis.LLH import LLH_webins2

# from ..models.models import Bkg_Model_wSA, Point_Source_Model, Point_Source_Model_Wuncoded,\
#             CompoundModel, Bkg_Model_wFlatA, Point_Source_Model_Binned_Rates
# from do_intllh_scan import kum_mode, kum_pdf, kum_logpdf, kum_deriv_logpdf, deriv2_kum_logpdf
# from do_InFoV_scan3 import Swift_Mask_Interactions, Source_Model_InFoV, Bkg_Model_wFlatA,\
#                             CompoundModel, Point_Source_Model_Binned_Rates,\
#                             theta_phi2imxy, bldmask2batxys, imxy2theta_phi,\
#                             get_fixture_struct, LLH_webins
# from do_OutFoV_scan2 import Source_Model_OutFoV
from ..models.models import (
    CompoundModel,
    Point_Source_Model_Binned_Rates,
    Sig_Bkg_Model,
    Bkg_Model_wFlatA,
    Source_Model_InFoV,
    Source_Model_InOutFoV,
)
from ..lib.coord_conv_funcs import theta_phi2imxy, imxy2theta_phi
from ..llh_analysis.do_manage2 import get_out_res_fnames

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
        default="out_job_table.csv",
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--log_fname",
        type=str,
        help="Name for the log file",
        default="llh_analysis_out_FoV",
    )
    args = parser.parse_args()
    return args


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


def get_new_Epeaks_gammas2scan(
    nllhs, Epeaks_done, gammas_done, dgamma=0.2, dlog10Ep=0.2, gam_steps=3, Ep_steps=3
):
    min_ind = np.nanargmin(nllhs)
    Epeak_bf = Epeaks_done[min_ind]
    gamma_bf = gammas_done[min_ind]

    Epeak_ax = np.logspace(
        np.log10(Epeak_bf) - dlog10Ep, np.log10(Epeak_bf) + dlog10Ep, Ep_steps
    )
    gamma_ax = np.linspace(gamma_bf - dgamma, gamma_bf + dgamma, gam_steps)

    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    bl = np.ones(len(Epeaks), dtype=bool)
    for i in range(len(Epeaks)):
        bl[i] = np.any(
            ~(np.isclose(Epeaks[i], Epeaks_done) & np.isclose(gammas[i], gammas_done))
        )
    Epeaks = Epeaks[bl]
    gammas = gammas[bl]

    Nspec_pnts = len(Epeaks)

    return Epeaks, gammas


def min_at_Epeaks_gammas(sig_miner, sig_mod, sig_bkg_mod, Epeaks, gammas):
    nllhs = []
    As = []
    flux_params = {"A": 1.0, "Epeak": 150.0, "gamma": -0.25}

    Npnts = len(gammas)

    for i in range(Npnts):
        flux_params["gamma"] = gammas[i]
        flux_params["Epeak"] = Epeaks[i]
        sig_mod.set_flux_params(flux_params)

        sig_pars = copy(flux_params)
        sig_pars["A"] = 1.0
        # Below are already defined in analysis_at_theta_phi
        # sig_pars['theta'] = thetas[ii]
        # sig_pars['phi'] = phis[ii]
        sig_bkg_mod.set_sig_params(sig_pars)

        pars, nllh, res = sig_miner.minimize()
        nllhs.append(nllh[0])
        As.append(pars[0][0])
    return nllhs, As


def analysis_at_theta_phi(
    theta,
    phi,
    rt_obj,
    bkg_bf_params_list,
    bkg_mod,
    flux_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
    timeIDs,
):
    bl_dmask = bkg_mod.bl_dmask

    # sig_mod = Source_Model_OutFoV(flux_mod, [ebins0,ebins1], bl_dmask, use_deriv=True)
    sig_mod = Source_Model_InOutFoV(
        flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True, use_tube_corr=True, use_under_corr=True
    )
    # sig_mod.flor_resp_dname = '/gpfs/scratch/jjd330/bat_data/flor_resps_ebins/'
    sig_mod.set_theta_phi(theta, phi)
    print("theta, phi set")

    # comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_miner = NLLH_ScipyMinimize_Wjacob("")
    sig_llh_obj = LLH_webins2(ev_data, ebins0, ebins1, bl_dmask, has_err=True)

    # sig_llh_obj.set_model(comp_mod)

    gamma_ax = np.linspace(-0.2, 1.8, 8 + 1)
    gamma_ax = np.linspace(-0.4, 1.6, 4 + 1)[1:-1]
    gamma_ax = np.linspace(-0.2, 2.2, 4 + 1)

    Epeak_ax = np.logspace(np.log10(45.0), 3, 10)  # +1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 5 + 1)[1:-1]
    Epeak_ax = np.logspace(np.log10(45.0), 3, 4 + 1)[1:-1]
    Epeak_ax = np.logspace(1.4, 3, 2 * 2 + 1)
    #     Epeak_ax = np.logspace(np.log10(25.0), 3, 3+1)
    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    # flux_params = {'A':1.0, 'gamma':0.5, 'Epeak':1e2}

    flux_params = {"A": 1.0, "gamma": gammas[0], "Epeak": Epeaks[0]}

    sig_mod.set_flux_params(flux_params)

    bkg_name = bkg_mod.name

    pars_ = {}
    # pars_['Signal_theta'] = theta
    # pars_['Signal_phi'] = phi
    # for pname,val in bkg_bf_params_list[0].items():
    # pars_['Background_'+pname] = val
    #    pars_[bkg_name+'_'+pname] = val
    # for pname,val in flux_params.items():
    #    pars_['Signal_'+pname] = val

    sig_bkg_mod = Sig_Bkg_Model(bl_dmask, sig_mod, bkg_mod, use_deriv=True)
    sig_pars = copy(flux_params)
    sig_pars["A"] = 1.0
    sig_pars["theta"] = theta  # np.mean(thetas)
    sig_pars["phi"] = phi  # np.mean(phis)
    sig_bkg_mod.set_bkg_params(bkg_bf_params_list[0])
    sig_bkg_mod.set_sig_params(sig_pars)

    sig_llh_obj.set_model(sig_bkg_mod)

    sig_miner.set_llh(sig_llh_obj)

    # fixed_pnames = list(pars_.keys())
    # fixed_vals = list(pars_.values())
    # trans = [None for i in range(len(fixed_pnames))]
    # sig_miner.set_trans(fixed_pnames, trans)
    # sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    # sig_miner.set_fixed_params(['Signal_A'], fixed=False)

    sig_miner.set_trans(["A"], [None])

    res_dfs = []

    ntbins = len(tbins0)

    for i in range(ntbins):
        t0 = tbins0[i]
        t1 = tbins1[i]
        timeID = timeIDs[i]
        dt = t1 - t0
        sig_llh_obj.set_time(tbins0[i], tbins1[i])

        # parss_ = {}
        # for pname,val in bkg_bf_params_list[i].items():
        # pars_['Background_'+pname] = val
        #    parss_[bkg_name+'_'+pname] = val
        #    pars_[bkg_name+'_'+pname] = val
        # sig_miner.set_fixed_params(list(parss_.keys()), values=list(parss_.values()))

        sig_bkg_mod.set_bkg_params(bkg_bf_params_list[i])

        res_dict = {"theta": theta, "phi": phi, "time": t0, "dur": dt, "timeID": timeID}

        nllhs, As = min_at_Epeaks_gammas(
            sig_miner, sig_mod, sig_bkg_mod, Epeaks, gammas
        )

        Epeaks2, gammas2 = get_new_Epeaks_gammas2scan(nllhs, Epeaks, gammas)
        nllhs2, As2 = min_at_Epeaks_gammas(
            sig_miner, sig_mod, sig_bkg_mod, Epeaks2, gammas2
        )

        nllhs = np.append(nllhs, nllhs2)
        As = np.append(As, As2)
        res_dict["Epeak"] = np.append(Epeaks, Epeaks2)
        res_dict["gamma"] = np.append(gammas, gammas2)

        pars_["A"] = 1e-10
        bkg_nllh = -sig_llh_obj.get_logprob(pars_)

        res_dict["nllh"] = np.array(nllhs)
        res_dict["A"] = np.array(As)
        res_dict["TS"] = np.sqrt(2 * (bkg_nllh - res_dict["nllh"]))
        res_dict["TS"][np.isnan(res_dict["TS"])] = 0.0
        res_dict["bkg_nllh"] = bkg_nllh

        res_dfs.append(pd.DataFrame(res_dict))
        logging.debug("done with %d of %d tbins" % (i + 1, ntbins))
    return pd.concat(res_dfs, ignore_index=True)


def do_analysis(
    proc_num,
    seed_tab,
    ev_data,
    flux_mod,
    rt_dir,
    ebins0,
    ebins1,
    bl_dmask,
    trigger_time,
    work_dir,
    bkg_fname,
    ignore_started=False,
):
    started_dname = os.path.join(work_dir, "started_outfov")

    nebins = len(ebins0)

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    bl_jobid = np.isclose(seed_tab["proc_group"], proc_num)
    N4jobid = np.sum(bl_jobid)

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    bkg_df, bkg_name, PSnames, bkg_mod, ps_mods = parse_bkg_csv(
        bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir
    )
    rt_obj = RayTraces(rt_dir)

    bkg_mod.has_deriv = False
    bkg_mod_list = [bkg_mod]
    Nsrcs = len(ps_mods)
    if Nsrcs > 0:
        bkg_mod_list += ps_mods
        for ps_mod in ps_mods:
            ps_mod.has_deriv = False
        bkg_mod = CompoundModel(bkg_mod_list)

    hp_ind_grps = seed_tab[bl_jobid].groupby("hp_ind")

    for hp_ind, df in hp_ind_grps:
        logging.info("Starting hp_ind: %d" % (hp_ind))
        theta = np.nanmean(df["theta"])
        phi = np.nanmean(df["phi"])
        ra = np.nanmean(df["ra"])
        dec = np.nanmean(df["dec"])
        logging.info("At theta, phi: %.2f, %.2f" % (theta, phi))
        logging.info("RA, Dec: %.2f, %.2f" % (ra, dec))

        ident_str = "hpind_%d" % (hp_ind)
        fname0 = ident_str + ".txt"
        fname = os.path.join(started_dname, fname0)

        fnames = os.listdir(started_dname)
        already_started = False
        if fname0 in fnames:
            already_started = True
        if already_started and not ignore_started:
            logging.info("Already started")
            continue
        save_fname = "res_hpind_%d_.csv" % (hp_ind)
        fnames = get_out_res_fnames()
        if save_fname in fnames:
            logging.info("Already has results")
            continue

        f = open(fname, "w")
        f.write("NONE")
        f.close()

        t0s = []
        t1s = []
        timeIDs = []
        bkg_params_list = []

        for row_ind, seed_row in df.iterrows():
            t0s.append(seed_row["time"])
            t1s.append(seed_row["time"] + seed_row["dur"])
            timeIDs.append(seed_row["timeID"])
            tmid = seed_row["time"] + (seed_row["dur"] / 2.0)
            bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
            bkg_params = {pname: bkg_row[pname] for pname in bkg_mod.param_names}
            bkg_params_list.append(bkg_params)

        res_df = analysis_at_theta_phi(
            theta,
            phi,
            rt_obj,
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
        gc.collect()
        res_df["hp_ind"] = hp_ind

        fname = os.path.join(work_dir, "res_hpind_%d_.csv" % (hp_ind))
        fname = "res_hpind_%d_.csv" % (hp_ind)

        res_df.to_csv(fname)
        logging.info("Saved results to")
        logging.info(fname)


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

    seed_tab = pd.read_csv(args.job_fname)
    #     if proc_num >= 0:
    #         bl = (seed_tab['proc_group']==proc_num)
    #     else:
    #         bl = np.ones(len(seed_tab), dtype=bool)
    #     seed_tab = seed_tab[bl]

    logging.info("Read in Seed Table, now to do analysis")

    do_analysis(
        proc_num,
        seed_tab,
        ev_data,
        flux_mod,
        rt_dir,
        ebins0,
        ebins1,
        bl_dmask,
        trigtime,
        work_dir,
        args.bkg_fname,
    )
    # do_analysis(square_tab, rate_res_tab, good_pix['imx'], good_pix['imy'], pl_flux,\
    #                 drm_obj, rt_dir,\
    #                 bkg_llh_obj, sig_llh_obj,\
    #                 conn, db_fname, trigtime, work_dir,bkg_fits_df)
    conn.close()

    logging.info("Done with all seeds for this proc")

    logging.info("Now checking for other unstarted seeds")

    Njobs = args.Njobs

    for i in range(Njobs):
        if i == proc_num:
            continue
        do_analysis(
            i,
            seed_tab,
            ev_data,
            flux_mod,
            rt_dir,
            ebins0,
            ebins1,
            bl_dmask,
            trigtime,
            work_dir,
            args.bkg_fname,
        )

    seed_tab_shuff = seed_tab.sample(frac=1)

    for i in range(Njobs):
        if i == proc_num:
            continue
        do_analysis(
            i,
            seed_tab_shuff,
            ev_data,
            flux_mod,
            rt_dir,
            ebins0,
            ebins1,
            bl_dmask,
            trigtime,
            work_dir,
            args.bkg_fname,
            ignore_started=True,
        )


if __name__ == "__main__":
    args = cli()

    main(args)
