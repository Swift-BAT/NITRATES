import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import argparse
import logging, traceback
import time
import pandas as pd
from scipy import optimize
from copy import copy

# import ..config

from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import (
    get_rate_fits_tab,
    guess_dbfname,
    get_seeds_tab,
    get_info_tab,
    get_files_tab,
    get_square_tab,
    get_full_sqlite_table_as_df,
)
from ..lib.drm_funcs import DRMs
from ..response.ray_trace_funcs import RayTraces, FootPrints
from ..llh_analysis.LLH import LLH_webins
from ..models.flux_models import Plaw_Flux
from ..models.models import (
    Bkg_Model_wFlatA,
    Point_Source_Model_Wuncoded,
    im_dist,
    Point_Source_Model_Binned_Rates,
    Bkg_and_Point_Source_Model,
    CompoundModel,
    Bkg_and_Point_Source_Model,
)
from ..llh_analysis.minimizers import NLLH_ScipyMinimize_Wjacob, NLLH_Minimizer
from ..config import EBINS0, EBINS1, solid_angle_dpi_fname, rt_dir, drm_dir, fp_dir


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
        "--peak_fname", type=str, help="Rate results file name", default="peaks.csv"
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
        "--log_fname", type=str, help="Name for the log file", default="intLLH"
    )
    args = parser.parse_args()
    return args


def integrated_LLH_webins(llh_obj, miner, params, pname2ignore="flat"):
    nebins = llh_obj.nebins

    log_prob = 0.0
    global Nfunc_calls
    bf_params = {}
    bf_params = copy(params)

    for j in range(nebins):
        params_ = copy(bf_params)
        miner.set_fixed_params(list(params_.keys()), values=list(params_.values()))
        e0_pnames = []
        for pname in miner.param_names:
            try:
                if int(pname[-1]) == j and not pname2ignore in pname:
                    if miner.param_info_dict[pname]["nuis"]:
                        e0_pnames.append(pname)
            except:
                pass
        miner.set_fixed_params(e0_pnames, fixed=False)

        llh_obj.set_ebin(j)

        bf_vals, nllh, res = miner.minimize()
        nllh = nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]

        I_mat = llh_obj.get_logprob_hess(bf_params)

        gauss_integral = np.log(1.0 / np.sqrt(np.linalg.det(I_mat / (2.0 * (np.pi)))))
        log_prob += -nllh + gauss_integral

    return log_prob, bf_params


def int_LLH2min_Agamma(A_gamma, llh_obj, miner, params):
    if A_gamma[1] < 0 or A_gamma[0] < 1e-8 or A_gamma[1] > 2.5:
        return np.inf
    params["Signal_A"] = A_gamma[0]
    params["Signal_gamma"] = A_gamma[1]
    miner.set_fixed_params(["Signal_A"], [A_gamma[0]])
    miner.set_fixed_params(["Signal_gamma"], [A_gamma[1]])
    return -integrated_LLH_webins(llh_obj, miner, params)[0]


def min_intLLH(miner, bf_params_):
    bf_A = bf_params_["Signal_A"]
    bf_gamma = bf_params_["Signal_gamma"]
    x0s = [[bf_A, bf_gamma]]
    x0s += [[0.8 * bf_A, 0.5], [1.25 * bf_A, 1.95]]
    ress = []
    nllhs = []
    for x0 in x0s:
        params_ = copy(bf_params_)
        res = optimize.minimize(
            int_LLH2min_Agamma,
            x0,
            method="Nelder-Mead",
            args=(miner.llh_obj, miner, params_),
            options={"fatol": 1e-3, "xatol": 1e-6, "maxfev": 480},
        )
        ress.append(res)
        nllhs.append(res.fun)
        logging.debug(res)
        if res.success:
            break
        logging.warning("Failed to minimize at: ")
        logging.warning(bf_params_)
        logging.warning("At seed: ")
        logging.warning(x0)

    min_ind = np.nanargmin(np.array(nllhs))
    bf_A = ress[min_ind].x[0]
    bf_gamma = ress[min_ind].x[1]
    return nllhs[min_ind], bf_A, bf_gamma


def min_intLLH_imgrid(miner, bf_params_, imstep=1e-3, dimxy=2e-3, ret_all=False):
    imax = np.arange(-dimxy, dimxy + imstep / 2.0, imstep)
    imgrids = np.meshgrid(imax, imax)
    imxs = imgrids[0].ravel() + bf_params_["Signal_imx"]
    imys = imgrids[1].ravel() + bf_params_["Signal_imy"]
    Npnts = len(imxs)
    nllhs = np.zeros_like(imxs)
    As = np.zeros_like(imxs)
    gammas = np.zeros_like(imxs)

    for i in range(Npnts):
        params_ = copy(bf_params_)
        params_["Signal_imx"] = imxs[i]
        params_["Signal_imy"] = imys[i]

        nllhs[i], As[i], gammas[i] = min_intLLH(miner, params_)

    if ret_all:
        return nllhs, As, gammas, imxs, imys

    min_ind = np.nanargmin(nllhs)
    return nllhs[min_ind], As[min_ind], gammas[min_ind], imxs[min_ind], imys[min_ind]


def min_intLLH_imlist(miner, bf_params_, imxs, imys):
    Npnts = len(imxs)
    nllhs = np.zeros_like(imxs)
    As = np.zeros_like(imxs)
    gammas = np.zeros_like(imxs)

    for i in range(Npnts):
        params_ = copy(bf_params_)
        params_["Signal_imx"] = imxs[i]
        params_["Signal_imy"] = imys[i]

        nllhs[i], As[i], gammas[i] = min_intLLH(miner, params_)

    return nllhs, As, gammas


def axs2grid_list(xax, yax, decimals=None):
    grids = np.meshgrid(xax, yax)
    xs = grids[0].ravel()
    ys = grids[1].ravel()
    if decimals is not None:
        xs = np.round(xs, decimals=decimals)
        ys = np.round(ys, decimals=decimals)

    return xs, ys


def bnds2grid_list(xbs, ybs, step, ystep=None, decimals=None):
    xstep = step
    if ystep is None:
        ystep = step

    xax = np.arange(xbs[0], xbs[1] + xstep / 2.0, xstep)
    yax = np.arange(ybs[0], ybs[1] + ystep / 2.0, ystep)

    return axs2grid_list(xax, yax, decimals=decimals)


def adaptive_square_scan(
    miner,
    bf_params_,
    im_step=2e-3,
    dimxy=4e-3,
    min_imstep=5e-4,
    imx_pname="Signal_imx",
    imy_pname="Signal_imy",
):
    nllhs, As, gammas, imxs_scan, imys_scan = min_intLLH_imgrid(
        miner, bf_params_, imstep=im_step, dimxy=dimxy, ret_all=True
    )

    nllh_ind_sort = np.argsort(nllhs)
    best_imxs = imxs_scan[nllh_ind_sort[:2]]
    best_imys = imys_scan[nllh_ind_sort[:2]]
    Npnts = len(best_imxs)

    imxs_all = np.copy(imxs_scan)
    imys_all = np.copy(imys_scan)
    nllhs_all = np.copy(nllhs)
    As_all = np.copy(As)
    Gs_all = np.copy(gammas)

    imxs2scan = np.empty(0)
    imys2scan = np.empty(0)

    fine_im_step = im_step / 2.0

    logging.info("Going to scan %d pnts" % (Npnts))

    for i in range(Npnts):
        xbs = (best_imxs[i] - im_step, best_imxs[i] + im_step)
        ybs = (best_imys[i] - im_step, best_imys[i] + im_step)
        imxs_, imys_ = bnds2grid_list(xbs, ybs, fine_im_step, decimals=4)
        imdists = np.zeros_like(imxs_)
        for ii in range(len(imdists)):
            imdists[ii] = np.min(im_dist(imxs_[ii], imys_[ii], imxs_all, imys_all))
        bl_im = imdists > (0.5 * fine_im_step)
        if np.sum(bl_im) > 0:
            imxs2scan = np.append(imxs2scan, imxs_[bl_im])
            imys2scan = np.append(imys2scan, imys_[bl_im])
            imxs_all = np.append(imxs_all, imxs_[bl_im])
            imys_all = np.append(imys_all, imys_[bl_im])

    nllhs2, As2, gammas2 = min_intLLH_imlist(miner, bf_params_, imxs2scan, imys2scan)

    logging.info("%d more pnts to scan" % (len(imxs2scan)))

    nllhs_all = np.append(nllhs_all, nllhs2)
    As_all = np.append(As_all, As2)
    Gs_all = np.append(Gs_all, gammas2)

    min_ind = np.argmin(nllhs_all)
    best_imx = imxs_all[min_ind]
    best_imy = imys_all[min_ind]

    fine_im_step = fine_im_step / 2.0
    imxs2scan = np.empty(0)
    imys2scan = np.empty(0)

    xbs = (best_imx - 2.0 * fine_im_step, best_imx + 2.0 * fine_im_step)
    ybs = (best_imy - 2.0 * fine_im_step, best_imy + 2.0 * fine_im_step)
    imxs_, imys_ = bnds2grid_list(xbs, ybs, fine_im_step, decimals=4)
    imdists = np.zeros_like(imxs_)
    for ii in range(len(imdists)):
        imdists[ii] = np.min(im_dist(imxs_[ii], imys_[ii], imxs_all, imys_all))
    bl_im = imdists > (0.5 * fine_im_step)
    if np.sum(bl_im) > 0:
        imxs2scan = np.append(imxs2scan, imxs_[bl_im])
        imys2scan = np.append(imys2scan, imys_[bl_im])
        imxs_all = np.append(imxs_all, imxs_[bl_im])
        imys_all = np.append(imys_all, imys_[bl_im])

    logging.info("%d more pnts to scan" % (len(imxs2scan)))

    nllhs2, As2, gammas2 = min_intLLH_imlist(miner, bf_params_, imxs2scan, imys2scan)

    nllhs_all = np.append(nllhs_all, nllhs2)
    As_all = np.append(As_all, As2)
    Gs_all = np.append(Gs_all, gammas2)

    return nllhs_all, imxs_all, imys_all, As_all, Gs_all


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

    bkg_mod = Bkg_Model_wFlatA(
        bl_dmask, solid_angle_dpi, nebins, use_deriv=True, use_prior=True
    )

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


def do_analysis(
    peaks_tab,
    pl_flux,
    drm_obj,
    rt_dir,
    fp_dir,
    bkg_llh_obj,
    sig_llh_obj,
    trigger_time,
    work_dir,
    jobid,
    bkg_fname,
    solid_angle_dpi_fname,
):
    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    nebins = len(ebins0)
    bl_dmask = sig_llh_obj.bl_dmask

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    rt_obj = RayTraces(rt_dir, max_nbytes=6e9)
    fp_obj = FootPrints(fp_dir)

    bkg_df, bkg_name, PSnames, bkg_mod, ps_mods = parse_bkg_csv(
        bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir
    )

    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    Npeaks = len(peaks_tab)

    for peak_ind, peak_row in peaks_tab.iterrows():
        logging.debug(
            "Starting timeID: %d, peak index: %d" % (peak_row["timeID"], peak_ind)
        )
        logging.debug(
            "Starting imx: %.3f, imy: %.3f" % (peak_row["imx"], peak_row["imy"])
        )

        res_dict = {}
        res_dict["peak_ind"] = peak_ind
        imx0 = peak_row["imx"]
        imy0 = peak_row["imy"]
        t0 = peak_row["time"]
        dt = peak_row["duration"]
        t1 = t0 + dt
        tmid = t0 + dt / 2.0
        timeid = peak_row["timeID"]
        sig_llh_obj.set_time(t0, t1)
        res_dict["timeID"] = timeid
        res_dict["time"] = t0
        res_dict["duration"] = dt
        res_dict["jobid"] = jobid

        bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]

        Nps = len(PSnames)
        if Nps > 0:
            bkg_mod = Bkg_and_Point_Source_Model(
                solid_ang_dpi,
                [ebins0, ebins1],
                rt_obj,
                bl_dmask,
                PSnames,
                bkg_row=bkg_row,
                bkg_err_fact=2.0,
            )

        bkg_mod.set_bkg_row(bkg_row)

        sig_mod = Point_Source_Model_Wuncoded(
            imx0,
            imy0,
            0.1,
            pl_flux,
            drm_obj,
            [ebins0, ebins1],
            rt_obj,
            fp_obj,
            bl_dmask,
            use_deriv=True,
            use_prior=True,
        )

        comp_mod = CompoundModel([bkg_mod, sig_mod])

        params_ = copy({pname: d["val"] for pname, d in comp_mod.param_dict.items()})
        params_["Signal_A"] = peak_row["Signal_A"]
        params_["Signal_gamma"] = peak_row["Signal_gamma"]
        sig_llh_obj.set_model(comp_mod)
        sig_miner.set_llh(sig_llh_obj)

        params = copy(params_)
        sig_nllhs, imxs, imys, As, gammas = adaptive_square_scan(sig_miner, params)

        params = copy(params_)
        params["Signal_A"] = 1e-8
        bkg_logl, ps = integrated_LLH_webins(sig_miner.llh_obj, sig_miner, params)
        res_dict["bkg_nllh"] = -bkg_logl

        TSs = np.sqrt(2.0 * (-bkg_logl - sig_nllhs))
        TSs[np.isnan(TSs)] = 0.0
        res_dict["imx"] = imxs
        res_dict["imy"] = imys
        res_dict["A"] = As
        res_dict["gamma"] = gammas
        res_dict["sig_nllh"] = sig_nllhs
        res_dict["TS"] = TSs

        fname = os.path.join(
            work_dir, "peak_scan_%d_%d_.csv" % (int(peak_ind), int(jobid))
        )
        df = pd.DataFrame(res_dict)
        df.to_csv(fname)
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

    # bkg_fits_df = pd.read_csv(args.bkg_fname)

    # rate_fits_df = get_rate_fits_tab(conn)
    # bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    time_starting = time.time()
    proc_num = args.job_id
    # init classes up here

    drm_dir = files_tab["drmDir"][0]
    if args.rt_dir is None:
        rt_dir = files_tab["rtDir"][0]
    else:
        rt_dir = args.rt_dir
    drm_obj = DRMs(drm_dir)
    # rt_obj = RayTraces(rt_dir, max_nbytes=1e10)
    work_dir = files_tab["workDir"][0]

    pl_flux = Plaw_Flux()

    ebins0 = np.array(config.EBINS0)
    ebins1 = np.array(config.EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    peaks_tab = pd.read_csv(args.peak_fname)
    if proc_num >= 0:
        bl = peaks_tab["jobID"] == proc_num
        peaks_tab = peaks_tab[bl]

    logging.info("Read in peaks tab")
    logging.info("%d peaks to scan" % (len(peaks_tab)))

    do_analysis(
        peaks_tab,
        pl_flux,
        drm_obj,
        rt_dir,
        fp_dir,
        bkg_llh_obj,
        sig_llh_obj,
        trigtime,
        work_dir,
        proc_num,
        args.bkg_fname,
        solid_angle_dpi_fname,
    )


if __name__ == "__main__":
    args = cli()

    main(args)
