import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback
import time

from ..analysis_seeds.bkg_rate_estimation import rate_obj_from_sqltab
from ..lib.sqlite_funcs import get_conn, write_result, write_results
from ..lib.dbread_funcs import (
    get_rate_fits_tab,
    guess_dbfname,
    get_seeds_tab,
    get_info_tab,
    get_files_tab,
)
from ..config import EBINS0, EBINS1
from ..models.flux_models import Plaw_Flux
from ..llh_analysis.minimizers import (
    NLLH_ScipyMinimize_Wjacob,
    imxy_grid_miner,
    NLLH_ScipyMinimize,
)
from ..lib.drm_funcs import DRMs
from ..response.ray_trace_funcs import RayTraces
from ..llh_analysis.LLH import LLH_webins
from ..models.models import Bkg_Model, Point_Source_Model, CompoundModel

# need to read rate fits from DB
# and read twinds
# and read/get event, dmask, and ebins
# then get bkg_llh_obj and a minimizer
# then loop over all time windows
# minimizing nllh and recording bf params


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posfname", type=str, help="File name of imxys to read", default=None
    )
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--dt0", type=float, help="Start time relative to trig_time", default=12.512
    )
    parser.add_argument(
        "--dt1", type=float, help="End time relative to trig_time", default=12.552
    )
    parser.add_argument("--dt_step", type=float, help="Time Step", default=0.03)
    parser.add_argument("--dt", type=float, help="Expsoure time", default=0.03)
    parser.add_argument(
        "--job_id", type=int, help="ID to tell it what seeds to do", default=-1
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--rt_dir", type=str, help="Directory with ray traces", default=None
    )
    args = parser.parse_args()
    return args


def do_analysis(
    imxs,
    imys,
    t0,
    t1,
    pl_flux,
    drm_obj,
    rt_obj,
    bkg_llh_obj,
    sig_llh_obj,
    bkg_rate_obj,
    job_id,
):
    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    bl_dmask = sig_llh_obj.bl_dmask

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")
    tmid = (t0 + t1) / 2.0

    bkg_mod = Bkg_Model(
        bkg_rate_obj, bl_dmask, t=tmid, bkg_err_fact=2.0, use_prior=False
    )

    logging.debug("bkg exp rates, errors")
    logging.debug(bkg_mod._rates)
    logging.debug(bkg_mod._errs)

    bkg_llh_obj.set_model(bkg_mod)

    bkg_miner.set_llh(bkg_llh_obj)

    bkg_miner.set_fixed_params(bkg_llh_obj.model.param_names)
    bkg_params = {
        pname: bkg_llh_obj.model.param_dict[pname]["val"]
        for pname in bkg_llh_obj.model.param_names
    }
    bkg_nllh = -bkg_llh_obj.get_llh(bkg_params)

    sig_mod = Point_Source_Model(
        np.mean(imxs),
        np.mean(imys),
        0.3,
        pl_flux,
        drm_obj,
        [ebins0, ebins1],
        rt_obj,
        bl_dmask,
        use_deriv=True,
    )
    sig_mod.drm_im_update = 0.2
    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_llh_obj.set_model(comp_mod)
    sig_miner.set_llh(sig_llh_obj)
    fixed_pars = [
        pname
        for pname in sig_miner.param_names
        if ("A" not in pname) or ("gamma" not in pname)
    ]
    sig_miner.set_fixed_params(fixed_pars)
    sig_miner.set_fixed_params(["Signal_A", "Signal_gamma"], fixed=False)
    sig_miner.set_trans(["Signal_A"], [None])
    # sig_miner.set_bounds(['Signal_A'], [(-5e-2, 1e1)])
    sig_miner.set_bounds(["Signal_gamma"], [(0.0, 2.5)])

    nllhs = np.zeros_like(imxs)
    TSs = np.zeros_like(imxs)
    As = np.zeros_like(imxs)
    gammas = np.zeros_like(imxs)

    logging.info(str(len(imxs)) + " positions to minimize")

    for ii in range(len(imxs)):
        try:
            sig_miner.set_fixed_params(
                ["Signal_imx", "Signal_imy"], [imxs[ii], imys[ii]]
            )
            pars, nllh, res = sig_miner.minimize()
            TSs[ii] = np.sqrt(2.0 * (bkg_nllh - nllh[0]))
            nllhs[ii] = nllh[0]
            As[ii] = pars[0][0]
            gammas[ii] = pars[0][1]

        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.error("Failed to minimize seed: ")

        if ii % 100 == 0:
            logging.info("Done with %d of %d positions" % (ii + 1, len(imxs)))

    fname = "t0_%.3f_t1_%.3f_%d" % (t0, t1, job_id)
    np.savez(
        fname,
        nllhs=nllhs,
        TSs=TSs,
        imxs=imxs,
        imys=imys,
        As=As,
        gammas=gammas,
        bkg_nllh=bkg_nllh,
    )


def main(args):
    fname = "llh_scan_" + str(args.job_id)

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
    rate_fits_df = get_rate_fits_tab(conn)
    bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    pos_file = np.load(args.posfname)

    bins = [np.linspace(-2, 2, 10 * 4 + 1), np.linspace(-1, 1, 10 * 2 + 1)]
    # bins = [np.linspace(-2,2,5*4+1),
    #         np.linspace(-1,1,5*2+1)]

    h = np.histogram2d(pos_file["imxs"], pos_file["imys"], bins=bins)[0]

    imx_inds, imy_inds = np.where(h > 0)

    Nbins_per_job = 2

    I0 = Nbins_per_job * args.job_id
    I1 = I0 + Nbins_per_job

    imxinds = imx_inds[I0:I1]
    imyinds = imy_inds[I0:I1]

    imx_bl = np.isin(pos_file["imx_inds"], imxinds)
    imy_bl = np.isin(pos_file["imy_inds"], imyinds)

    pos_bl = imx_bl & imy_bl
    imxs = pos_file["imxs"][pos_bl]
    imys = pos_file["imys"][pos_bl]

    time_starting = time.time()
    proc_num = args.job_id
    # init classes up here

    drm_dir = files_tab["drmDir"][0]
    if args.rt_dir is None:
        rt_dir = files_tab["rtDir"][0]
    else:
        rt_dir = args.rt_dir
    drm_obj = DRMs(drm_dir)
    rt_obj = RayTraces(rt_dir, max_nbytes=1e10)

    pl_flux = Plaw_Flux()

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    tbins0 = np.arange(args.dt0, args.dt1, args.dt_step) + trigtime
    tbins1 = tbins0 + args.dt
    Ntbins = len(tbins0)
    logging.info("tbins0: ")
    logging.info(tbins0)
    logging.info("tbins1: ")
    logging.info(tbins1)
    logging.info("Ntbins: %d" % (Ntbins))

    bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    conn = get_conn(db_fname)

    for i in range(Ntbins):
        t0 = tbins0[i]
        dt = args.dt

        bkg_llh_obj.set_time(t0, dt)
        sig_llh_obj.set_time(t0, dt)

        do_analysis(
            imxs,
            imys,
            t0,
            t0 + dt,
            pl_flux,
            drm_obj,
            rt_obj,
            bkg_llh_obj,
            sig_llh_obj,
            bkg_rates_obj,
            args.job_id,
        )
    conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
