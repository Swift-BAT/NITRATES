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
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
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
    seed_tab,
    pl_flux,
    drm_obj,
    rt_obj,
    bkg_llh_obj,
    sig_llh_obj,
    bkg_rate_obj,
    conn,
    db_fname,
    grid_width=2e-3,
    grid_step=1e-3,
):
    seed_t_gs = seed_tab.groupby("timeID")
    N_twinds = seed_t_gs.ngroups

    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    bl_dmask = sig_llh_obj.bl_dmask

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    for seed_g in seed_t_gs:
        timeID = seed_g[0]
        seed_tab = seed_g[1]
        Nseeds = len(seed_tab)

        t0 = np.nanmean(seed_tab["time"])
        dt = np.nanmean(seed_tab["duration"])
        t1 = t0 + dt
        tmid = (t0 + t1) / 2.0

        bkg_llh_obj.set_time(t0, dt)
        sig_llh_obj.set_time(t0, dt)

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

        rows = []
        sig_param_dicts = []
        sig_nllhs = []
        TSs = []

        for row in seed_tab.itertuples():
            try:
                blipID = row.blipID

                sig_mod = Point_Source_Model(
                    row.imx,
                    row.imy,
                    grid_width,
                    pl_flux,
                    drm_obj,
                    [ebins0, ebins1],
                    rt_obj,
                    bl_dmask,
                    use_deriv=True,
                )
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

                param_list, nllhs, imxs, imys = imxy_grid_miner(
                    sig_miner,
                    row.imx - grid_width / 2.0,
                    row.imy - grid_width / 2.0,
                    row.imx + grid_width / 2.0,
                    row.imy + grid_width / 2.0,
                    dimxy=grid_step,
                )

                best_ind = np.argmin(nllhs)
                bf_params = param_list[best_ind]
                sig_nllh = nllhs[best_ind]
                imx = imxs[best_ind]
                imy = imys[best_ind]

                sig_param_dict = {}
                i = 0
                for pname in sig_miner.param_names:
                    if pname in sig_miner.fixed_params:
                        if "imx" in pname:
                            sig_param_dict[pname] = imx
                        elif "imy" in pname:
                            sig_param_dict[pname] = imy
                        else:
                            sig_param_dict[pname] = sig_miner.param_info_dict[pname][
                                "val"
                            ]
                    else:
                        sig_param_dict[pname] = bf_params[i]
                        i += 1

                logging.info("sig_param_dict: ")
                logging.info(str(sig_param_dict))
                TS = np.sqrt(2.0 * (bkg_nllh - sig_nllh))
                if np.isnan(TS):
                    TS = 0.0
                TSs.append(TS)
                sig_nllhs.append(sig_nllh)
                rows.append(row)
                sig_param_dicts.append(sig_param_dict)
                # try:
                #     write_result(conn, row, sig_param_dict,\
                #                 bkg_nllh, sig_nllh, TS)
                # except Exception as E:
                #     logging.error(E)
                #     logging.error("Problem writing to DB")
                #     conn.close()
                #     conn = get_conn(db_fname, timeout=30.0)
                #     try:
                #         write_result(conn, row, sig_param_dict,\
                #                     bkg_nllh, sig_nllh, TS)
                #     except Exception as E:
                #         logging.error(E)
                #         logging.error("Problem writing to DB")
                #         logging.error("Couldn't write ")
                #         logging.error(str(sig_param_dict))
                #         logging.error("to DB")

            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Failed to minimize seed: ")
                logging.error(row)

        try:
            write_results(conn, rows, sig_param_dicts, bkg_nllh, sig_nllhs, TSs)
        except Exception as E:
            logging.error(E)
            logging.error("Problem writing to DB")
            conn.close()
            conn = get_conn(db_fname, timeout=30.0)
            try:
                write_results(conn, rows, sig_param_dicts, bkg_nllh, sig_nllhs, TSs)
            except Exception as E:
                logging.error(E)
                logging.error("Problem writing to DB")
                logging.error("Couldn't write ")
                # logging.error(str(sig_param_dict))
                logging.error("to DB")


def main(args):
    fname = "llh_analysis_" + str(args.job_id)

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    t_0 = time.time()

    while (time.time() - t_0) < 4 * 3600.0:
        try:
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
            break
        except Exception as E:
            logging.error(str(E))
            logging.error(traceback.format_exc())
            time.sleep(30.0)

    time_starting = time.time()
    proc_num = args.job_id
    # init classes up here

    drm_dir = files_tab["drmDir"][0]
    if args.rt_dir is None:
        rt_dir = files_tab["rtDir"][0]
    else:
        rt_dir = args.rt_dir
    drm_obj = DRMs(drm_dir)
    rt_obj = RayTraces(rt_dir)

    pl_flux = Plaw_Flux()

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    while (time.time() - t_0) < 6 * 3600.0:
        conn = get_conn(db_fname)
        try:
            if proc_num >= 0:
                seeds_tab = get_seeds_tab(conn, proc_group=proc_num)
            else:
                seeds_tab = get_seeds_tab(conn)
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.warning("Failed to get seed tab, will try again")
            conn.close()
            time.sleep(30.0)
            continue

        new_seeds = seeds_tab["done"] == 0
        seed_tab = seeds_tab[new_seeds]
        Nseeds_todo = np.sum(new_seeds)
        logging.info(str(Nseeds_todo) + " new seeds")

        if Nseeds_todo == 0:
            conn.close()
            time.sleep(30.0)
            continue

        # seed_prio_gs = seed_tab.groupby('priority')

        # Nprio = seed_prio_gs.ngroups

        min_priority = np.min(seed_tab["priority"])

        logging.info("min priority is " + str(min_priority))

        prio_bl = seed_tab["priority"] == min_priority

        Nmin_prio = np.sum(prio_bl)

        logging.info(str(Nmin_prio) + " new seeds with priority " + str(min_priority))

        seed_tab = seed_tab[prio_bl]

        do_analysis(
            seed_tab,
            pl_flux,
            drm_obj,
            rt_obj,
            bkg_llh_obj,
            sig_llh_obj,
            bkg_rates_obj,
            conn,
            db_fname,
        )
        conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
