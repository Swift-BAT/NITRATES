import numpy as np
from astropy.io import fits
from astropy.table import Table
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
    square_tab,
    sig_img_tab,
    pl_flux,
    drm_obj,
    rt_dir,
    bkg_llh_obj,
    sig_llh_obj,
    bkg_rate_obj,
    conn,
    db_fname,
    trigger_time,
    work_dir,
    TSwrite=3.0,
):
    conn.close()

    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    bl_dmask = sig_llh_obj.bl_dmask

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    for square_ind, square_row in square_tab.iterrows():
        logging.info("Starting squareID: %d" % (square_row["squareID"]))

        rt_obj = RayTraces(rt_dir, max_nbytes=8e9)

        tab = Table()
        res_dfs2write = []

        logging.info("%d timeIDs to do" % (len(sig_img_tab)))

        for sig_ind, sig_row in sig_img_tab.iterrows():
            logging.debug(
                "Starting timeID: %d, table index: %d" % (sig_row["timeID"], sig_ind)
            )

            res_dict = {}
            res_dict = {"squareID": square_row["squareID"], "timeID": sig_row["timeID"]}

            sig_pix = np.load(sig_row["fname"], mmap_mode="r")
            sig_bl = (
                (sig_pix["imx"] >= square_row["imx0"])
                & (sig_pix["imx"] < square_row["imx1"])
                & (sig_pix["imy"] >= square_row["imy0"])
                & (sig_pix["imy"] < square_row["imy1"])
            )
            Npix = np.sum(sig_bl)
            logging.debug("%d Pixels to minimize at" % (Npix))

            if Npix < 1:
                # update_square_stat(conn, sig_row['timeID'],\
                #                     square_row['squareID'])
                continue

            imxs = sig_pix["imx"][sig_bl]
            imys = sig_pix["imy"][sig_bl]
            snrs = sig_pix["snr"][sig_bl]

            t0 = sig_row["time"]
            dt = sig_row["duration"]
            tmid = t0 + dt / 2.0
            t1 = t0 + dt
            res_dict["time"] = t0
            res_dict["duration"] = dt

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
            res_dict["bkg_nllh"] = bkg_nllh

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

            TSs = np.zeros(Npix)
            sig_nllhs = np.zeros(Npix)
            As = np.zeros(Npix)
            gammas = np.zeros(Npix)

            for ii in range(Npix):
                try:
                    sig_miner.set_fixed_params(
                        ["Signal_imx", "Signal_imy"], [imxs[ii], imys[ii]]
                    )
                    pars, nllh, res = sig_miner.minimize()
                    TS = np.sqrt(2.0 * (bkg_nllh - nllh[0]))
                    # if TS >= TS_min:
                    if np.isnan(TS):
                        TS = 0.0
                    TSs[ii] = TS
                    sig_nllhs[ii] = nllh[0]
                    As[ii] = pars[0][0]
                    gammas[ii] = pars[0][1]

                except Exception as E:
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    logging.error("Failed to minimize seed: ")
                    logging.error((imxs[ii], imys[ii]))

            # best_ind = np.nanargmax(TSs)
            # res_dict['TS'] = TSs[best_ind]
            # res_dict['imx'] = imxs[best_ind]
            # res_dict['imy'] = imys[best_ind]
            # res_dict['A'] = As[best_ind]
            # res_dict['ind'] = gammas[best_ind]
            # res_dict['sig_nllh'] = sig_nllhs[best_ind]
            # fname = os.path.join(work_dir,\
            #         'res_%d_%d_.fits' %(res_dict['timeID'],\
            #         res_dict['squareID']))

            TSbl = TSs >= TSwrite
            if np.sum(TSbl) > 0:
                res_dict["TS"] = TSs[TSbl]
                res_dict["imx"] = imxs[TSbl]
                res_dict["imy"] = imys[TSbl]
                res_dict["snr"] = snrs[TSbl]
                res_dict["A"] = As[TSbl]
                res_dict["ind"] = gammas[TSbl]
                res_dict["sig_nllh"] = sig_nllhs[TSbl]
                # res_dict['fname'] = fname
                res_dfs2write.append(pd.DataFrame(res_dict))

            # try:
            #     write_square_res_line(conn, res_dict)
            # except Exception as E:
            #     logging.error(E)
            #     logging.error(traceback.format_exc())
            #     logging.error("Problem writing to DB")
            #     conn.close()
            #     conn = get_conn(db_fname, timeout=30.0)
            #     try:
            #         write_square_res_line(conn, res_dict)
            #     except Exception as E:
            #         logging.error(E)
            #         logging.error(traceback.format_exc())
            #         logging.error("Problem writing to DB")
            #         logging.error("Couldn't write ")
            #         logging.error(res_dict)
            #         logging.error("to DB")

            # tab['TS'] = TSs
            # tab['sig_nllh'] = sig_nllhs
            # tab['A'] = As
            # tab['gamma'] = gammas
            # tab['imx'] = imxs
            # tab['imy'] = imys

        fname = os.path.join(work_dir, "res_%d_.csv" % (res_dict["squareID"]))

        res_df = pd.concat(res_dfs2write)
        res_written = False
        conn = get_conn(db_fname)
        try:
            write_square_results(conn, res_df)
            res_written = True
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.error("Problem writing to DB")
            conn.close()
            conn = get_conn(db_fname, timeout=30.0)
            try:
                write_square_results(conn, res_df)
                res_written = True
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Problem writing to DB")
                logging.info("Instead writing results to file: ")
                logging.info(fname)
                res_df.to_csv(fname)

        if res_written:
            try:
                update_square_stat(conn, sig_img_tab["timeID"], square_ind)
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Problem writing to DB")
                conn.close()
                conn = get_conn(db_fname, timeout=30.0)
                try:
                    update_square_stat(conn, sig_img_tab["timeID"], square_ind)
                except Exception as E:
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    logging.error("Problem writing to DB")
        conn.close()

        # tab.write(fname, overwrite=True)


def main(args):
    fname = "llh_analysis_from_sigpixels_" + str(args.job_id)

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

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    conn = get_conn(db_fname)
    if proc_num >= 0:
        square_tab = get_square_tab(conn, proc_group=proc_num)
    else:
        square_tab = get_square_tab(conn)

    sig_img_tab = get_full_sqlite_table_as_df(conn, "SigImages")

    logging.info("Read in Square and Img Tables, now to do analysis")

    do_analysis(
        square_tab,
        sig_img_tab,
        pl_flux,
        drm_obj,
        rt_dir,
        bkg_llh_obj,
        sig_llh_obj,
        bkg_rates_obj,
        conn,
        db_fname,
        trigtime,
        work_dir,
    )
    conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
