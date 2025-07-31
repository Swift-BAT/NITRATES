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
from ..config import EBINS0, EBINS1, solid_angle_dpi_fname
from ..models.flux_models import Plaw_Flux
from ..llh_analysis.minimizers import (
    NLLH_ScipyMinimize_Wjacob,
    imxy_grid_miner,
    NLLH_ScipyMinimize,
)
from ..lib.drm_funcs import DRMs
from ..response.ray_trace_funcs import RayTraces
from ..llh_analysis.LLH import LLH_webins
from ..models.models import Bkg_Model_wSA, Point_Source_Model, CompoundModel

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
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default="pc_2.img"
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
    args = parser.parse_args()
    return args


def do_analysis(
    square_tab,
    rate_res_tab,
    pc_imxs,
    pc_imys,
    pl_flux,
    drm_obj,
    rt_dir,
    bkg_llh_obj,
    sig_llh_obj,
    conn,
    db_fname,
    trigger_time,
    work_dir,
    bkg_df,
    TSwrite=4.5,
):
    conn.close()

    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    nebins = len(ebins0)
    bl_dmask = sig_llh_obj.bl_dmask

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    bkg_miner = NLLH_ScipyMinimize("")
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    for square_ind, square_row in square_tab.iterrows():
        logging.info("Starting squareID: %d" % (square_row["squareID"]))

        rt_obj = RayTraces(rt_dir, max_nbytes=6e9)

        im_bl = (
            (pc_imxs >= square_row["imx0"])
            & (pc_imxs < square_row["imx1"])
            & (pc_imys >= square_row["imy0"])
            & (pc_imys < square_row["imy1"])
        )
        Npix = np.sum(im_bl)
        logging.debug("%d Pixels to minimize at" % (Npix))
        imxs = pc_imxs[im_bl]
        imys = pc_imys[im_bl]

        tab = Table()
        res_dfs2write = []

        bl = rate_res_tab["squareID"] == square_row["squareID"]

        logging.info("%d timeIDs to do" % (np.sum(bl)))

        rate_ress = rate_res_tab[bl]

        for rate_ind, rate_row in rate_ress.iterrows():
            logging.debug(
                "Starting timeID: %d, table index: %d" % (rate_row["timeID"], rate_ind)
            )

            res_dict = {}
            res_dict = {
                "squareID": square_row["squareID"],
                "timeID": rate_row["timeID"],
            }

            t0 = rate_row["time"]
            dt = rate_row["dur"]
            tmid = t0 + dt / 2.0
            t1 = t0 + dt
            res_dict["time"] = t0
            res_dict["duration"] = dt

            bkg_llh_obj.set_time(t0, t1)
            sig_llh_obj.set_time(t0, t1)

            # bkg_mod = Bkg_Model(bkg_rate_obj, bl_dmask, t=tmid,\
            #                 bkg_err_fact=2.0, use_prior=False)
            bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
            bkg_mod = Bkg_Model_wSA(bl_dmask, solid_ang_dpi, nebins, param_vals=bkg_row)

            # logging.debug("bkg exp rates, errors")
            # logging.debug(bkg_mod._rates)
            # logging.debug(bkg_mod._errs)

            bkg_llh_obj.set_model(bkg_mod)

            bkg_miner.set_llh(bkg_llh_obj)

            bkg_params = {
                pname: bkg_row[pname] for pname in bkg_llh_obj.model.param_names
            }
            # bkg_miner.set_fixed_params(bkg_llh_obj.model.param_names)
            bkg_miner.set_fixed_params(
                list(bkg_params.keys()), values=list(bkg_params.values())
            )

            # bkg_params = {pname:bkg_llh_obj.model.param_dict[pname]['val'] for\
            #                 pname in bkg_llh_obj.model.param_names}
            bkg_nllh = -bkg_llh_obj.get_llh(bkg_params)
            res_dict["bkg_nllh"] = bkg_nllh
            logging.debug("bkg_param_dict: ")
            logging.debug(bkg_miner.param_info_dict)
            logging.debug("bkg_nllh: %.3f" % (bkg_nllh))

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

            logging.debug("Max TS: %.2f" % (np.nanmax(TSs)))

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
                logging.info("%d above TS of %.1f" % (np.sum(TSbl), TSwrite))
                res_dict["TS"] = TSs[TSbl]
                res_dict["imx"] = imxs[TSbl]
                res_dict["imy"] = imys[TSbl]
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

        fname = os.path.join(work_dir, "res_%d_.csv" % (square_row["squareID"]))

        if len(res_dfs2write) > 0:
            res_df = pd.concat(res_dfs2write)
            res_df.to_csv(fname)
            logging.info("Saved results to")
            logging.info(fname)
        # res_written = False
        # conn = get_conn(db_fname)
        # try:
        #     write_square_results(conn, res_df)
        #     res_written = True
        # except Exception as E:
        #     logging.error(E)
        #     logging.error(traceback.format_exc())
        #     logging.error("Problem writing to DB")
        #     conn.close()
        #     conn = get_conn(db_fname, timeout=30.0)
        #     try:
        #         write_square_results(conn, res_df)
        #         res_written = True
        #     except Exception as E:
        #         logging.error(E)
        #         logging.error(traceback.format_exc())
        #         logging.error("Problem writing to DB")
        #         logging.info("Instead writing results to file: ")
        #         logging.info(fname)
        #         res_df.to_csv(fname)
        #
        # if res_written:
        #     try:
        #         update_square_stat(conn, sig_img_tab['timeID'], square_ind)
        #     except Exception as E:
        #         logging.error(E)
        #         logging.error(traceback.format_exc())
        #         logging.error("Problem writing to DB")
        #         conn.close()
        #         conn = get_conn(db_fname, timeout=30.0)
        #         try:
        #             update_square_stat(conn, sig_img_tab['timeID'], square_ind)
        #         except Exception as E:
        #             logging.error(E)
        #             logging.error(traceback.format_exc())
        #             logging.error("Problem writing to DB")
        # conn.close()

        # tab.write(fname, overwrite=True)


def main(args):
    fname = "llh_analysis_from_rate_seeds_" + str(args.job_id)

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

    PC = fits.open(args.pcfname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key="T")

    pcbl = pc >= 0.1
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    conn = get_conn(db_fname)
    if proc_num >= 0:
        square_tab = get_square_tab(conn, proc_group=proc_num)
    else:
        square_tab = get_square_tab(conn)

    rate_res_tab = pd.read_csv(args.rate_fname)

    logging.info("Read in Square and Rates Tables, now to do analysis")

    do_analysis(
        square_tab,
        rate_res_tab,
        pc_imxs,
        pc_imys,
        pl_flux,
        drm_obj,
        rt_dir,
        bkg_llh_obj,
        sig_llh_obj,
        conn,
        db_fname,
        trigtime,
        work_dir,
        bkg_fits_df,
    )
    conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
