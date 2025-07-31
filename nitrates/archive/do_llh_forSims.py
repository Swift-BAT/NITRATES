import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import time
import pandas as pd

# import ..config

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
    get_twinds_tab,
    get_seeds_tab,
    get_info_tab,
    get_files_tab,
    get_square_tab,
    get_full_sqlite_table_as_df,
)
from ..config import EBINS0, EBINS1, solid_angle_dpi_fname, fp_dir
from ..models.flux_models import Plaw_Flux
from ..llh_analysis.minimizers import (
    NLLH_ScipyMinimize_Wjacob,
    imxy_grid_miner,
    NLLH_ScipyMinimize,
)
from ..lib.drm_funcs import DRMs
from ..response.ray_trace_funcs import RayTraces, FootPrints
from ..llh_analysis.LLH import LLH_webins
from ..models.models import (
    Bkg_Model_wSA,
    Point_Source_Model,
    Point_Source_Model_Wuncoded,
    CompoundModel,
    Bkg_Model_wFlatA,
    Point_Source_Model_Binned_Rates,
)
from .do_intllh_scan import (
    kum_mode,
    kum_pdf,
    kum_logpdf,
    kum_deriv_logpdf,
    deriv2_kum_logpdf,
)

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
        default="job_table2.csv",
    )
    parser.add_argument(
        "--rate_fname",
        type=str,
        help="Rate results file name",
        default="rate_seeds2.csv",
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
        "--sim_dir", type=str, help="Name of the simulation directory", default=None
    )
    parser.add_argument(
        "--log_fname", type=str, help="Name for the log file", default="llh_sim"
    )
    parser.add_argument(
        "--dur_min", type=float, help="Min duration to use", default=0.5
    )
    parser.add_argument(
        "--dur_max", type=float, help="Max duration to use", default=4.096
    )
    parser.add_argument(
        "--dt_min", type=float, help="Min time from trig time to use", default=-6.144
    )
    parser.add_argument(
        "--dt_max", type=float, help="Max time from trig time to use", default=4.096
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


def do_analysis(
    sim_params_df,
    sim_tab,
    twind_df,
    pl_flux,
    drm_obj,
    rt_dir,
    fp_dir,
    ev_data,
    bl_dmask,
    ebins0,
    ebins1,
    conn,
    db_fname,
    trigger_time,
    work_dir,
    sim_dir,
    bkg_fname,
    TSwrite=4.5,
):
    conn.close()

    nebins = len(ebins0)

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

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

    for sim_param_ind, sim_param_row in sim_params_df.iterrows():
        logging.info("Starting params_id: %d" % (sim_param_row["params_id"]))

        bl = sim_tab["params_id"] == sim_param_row["params_id"]
        sim_df = sim_tab[bl]

        rt_obj = RayTraces(rt_dir, max_nbytes=6e9)
        fp_obj = FootPrints(fp_dir)

        res_dfs2write = []

        for sim_ind, sim_row in sim_df.iterrows():
            logging.debug(
                "Starting simid: %d, table index: %d" % (sim_row["simid"], sim_ind)
            )
            tstart = sim_row["tstart"]
            tstop = tstart + sim_row["dur"]
            dur = sim_row["dur"]

            imx_sim = sim_row["imx"]
            imy_sim = sim_row["imy"]

            ev_sim_tab = Table.read(sim_row["fname"])

            evdata_ = ev_data.copy()
            evdata = vstack([evdata_, ev_sim_tab])
            evdata.sort("TIME")

            bkg_llh_obj = LLH_webins(evdata, ebins0, ebins1, bl_dmask)
            sig_llh_obj = LLH_webins(evdata, ebins0, ebins1, bl_dmask)

            xax = np.linspace(-1e-3, 1e-3, 2 + 1) + imx_sim
            yax = np.linspace(-1e-3, 1e-3, 2 + 1) + imy_sim
            grids = np.meshgrid(xax, yax)
            imxs = grids[0].ravel()
            imys = grids[1].ravel()
            Npix = len(imxs)

            bl = (twind_df["time"] < (tstop - 0.1 * dur)) & (
                (twind_df["duration"] + twind_df["time"]) > (tstart + 0.1 * dur)
            )
            tgrps = twind_df[bl].groupby("timeID")
            logging.debug("%d timeIDs to do" % (len(tgrps)))

            res_dicts = []

            for timeID, tdf in tgrps:
                res_dict = {}
                res_dict["params_id"] = sim_param_row["params_id"]
                res_dict["simid"] = sim_row["simid"]
                res_dict["timeID"] = timeID

                t0 = tdf["time"].values[0]
                dt = tdf["duration"].values[0]
                tmid = t0 + dt / 2.0
                t1 = t0 + dt
                res_dict["time"] = t0
                res_dict["duration"] = dt

                bkg_llh_obj.set_time(t0, t1)
                sig_llh_obj.set_time(t0, t1)

                bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]

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
                imx_, imy_ = np.nanmean(imxs), np.nanmean(imys)

                # sig_mod = Point_Source_Model(imx_,\
                #                         imy_, 0.3,\
                #                         pl_flux, drm_obj,\
                #                         [ebins0,ebins1], rt_obj, bl_dmask,\
                #                         use_deriv=True)

                sig_mod = Point_Source_Model_Wuncoded(
                    imx_,
                    imy_,
                    0.3,
                    pl_flux,
                    drm_obj,
                    [ebins0, ebins1],
                    rt_obj,
                    fp_obj,
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

                best_ind = np.nanargmax(TSs)
                res_dict["TS"] = TSs[best_ind]
                res_dict["imx"] = imxs[best_ind]
                res_dict["imy"] = imys[best_ind]
                res_dict["A"] = As[best_ind]
                res_dict["ind"] = gammas[best_ind]
                res_dict["sig_nllh"] = sig_nllhs[best_ind]
                res_dicts.append(res_dict)
                # fname = os.path.join(work_dir,\
                #         'res_%d_%d_.fits' %(res_dict['timeID'],\
                #         res_dict['squareID']))

                # TSbl = (TSs>=TSwrite)
                # if np.sum(TSbl) > 0:
                #     logging.info("%d above TS of %.1f"%(np.sum(TSbl),TSwrite))
                #     res_dict['TS'] = TSs[TSbl]
                #     res_dict['imx'] = imxs[TSbl]
                #     res_dict['imy'] = imys[TSbl]
                #     res_dict['A'] = As[TSbl]
                #     res_dict['ind'] = gammas[TSbl]
                #     res_dict['sig_nllh'] = sig_nllhs[TSbl]
                #     # res_dict['fname'] = fname
                #     res_dfs2write.append(pd.DataFrame(res_dict))

            res_df = pd.DataFrame(res_dicts)
            res_dfs2write.append(res_df)

        fname = os.path.join(
            sim_dir, "res_paramsID_%d_.csv" % (sim_param_row["params_id"])
        )

        res_df = pd.concat(res_dfs2write, ignore_index=True)
        res_df.to_csv(fname, index=False)
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
    # ev_data = fits.open(evfname)[1].data
    ev_data = Table.read(evfname)
    dmask_fname = files_tab["detmask"][0]
    dmask = fits.open(dmask_fname)[0].data
    bl_dmask = dmask == 0.0
    logging.debug("Opened up event and detmask files")

    bkg_fits_df = pd.read_csv(args.bkg_fname)

    sim_params_df = pd.read_csv(os.path.join(args.sim_dir, "sim_param_table.csv"))
    sim_tab = pd.read_csv(os.path.join(args.sim_dir, "sim_table.csv"))

    bl = sim_params_df["im_id"] == args.job_id
    sim_params_df = sim_params_df[bl]
    Nsims = len(sim_params_df)

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

    twind_df = get_twinds_tab(conn)

    bl = (
        ((twind_df["time"] - trigtime) >= args.dt_min)
        & ((twind_df["duration"] + twind_df["time"] - trigtime) < args.dt_max)
        & (twind_df["duration"] >= args.dur_min)
        & (twind_df["duration"] <= args.dur_max)
    )

    twind_df = twind_df[bl]
    logging.info("Got TimeWindows table")

    do_analysis(
        sim_params_df,
        sim_tab,
        twind_df,
        pl_flux,
        drm_obj,
        rt_dir,
        fp_dir,
        ev_data,
        bl_dmask,
        ebins0,
        ebins1,
        conn,
        db_fname,
        trigtime,
        work_dir,
        args.sim_dir,
        args.bkg_fname,
    )
    # do_analysis(square_tab, rate_res_tab, good_pix['imx'], good_pix['imy'], pl_flux,\
    #                 drm_obj, rt_dir,\
    #                 bkg_llh_obj, sig_llh_obj,\
    #                 conn, db_fname, trigtime, work_dir,bkg_fits_df)
    conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
