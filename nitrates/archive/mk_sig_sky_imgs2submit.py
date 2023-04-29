import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback, time

from ..lib.funcs2run_bat_tools import (
    std_grb,
    do_bkg,
    do_pc,
    mk_sig_imgs_mp,
    mk_sky_sig_img,
    mk_sig_imgs_pix,
)
from ..lib.sqlite_funcs import get_conn, timeID2time_dur, write_cat2db
from ..lib.dbread_funcs import (
    get_top_rate_timeIDs,
    get_files_tab,
    guess_dbfname,
    get_info_tab,
)
from ..config import EBINS0, EBINS1


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument("--workdir", type=str, help="Work Directory", default=None)
    parser.add_argument(
        "--savedir", type=str, help="Directory to save tmp files to", default="/tmp/"
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name of the database", default=None
    )
    parser.add_argument("--Nproc", type=int, help="Number of procs to use", default=1)
    parser.add_argument(
        "--dt0", type=float, help="Start time relative to trig_time", default=-15.12
    )
    parser.add_argument(
        "--dt1", type=float, help="End time relative to trig_time", default=15.12
    )
    parser.add_argument("--TScut", type=float, help="Rate TS to cut at", default=2.0)
    parser.add_argument("--trigtime", type=float, help="trigger time in MET secs")
    args = parser.parse_args()
    return args


def main(args):
    fname = "mk_sig_sky_imgs_%.2f_%.2f" % (args.dt0, args.dt1)

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    time_starting = time.time()

    if args.dbfname is None or "none" in args.dbfname:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB: " + db_fname)
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    if args.trigtime is None:
        trig_time = info_tab["trigtimeMET"][0]
    else:
        trig_time = args.trigtime
    min_ev_time = info_tab["tstartMET"][0]

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    if args.workdir is None:
        work_dir = files_tab["workDir"][0]
    else:
        work_dir = args.workdir
    if args.evfname is None:
        ev_fname = files_tab["evfname"][0]
    else:
        ev_fname = args.evfname
    if args.dmask is None:
        dmask = files_tab["detmask"][0]
    else:
        dmask = args.dmask
    att_fname = files_tab["attfname"][0]

    logging.info("Making PC image if it doesn't already exist")
    pc_fname = do_pc(dmask, att_fname, work_dir, ovrsmp=2)

    bkg_t1 = trig_time + args.dt0 - 10.0
    bkg_t0 = max(min_ev_time, bkg_t1 - 20.0)
    bkg_dt = bkg_t1 - bkg_t0

    logging.info("Making the bkg DPI")
    dpi_bkg_fname = do_bkg(bkg_t0, bkg_t1, ev_fname, dmask, work_dir)
    logging.info("DPI bkg from MET %.3f - %.3f" % (bkg_t0, bkg_t1))

    timeIDs, RateTSvals = get_top_rate_timeIDs(conn, TScut=args.TScut, ret_TSs=True)

    times = np.zeros(len(timeIDs))
    dts = np.zeros(len(timeIDs))

    logging.info("Getting times and exposures")
    for i in range(len(timeIDs)):
        times[i], dts[i] = timeID2time_dur(timeIDs[i], trig_time)

    tbl = ((times - trig_time) >= args.dt0) & ((times - trig_time) < args.dt1)
    times = times[tbl]
    dts = dts[tbl]
    Nimgs = np.sum(tbl)
    RateTSs = RateTSvals[tbl]
    snr_cuts = 10 * np.ones(np.sum(tbl))
    snr_cuts[(RateTSs > 2)] = 3
    snr_cuts[(RateTSs > 2.5)] = 2.5
    snr_cuts[(RateTSs > 3)] = 2.0
    snr_cuts[(RateTSs > 4)] = 1.5

    # conn.close()

    logging.info("Starting image creation and search for %d time frames" % (Nimgs))

    savedir = os.path.join(args.savedir, str(int(trig_time)))
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    mk_sig_imgs_pix(
        times,
        dts,
        ev_fname,
        dpi_bkg_fname,
        pc_fname,
        att_fname,
        dmask,
        savedir,
        work_dir,
        trig_time,
        conn,
        db_fname,
        oversamp=2,
        snr_cuts=snr_cuts,
    )

    logging.info("Done with mk_sig_sky_imgs, now exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
