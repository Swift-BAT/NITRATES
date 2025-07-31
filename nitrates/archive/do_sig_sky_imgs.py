import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback, time

from ..lib.funcs2run_bat_tools import (
    std_grb,
    do_bkg,
    do_pc,
    get_sig_pix_mp,
    mk_sky_sig_img,
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
    parser.add_argument(
        "--dbfname", type=str, help="Name of the database", default=None
    )
    parser.add_argument(
        "--TScut", type=float, help="Min rate TS to make an image for", default=2
    )
    parser.add_argument(
        "--Nimgs", type=int, help="Number of images to make", default=50
    )
    parser.add_argument("--Nproc", type=int, help="Number of procs to use", default=4)
    parser.add_argument("--os", type=int, help="Oversampling", default=2)
    args = parser.parse_args()
    return args


def main(args):
    fname = "mk_sig_sky_img"

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    f = open(fname + ".pid", "w")
    f.write(str(os.getpid()))
    f.close()

    logging.info("Wrote pid: %d" % (os.getpid()))

    time_starting = time.time()

    # ebins0 = np.append([EBINS0[0]], EBINS0[:-1])
    # ebins1 = np.append([EBINS1[-1]], EBINS1[1:])
    # ebins0 = EBINS0[0]
    # ebins1 = EBINS1[-1]
    # Nebins = len(ebins0)
    # logging.info("ebins0:")
    # logging.info(ebins0)
    # logging.info("ebins1:")
    # logging.info(ebins1)

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB: " + db_fname)
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    trig_time = info_tab["trigtimeMET"][0]
    min_ev_time = info_tab["tstartMET"][0]

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    work_dir = files_tab["workDir"][0]
    ev_fname = files_tab["evfname"][0]
    dmask = files_tab["detmask"][0]
    att_fname = files_tab["attfname"][0]

    TScut = args.TScut

    logging.info("Making PC image if it doesn't already exist")
    pc_fname = do_pc(dmask, att_fname, work_dir)

    bkg_t1 = trig_time - 20.0
    bkg_t0 = max(min_ev_time, bkg_t1 - 20.0)
    bkg_dt = bkg_t1 - bkg_t0

    dpi_bkg_fnames = []
    logging.info("Making the bkg DPIs")
    # for i in xrange(Nebins):
    dpi_bkg_fname = do_bkg(bkg_t0, bkg_t1, ev_fname, dmask, work_dir)
    logging.info("DPI bkg from MET %.3f - %.3f" % (bkg_t0, bkg_t1))

    timeIDs, RateTSvals = get_top_rate_timeIDs(conn, TScut=TScut, ret_TSs=True)
    Nimgs = len(timeIDs)
    conn.close()
    times = np.zeros(Nimgs)
    dts = np.zeros(Nimgs)

    logging.info("Getting times and exposures")
    for i in range(Nimgs):
        times[i], dts[i] = timeID2time_dur(timeIDs[i], trig_time)

    logging.info("Starting image creation and search for %d time frames" % (Nimgs))

    get_sig_pix_mp(
        args.Nproc,
        times,
        dts,
        ev_fname,
        dpi_bkg_fname,
        pc_fname,
        att_fname,
        dmask,
        work_dir,
        db_fname=db_fname,
        timeIDs=timeIDs,
        RateTSs=RateTSvals,
        oversamp=args.os,
    )

    logging.info("Done with do_sig_sky_imgs, now exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
