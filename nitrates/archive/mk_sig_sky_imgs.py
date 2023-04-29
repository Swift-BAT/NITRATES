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
        "--dbfname", type=str, help="Name of the database", default=None
    )
    parser.add_argument("--Nproc", type=int, help="Number of procs to use", default=4)
    parser.add_argument("--oversamp", type=int, help="Overampling to use", default=2)
    parser.add_argument(
        "--dt0", type=float, help="Start time relative to trig_time", default=-5.12
    )
    parser.add_argument(
        "--dt1", type=float, help="End time relative to trig_time", default=5.12
    )
    parser.add_argument("--dt_step", type=float, help="Time Step", default=0.512)
    parser.add_argument("--dt", type=float, help="Expsoure time", default=1.024)
    parser.add_argument("--trigtime", type=float, help="Trigger Time", default=None)
    parser.add_argument(
        "--detapp", help="Use the detection aperture", action="store_true"
    )
    parser.add_argument("--norebal", help="Don't rebalance image", action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    fname = "mk_sig_sky_imgs"

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
    pc_fname = do_pc(
        dmask, att_fname, work_dir, ovrsmp=args.oversamp, detapp=args.detapp
    )

    bkg_t1 = trig_time - 20.0
    bkg_t0 = max(min_ev_time, bkg_t1 - 20.0)
    bkg_dt = bkg_t1 - bkg_t0

    logging.info("Making the bkg DPI")
    dpi_bkg_fname = do_bkg(bkg_t0, bkg_t1, ev_fname, dmask, work_dir)
    logging.info("DPI bkg from MET %.3f - %.3f" % (bkg_t0, bkg_t1))

    tbins0 = np.arange(args.dt0, args.dt1, args.dt_step)[:-1] + trig_time
    tbins1 = tbins0 + args.dt
    Ntbins = len(tbins0)
    Nimgs = Ntbins
    dts = tbins1 - tbins0

    conn.close()

    logging.info("Starting image creation and search for %d time frames" % (Nimgs))

    rebal = True
    if args.norebal:
        rebal = False
    logging.info("Rebal is: " + str(rebal))

    mk_sig_imgs_mp(
        args.Nproc,
        tbins0,
        dts,
        ev_fname,
        dpi_bkg_fname,
        pc_fname,
        att_fname,
        dmask,
        work_dir,
        oversamp=args.oversamp,
        detapp=args.detapp,
        rebal=rebal,
    )

    logging.info("Done with mk_sig_sky_imgs, now exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
