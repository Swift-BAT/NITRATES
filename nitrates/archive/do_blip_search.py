import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback, time

from ..lib.funcs2run_bat_tools import std_grb, do_bkg, do_pc, do_search_mp
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
        "--Nimgs", type=int, help="Number of images to make", default=50
    )
    parser.add_argument("--Nproc", type=int, help="Number of procs to use", default=2)
    args = parser.parse_args()
    return args


def main(args):
    fname = "blip_search"

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
    ebins0 = [EBINS0[0]]
    ebins1 = [EBINS1[-1]]
    Nebins = len(ebins0)
    logging.info("ebins0:")
    logging.info(ebins0)
    logging.info("ebins1:")
    logging.info(ebins1)

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

    Nimgs = args.Nimgs

    logging.info("Making PC image if it doesn't already exist")
    pc_fname = do_pc(dmask, att_fname, work_dir)

    bkg_t1 = trig_time - 20.0
    bkg_t0 = max(min_ev_time, bkg_t1 - 20.0)
    bkg_dt = bkg_t1 - bkg_t0

    dpi_bkg_fnames = []
    logging.info("Making the bkg DPIs")
    for i in range(Nebins):
        dpi_bkg_fnames.append(
            do_bkg(
                bkg_t0, bkg_t1, ev_fname, dmask, work_dir, e0=ebins0[i], e1=ebins1[i]
            )
        )
    logging.info("DPI bkg from MET %.3f - %.3f" % (bkg_t0, bkg_t1))

    timeIDs = get_top_rate_timeIDs(conn, N=Nimgs)
    conn.close()
    times = np.zeros(Nimgs)
    dts = np.zeros(Nimgs)

    logging.info("Getting times and exposures")
    for i in range(Nimgs):
        times[i], dts[i] = timeID2time_dur(timeIDs[i], trig_time)

    logging.info("Starting image creation and search for %d time frames" % (Nimgs))

    if args.Nproc > 1:
        do_search_mp(
            args.Nproc,
            times,
            dts,
            ev_fname,
            dpi_bkg_fnames,
            pc_fname,
            att_fname,
            dmask,
            work_dir,
            e0=ebins0,
            e1=ebins1,
            db_fname=db_fname,
            timeIDs=timeIDs,
        )

    else:
        for i in range(Nimgs):
            cat_fname = std_grb(
                times[i],
                dts[i],
                ev_fname,
                dpi_bkg_fname,
                att_fname,
                dmask,
                work_dir,
                pc=pc_fname,
            )
            logging.info("Finished search of time frame %d of %d" % ((i + 1), Nimgs))

            conn = get_conn(db_fname)
            try:
                write_cat2db(conn, cat_fname, timeIDs[i])
                logging.info("Wrote results from " + cat_fname + " into DB")
            except Exception as E:
                logging.error(str(E))
                logging.error(traceback.format_exc())
                logging.warning(
                    "Failed to write results from " + cat_fname + " into DB"
                )
                logging.info("Trying again")
                conn.close()
                time.sleep(1.0)
                conn = get_conn(db_fname)
                try:
                    write_cat2db(conn, cat_fname, timeIDs[i])
                    logging.info("Wrote results from " + cat_fname + " into DB")
                except Exception as E:
                    logging.error(str(E))
                    logging.error(traceback.format_exc())
                    logging.error(
                        "Failed to write results from " + cat_fname + " into DB"
                    )
                    logging.error("And not trying again")
            conn.close()

    logging.info("Done with do_blip_search, now exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
