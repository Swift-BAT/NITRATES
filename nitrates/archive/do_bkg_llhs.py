import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback

from ..analysis_seeds.bkg_rate_estimation import rate_obj_from_sqltab
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_rate_fits_tab, guess_dbfname
from ..config import EBINS0, EBINS1

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
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    args = parser.parse_args()
    return args


def do_bkg_analysis_mp(i_proc, nprocs):
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
    dmask = files_tab["detmask"][0].data
    bl_dmask = dmask == 0.0

    logging.debug("Opened up event and detmask files")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # probably get times from twind table

    rate_fits_df = get_rate_fits_tab(conn)

    bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    twind_df = get_twinds_tab(conn)

    logging.info("Got TimeWindows table")

    logging.info("Getting rate fits from DB")

    min_bin_size = np.min(twind_df["duration"])

    logging.info("Smallest duration to test is %.3fs" % (min_bin_size))

    llh_bkg = get_bkg_llh_obj(
        ev_data,
        ebins0,
        ebins1,
        bl_dmask,
        bkg_rates_obj,
        twind_df["time"].values[0],
        min_bin_size,
    )
    miner.set_llh(llh_bkg)

    t_bins0 = twind_df["time"].values
    t_bins1 = twind_df["time_end"].values

    t_bins0 = t_bins0[i_proc:][::nprocs]
    t_bins1 = t_bins1[i_proc:][::nprocs]

    ntbins = len(t_bins0)

    logging.debug("There are %d time0 bins" % (len(t_bins0)))
    logging.debug("There are %d time1 bins" % (len(t_bins1)))
    logging.debug(
        "min(t_bins0), max(t_bins0): %.3f, %.3f" % (np.min(t_bins0), np.max(t_bins0))
    )
    logging.debug(
        "min(t_bins1), max(t_bins1): %.3f, %.3f" % (np.min(t_bins1), np.max(t_bins1))
    )

    for i in range(ntbins):
        # pretty sure I don't have to do miner.set_llh() again
        dt = t_bins1[i] - t_bins0[i]
        llh_bkg.set_time(t_bins0[i], dt)
        bf_vals, bf_nllhs, ress = miner.minimize()

        # then write the result


def main(args):
    logging.basicConfig(
        filename="bkg_llh_analysis.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    # probably want to move this entire thing to a seperate function
    # so that they can be launched as seperate processes

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
    dmask = files_tab["detmask"][0].data
    bl_dmask = dmask == 0.0

    logging.debug("Opened up event and detmask files")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # probably get times from twind table

    rate_fits_df = get_rate_fits_tab(conn)

    bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    twind_df = get_twinds_tab(conn)

    logging.info("Got TimeWindows table")

    logging.info("Getting rate fits from DB")

    min_bin_size = np.min(twind_df["duration"])

    logging.info("Smallest duration to test is %.3fs" % (min_bin_size))

    exp_groups = twind_df.groupby("duration")

    nexps = len(exp_groups)

    miner = NLLH_ScipyMinimize("")

    llh_bkg = get_bkg_llh_obj(
        ev_data,
        ebins0,
        ebins1,
        bl_dmask,
        bkg_rates_obj,
        twind_df["time"].values[0],
        min_bin_size,
    )
    miner.set_llh(llh_bkg)

    for ii, exp_group in enumerate(exp_groups):
        logging.info("Starting duration size %d of %d" % (ii + 1, nexps))

        df_twind = exp_group[1]

        t_bins0 = df_twind["time"].values
        t_bins1 = df_twind["time_end"].values

        dt = t_bins1[0] - t_bins0[0]

        ntbins = len(t_bins0)

        logging.debug("There are %d time0 bins" % (len(t_bins0)))
        logging.debug("There are %d time1 bins" % (len(t_bins1)))
        logging.debug(
            "min(t_bins0), max(t_bins0): %.3f, %.3f"
            % (np.min(t_bins0), np.max(t_bins0))
        )
        logging.debug(
            "min(t_bins1), max(t_bins1): %.3f, %.3f"
            % (np.min(t_bins1), np.max(t_bins1))
        )

        for i in range(ntbins):
            # pretty sure I don't have to do miner.set_llh() again
            llh_bkg.set_time(t_bins0[i], dt)
            bf_vals, bf_nllhs, ress = miner.minimize()

            # then write the result


if __name__ == "__main__":
    args = cli()

    main(args)
