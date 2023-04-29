import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback

from ..analysis_seeds.bkg_rate_estimation import get_avg_lin_cub_rate_quad_obs
from ..config import quad_dicts, EBINS0, EBINS1
from ..lib.sqlite_funcs import write_rate_fits_from_obj, get_conn
from ..lib.dbread_funcs import get_info_tab, guess_dbfname, get_files_tab
from ..lib.event2dpi_funcs import filter_evdata


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--twind",
        type=float,
        help="Number of seconds to go +/- from the trigtime",
        default=15,
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename="bkg_rate_estimation.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

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
    dmfname = files_tab["detmask"][0]
    ev_data = fits.open(evfname)[1].data
    dmask = fits.open(dmfname)[0].data
    tmin = trigtime - 100.0
    tmax = trigtime + 100.0
    # logging.info('Filtering Event Data')
    # evdata = filter_evdata(evdata0, dmask, ebins0[0], ebins1[-1], tmin, tmax)
    logging.debug("Opened up event and detmask files")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    logging.info("Getting rate objects for each quad")
    avg_quad_obj, lin_quad_obj, cub_quad_obj = get_avg_lin_cub_rate_quad_obs(
        quad_dicts,
        ev_data,
        trigtime,
        ebins0,
        ebins1,
        poly_trng=args.twind,
        trng=args.twind + 45,
    )

    logging.info("Now writing rate fit results to DB")

    for k, obj in avg_quad_obj.items():
        write_rate_fits_from_obj(conn, obj, quad_dicts[k]["id"])

    for k, obj in lin_quad_obj.items():
        write_rate_fits_from_obj(conn, obj, quad_dicts[k]["id"])


if __name__ == "__main__":
    args = cli()

    main(args)
