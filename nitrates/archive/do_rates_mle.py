import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback

from ..config import quad_dicts, EBINS0, EBINS1, drm_quad_dir
from ..lib.sqlite_funcs import get_conn, append_rate_tab
from ..lib.dbread_funcs import (
    get_info_tab,
    guess_dbfname,
    get_files_tab,
    get_twinds_tab,
)
from ..analysis_seeds.bkg_rate_estimation import get_quad_rate_objs_from_db
from ..analysis_seeds.mle_rates_for_realtime import (
    do_rate_mle,
    do_rate_mle_mp,
    get_abs_cor_rates,
    get_cnts_intp_obj,
    get_quad_cnts_tbins,
)
from ..lib.counting_and_quad_funcs import get_quad_cnts_tbins_fast
from ..lib.drm_funcs import get_ebin_ind_edges


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--nproc", type=int, help="Number of processors to use", default=4
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename="rates_llh_analysis.log",
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

    drm_dir = files_tab["drmDir"][0]

    evfname = files_tab["evfname"][0]
    # dmfname = files_tab['detmask'][0]
    ev_data = fits.open(evfname)[1].data
    # dmask = fits.open(dmfname)[0].data
    tmin = trigtime - 100.0
    tmax = trigtime + 100.0
    # logging.info('Filtering Event Data')
    # evdata = filter_evdata(evdata0, dmask, ebins0[0], ebins1[-1], tmin, tmax)
    logging.debug("Opened up event file")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # probably get times from twind table

    twind_df = get_twinds_tab(conn)

    logging.info("Got TimeWindows table")

    logging.info("Getting rate fits from DB")

    avg_rate_quad_dict, lin_rate_quad_dict = get_quad_rate_objs_from_db(
        conn, quad_dicts
    )

    logging.info("Finished making quad rate fit objects")

    min_bin_size = np.min(twind_df["duration"])

    logging.info("Smallest duration to test is %.3fs" % (min_bin_size))

    exp_groups = twind_df.groupby("duration")

    nexps = len(exp_groups)

    ind_ax = np.linspace(-1.5, 3.5, 10 * 5 + 1)

    for i, exp_group in enumerate(exp_groups):
        logging.info("Starting duration size %d of %d" % (i + 1, nexps))

        df_twind = exp_group[1]

        t_bins0 = df_twind["time"].values
        t_bins1 = df_twind["time_end"].values

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

        quad_cnts_mat = get_quad_cnts_tbins_fast(
            t_bins0, t_bins1, ebins0, ebins1, ev_data
        )

        logging.info("Finished making quad_cnts_mat")

        for direction, quad_dict in quad_dicts.items():
            # I still have to import and define like all of this
            # how to store rate results in the loop?
            # put into a DataFrame? would make it easy to write it

            if quad_dict["id"] > 4:
                continue

            drm = fits.open(os.path.join(drm_dir, quad_dict["drm_fname"]))

            ebin_ind_edges = get_ebin_ind_edges(drm, ebins0, ebins1)

            imx = quad_dict["imx"]
            imy = quad_dict["imy"]

            abs_cor = get_abs_cor_rates(imx, imy, drm)

            cnts_intp = get_cnts_intp_obj(ind_ax, drm, ebin_ind_edges, abs_cor)

            cnts_per_tbin = np.sum(
                [quad_cnts_mat[:, :, q] for q in quad_dict["quads"]], axis=0
            )

            logging.info("Doing rate MLE analysis for " + direction)

            if args.nproc > 1:
                bkg_llh_tbins, llhs, bf_nsigs, bf_inds = do_rate_mle_mp(
                    cnts_per_tbin,
                    lin_rate_quad_dict[direction],
                    cnts_intp,
                    t_bins0,
                    t_bins1,
                    nproc=args.nproc,
                    bkg_err_fact=2.0,
                )
            else:
                bkg_llh_tbins, llhs, bf_nsigs, bf_inds = do_rate_mle(
                    cnts_per_tbin,
                    lin_rate_quad_dict[direction],
                    cnts_intp,
                    t_bins0,
                    t_bins1,
                    bkg_err_fact=2.0,
                )

            logging.info("Finished rate MLE analysis for " + direction)

            append_rate_tab(
                conn, df_twind, quad_dict["id"], bkg_llh_tbins, llhs, bf_nsigs, bf_inds
            )

            logging.info("Appended rate results to DB")


if __name__ == "__main__":
    args = cli()

    main(args)
