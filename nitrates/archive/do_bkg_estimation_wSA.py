import numpy as np
from astropy.io import fits
import os
import argparse
import logging, traceback
import pandas as pd

from ..analysis_seeds.bkg_rate_estimation import get_avg_lin_cub_rate_quad_obs
from ..config import quad_dicts, EBINS0, EBINS1, solid_angle_dpi_fname
from ..lib.sqlite_funcs import write_rate_fits_from_obj, get_conn
from ..lib.dbread_funcs import get_info_tab, guess_dbfname, get_files_tab
from ..lib.event2dpi_funcs import filter_evdata
from ..models.models import Bkg_Model_wSA
from ..llh_analysis.LLH import LLH_webins
from ..llh_analysis.minimizers import NLLH_ScipyMinimize, NLLH_ScipyMinimize_Wjacob


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
    parser.add_argument("--bkg_dur", type=float, help="bkg duration", default=28.0)
    parser.add_argument(
        "--bkg_nopost",
        help="Don't use time after signal window for bkg",
        action="store_true",
    )
    parser.add_argument(
        "--bkg_nopre",
        help="Don't use time before signal window for bkg",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename="bkg_rate_estimation_wSA.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    if args.bkg_nopost and args.bkg_nopre:
        raise Exception("Can't have no pre and no post")

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
    tstart = trigtime - args.twind
    tstop = trigtime + args.twind

    evfname = files_tab["evfname"][0]
    dmfname = files_tab["detmask"][0]
    ev_data = fits.open(evfname)[1].data
    dmask = fits.open(dmfname)[0].data
    bl_dmask = dmask == 0.0
    logging.debug("Opened up event and detmask files")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    nebins = len(ebins0)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    solid_angle_dpi = np.load(solid_angle_dpi_fname)

    bkg_mod = Bkg_Model_wSA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)

    bkg_miner = NLLH_ScipyMinimize_Wjacob("")

    bkg_tstep = 1 * 1.024
    # bkg_dur = 40*1.024
    # sig_wind = 20*1.024
    bkg_dur = args.bkg_dur * 1.024
    sig_wind = 10 * 1.024

    bkg_t0s = (
        np.arange(-args.twind * 1.024, args.twind * 1.024 + 1, bkg_tstep)
        + trigtime
        - (bkg_dur + sig_wind) / 2.0
    )

    llh_bkg = LLH_webins(
        ev_data,
        ebins0,
        ebins1,
        bl_dmask,
        t0=trigtime,
        t1=trigtime + bkg_dur,
        model=bkg_mod,
    )
    bkg_miner.set_llh(llh_bkg)

    bkg_bf_dicts = []
    bkg_tax = []

    logging.info("%d times to do" % (len(bkg_t0s)))

    for i in range(len(bkg_t0s)):
        if args.bkg_nopost:
            t0s = np.array([bkg_t0s[i]])
        elif args.bkg_nopre:
            t0s = np.array([bkg_t0s[i] + bkg_dur / 2.0 + sig_wind])
        else:
            t0s = np.array([bkg_t0s[i], bkg_t0s[i] + bkg_dur / 2.0 + sig_wind])
        t1s = t0s + bkg_dur / 2.0
        # bkg_tax.append((t0s[0]+t1s[1])/2.)
        bkg_tax.append(bkg_t0s[i] + (bkg_dur + sig_wind) / 2.0)
        llh_bkg.set_time(t0s, t1s)

        bf_params = {}
        for e0 in range(nebins):
            bkg_miner.set_fixed_params(bkg_miner.param_names)
            e0_pnames = [
                pname for pname in bkg_miner.param_names if int(pname[-1]) == e0
            ]
            #         print e0_pnames
            bkg_miner.set_fixed_params(e0_pnames, fixed=False)
            llh_bkg.set_ebin(e0)

            bf_vals, bkg_nllh, res = bkg_miner.minimize()
            for ii, pname in enumerate(e0_pnames):
                bf_params[pname] = bf_vals[0][ii]
        bkg_bf_dicts.append(bf_params)

    bkg_tax = np.array(bkg_tax)
    bkg_df = pd.DataFrame(bkg_bf_dicts)
    bkg_df["time"] = bkg_tax
    bkg_df["dt"] = bkg_tax - trigtime

    save_fname = "bkg_estimation.csv"
    logging.info("Saving results in a DataFrame to file: ")
    logging.info(save_fname)
    bkg_df.to_csv(save_fname, index=False)


if __name__ == "__main__":
    args = cli()

    main(args)
