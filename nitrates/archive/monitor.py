import os
import argparse
import logging, traceback
import time

from ..lib.dbread_funcs import (
    get_info_tab,
    guess_dbfname,
    get_files_tab,
    get_twinds_tab,
)
from ..lib.sqlite_funcs import get_conn

# from dbread import get_files_tab
from ..lib.helper_funcs import send_email, send_email_attach

# from plot_funcs import mk_rates_plot


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument("--gwname", type=str, help="GW identifier", default=None)
    args = parser.parse_args()
    return args


def is_there_event_data(conn):
    # return True if there's event data, False if there's not
    # check the files table
    ev = False

    df = get_files_tab(conn)
    if len(df) > 0:
        ev = True

    return ev


def email_stuff():
    return


def main(args):
    # have several functions that are run on each loop
    # the function can stop once it already did it's job
    # like making a plot of the rates, if there's a rate plot
    # then the don't do the function or just have the func return

    # send email when there's event data
    # send rates plot when the rate fits are done
    # send the top results of the rates analysis
    # send the top snr of the blips
    # periodically send the top results of the LLH analysis
    # I can probably take the tail of the sorted df and dump .to_html()
    # into the email
    # Possibly try to figure out if something with the submitted
    # jobs have gone wrong, like looking if some job_ids are lacking

    # should probably shift emailing focus to just where there's
    # problems or maybe at the very end
    # Have things like the plots and html tables put into
    # the realtime_workdir/results/S*/ folder

    logging.basicConfig(
        filename="monitor.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    script_start = time.time()
    dt = 0.0

    isev = False

    if args.gwname is not None:
        gwname = args.gwname

    if args.dbfname is None:
        db_fname = guess_dbfname()
    else:
        db_fname = args.dbfname

    while dt < (24.0 * 3600.0):
        conn = get_conn(db_fname)

        if not isev:
            isev = is_there_event_data(conn)
            # if isev: should this be deleted? incomplete code

        fnames = os.listdir(".")

        #

        conn.close()
        time.sleep(5 * 60.0)


if __name__ == "__main__":
    args = cli()

    main(args)
