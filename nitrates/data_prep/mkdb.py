import numpy as np
import sqlite3
from astropy.io import fits
from astropy.time import Time
import os
import sys
import argparse
import logging, traceback

from ..lib.time_funcs import met2astropy
from ..lib.sqlite_funcs import (
    get_sql_tab_list,
    create_tables,
    get_conn,
    setup_tab_info,
    setup_tab_twinds,
    setup_files_tab,
    setup_tab_twind_status,
)
from ..lib.dbread_funcs import get_info_tab, get_twinds_tab
from ..lib.event2dpi_funcs import filter_evdata


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drm_dir", type=str, help="drm_directory", default=None)
    parser.add_argument("--rt_dir", type=str, help="rt_directory", default=None)
    parser.add_argument(
        "--work_dir", type=str, help="Directory to work in", default=None
    )
    parser.add_argument(
        "--data_dbfname",
        type=str,
        help="DB file name with information on the BAT data already downloaded from the QL site",
        default="/gpfs/group/jak51/default/nitrates_realtime/NITRATES/data_scraping/BATQL.db",
    )
    parser.add_argument(
        "--att_dname",
        type=str,
        help="Directory name that contains merged attfiles over chunks of time",
        default="/gpfs/group/jak51/default/realtime_workdir/merged_atts/",
    )
    parser.add_argument(
        "--enb_dname",
        type=str,
        help="Directory name that contains merged enable/disable files over chunks of time",
        default="/gpfs/group/jak51/default/realtime_workdir/merged_enbs/",
    )
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="detmask file name", default=None)
    parser.add_argument("--obsid", type=str, help="Obsid", default=None)
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--att_fname", type=str, help="Fname for that att file", default=None
    )
    parser.add_argument(
        "--trig_time",
        type=str,
        help="Time of trigger, in either MET or a datetime string",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.work_dir == "/gpfs/scratch/jjd330/bat_data/" and args.obsid is not None:
        work_dir = os.path.join(args.work_dir, args.obsid)
    else:
        work_dir = args.work_dir

    bat_ml_dir = os.getcwd()

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    os.chdir(work_dir)

    logging.basicConfig(
        filename="making_db.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    logging.info("Creating Database")

    if args.dbfname is None:
        dbfname = "results.db"
    elif args.dbfname[-2:] != "db":
        dbfname = args.dbfname + ".db"
    else:
        dbfname = args.dbfname

    conn = get_conn(dbfname)

    logging.info("Database %s created" % (dbfname))

    sql_tab_list = get_sql_tab_list()

    logging.info("Initializing tables")

    create_tables(conn, sql_tab_list)

    logging.info("Tables Created")

    conn.close()

    # DB has been Initialized
    # Now it's time to wait for data

    # if args.att_fname is None:
    #     logging.info("Looking for att file")
    #     if os.path.exists('auxil/'):
    #         aux_dir = 'auxil/'
    #         att_fname = [fname for fname in os.listdir(aux_dir) if 'pat' in fname]
    #         if len(att_fname) == 1:
    #             att_fname = os.path.join(aux_dir, att_fname[0])
    #         else:
    #             att_fname = [fname for fname in os.listdir(aux_dir) if 'sat' in fname]
    #             if len(att_fname) == 1:
    #                 att_fname = os.path.join(aux_dir, att_fname[0])
    #             else:
    #                 att_fname = None
    #     else:
    #         att_fname = None
    # else:
    #     att_fname = None
    #
    # if att_fname is None:
    #     logging.info("No att file available")
    # else:
    #     logging.info("Attitude is located in " + att_fname)
    #
    #
    #
    #
    # if args.evfname is None:
    #     logging.info("Looking for bat event file")
    #     if os.path.exists('bat/event/'):
    #         ev_dir = 'bat/event/'
    #         evfname = [fname for fname in os.listdir(ev_dir) if 'bevs' in fname]
    #         if len(evfname) > 0:
    #             evfname = os.path.join(ev_dir, evfname[0])
    #         else:
    #             evfname = None
    #     else:
    #         evfname = None
    # else:
    #     evfname = args.evfname
    #
    # if evfname is None:
    #     logging.info("No bat event file available")
    # else:
    #     logging.info("Bat event data is located in " + evfname)
    #
    #
    # if evfname is None:
    #     logging.info("No BAT data, nothing else to do")
    #     return
    #
    # if 'T' in args.trig_time:
    #     apy_trig_time = Time(args.trig_time, format='isot')
    # elif '-' in args.trig_time:
    #     apy_trig_time = Time(args.trig_time, format='iso')
    # else:
    #     met_trig_time = float(args.trig_time)
    #     if evfname is not None:
    #         apy_trig_time = met2astropy(met_trig_time, evfname)
    #     else:
    #         apy_trig_time = None
    #
    #
    # ev_data = fits.open(evfname)
    # tmin = np.min(ev_data[1].data['TIME'])
    #
    # logging.info("Writing the info table")
    # logging.debug("apy_trig_time: " + str(apy_trig_time))
    # setup_tab_info(conn, ev_data, apy_trig_time)
    #
    # sys.path.append(bat_ml_dir)
    #
    # if args.rt_dir is None:
    #     from ..config import rt_dir
    # else:
    #     rt_dir = args.rt_dir
    # if args.drm_dir is None:
    #     from ..config import drm_dir
    # else:
    #     drm_dir = args.drm_dir
    #
    # detmask = fits.open(args.dmask)[0].data
    # evdata = filter_evdata(ev_data[1].data, detmask, 14., 195., tmin-1., 1e10)
    # ev_data[1].data = evdata
    # ev_fname = os.path.join(work_dir, 'filter_evdata.fits')
    # ev_data.writeto(ev_fname)
    #
    # # logging.info("Writing the Files table")
    # # setup_files_tab(conn, ev_fname, att_fname, args.dmask, rt_dir, drm_dir, work_dir, bat_ml_dir)
    #
    # tab_info = get_info_tab(conn)
    #
    # logging.info("Writing the TimeWindows Table")
    # setup_tab_twinds(conn, tab_info['trigtimeMET'][0], tmin)
    #
    # twind_df = get_twinds_tab(conn)
    # timeIDs = twind_df['timeID'].values
    #
    # setup_tab_twind_status(conn, timeIDs)


if __name__ == "__main__":
    args = cli()

    main(args)
