import logging, traceback, argparse
import requests
import urllib.request, urllib.error, urllib.parse
import os
from bs4 import BeautifulSoup
from astropy.io import fits
from astropy.time import Time
import numpy as np
import sys
import pandas as pd

from .db_ql_funcs import (
    get_conn,
    get_qlevent_db_tab,
    write_new_obsid_line,
    update_obsid_line,
    get_db_tab,
    write_event_files2db,
)

# Meant to be run as a cronjob


def listFD(url, ext=""):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith(ext)
    ]


def get_obsid_dict(url):
    obsid_keys = ["ver", "url"]
    obsid_dict = {}
    url_list = listFD(url)
    for url_ in url_list:
        flist = url_.split("/")
        f0 = [f_ for f_ in flist if "sw0" in f_]
        if len(f0) > 0:
            # print f0[0]
            f0s = f0[0].split(".")
            obsid = f0s[0][2:]
            ver = int(f0s[1])
            if obsid in list(obsid_dict.keys()):
                if ver <= obsid_dict[obsid]["ver"]:
                    continue
            obsid_dict[obsid] = {}
            obsid_dict[obsid]["ver"] = ver
            obsid_dict[obsid]["url"] = url_
    return obsid_dict


def get_bat_files_from_list_url(url, aux=False):
    data = urllib.request.urlopen(url).read().decode("utf-8")
    data = data.split("\n")
    bat_files = []
    aux_files = []
    for line in data:
        if "bat" in line:
            bat_files.append(line)
        elif "aux" in line:
            aux_files.append(line)
    if aux:
        return bat_files, aux_files
    return bat_files


def get_urls2download(obsid_url):
    urls2download = []

    file_list_url = os.path.join(obsid_url, "dts.list")
    file_list_url2 = os.path.join(obsid_url, "dts_ql.list")
    try:
        bat_files = get_bat_files_from_list_url(file_list_url)
    except Exception as E:
        logging.error(E)
        try:
            logging.info("Trying dts_ql.list instead")
            bat_files = get_bat_files_from_list_url(file_list_url2)
        except Exception as E:
            logging.info("Still didn't work")
            logging.error(E)
            return urls2download

    event_files = [fn for fn in bat_files if "event" in fn]

    if len(event_files) == 0:
        logging.debug("No BAT event data under " + obsid_url)
        return urls2download

    data_url = os.path.join(obsid_url, "data")

    for evf in event_files:
        url = os.path.join(data_url, evf)
        urls2download.append(url)

    return urls2download


def download_file(url, fname):
    try:
        urllib.request.urlretrieve(url, fname)
    except Exception as E:
        logging.error(E)
        try:
            logging.info("Let's try again")
            urllib.request.urlretrieve(url, fname)
        except Exception as E:
            logging.error(E)


def met2mjd(met, utcf, mjdref=51910.0):
    mjd = mjdref + (met + utcf) / 86400.0

    return mjd


def met2isot(met, utcf, mjdref=51910.0):
    mjd = mjdref + (met + utcf) / 86400.0
    apy_time = Time(mjd, format="mjd")

    return apy_time.isot


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save data to",
        default="/storage/group/jak51/default/realtime_workdir/",
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name of the sqlite database", default=None
    )
    parser.add_argument(
        "--htmldir",
        type=str,
        help="bash script to run analysis",
        default="/storage/group/jak51/default/realtime_workdir/htmls",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Want to find all new obsid.ver's that have event data
    # So need to query qlDB for all obsid.ver's
    # Then grab all obsid.ver's from base_url
    # Check if any are new
    # For new ones, check to see if the base_url/obsid.ver/data/bat/event/ exists
    # If so record the url, along with the att url and dmask url
    # then download the files and record the file_names
    # also get the start and stop times of the event data and record them

    log_fname = os.path.join(args.save_dir, "get_quicklook_event_data_test.log")

    logging.basicConfig(
        filename=log_fname,
        level=logging.INFO,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    logging.info("Looking for event data using QuickLook url: ")

    conn = get_conn(args.dbfname)

    # get quick look table from local DB
    ql_db_tab = get_db_tab(conn, "SwiftQLeventOBS")

    base_url = "https://swift.gsfc.nasa.gov/data/swift/.original/"

    logging.info(base_url)

    # get the obsid and vers from the ql website
    obsid_dict = get_obsid_dict(base_url)

    for obsid, obs_dict in list(obsid_dict.items()):
        new_obsid = True

        logging.debug(obsid)
        # check if we've already seen this obsid and ver
        # and do nothing if we have seen it
        if np.isin(obsid, ql_db_tab["obsid"]):
            logging.debug("obsid already in DB")
            new_obsid = False
            db_ver = np.max(ql_db_tab["ver"][(ql_db_tab["obsid"] == obsid)])
            logging.debug("QL ver " + str(obs_dict["ver"]))
            logging.debug("DB ver " + str(db_ver))
            if obs_dict["ver"] <= db_ver:
                continue

        # check if there's event data
        urls2download = get_urls2download(obs_dict["url"])

        db_obs_dict = {}
        ev_file_dicts = []
        db_data_dicts = []
        db_obs_dict["ver"] = obs_dict["ver"]

        # if no urls, means no event data
        if len(urls2download) > 0:
            logging.info(
                str(len(urls2download)) + " urls to download for obsid " + obsid
            )

            obsid_dir = os.path.join(args.save_dir, obsid)
            db_obs_dict["obsDname"] = obsid_dir
            if not os.path.exists(obsid_dir):
                os.mkdir(obsid_dir)
            bat_dir = os.path.join(obsid_dir, "bat")
            if not os.path.exists(bat_dir):
                os.mkdir(bat_dir)
            ev_dir = os.path.join(bat_dir, "event")
            if not os.path.exists(ev_dir):
                os.mkdir(ev_dir)

            for url2down in urls2download:
                ev_file_dict = {}
                dname = ev_dir

                fname = os.path.join(dname, url2down.split("/")[-1])
                if not ("bevtr" in url2down):
                    ev_file_dict["eventFname"] = fname
                    ev_file_dict["eventURL"] = url2down
                    ev_file_dict["ver"] = obs_dict["ver"]
                    download_file(url2down, fname)
                    ev_file_dicts.append(ev_file_dict)

            # now I need to record the urls and file names
            # I might want to change up the DB here since there's
            # several att files and hk files
            # maybe just have the obsid directory and the event start and stop times
            # not sure about the event file though, since there still might be more than one
            # maybe have a NeventFiles column
            # then just pick one event file url for the url

            for ev_file_dict in ev_file_dicts:
                ev_file = fits.open(ev_file_dict["eventFname"])
                gti = ev_file[2]
                utcf = float(gti.header["UTCFINIT"])
                obs_mode = ev_file[1].header["OBS_MODE"]
                Ngtis = len(gti.data["START"])
                for ii in range(Ngtis):
                    db_data_dict = {}
                    db_data_dict["eventFname"] = ev_file_dict["eventFname"]
                    db_data_dict["eventURL"] = ev_file_dict["eventURL"]
                    db_data_dict["ver"] = ev_file_dict["ver"]
                    db_data_dict["obs_mode"] = obs_mode
                    db_data_dict["METstart"] = gti.data["START"][ii]
                    db_data_dict["METstop"] = gti.data["STOP"][ii]
                    db_data_dict["UTCstop"] = met2isot(db_data_dict["METstop"], utcf)
                    db_data_dict["UTCstart"] = met2isot(db_data_dict["METstart"], utcf)
                    db_data_dicts.append(db_data_dict)

                    logging.info("Event data UTCStart = " + db_data_dict["UTCstart"])
                    logging.info("Event data UTCStop = " + db_data_dict["UTCstop"])

        conn = get_conn(args.dbfname)

        if new_obsid:
            try:
                logging.debug("Making DB row for obsid " + obsid)
                write_new_obsid_line(
                    conn, obsid, data_dict=db_obs_dict, table="SwiftQLeventOBS"
                )
            except Exception as E:
                logging.error(E)
                logging.warn("Troub making obsid line for obside " + obsid)

        else:
            try:
                logging.debug("Updating DB for obsid " + obsid)
                update_obsid_line(conn, obsid, db_obs_dict, table="SwiftQLeventOBS")
            except Exception as E:
                logging.error(E)
                logging.warn("Troub updating obsid line for obside " + obsid)

        if len(db_data_dicts) > 0:
            logging.info("Writing Event Files to DB for obsid " + obsid)
            write_event_files2db(conn, obsid, db_data_dicts)

        conn.close()

    #conn = get_conn(args.dbfname)
    #table_name = "SwiftQLevent"
    #pd.set_option("display.max_colwidth", -1)
    #ql_db_tab = get_db_tab(conn, table_name)
    #html_file = os.path.join(args.htmldir, table_name + ".html")
    #    ql_db_tab.sort_values('METstart', ascending=False).to_html(\
    #            html_file, render_links=True, float_format='{0:.4f}'.format)
    #conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
