import logging, traceback, argparse
import requests
import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import os
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


def from_top_url_get_obsids(top_url):
    urls = listFD(top_url)
    obsids = []
    obsid_urls = []
    for url in urls:
        flist = url.split("/")
        logging.debug(url)
        logging.debug(flist)
        try:
            if flist[-2][0] == "0":
                obsids.append(flist[-2])
                obsid_urls.append(url)
        except:
            pass

    return obsids, obsid_urls


def get_ev_url(obs_url):
    urls = listFD(obs_url)
    ev_url = []
    for url in urls:
        if "msbevshsp_uf.evt" in url:
            ev_url.append(url)
    return ev_url


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
    data = urllib.request.urlopen(url).read()
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
    parser.add_argument(
        "--loglevel", type=int, help="loglevel, 10=debug, 20=info", default=20
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

    log_fname = os.path.join(args.save_dir, "get_trig_event_data_test.log")

    logging.basicConfig(
        filename=log_fname,
        level=args.loglevel,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    logging.info("Looking for event data using QuickLook url: ")

    conn = get_conn(args.dbfname)

    # get quick look table from local DB
    table_name = "SwiftQLtdrss"
    try:
        ql_db_tab = get_db_tab(conn, table_name)
    except Exception as E:
        logging.error(E)
        logging.info("Trouble getting DB")

    conn.close()

    base_url = "https://swift.gsfc.nasa.gov/data/swift/qltdrss/"

    logging.info(base_url)

    # get top folders that go like "009478xx"
    top_urls = listFD(base_url)
    logging.info("%d top urls" % (len(top_urls)))

    new_obs_dicts = []

    for url in top_urls:
        logging.info("Checking url: ")
        logging.info(url)
        obsids, obsid_urls = from_top_url_get_obsids(url)
        obsids = np.array(obsids)
        obsid_urls = np.array(obsid_urls)
        logging.info("%d total obsids here" % (len(obsids)))
        try:
            bl = ~np.isin(obsids, ql_db_tab["obsid"])
        except Exception as E:
            logging.error(E)
            bl = np.ones(len(obsids), dtype=bool)
        new_obsids = obsids[bl]
        Nnew = len(new_obsids)
        Nobsids = len(obsids)
        logging.info("%d obsids to check" % (Nobsids))
        if Nnew < 1:
            continue
        obsid_urls = obsid_urls
        for i in range(Nobsids):
            ev_urls = get_ev_url(obsid_urls[i])
            if len(ev_urls) == 0:
                continue
            for ev_url in ev_urls:
                # ev_url = ev_url[0]
                if np.isin(ev_url, ql_db_tab["eventURL"]):
                    continue
                dname = os.path.join(args.save_dir, obsids[i])
                if not os.path.exists(dname):
                    os.mkdir(dname)
                fname = os.path.join(dname, ev_url.split("/")[-1])
                logging.info("Downloading " + ev_url)
                download_file(ev_url, fname)
                logging.info("Saved to " + fname)
                ev_file = fits.open(fname)
                gti = ev_file[2]
                # obs_mode = ev_file[1].header['OBS_MODE']
                Ngtis = len(gti.data["START"])
                for ii in range(Ngtis):
                    obs_dict = {"obsid": obsids[i], "eventURL": ev_url}
                    obs_dict["eventFname"] = fname
                    obs_dict["METstart"] = gti.data["START"][ii]
                    obs_dict["METstop"] = gti.data["STOP"][ii]
                    obs_dict["Duration"] = obs_dict["METstop"] - obs_dict["METstart"]
                    logging.info("Event data METStart = %.3f" % (obs_dict["METstart"]))
                    logging.info("Event data METStop = %.3f" % (obs_dict["METstop"]))
                    try:
                        utcf = float(gti.header["UTCFINIT"])
                        obs_dict["UTCstop"] = met2isot(obs_dict["METstop"], utcf)
                        obs_dict["UTCstart"] = met2isot(obs_dict["METstart"], utcf)
                        logging.info("Event data UTCStart = " + obs_dict["UTCstart"])
                        logging.info("Event data UTCStop = " + obs_dict["UTCstop"])
                    except:
                        obs_dict["UTCstop"] = "2000-01-00T00:00:00.000"
                        obs_dict["UTCstart"] = "2000-01-00T00:00:00.000"
                    new_obs_dicts.append(obs_dict)

    if len(new_obs_dicts) > 1:
        logging.info("Writing to DB now")
        df = pd.DataFrame(new_obs_dicts)
        conn = get_conn(args.dbfname)
        df.to_sql("SwiftQLtdrss", conn, if_exists="append", index=False)
        conn.close()
        logging.info("Done writing to DB now")

        conn = get_conn(args.dbfname)
        pd.set_option("display.max_colwidth", -1)
        ql_db_tab = get_db_tab(conn, table_name)
        html_file = os.path.join(args.htmldir, table_name + ".html")
        ql_db_tab.sort_values("METstart", ascending=False).to_html(
            html_file, render_links=True, float_format="{0:.4f}".format
        )
        conn.close()
        logging.info("Made HTML file")
    logging.info("Done: exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
