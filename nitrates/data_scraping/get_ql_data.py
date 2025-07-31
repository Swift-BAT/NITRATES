import logging, traceback, argparse
import requests
import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table, vstack, unique
import sys
from .db_ql_funcs import (
    get_conn,
    get_ql_db_tab,
    write_new_obsid_line,
    update_obsid_line,
)

# Meant to be run as a cronjob


def now_in_met(utcf=0.0):
    mjdref = 51910.0
    met = (Time.now().mjd - mjdref) * 86400.0 - utcf
    return met


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
        bat_files, aux_files = get_bat_files_from_list_url(file_list_url, aux=True)
    except Exception as E:
        logging.warning("Problem with url: " + file_list_url)
        logging.error(E)
        try:
            logging.info("Trying dts_ql.list instead")
            bat_files, aux_files = get_bat_files_from_list_url(file_list_url2, aux=True)
        except Exception as E:
            logging.info("Still didn't work")
            logging.warning("Problem with url: " + file_list_url2)
            logging.error(E)
            return urls2download

    data_url = os.path.join(obsid_url, "data")

    hk_files = [fn for fn in bat_files if "hk" in fn]

    for hkf in hk_files:
        if ("bdecb" in hkf) or ("bdqcb" in hkf) or ("bgocb" in hkf):
            url = os.path.join(data_url, hkf)
            urls2download.append(url)

    for auxf in aux_files:
        if ("pat" in auxf) or ("sat" in auxf):
            url = os.path.join(data_url, auxf)
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

    log_fname = os.path.join(args.save_dir, "get_quicklook_data2_test.log")

    logging.basicConfig(
        filename=log_fname,
        level=logging.INFO,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    conn = get_conn(args.dbfname)

    # get quick look table from local DB
    ql_db_tab = get_ql_db_tab(conn)

    base_url = "https://swift.gsfc.nasa.gov/data/swift/.original/"

    # get the obsid and vers from the ql website
    obsid_dict = get_obsid_dict(base_url)

    new_atts = []
    new_enb_tabs = []

    for obsid, obs_dict in list(obsid_dict.items()):
        att_dict = {}
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

        # if no urls, means no data
        if len(urls2download) == 0:
            continue

        logging.info(str(len(urls2download)) + " urls to download for obsid " + obsid)

        db_data_dict = {}

        obsid_dir = os.path.join(args.save_dir, obsid)
        if not os.path.exists(obsid_dir):
            os.mkdir(obsid_dir)
        bat_dir = os.path.join(obsid_dir, "bat")
        if not os.path.exists(bat_dir):
            os.mkdir(bat_dir)
        hk_dir = os.path.join(bat_dir, "hk")
        if not os.path.exists(hk_dir):
            os.mkdir(hk_dir)
        aux_dir = os.path.join(obsid_dir, "auxil")
        if not os.path.exists(aux_dir):
            os.mkdir(aux_dir)

        for url2down in urls2download:
            if "auxil" in url2down:
                dname = aux_dir
                fname = os.path.join(dname, url2down.split("/")[-1])
                if "pat" in url2down:
                    db_data_dict["patFname"] = fname
                elif "sat" in url2down:
                    db_data_dict["satFname"] = fname
            elif "hk" in url2down:
                dname = hk_dir
                fname = os.path.join(dname, url2down.split("/")[-1])
                if "bdecb" in url2down:
                    db_data_dict["DetEnbFname"] = fname
                elif "bdqcb" in url2down:
                    db_data_dict["DetQualFname"] = fname
                elif "bgocb" in url2down:
                    db_data_dict["GainOffFname"] = fname

            download_file(url2down, fname)

        # now I need to record the urls and file names
        # I might want to change up the DB here since there's
        # several att files and hk files
        # maybe just have the obsid directory and the event start and stop times
        # not sure about the event file though, since there still might be more than one
        # maybe have a NeventFiles column
        # then just pick one event file url for the url

        db_data_dict["ver"] = obs_dict["ver"]
        db_data_dict["obsDname"] = obsid_dir

        if "patFname" in list(db_data_dict.keys()):
            att_fname = db_data_dict["patFname"]
            att_file = fits.open(att_fname)[1]
            att_dict["fname"] = att_fname
        elif "satFname" in list(db_data_dict.keys()):
            att_fname = db_data_dict["satFname"]
            att_file = fits.open(att_fname)[1]
            att_dict["fname"] = att_fname
        else:
            att_file = None

        if "DetEnbFname" in list(db_data_dict.keys()):
            new_enb_tabs.append(Table.read(db_data_dict["DetEnbFname"]))

        if att_file is not None:
            db_data_dict["UTCstart"] = att_file.header["DATE-OBS"]
            db_data_dict["UTCstop"] = att_file.header["DATE-END"]
            db_data_dict["METstart"] = att_file.header["TSTART"]
            db_data_dict["METstop"] = att_file.header["TSTOP"]
            att_dict["tstart"] = db_data_dict["METstart"]
            att_dict["tstop"] = db_data_dict["METstop"]
            new_atts.append(att_dict)

            logging.info("att UTCStart = " + db_data_dict["UTCstart"])
            logging.info("att UTCStop = " + db_data_dict["UTCstop"])

        conn = get_conn(args.dbfname)

        if new_obsid:
            logging.debug("Making DB row for obsid " + obsid)
            try:
                write_new_obsid_line(conn, obsid, data_dict=db_data_dict)
            except Exception as E:
                logging.warn("Trouble writing to DB")
                logging.error(E)
                continue
        else:
            logging.debug("Updating DB for obsid " + obsid)
            try:
                update_obsid_line(conn, obsid, db_data_dict)
            except Exception as E:
                logging.warn("Trouble writing to DB")
                logging.error(E)
                continue

        conn.close()

    no_new_atts = False
    no_new_enbs = True
    if len(new_atts) == 0:
        no_new_atts = True
        logging.info("No new att files to merge")
        # return

    N_new_enbs = len(new_enb_tabs)
    logging.info(str(N_new_enbs) + " new Enb/Dis files, merging them now")

    if N_new_enbs > 0:
        no_new_enbs = False
        enb_tab = vstack(new_enb_tabs, metadata_conflicts="silent")

    if no_new_atts and no_new_enbs:
        return

    att_tab = Table(new_atts)

    met_now = now_in_met()

    met_step = int(1e5)
    met_steps = 7

    met_min = int(np.round(met_now, decimals=-5)) - 5 * met_step

    logging.info(
        "Looking for new att files from met %d to %d"
        % (met_min, met_min + 6 * met_step)
    )

    att_merged_dname = os.path.join(args.save_dir, "merged_atts")
    acs_merged_dname = os.path.join(args.save_dir, "merged_acs")
    enb_merged_dname = os.path.join(args.save_dir, "merged_enbs")

    if not os.path.exists(att_merged_dname):
        os.mkdir(att_merged_dname)
    if not os.path.exists(acs_merged_dname):
        os.mkdir(acs_merged_dname)
    if not os.path.exists(enb_merged_dname):
        os.mkdir(enb_merged_dname)

    for i in range(met_steps):
        met0 = met_min + i * met_step
        met1 = met0 + met_step

        met00 = met0 - 5000
        met11 = met1 + 5000

        att_merged_fname = os.path.join(
            att_merged_dname, str(met0)[:5] + "_" + str(met1)[:5] + ".fits"
        )
        acs_merged_fname = os.path.join(
            acs_merged_dname, str(met0)[:5] + "_" + str(met1)[:5] + ".fits"
        )
        enb_merged_fname = os.path.join(
            enb_merged_dname, str(met0)[:5] + "_" + str(met1)[:5] + ".fits"
        )

        bl0 = (att_tab["tstart"] >= met0) & (att_tab["tstart"] < met1)
        bl1 = (att_tab["tstop"] >= met0) & (att_tab["tstop"] < met1)
        bl = bl0 | bl1
        Natts = np.sum(bl)
        logging.info(str(Natts) + " atts with times in met %d to %d" % (met0, met1))
        if Natts > 0:
            AttTab = att_tab[bl]
            att_tab_list = []
            acs_tab_list = []

            for i in range(Natts):
                tab = Table.read(AttTab[i]["fname"])
                bl = (tab["TIME"] > met00) & (tab["TIME"] < met11)
                tab = tab[bl]
                tab["TIME"] = np.round(tab["TIME"], decimals=1)
                att_tab_list.append(tab)

                tab = Table.read(AttTab[i]["fname"], hdu=2)
                bl = (tab["TIME"] > met00) & (tab["TIME"] < met11)
                tab = tab[bl]
                tab["TIME"] = np.round(tab["TIME"], decimals=1)
                acs_tab_list.append(tab)

            if os.path.exists(att_merged_fname):
                att_tab_list.append(Table.read(att_merged_fname))
            if os.path.exists(acs_merged_fname):
                acs_tab_list.append(Table.read(acs_merged_fname))

            logging.info("Done opening att tables, now stacking them")
            att_tab_merged = vstack(att_tab_list, metadata_conflicts="silent")
            # uniq_times, uniq_inds = np.unique(att_tab_merged['TIME'], return_index=True)
            # att_tab_merged = att_tab_merged[uniq_inds]
            att_tab_merged = unique(att_tab_merged, keys="TIME", keep="last")
            logging.info(
                "Removed att row duplicates, now saving to " + att_merged_fname
            )
            att_tab_merged.write(att_merged_fname, overwrite=True)

            logging.info("Now stacking ACS tables")
            acs_tab_merged = vstack(acs_tab_list, metadata_conflicts="silent")
            uniq_times, uniq_inds = np.unique(acs_tab_merged["TIME"], return_index=True)
            acs_tab_merged = acs_tab_merged[uniq_inds]
            logging.info(
                "Removed ACS row duplicates, now saving to " + acs_merged_fname
            )
            acs_tab_merged.write(acs_merged_fname, overwrite=True)

        enb_bl = (enb_tab["TIME"] >= met00) & (enb_tab["TIME"] < met11)
        N_enb_rows = np.sum(enb_bl)
        logging.info(
            str(N_enb_rows)
            + " enb/disable files with times in met %d to %d" % (met0, met1)
        )
        if N_enb_rows > 0:
            enb_tab_i = enb_tab[enb_bl]

            if os.path.exists(enb_merged_fname):
                enb_old_tab = Table.read(enb_merged_fname)
                enb_tab_i = vstack(
                    [enb_tab_i, enb_old_tab], metadata_conflicts="silent"
                )

            uniq_times, uniq_inds = np.unique(enb_tab_i["TIME"], return_index=True)
            enb_tab_i = enb_tab_i[uniq_inds]
            logging.info(
                "Removed Enb/Dis row duplicates, now saving to " + enb_merged_fname
            )
            enb_tab_i.write(enb_merged_fname, overwrite=True)


if __name__ == "__main__":
    args = cli()

    main(args)
