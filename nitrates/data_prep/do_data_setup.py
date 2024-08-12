import numpy as np
import sqlite3
import pandas as pd
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.table import Table, vstack
import os
import shutil
import sys
import time
from datetime import datetime
import argparse
import logging, traceback

from ..lib.time_funcs import met2astropy, utc2met, met2utc_str, apy_time2met
from ..lib.sqlite_funcs import (
    get_conn,
    setup_tab_info,
    setup_tab_twinds,
    setup_files_tab,
    setup_tab_twind_status,
)
from ..lib.dbread_funcs import (
    get_info_tab,
    get_twinds_tab,
    get_full_sqlite_table_as_df,
    guess_dbfname,
)
from ..lib.event2dpi_funcs import filter_evdata, mask_detxy
from ..lib.dmask_funcs import (
    combine_detmasks,
    get_hotpix_map,
    find_rate_spike_dets2mask,
)
from ..lib.funcs2run_bat_tools import do_pc
from ..lib.coord_conv_funcs import convert_radec2imxy, convert_imxy2radec
from ..lib.wcs_funcs import world2val
from ..lib.hp_funcs import pc_gwmap2good_pix
from ..lib.gti_funcs import (
    add_bti2gti,
    bti2gti,
    gti2bti,
    union_gtis,
    flags2gti,
    get_btis_for_glitches,
    check_if_in_GTI,
    find_and_remove_cr_glitches,
)
from ..data_scraping.db_ql_funcs import get_gainoff_fname
from ..data_scraping.api_funcs import get_sao_file
from ..HeasoftTools.bat_tool_funcs import bateconvert
from ..lib.search_config import Config

def query_data_metslice(conn, met0, met1, table_name="SwiftQLevent"):
    sql = """SELECT * FROM %s
        Where METstart < '%s'
        and METstop > '%s' """ % (
        table_name,
        met0,
        met1,
    )

    df = pd.read_sql(sql, conn)
    return df


def query_data_utcslice(conn, utc0, utc1, table_name="SwiftQLevent"):
    sql = """SELECT * FROM %s
        Where UTCstart < '%s'
        and UTCstop > '%s' """ % (
        table_name,
        utc0,
        utc1,
    )

    df = pd.read_sql(sql, conn)

    return df


def evfnames2write(
    evfnames,
    dmask,
    save_dir,
    acs_tab,
    save_fname="filter_evdata.fits",
    emin=14.0,
    emax=500.0,
    tmin=0.0,
    tmax=np.inf,
):
    tabs = []
    gti_pnts = []
    gti_slews = []
    gti_slewpnts = []
    all_gtis = []
    for evf in evfnames:
        tab = Table.read(evf)
        if "bevshpo" in evf or "filter_evdata" in evf:
            tab["SLEW"] = np.zeros(len(tab), dtype=np.int64)
            gti_pnts.append(Table.read(evf, hdu=2))
        elif "bevshsl" in evf:
            tab["SLEW"] = np.ones(len(tab), dtype=np.int64)
            gti_slews.append(Table.read(evf, hdu=2))
        elif "bevshsp" in evf:
            tab["SLEW"] = 2 * np.ones(len(tab), dtype=np.int64)
            gti_slewpnts.append(Table.read(evf, hdu=2))
        else:
            tab["SLEW"] = 2 * np.ones(len(tab), dtype=np.int64)
            gti_slewpnts.append(Table.read(evf, hdu=2))
        tabs.append(tab)
    ev_data = vstack(tabs)
    ev_data.sort(keys="TIME")

    ev_data0 = filter_evdata(ev_data, dmask, emin, emax, tmin, tmax)

    if len(gti_pnts) > 0:
        gti_pnt = vstack(gti_pnts)
        gti_pnt.sort(keys="START")
        all_gtis.append(gti_pnt)
    else:
        gti_pnt = Table(names=("START", "STOP"))
    gti_pnt_hdu = fits.BinTableHDU(gti_pnt, name="GTI_POINTING")
    if len(gti_slews) > 0:
        gti_slew = vstack(gti_slews)
        gti_slew.sort(keys="START")
        all_gtis.append(gti_slew)
    else:
        gti_slew = Table(names=("START", "STOP"))
    gti_slew_hdu = fits.BinTableHDU(gti_slew, name="GTI_SLEW")
    if len(gti_slewpnts) > 0:
        gti_slewpnt = vstack(gti_slewpnts)
        gti_slewpnt.sort(keys="START")
        all_gtis.append(gti_slewpnt)
        gti_pnt_ = flags2gti(acs_tab["TIME"], acs_tab["FLAGS"][:, 0])
        bti_pnt = gti2bti(gti_pnt_)
        gti_data_pnt = add_bti2gti(bti_pnt, gti_slewpnt)
        gti_pnts.append(gti_data_pnt)
        gti_pnt = vstack(gti_pnts)
        gti_pnt.sort(keys="START")
        gti_pnt_hdu = fits.BinTableHDU(gti_pnt, name="GTI_POINTING")
    else:
        gti_slewpnt = Table(names=("START", "STOP"))
    gti_slewpnt_hdu = fits.BinTableHDU(gti_slewpnt, name="GTI_SLEW_POINTING")

    if len(all_gtis) > 1:
        gti_tot = union_gtis(all_gtis)
    else:
        gti_tot = all_gtis[0]

    glitch_btis = get_btis_for_glitches(
        ev_data0, gti_tot["START"][0], gti_tot["STOP"][-1]
    )
    for bti in glitch_btis:
        logging.info("Found glitch bti: ")
        logging.info(bti)
        gti_pnt = add_bti2gti(bti, gti_pnt)
        gti_pnt_hdu = fits.BinTableHDU(gti_pnt, name="GTI_POINTING")

    ev_data0 = find_and_remove_cr_glitches(ev_data0, gti_pnt)

    ev_hdu = fits.BinTableHDU(ev_data0, name="EVENTS")
    gti_tot_hdu = fits.BinTableHDU(gti_tot, name="GTI")

    primary_hdu = fits.PrimaryHDU()
    hdu_list = fits.HDUList(
        [primary_hdu, ev_hdu, gti_tot_hdu, gti_pnt_hdu, gti_slew_hdu, gti_slewpnt_hdu]
    )
    ev_fname = os.path.join(save_dir, save_fname)
    hdu_list.writeto(ev_fname, overwrite=True)

    return ev_fname


def get_event_failed_trigs(args, UTCFINIT_guess=-24.0):
    conn_data = get_conn(args.data_dbfname)

    trig_time = args.trig_time
    MET = False
    if "T" in trig_time:
        apy_trig_time = Time(args.trig_time, format="isot")
        logging.debug("trig_time: ")
        logging.debug(apy_trig_time)
    elif "-" in args.trig_time:
        apy_trig_time = Time(args.trig_time, format="iso")
        logging.debug("trig_time: ")
        logging.debug(apy_trig_time)
    else:
        met_trig_time = float(args.trig_time)
        MET = True
        logging.debug("met_trig_time: ")
        logging.debug(met_trig_time)

    if not MET:
        met_trig_time = apy_time2met(apy_trig_time, None, utcf=UTCFINIT_guess)

    t_buff = 60.0
    # t_bounds = (met_trig_time - t_buff, met_trig_time + t_buff)
    t_bounds = (met_trig_time + t_buff, met_trig_time - t_buff)
    logging.debug("t_bounds: ")
    logging.debug(t_bounds)

    ev_data_table = query_data_metslice(
        conn_data, t_bounds[0], t_bounds[1], table_name="SwiftQLtdrss"
    )

    N_evfiles = len(ev_data_table)
    # logging.info(str(N_evfiles) + " event files found")
    if N_evfiles == 0:
        return
    logging.info(str(N_evfiles) + " failed trig event files found")

    times2cover = np.arange(int(met_trig_time) - 30, int(met_trig_time) + 30)
    times_coverd = np.zeros(len(times2cover), dtype=bool)

    ev_fnames = []
    for index, row in ev_data_table.iterrows():
        evfile = fits.open(row["eventFname"])
        min_ev_time = np.min(evfile[1].data["TIME"])
        max_ev_time = np.max(evfile[1].data["TIME"])
        bl = (times2cover > min_ev_time) & (times2cover < max_ev_time)
        times_coverd[bl] = True
        if np.sum(bl) > 0:
            if not "ENERGY" in evfile[1].data.columns.names:
                try:
                    go_fname = get_gainoff_fname(conn_data, met_trig_time)
                    infile = row["eventFname"]
                    outfile = infile + "2"
                    bateconvert(infile, outfile, go_fname)
                    ev_fnames.append(outfile)
                    utcf = fits.open(go_fname)[1].header["UTCFINIT"]
                    fits.setval(outfile, "UTCFINIT", value=utcf, ext=1)
                except Exception as E:
                    logging.warning("Trouble running bateconvert")
                    logging.error(E)
                    logging.error(traceback.format_exc())
            else:
                ev_fnames.append(row["eventFname"])

    ev_fnames = list(np.unique(np.array(ev_fnames)))
    logging.info("Good Event Fnames: ")
    logging.info(ev_fnames)
    if len(ev_fnames) > 1:
        for i, evfname in enumerate(ev_fnames):
            exposure = fits.open(evfname)[2].header["EXPOSURE"]
            gti = Table.read(evfname, hdu="GTI")
            for row in gti:
                for j in range(i + 1, len(ev_fnames)):
                    exp = fits.open(ev_fnames[j])[2].header["EXPOSURE"]
                    gti_ = Table.read(ev_fnames[j], hdu="GTI")
                    for row_ in gti_:
                        if row_["START"] >= (row["STOP"] - 1e-3):
                            continue
                        if row_["STOP"] <= (row["START"] + 1e-3):
                            continue
                        if exposure >= exp:
                            ev_fnames.pop(j)
                        else:
                            ev_fnames.pop(i)
                        break

    logging.info("Good Event Fnames: ")
    logging.info(ev_fnames)

    if np.sum(times_coverd) < 0.85 * len(times_coverd):
        return
    else:
        return ev_fnames


def get_event(args):
    if args.evfname is not None:
        evfname = args.evfname
        return [evfname]
    elif args.Obsid_Dir is not None:
        bat_ev_dir = os.path.join(args.Obsid_Dir, "bat", "event")
        bat_ev_fnames = [
            os.path.join(bat_ev_dir, fname)
            for fname in os.listdir(bat_ev_dir)
            if "bevtr" not in fname
        ]
        return bat_ev_fnames
    else:
        conn_data = get_conn(args.data_dbfname)

        trig_time = args.trig_time
        MET = False
        if "T" in trig_time:
            apy_trig_time = Time(args.trig_time, format="isot")
            logging.debug("trig_time: ")
            logging.debug(apy_trig_time)
        elif "-" in args.trig_time:
            apy_trig_time = Time(args.trig_time, format="iso")
            logging.debug("trig_time: ")
            logging.debug(apy_trig_time)
        else:
            met_trig_time = float(args.trig_time)
            MET = True
            logging.debug("met_trig_time: ")
            logging.debug(met_trig_time)

        if MET:
            t_buff = 60.0
            # t_bounds = (met_trig_time - t_buff, met_trig_time + t_buff)
            t_bounds = (met_trig_time + t_buff, met_trig_time - t_buff)
            logging.debug("t_bounds: ")
            logging.debug(t_bounds)
            ev_data_table = query_data_metslice(conn_data, t_bounds[0], t_bounds[1])
        else:
            t0 = Time(51910, format="mjd")
            t_buff = TimeDelta(60.0, format="sec")
            # t_bounds = (apy_trig_time - t_buff, apy_trig_time + t_buff)
            t_bounds = (apy_trig_time + t_buff, apy_trig_time - t_buff)
            logging.debug("t_bounds: ")
            logging.debug(t_bounds)
            ev_data_table = query_data_utcslice(conn_data, t_bounds[0], t_bounds[1])

        N_evfiles = len(ev_data_table)
        # logging.info(str(N_evfiles) + " event files found")
        if N_evfiles == 0:
            return
        idx = (
            ev_data_table.groupby(["eventFname"])["ver"].transform(max)
            == ev_data_table["ver"]
        )
        ev_data_table = ev_data_table[idx]
        N_evfiles = len(ev_data_table)
        logging.info(str(N_evfiles) + " event files found")

        tstarts = Time(ev_data_table.UTCstart.values.astype(str), format="isot")
        tstops = Time(ev_data_table.UTCstop.values.astype(str), format="isot")
        logging.info("Tstarts: ")
        logging.info(tstarts.isot)
        logging.info("Tstopts: ")
        logging.info(tstops.isot)
        if not MET:
            for index, row in ev_data_table.iterrows():
                try:
                    evfile = fits.open(row["eventFname"])
                    met_trig_time = utc2met(apy_trig_time.isot, evfile)
                    break
                except Exception as E:
                    logging.warning("Trouble openning or reading an event file")
                    logging.error(E)
                    logging.error(traceback.format_exc())

        # times2cover = np.arange(int(met_trig_time)-45,int(met_trig_time)+45)
        times2cover = np.arange(int(met_trig_time) - 30, int(met_trig_time) + 30)
        times_coverd = np.zeros(len(times2cover), dtype=bool)

        ev_fnames = []
        for index, row in ev_data_table.iterrows():
            evfile = fits.open(row["eventFname"])
            min_ev_time = np.min(evfile[1].data["TIME"])
            max_ev_time = np.max(evfile[1].data["TIME"])
            bl = (times2cover > min_ev_time) & (times2cover < max_ev_time)
            times_coverd[bl] = True
            if np.sum(bl) > 0:
                if not "ENERGY" in evfile[1].data.columns.names:
                    try:
                        go_fname = get_gainoff_fname(conn_data, met_trig_time)
                        infile = row["eventFname"]
                        outfile = infile + "2"
                        bateconvert(infile, outfile, go_fname)
                        ev_fnames.append(outfile)
                    except Exception as E:
                        logging.warning("Trouble running bateconvert")
                        logging.error(E)
                        logging.error(traceback.format_exc())
                else:
                    ev_fnames.append(row["eventFname"])

        logging.debug("Good Event Fnames: ")
        logging.debug(ev_fnames)

        ev_fnames = np.unique(np.array(ev_fnames))

        logging.info("Good Event Fnames: ")
        logging.info(ev_fnames)

        if np.sum(times_coverd) < 0.85 * len(times_coverd):
            return
        else:
            return ev_fnames

        # ev_fnames = ev_data_table['eventFname'].values
        # good_ev_fnames = []
        # better_ev_fnames = []
        # for ev_fname in ev_fnames:
        #     try:
        #         evfile = fits.open(ev_fname)
        #         if not MET:
        #             met_trig_time = utc2met(apy_trig_time.isot, evfile)
        #         trig_time_bnds = (met_trig_time - 30., met_trig_time + 30.)
        #         min_ev_time = np.min(evfile[1].data['TIME'])
        #         max_ev_time = np.max(evfile[1].data['TIME'])
        #         if (min_ev_time < trig_time_bnds[1]) and\
        #                 (max_ev_time > trig_time_bnds[0]):
        #             good_ev_fnames.append(ev_fname)
        #         if (min_ev_time < met_trig_time) and (max_ev_time > met_trig_time):
        #             better_ev_fnames.append(ev_fname)
        #     except Exception as E:
        #         logging.warning("Trouble openning or reading an event file")
        #         logging.error(E)
        #         logging.error(traceback.format_exc())
        #
        # N_good_evfiles = len(good_ev_fnames)
        # logging.info(str(N_good_evfiles) +\
        #         "  actually good event files found")
        # if N_good_evfiles == 0:
        #     return
        # if N_good_evfiles == 1:
        #     return good_ev_fnames[0]
        # if len(better_ev_fnames) == 1:
        #     return better_ev_fnames[0]
        #
        # # if there's multiple event files that elapse the
        # # trigger time, then choose the one that
        # # covers the most time in the interval of +/- 60s
        # exps = []
        # for ev_fname in better_ev_fnames:
        #     evfile = fits.open(ev_fname)
        #     if not MET:
        #         met_trig_time = utc2met(apy_trig_time.isot, evfile)
        #     trig_time_bnds = (met_trig_time - 60., met_trig_time + 60.)
        #     min_ev_time = np.min(evfile[1].data['TIME'])
        #     max_ev_time = np.max(evfile[1].data['TIME'])
        #     met0 = max(min_ev_time, trig_time_bnds[0])
        #     met1 = min(max_ev_time, trig_time_bnds[1])
        #     exps.append(met1 - met0)
        # best_ev_ind = np.argmax(exps)
        # return better_ev_fnames[best_ev_ind]


def get_dmask(args, evdata):
    """
    Combine the global detmask with the detector enable/disable
    map and then make the hotpix map and combine it with that too

    """

    if args.dmask is not None:
        if "bdecb" in args.dmask:
            det_enb_mask = fits.open(args.dmask)[1].data["FLAG"][-1]
        else:
            return fits.open(args.dmask)[0].data

    elif args.Obsid_Dir is not None:
        bat_hk_dir = os.path.join(args.Obsid_Dir, "bat", "hk")
        bdecb_fname = [
            os.path.join(bat_hk_dir, fname)
            for fname in os.listdir(bat_hk_dir)
            if "bdecb" in fname
        ][0]
        det_enb_mask = fits.open(bdecb_fname)[1].data["FLAG"][-1]

    if args.dmask is None and args.Obsid_Dir is None:
        min_ev_time = np.min(evdata["TIME"])
        max_ev_time = np.max(evdata["TIME"])
        mid_ev_time = (min_ev_time + max_ev_time) / 2.0

        enb_dname = args.enb_dname

        enb_fnames = [fname for fname in os.listdir(enb_dname) if ".fits" in fname]

        enb_t0s = [1e4 * float(fname.split("_")[0]) for fname in enb_fnames]
        enb_t1s = [
            1e4 * float((fname.split("_")[1]).split(".")[0]) for fname in enb_fnames
        ]

        max_dt0 = 30 * 60.0
        max_dt1 = 90 * 60.0

        enb_tab = None
        for i in range(len(enb_fnames)):
            if (mid_ev_time > enb_t0s[i]) and (mid_ev_time < enb_t1s[i]):
                enb_tab = Table.read(os.path.join(args.enb_dname, enb_fnames[i]))

        if enb_tab is None:
            return

        best_ind = np.argmin(np.abs(mid_ev_time - enb_tab["TIME"]))
        if np.abs(mid_ev_time - enb_tab["TIME"][best_ind]) > max_dt1:
            return
        det_enb_mask = enb_tab["FLAG"][best_ind]
        if np.abs(mid_ev_time - enb_tab["TIME"][best_ind]) > max_dt0:
            logging.warn(
                "Using enb/disb map that's from more than half an hour off of trigtime"
            )

    dmask_enb_glob = det_enb_mask

    bl_dmask_enb_glob = dmask_enb_glob == 0
    ndets_enb_glob = np.sum(bl_dmask_enb_glob)

    mask_zeros = False
    # if len(evdata) > 1e5:
    if len(evdata) / float(ndets_enb_glob) > 14:
        mask_zeros = True
    hotpix_map = get_hotpix_map(evdata, bl_dmask_enb_glob, mask_zeros=mask_zeros)

    dmask = combine_detmasks([dmask_enb_glob, hotpix_map])

    return dmask


def get_att(args, MET_time):
    if args.att_fname is not None:
        return Table.read(args.att_fname)

    if args.Obsid_Dir is not None:
        aux_dir = os.path.join(args.Obsid_Dir, "auxil")
        att_fnames = [
            os.path.join(aux_dir, fname)
            for fname in os.listdir(aux_dir)
            if "pat" in fname
        ]
        if len(att_fnames) > 0:
            att_fname = att_fnames[0]
        else:
            att_fname = [
                os.path.join(aux_dir, fname)
                for fname in os.listdir(aux_dir)
                if "sat" in fname
            ][0]
        return Table.read(att_fname)

    # min_ev_time = np.min(evdata['TIME'])
    # max_ev_time = np.max(evdata['TIME'])
    # mid_ev_time = (min_ev_time + max_ev_time)/2.

    att_dname = args.att_dname

    att_fnames = [fname for fname in os.listdir(att_dname) if ".fits" in fname]

    att_t0s = [1e4 * float(fname.split("_")[0]) for fname in att_fnames]
    att_t1s = [1e4 * float((fname.split("_")[1]).split(".")[0]) for fname in att_fnames]

    max_dt = 30.0

    att_fname = None
    for i in range(len(att_fnames)):
        if (MET_time > att_t0s[i]) and (MET_time < att_t1s[i]):
            att_fname = att_fnames[i]

    if att_fname is None:
        return

    att_tab = Table.read(os.path.join(args.att_dname, att_fname))
    min_dt = np.min(np.abs(MET_time - att_tab["TIME"]))
    logging.info("Min att dt from trigtime is %.2f" % (min_dt))
    if min_dt > max_dt:
        return
    return att_tab


def get_acs(args, trigtime):
    if args.acs_fname is not None:
        return Table.read(args.acs_fname, hdu="ACS_DATA")

    if args.Obsid_Dir is not None:
        aux_dir = os.path.join(args.Obsid_Dir, "auxil")
        att_fnames = [
            os.path.join(aux_dir, fname)
            for fname in os.listdir(aux_dir)
            if "pat" in fname
        ]
        if len(att_fnames) > 0:
            att_fname = att_fnames[0]
        else:
            att_fname = [
                os.path.join(aux_dir, fname)
                for fname in os.listdir(aux_dir)
                if "sat" in fname
            ][0]
        return Table.read(att_fname, hdu=2)

    acs_dname = args.acs_dname

    acs_fnames = [fname for fname in os.listdir(acs_dname) if ".fits" in fname]

    acs_t0s = [1e4 * float(fname.split("_")[0]) for fname in acs_fnames]
    acs_t1s = [1e4 * float((fname.split("_")[1]).split(".")[0]) for fname in acs_fnames]

    max_dt = 30.0

    acs_fname = None
    for i in range(len(acs_fnames)):
        if (trigtime > acs_t0s[i]) and (trigtime < acs_t1s[i]):
            acs_fname = acs_fnames[i]

    if acs_fname is None:
        return

    acs_tab = Table.read(os.path.join(args.acs_dname, acs_fname))
    min_dt = np.min(np.abs(trigtime - acs_tab["TIME"]))
    if min_dt > max_dt:
        return
    return acs_tab

def get_sao(args):

    fname = get_sao_file(args.trig_time)
    new_fname = os.path.join(args.work_dir,'sao.fits')
    shutil.move(fname, new_fname)
    return new_fname

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drm_dir",
        type=str,
        help="drm_directory",
        default="/gpfs/group/jak51/default/responses/drms/",
    )
    parser.add_argument("--rt_dir", type=str, help="rt_directory", default=None)
    parser.add_argument(
        "--work_dir", type=str, help="Directory to work in", default=None
    )
    parser.add_argument(
        "--data_dbfname",
        type=str,
        help="DB file name with information on the BAT data already downloaded from the QL site",
        default="/storage/group/jak51/default/realtime_workdir/BATQL.db",
    )
    parser.add_argument(
        "--att_dname",
        type=str,
        help="Directory name that contains merged attfiles over chunks of time",
        default="/storage/group/jak51/default/realtime_workdir/merged_atts/",
    )
    parser.add_argument(
        "--acs_dname",
        type=str,
        help="Directory name that contains merged acsfiles over chunks of time",
        default="/storage/group/jak51/default/realtime_workdir/merged_acs/",
    )
    parser.add_argument(
        "--enb_dname",
        type=str,
        help="Directory name that contains merged enable/disable files over chunks of time",
        default="/storage/group/jak51/default/realtime_workdir/merged_enbs/",
    )
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="detmask file name", default=None)
    parser.add_argument("--obsid", type=str, help="Obsid", default=None)
    parser.add_argument(
        "--Obsid_Dir", type=str, help="Obsid directory to find data files", default=None
    )
    parser.add_argument(
        "--dbfname", type=str, help="File name of the analysis database", default=None
    )
    parser.add_argument(
        "--att_fname", type=str, help="Fname for that att file", default=None
    )
    parser.add_argument(
        "--acs_fname", type=str, help="Fname for the acs file", default=None
    )
    parser.add_argument(
        "--trig_time",
        type=str,
        help="Time of trigger, in either MET or a datetime string",
    )
    parser.add_argument(
        "--api_token",
        type=str,
        help="EchoAPI key for interactions.",
        default=None
    )

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename="data_setup.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    if args.api_token is not None:
        try:
            from EchoAPI import API
        except ImportError:
            return print("EchoAPI required, exiting.")
        #look for file called 'config.json' in working directory
        #if not present, use cli args
        config_filename=os.path.join(args.work_dir,'config.json')
        if os.path.exists(config_filename):
            search_config = Config(config_filename)
            args.trig_time = search_config.trigtime
            api = API(api_token = args.api_token)
        else:
            logging.error('Api_token passed but no config.json file found. Exiting.')
            return False

    # Need to find the event, detmask, and att files
    # then prepare them for use if needed and save the new file
    # Ex: filtering event data and if dmask is a bdecb file then
    # combine it with the global mask and make a hotpix map to also
    # combine it with

    from ..config import rt_dir, drm_dir, bat_ml_dir

    if args.dbfname is None:
        dbfname = guess_dbfname()
    else:
        dbfname = args.dbfname

    start_time = time.time()
    dt = 0.0
    loop_wait_time = 30.0
    loop_wait_err = 120.0

    while dt < 24 * 3600:
        try:
            dt = time.time() - start_time

            evfname = get_event(args)
            utcf = None
            if evfname is None:
                evfname = get_event_failed_trigs(args)
                if evfname is None:
                    logging.info("No event data yet")
                    time.sleep(loop_wait_time)
                    continue

            if len(evfname) == 1:
                ev_data = Table.read(evfname[0])
            else:
                tabs = []
                for evf in evfname:
                    tabs.append(Table.read(evf))
                ev_data = vstack(tabs)
                ev_data.sort(keys="TIME")


            MET = False
            if "T" in args.trig_time:
                trig_time = Time(args.trig_time, format="isot")
            elif "-" in args.trig_time:
                trig_time = Time(args.trig_time, format="iso")
            else:
                trig_time = float(args.trig_time)
                MET = True

            if isinstance(trig_time, float):
                trigtimeMET = trig_time
                trigtimeUTC = met2utc_str(trigtimeMET, evfname[0])
            else:
                trigtimeMET = utc2met(trig_time.isot, evfname[0])
                trigtimeUTC = trig_time.iso

            # ev_data0 = filter_evdata(ev_data[1].data, None, 14.0, 350.0, 0., np.inf)
            ev_data0 = filter_evdata(
                ev_data, None, 14.0, 500.0, trigtimeMET - 2e3, trigtimeMET + 2e3
            )

            att_tab = get_att(args, trigtimeMET)
            if att_tab is None:
                logging.info("No att info yet")
                time.sleep(loop_wait_time)
                continue

            acs_tab = get_acs(args, trigtimeMET)
            if acs_tab is None:
                logging.info("No acs info yet")
                time.sleep(loop_wait_time)
                continue

            dmask = get_dmask(args, ev_data0)
            if dmask is None:
                logging.info("No dmask yet")
                time.sleep(loop_wait_time)
                continue

            break
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            time.sleep(loop_wait_err)


    if evfname is None or dmask is None or acs_tab is None:
        logging.error("No data found, exiting now.")
        if args.api_token is not None:
            try:
                api.report(search_config.queueID,complete=True)
            except Exception as e:
                logging.error(e)
                logging.error('Could not report done to Queue via EchoAPI.')
                
        return -1

    ev_fname = evfnames2write(evfname, dmask, args.work_dir, acs_tab)

    try:
        sao_fname = get_sao(args)
    except Exception as E:
        logging.warning("Couldn't get sao file")
        logging.error(E)
        logging.error(traceback.format_exc())        

    logging.info("Finally got all the data")

    if args.api_token is not None:
        try:
            api.post_log(trigger=search_config.triggerID, config_id=search_config.id, AllDataFound=datetime.utcnow().isoformat())
        except Exception as e:
            logging.error(e)
            logging.error('Could not post to log via EchoAPI.')

    try:
        mask_vals = mask_detxy(dmask, ev_data0)
        blev = mask_vals == 0.0
        bad_dets = find_rate_spike_dets2mask(ev_data0[blev])
        logging.debug("Bad Dets List: ")
        logging.debug(bad_dets)
        for bad_det in bad_dets:
            dmask[bad_det[0], bad_det[1]] = 1
    except Exception as E:
        logging.warning("Messed up getting bad dets from rate spikes")
        logging.error(E)
        logging.error(traceback.format_exc())

    GTI_pnt = Table.read(ev_fname, hdu="GTI_POINTING")

    # evdata = ev_data0[(ev_data0['ENERGY']<195.)]
    # evdata = filter_evdata(ev_data0, dmask, 14., 195., 0., np.inf)
    # # ev_data[1].data = evdata
    # ev_fname = os.path.join(args.work_dir, 'filter_evdata.fits')
    # # ev_data.writeto(ev_fname, overwrite=True)
    # ev_hdu = fits.BinTableHDU(evdata, name='EVENTS')
    # gti = fits.open(evfname[0])[2]
    # gti_tab = Table()
    # gti_tab['START'] = [np.min(evdata['TIME'])]
    # gti_tab['STOP'] = [np.max(evdata['TIME'])]
    # primary_hdu = fits.PrimaryHDU()
    # gti_hdu = fits.BinTableHDU(gti_tab, header=gti.header, name='GTI')
    # hdu_list = fits.HDUList([primary_hdu, ev_hdu, gti_hdu])
    # hdu_list.writeto(ev_fname, overwrite=True)
    logging.info("Wrote filtered event data to")
    logging.info(ev_fname)
    # evdata.write(ev_fname, overwrite=True)

    att_fname = os.path.join(args.work_dir, "attitude.fits")
    att_tab.write(att_fname, overwrite=True)
    logging.info("Wrote att file to")
    logging.info(att_fname)

    acs_fname = os.path.join(args.work_dir, "acs.fits")
    acs_tab.write(acs_fname, overwrite=True)
    logging.info("Wrote acs file to")
    logging.info(acs_fname)

    dmask_fname = os.path.join(args.work_dir, "detmask.fits")
    hdu = fits.PrimaryHDU(dmask)
    hdu.writeto(dmask_fname, overwrite=True)
    logging.info("Wrote dmask to")
    logging.info(dmask_fname)

    conn = get_conn(dbfname)

    trig_time = args.trig_time
    MET = False
    if "T" in trig_time:
        trig_time = Time(args.trig_time, format="isot")
    elif "-" in args.trig_time:
        trig_time = Time(args.trig_time, format="iso")
    else:
        trig_time = float(args.trig_time)
        MET = True

    setup_tab_info(conn, ev_fname, trig_time)

    # not needed since this is defined in config.py. Only rt_dir is really needed but keeping it for now.
    # if args.rt_dir is not None:
    #    rt_dir = args.rt_dir
    # if args.drm_dir is not None:
    #    drm_dir = args.drm_dir
    # else:
    #    drm_dir = '.'

    logging.info("Writing the Files table")
    setup_files_tab(
        conn,
        ev_fname,
        att_fname,
        dmask_fname,
        rt_dir,
        drm_dir,
        args.work_dir,
        bat_ml_dir,
    )

    tab_info = get_info_tab(conn)

    try:
        pc_fname = do_pc(
            "detmask.fits", "attitude.fits", args.work_dir, ovrsmp=2, detapp=True
        )
    except Exception as e:
        logging.error(e)
        logging.error("Error making PC")

    sky_map_fnames = [
        fname
        for fname in os.listdir(args.work_dir)
        if "cWB.fits.gz" in fname or "bayestar" in fname or "skymap" in fname
    ]

    logging.info("Done, exiting now")


if __name__ == "__main__":
    args = cli()

    main(args)
