import numpy as np
import sqlite3
import pandas as pd
from astropy.io import fits
import logging, traceback

from ..lib.time_funcs import utc2met, met2astropy, met2utc_str, met2mjd
from ..lib.gti_funcs import check_if_in_GTI

quad_dicts = {
    "all": {
        "quads": [0, 1, 2, 3],
        "drm_fname": "drm_0.200_0.150_.fits",
        "imx": 0.2,
        "imy": 0.15,
        "id": 0,
    },
    "left": {
        "quads": [0, 1],
        "drm_fname": "drm_1.000_0.150_.fits",
        "imx": 1.0,
        "imy": 0.15,
        "id": 1,
    },
    "top": {
        "quads": [1, 2],
        "drm_fname": "drm_0.000_-0.500_.fits",
        "imx": 0.0,
        "imy": -0.5,
        "id": 2,
    },
    "right": {
        "quads": [2, 3],
        "drm_fname": "drm_-1.000_0.150_.fits",
        "imx": -1.0,
        "imy": 0.15,
        "id": 3,
    },
    "bottom": {
        "quads": [3, 0],
        "drm_fname": "drm_0.000_0.450_.fits",
        "imx": 0.0,
        "imy": 0.45,
        "id": 4,
    },
    "quad0": {
        "quads": [0],
        "drm_fname": "drm_1.000_0.500_.fits",
        "imx": 1.0,
        "imy": 0.5,
        "id": 5,
    },
    "quad1": {
        "quads": [1],
        "drm_fname": "drm_0.800_-0.400_.fits",
        "imx": 0.8,
        "imy": -0.4,
        "id": 6,
    },
    "quad2": {
        "quads": [2],
        "drm_fname": "drm_-0.750_-0.450_.fits",
        "imx": -0.75,
        "imy": -0.45,
        "id": 7,
    },
    "quad3": {
        "quads": [3],
        "drm_fname": "drm_-1.100_0.500_.fits",
        "imx": -1.1,
        "imy": 0.5,
        "id": 8,
    },
}


def get_sql_tab_list():
    sql_tab_twinds = """CREATE TABLE TimeWindows
            (timeID INT PRIMARY KEY NOT NULL,
            time REAL,
            duration REAL,
            time_end REAL,
            settled INTEGER,
            SAA INTEGER,
            sat_lat REAL,
            sat_lon REAL
            quat0 REAL,
            quat1 REAL,
            quat2 REAL,
            quat3 REAL
            );"""

    sql_tab_twind_stat = """CREATE TABLE TimeWindowStatus
            (timeID INT PRIMARY KEY NOT NULL,
            BlipsFounds INTEGER,
            SeedsAssigned INTEGER,
            llhDone INTEGER
            );"""

    sql_tab_quads = """CREATE TABLE Quadrants
            (quadID INT PRIMARY KEY NOT NULL,
            name TEXT,
            imx REAL,
            imy REAL,
            drmfname TEXT,
            quad0 INTEGER,
            quad1 INTEGER,
            quad2 INTEGER,
            quad3 INTEGER,
            type TEXT
            );"""

    sql_tab_ratefits = """CREATE TABLE RateFits
            (time REAL,
            quadID INT,
            pre INT,
            post INT,
            ebin INT,
            deg INT,
            rate REAL,
            error REAL,
            chi2 REAL,
            FOREIGN KEY(quadID) REFERENCES Quadrants(quadID)
            );"""

    # (rateQuadID INT PRIMARY KEY NOT NULL,
    sql_tab_rates = """CREATE TABLE Rates
            (timeID INT,
            quadID INT,
            time REAL,
            duration REAL,
            bkg_nllh REAL,
            sig_nllh REAL,
            TS REAL,
            nsig REAL,
            PlawInd REAL,
            FOREIGN KEY(quadID) REFERENCES Quadrants(quadID),
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_blips = """CREATE TABLE Blips
            (blipID INT NOT NULL,
            timeID INT,
            time REAL,
            duration REAL,
            snr REAL,
            imx REAL,
            imy REAL,
            pc REAL,
            ebin INT,
            PRIMARY KEY(blipID, timeID)
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_img = """CREATE TABLE ImageSigs
            (timeID INT,
            snr REAL,
            imx REAL,
            imy REAL,
            imx_ind INT,
            imy_ind INT,
            proc_group INT,
            done INT,
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_sigimg = """CREATE TABLE SigImages
            (timeID INT,
            time REAL,
            duration REAL,
            fname TEXT,
            Npix INT,
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_job_square = """CREATE TABLE JobSquare
            (squareID INT,
            proc_group INT,
            imx0 REAL,
            imx1 REAL,
            imy0 REAL,
            imy1 REAL);"""

    sql_tab_square_stat = """CREATE TABLE SquareStatus
            (timeID INT,
            squareID INT,
            done INT,
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_square_res = """CREATE TABLE SquareResults
            (squareID INT,
            timeID INT,
            time REAL,
            duration REAL,
            TS REAL,
            bkg_nllh REAL,
            sig_nllh REAL,
            imx REAL,
            imy REAL,
            A REAL,
            ind REAL,
            snr REAL,
            fname TEXT,
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_res = """CREATE TABLE Results
            (blipID INT,
            timeID INT,
            time REAL,
            duration REAL,
            TS REAL,
            bkg_nllh REAL,
            sig_nllh REAL,
            imx REAL,
            imy REAL,
            A REAL,
            ind REAL,
            bkg_rates BLOB,
            FOREIGN KEY(blipID) REFERENCES Blips(blipID),
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_seeds = """CREATE TABLE Seeds
            (blipID INT,
            timeID INT,
            time REAL,
            duration REAL,
            imx REAL,
            imy REAL,
            proc_group INT,
            done INT,
            priority INT,
            FOREIGN KEY(blipID) REFERENCES Blips(blipID),
            FOREIGN KEY(timeID) REFERENCES TimeWindows(timeID));"""

    sql_tab_info = """CREATE TABLE DataInfo
            (tstartMET REAL,
            tstopMET REAL,
            trigtimeMET REAL,
            utcf REAL,
            mjdref REAL,
            tstartUTC TEXT,
            tstopUTC TEXT,
            trigtimeUTC TEXT);"""

    sql_tab_files = """CREATE TABLE Files
            (evfname TEXT,
            detmask TEXT,
            attfname TEXT,
            drmDir TEXT,
            rtDir TEXT,
            workDir TEXT,
            batmlDir TEXT,
            SkymapFname TEXT,
            pcodeFname TEXT);"""

    sql_tab_list = [
        sql_tab_info,
        sql_tab_quads,
        sql_tab_twinds,
        sql_tab_ratefits,
        sql_tab_rates,
        sql_tab_blips,
        sql_tab_img,
        sql_tab_res,
        sql_tab_files,
        sql_tab_seeds,
        sql_tab_twind_stat,
        sql_tab_sigimg,
        sql_tab_job_square,
        sql_tab_square_stat,
        sql_tab_square_res,
    ]

    return sql_tab_list


def create_tables(conn, sql_tab_list):
    for sql_tab in sql_tab_list:
        conn.execute(sql_tab)

    conn.commit()


def create_table(conn, tab_name):
    sql_tab_list = get_sql_tab_list()
    for sql_tab in sql_tab_list:
        if tab_name in sql_tab:
            conn.execute(sql_tab)

    conn.commit()


# def make_timeID(time, duration, tmin):
#
#     str0 = str(int(1000*(time - tmin)))
#     str1 = str(int(1000*(duration))) if duration >= 1 else\
#             '0' + str(int(1000*(duration)))
#
#     timeID = int(str0 + str1)
#     return timeID


def make_timeID(time, duration, trig_time):
    str0 = str(int(1000 * (time - trig_time)))
    str1 = (
        str(int(1000 * (duration)))
        if duration >= 1
        else "0" + str(int(1000 * (duration)))
    )

    timeID = int(str0 + str1)
    return timeID


def make_timeIDs(times, durs, trig_time):
    dts = times - trig_time
    dts = np.round(dts * 1000, decimals=0).astype(np.int64)
    str0s = dts.astype(np.str_)
    durs_ = np.round(durs * 1000, decimals=0).astype(np.int64)
    str1s = durs_.astype(np.str_)

    bl = durs < 1
    s_ = np.empty(len(bl), dtype=np.str_)
    s_[bl] = "0"
    s_[~bl] = ""
    str1s = np.char.add(s_, str1s)
    timeIDs = np.char.add(str0s, str1s)
    return timeIDs


def timeID2time_dur(timeID, trig_time):
    strID = str(int(timeID))
    if len(strID) <= 4:
        dur = float(strID) / 1000.0
        dt = 0.0
    elif strID[-5:] == '16384':
        dur = float(strID[-5:]) / 1000.0
        if len(strID) == 5:
            dt = 0.0
        else:
            dt = float(strID[:-5]) / 1000.0
    else:
        dur = float(strID[-4:]) / 1000.0
        dt = float(strID[:-4]) / 1000.0
    time = dt + trig_time
    return time, dur


def write_sigimg_line(conn, time, duration, trig_time, fname, Npix):
    timeID = make_timeID(time, duration, trig_time)
    sql = """INSERT INTO SigImages (timeID, time, duration, fname, Npix)
            VALUES (?, ?, ?, ?, ?);"""
    conn.execute(sql, (timeID, time, duration, fname, Npix))
    conn.commit()


def write_JobSquare_line(conn, data_dict):
    sql = """INSERT INTO JobSquare (squareID, proc_group, imx0, imx1, imy0, imy1)
            VALUES (?, ?, ?, ?, ?, ?);"""
    conn.execute(
        sql,
        (
            data_dict["squareID"],
            data_dict["proc_group"],
            data_dict["imx0"],
            data_dict["imx1"],
            data_dict["imy0"],
            data_dict["imy1"],
        ),
    )
    conn.commit()


def update_square_stat(conn, timeID, squareID):
    sql = "UPDATE SquareStatus set done=1 where squareID=? AND timeID=?;"
    if np.isscalar(timeID):
        with conn:
            conn.execute(sql, (squareID, timeID))
    else:
        vals = []
        for i in range(len(timeID)):
            vals.append((squareID, timeID[i]))
        with conn:
            conn.executemany(sql, vals)


def write_square_res_line(conn, data_dict):
    col_names = []
    values = []
    for k, val in data_dict.items():
        col_names.append(k)
        values.append(val)

    sql = (
        "INSERT INTO SquareResults ("
        + ",".join(col_names)
        + ") values("
        + ",".join(["?"] * len(values))
        + ")"
    )

    sql2 = "UPDATE SquareStatus set done=1 where squareID=? AND timeID=?;"
    with conn:
        conn.execute(sql, values)
        conn.execute(sql2, (data_dict["squareID"], data_dict["timeID"]))


def write_square_results(conn, res_df):
    sql = "UPDATE SquareStatus set done=1 where squareID=? AND timeID=?;"
    res_df.to_sql("SquareResults", conn, if_exists="append", index=False)
    # dfg = res_df.groupby(['squareID','timeID'])
    # with conn:
    #     conn.executemany(sql, dfg.groups.keys())


def setup_square_stat(conn, timeIDs, squareIDs):
    sql = "INSERT INTO SquareStatus (timeID, squareID) VALUES (?,?);"

    vals = []

    for timeID in timeIDs:
        for squareID in squareIDs:
            vals.append((timeID, squareID))

    with conn:
        conn.executemany(sql, vals)


def write_twind_line(conn, timeID, data_dict):
    data_dict["timeID"] = timeID
    col_names = []
    values = []
    for k, val in data_dict.items():
        col_names.append(k)
        values.append(val)

    sql = (
        "INSERT INTO TimeWindows ("
        + ",".join(col_names)
        + ") values("
        + ",".join(["?"] * len(values))
        + ")"
    )

    conn.execute(sql, values)
    conn.commit()


def setup_tab_twinds(
    conn,
    trig_time,
    ntdbls=4,
    min_bin_size=0.256,
    att_tab=None,
    acs_tab=None,
    acs_att_dt_max=30.0,
    t_wind=15,
    tmin=None,
    tmax=None,
    GTI=None,
):
    bin_size = min_bin_size
    tstep = bin_size / 4.0
    tstarts = np.arange(-t_wind * 1.024, t_wind * 1.024, tstep) + trig_time
    tstops = tstarts + bin_size
    if tmin is not None:
        bl = tstarts >= (tmin + trig_time)
        tstarts = tstarts[bl]
        tstops = tstops[bl]
    if tmax is not None:
        bl = tstops < (tmax + trig_time)
        tstarts = tstarts[bl]
        tstops = tstops[bl]

    for ii in range(ntdbls):
        for i in range(len(tstarts)):
            timeID = make_timeID(tstarts[i], bin_size, trig_time)
            data_dict = {}
            data_dict["time"] = tstarts[i]
            data_dict["duration"] = bin_size
            data_dict["time_end"] = tstarts[i] + bin_size

            if GTI is not None:
                if not check_if_in_GTI(GTI, data_dict["time"], data_dict["time_end"]):
                    continue

            if att_tab is not None:
                tmid = (tstarts[i] + tstops[i]) / 2.0
                att_ind = np.argmin(np.abs(tmid - att_tab["TIME"]))
                if np.abs(tmid - att_tab["TIME"][att_ind]) < acs_att_dt_max:
                    att_quat = att_tab["QPARAM"][att_ind]
                    data_dict["quat0"] = att_quat[0]
                    data_dict["quat1"] = att_quat[1]
                    data_dict["quat2"] = att_quat[2]
                    data_dict["quat3"] = att_quat[3]

            if acs_tab is not None:
                tmid = (tstarts[i] + tstops[i]) / 2.0
                acs_ind = np.argmin(np.abs(tmid - acs_tab["TIME"]))
                if np.abs(tmid - acs_tab["TIME"][acs_ind]) < acs_att_dt_max:
                    data_dict["settled"] = acs_tab["FLAGS"][acs_ind][1]
                    data_dict["SAA"] = acs_tab["FLAGS"][acs_ind][2]
                    data_dict["sat_lon"] = acs_tab["POSITION"][acs_ind][0]
                    data_dict["sat_lat"] = acs_tab["POSITION"][acs_ind][1]

            # sql = '''INSERT INTO TimeWindows (timeID, time, duration, time_end)
            #         VALUES (%d, %f, %f, %f);'''\
            #         %(timeID, tstarts[i], bin_size, tstops[i])

            write_twind_line(conn, timeID, data_dict)

            # conn.execute(sql)

        bin_size *= 2
        tstep *= 2
        tstarts = np.arange(-t_wind * 1.024, t_wind * 1.024, tstep) + trig_time
        tstops = tstarts + bin_size
        if tmin is not None:
            bl = tstarts >= (tmin + trig_time)
            tstarts = tstarts[bl]
            tstops = tstops[bl]
        if tmax is not None:
            bl = tstops < (tmax + trig_time)
            tstarts = tstarts[bl]
            tstops = tstops[bl]

    conn.commit()


def setup_tab_twind_status(conn, timeIDs):
    for i in range(len(timeIDs)):
        sql = """INSERT INTO TimeWindowStatus (timeID, BlipsFounds, SeedsAssigned, llhDone)
                VALUES (%d, %d, %d, %d);""" % (
            timeIDs[i],
            0,
            0,
            0,
        )
        conn.execute(sql)
    conn.commit()


def get_conn(db_fname, timeout=10.0):
    conn = sqlite3.connect(db_fname, timeout=timeout)

    return conn


def setup_files_tab(
    conn, evfname, att_fname, detmask, rt_dir, drm_dir, work_dir, bat_ml_dir
):
    sql = """INSERT INTO Files
            (evfname, detmask, attfname, drmDir, rtDir, workDir, batmlDir)
            VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s");""" % (
        evfname,
        detmask,
        att_fname,
        drm_dir,
        rt_dir,
        work_dir,
        bat_ml_dir,
    )
    conn.execute(sql)
    conn.commit()


def setup_tab_info(conn, ev_fname, trigtime):
    event_file = fits.open(ev_fname)

    tmin = np.min(event_file[1].data["TIME"])
    tmax = np.max(event_file[1].data["TIME"])

    tmin_utc = met2utc_str(tmin, event_file)
    tmax_utc = met2utc_str(tmax, event_file)

    if isinstance(trigtime, float):
        trigtimeMET = trigtime
        trigtimeUTC = met2utc_str(trigtimeMET, event_file)
    else:
        trigtimeMET = utc2met(trigtime.isot, event_file)
        trigtimeUTC = trigtime.iso

    print(("trigtimeUTC: ", trigtimeUTC))
    print(("tmin_UTC: ", tmin_utc))

    sql = """INSERT INTO DataInfo
            (tstartMET, tstopMET, trigtimeMET, tstartUTC, tstopUTC, trigtimeUTC)
            VALUES (%f, %f, %f, "%s", "%s", "%s");""" % (
        tmin,
        tmax,
        trigtimeMET,
        tmin_utc,
        tmax_utc,
        trigtimeUTC,
    )
    conn.execute(sql)

    conn.commit()


def write_rate_fit_line(conn, rate_obj, ind, ebin, quadID, rate, err, chi2):
    sql = """INSERT INTO RateFits
            (time, quadID, pre, post, ebin,
            deg, rate, error, chi2) VALUES
            (%f, %d, %d, %d, %d, %d, %f, %f, %f);
            """ % (
        rate_obj.t_poly_ax[ind],
        quadID,
        1,
        int(rate_obj.post),
        ebin,
        rate_obj.deg,
        rate,
        err,
        chi2,
    )

    conn.execute(sql)
    conn.commit()


def write_rate_fits_from_obj(conn, rate_obj, quadID):
    for i in range(rate_obj.n_lin_pnts):
        rates, errs, chi2s = rate_obj.get_rate(rate_obj.t_poly_ax[i], chi2=True)
        for j in range(rate_obj.nebins):
            if ~np.isfinite(chi2s[j]):
                chi2s[j] = 0.0
            if ~np.isfinite(errs[j]):
                errs[j] = 0.0
            write_rate_fit_line(
                conn, rate_obj, i, j, quadID, rates[j], errs[j], chi2s[j]
            )


def append_rate_tab(conn, df_twind, quadID, bkg_nllh, sig_nllh, bf_nsigs, bf_inds):
    TS = np.sqrt(2.0 * (bkg_nllh - sig_nllh))
    TS[(np.isnan(TS))] = 0.0
    df_dict = {
        "timeID": df_twind.timeID,
        "quadID": quadID,
        "time": df_twind.time,
        "duration": df_twind.duration,
        "bkg_nllh": bkg_nllh,
        "sig_nllh": sig_nllh,
        "TS": TS,
        "nsig": bf_nsigs,
        "PlawInd": bf_inds,
    }
    df = pd.DataFrame(df_dict)
    df.to_sql("Rates", conn, if_exists="append", index=False)


def update_twind_stat(conn, timeID, col="BlipsFounds", val=1):
    sql = "update TimeWindowStatus set %s=%d where timeID=%d" % (col, val, timeID)

    conn.execute(sql)
    conn.commit()


def write_cat2db(conn, cat_fname, timeID):
    cat = fits.open(cat_fname)[1].data

    blipIDs = np.arange(len(cat), dtype=np.int64)

    df_dict = {
        "blipID": blipIDs,
        "timeID": timeID,
        "time": cat["TIME"],
        "duration": cat["EXPOSURE"],
        "snr": cat["SNR"],
        "imx": cat["IMX"],
        "imy": cat["IMY"],
        "pc": cat["PCODEFR"],
    }
    df = pd.DataFrame(df_dict)
    df.to_sql("Blips", conn, if_exists="append", index=False)

    update_twind_stat(conn, timeID)


def write_cats2db(conn, cat_fnames, timeID):
    blipID0 = 0
    for i, cat_fname in enumerate(cat_fnames):
        cat = fits.open(cat_fname)[1].data

        blipIDs = np.arange(len(cat), dtype=np.int64) + blipID0
        blipID0 = np.max(blipIDs) + 1

        df_dict = {
            "blipID": blipIDs,
            "timeID": timeID,
            "time": cat["TIME"],
            "duration": cat["EXPOSURE"],
            "snr": cat["SNR"],
            "imx": cat["IMX"],
            "imy": cat["IMY"],
            "pc": cat["PCODEFR"],
            "ebin": i,
        }
        df = pd.DataFrame(df_dict)
        try:
            df.to_sql("Blips", conn, if_exists="append", index=False)
        except sqlite3.IntegrityError:
            continue

    update_twind_stat(conn, timeID)


def write_seed_line(
    conn, blipID, timeID, time, duration, imx, imy, proc_group, priority
):
    sql = """INSERT INTO Seeds
            (blipID, timeID, time, duration,\
            imx, imy, proc_group, priority, done) VALUES
            (%d, %d, %f, %f, %f, %f, %d, %d, %d);
            """ % (
        blipID,
        timeID,
        time,
        duration,
        imx,
        imy,
        proc_group,
        priority,
        0,
    )

    conn.execute(sql)
    conn.commit()


def write_seeds(conn, blip_df):
    """
    Writes seeds to the seed table

    Parameters:
    conn: sqlite connection object
    blip_df: DataFrame, a blip table with proc_group and priority columns added

    """

    for i in range(len(blip_df)):
        write_seed_line(
            conn,
            blip_df["blipID"][i],
            blip_df["timeID"][i],
            blip_df["time"][i],
            blip_df["duration"][i],
            blip_df["imx"][i],
            blip_df["imy"][i],
            blip_df["proc_group"][i],
            blip_df["priority"][i],
        )

    update_twind_stat(conn, blip_df["timeID"][0], col="SeedsAssigned")


def write_result(conn, seed_tab_row, sig_param_dict, bkg_nllh, sig_nllh, TS):
    """
    Writes a result row to the result table and updates the seed table as done

    Parameters:
    conn: sqlite connection object
    seed_tab_row: named tuple, a row of the seed table as a named tuple
    sig_param_dict: dictionary, a dict of the best fit signal paramters
    bkg_nllh: float, the background NLLH
    sig_nllh: float, the signal NLLH
    TS: float, the test statistic value
    """

    imx = sig_param_dict["Signal_imx"]
    imy = sig_param_dict["Signal_imy"]
    A = sig_param_dict["Signal_A"]
    ind = sig_param_dict["Signal_gamma"]
    bkg_pnames = sorted([k for k in list(sig_param_dict.keys()) if "bkg_rate" in k])
    bkg_rates = "'"
    for bkg_pname in bkg_pnames:
        bkg_rates += "%.3f, " % (sig_param_dict[bkg_pname])
    bkg_rates = bkg_rates[:-2] + "'"  # + ")"

    logging.debug(
        (
            seed_tab_row.blipID,
            seed_tab_row.timeID,
            seed_tab_row.time,
            seed_tab_row.duration,
            TS,
            bkg_nllh,
            sig_nllh,
            imx,
            imy,
            A,
            ind,
            bkg_rates,
        )
    )
    sql = """INSERT INTO Results
            (blipID, timeID, time, duration, TS,
            bkg_nllh, sig_nllh, imx, imy, A, ind, bkg_rates) VALUES
            (%d, %d, %f, %f, %f, %f, %f, %f, %f, %f, %f, %s);
            """ % (
        seed_tab_row.blipID,
        seed_tab_row.timeID,
        seed_tab_row.time,
        seed_tab_row.duration,
        TS,
        bkg_nllh,
        sig_nllh,
        imx,
        imy,
        A,
        ind,
        bkg_rates,
    )
    #    sql = '''INSERT INTO Results
    #            (blipID, timeID, time, duration, TS, bkg_nllh, sig_nllh, imx, imy, A, ind)
    #            VALUES
    #            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'''#\
    #            %(seed_tab_row.blipID, seed_tab_row.timeID, seed_tab_row.time,\
    #            seed_tab_row.duration, TS, bkg_nllh, sig_nllh,\
    #                imx, imy, A, ind)
    #
    #    sql = '''INSERT INTO Results
    #            (blipID, timeID, TS, imx, imy, A, ind)
    #            VALUES
    #            (?, ?, ?, ?, ?, ?, ?);'''

    logging.debug("sql snippet: ")
    logging.debug(sql)

    conn.execute(sql)
    # conn.execute(sql, (seed_tab_row.blipID, seed_tab_row.timeID, TS, imx, imy, A, ind))
    conn.commit()

    sql = "update Seeds set done=1 where blipID=%d AND timeID=%d" % (
        seed_tab_row.blipID,
        seed_tab_row.timeID,
    )

    conn.execute(sql)
    conn.commit()


def write_results(conn, seed_tab_rows, sig_param_dicts, bkg_nllh, sig_nllhs, TSs):
    """
    Writes a result row to the result table and updates the seed table as done

    Parameters:
    conn: sqlite connection object
    seed_tab_rows: list of named tuples, a row of the seed table as a named tuple
    sig_param_dicts: list of dictionaries, a dict of the best fit signal paramters
    bkg_nllh: float, the background NLLH
    sig_nllhs: list of floats, the signal NLLH
    TSs: list of floats, the test statistic value
    """

    nrows = len(TSs)
    data = []
    for i in range(nrows):
        imx = sig_param_dicts[i]["Signal_imx"]
        imy = sig_param_dicts[i]["Signal_imy"]
        A = sig_param_dicts[i]["Signal_A"]
        ind = sig_param_dicts[i]["Signal_gamma"]
        bkg_pnames = sorted(
            [k for k in list(sig_param_dicts[i].keys()) if "bkg_rate" in k]
        )
        bkg_rates = "'"
        for bkg_pname in bkg_pnames:
            bkg_rates += "%.3f, " % (sig_param_dicts[i][bkg_pname])
        bkg_rates = bkg_rates[:-2] + "'"  # + ")"

        data.append(
            (
                seed_tab_rows[i].blipID,
                seed_tab_rows[i].timeID,
                seed_tab_rows[i].time,
                seed_tab_rows[i].duration,
                TSs[i],
                bkg_nllh,
                sig_nllhs[i],
                imx,
                imy,
                A,
                ind,
                bkg_rates,
            )
        )

        # logging.debug((seed_tab_row.blipID, seed_tab_row.timeID, seed_tab_row.time,\
        #         seed_tab_row.duration, TS, bkg_nllh, sig_nllh,\
        #             imx, imy, A, ind, bkg_rates))
    sql = """INSERT INTO Results
            (blipID, timeID, time, duration, TS,
            bkg_nllh, sig_nllh, imx, imy, A, ind, bkg_rates) VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """

    logging.debug("sql snippet: ")
    logging.debug(sql)

    data_seeds = []

    for row in seed_tab_rows:
        data_seeds.append((row.blipID, row.timeID))

    sql2 = "update Seeds set done=1 where blipID=? AND timeID=?"

    with conn:
        conn.executemany(sql, data)
        conn.executemany(sql2, data_seeds)
    # conn.commit()

    # conn.execute(sql)
    # conn.commit()


def write_results_fromSigImg(
    conn, seed_tab_rows, t0s, dts, sig_param_dicts, bkg_nllhs, sig_nllhs, TSs
):
    """
    Writes a result row to the result table and updates the seed table as done

    Parameters:
    conn: sqlite connection object
    seed_tab_rows: list of dataframe rows, a row of the seed table
    t0s: list of floats, starting time of the time bin
    dts: list of floats, duration of time bin
    sig_param_dicts: list of dictionaries, a dict of the best fit signal paramters
    bkg_nllhs: list of floats, the background NLLH
    sig_nllhs: list of floats, the signal NLLH
    TSs: list of floats, the test statistic value
    """

    nrows = len(TSs)
    data = []
    for i, row in enumerate(seed_tab_rows):
        imx = sig_param_dicts[i]["Signal_imx"]
        imy = sig_param_dicts[i]["Signal_imy"]
        A = sig_param_dicts[i]["Signal_A"]
        ind = sig_param_dicts[i]["Signal_gamma"]
        # bkg_pnames = sorted([k for k in sig_param_dicts[i].keys() if 'bkg_rate' in k])
        # bkg_rates = "'"
        # for bkg_pname in bkg_pnames:
        #     bkg_rates += "%.3f, " %(sig_param_dicts[i][bkg_pname])
        # bkg_rates = bkg_rates[:-2] + "'" #+ ")"

        data.append(
            (
                row["timeID"],
                t0s[i],
                dts[i],
                TSs[i],
                bkg_nllhs[i],
                sig_nllhs[i],
                imx,
                imy,
                A,
                ind,
            )
        )

    sql = """INSERT INTO Results
            (timeID, time, duration, TS,
            bkg_nllh, sig_nllh, imx, imy, A, ind) VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """

    logging.debug("sql snippet: ")
    logging.debug(sql)

    data_seeds = []

    for row in seed_tab_rows:
        data_seeds.append((row["timeID"], row["imx"], row["imy"]))

    sql2 = "update ImageSigs set done=1 where timeID=? AND imx=? AND imy=?"

    with conn:
        conn.executemany(sql, data)
        conn.executemany(sql2, data_seeds)
