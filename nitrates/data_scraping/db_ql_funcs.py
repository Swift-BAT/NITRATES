import sqlite3
import logging, traceback
import pandas as pd
import numpy as np
from astropy.io import fits


def get_conn(db_fname):
    conn = sqlite3.connect(db_fname)

    return conn


def delete_obsid(conn, obsid, table):
    sql = "DELETE FROM %s WHERE obsid=%s" % (table, obsid)
    cur = conn.cursor()
    cur.execute(sql)


def write_event_files2db(conn, obsid, data_dicts, table="SwiftQLevent"):
    # need to remove all rows with obsid
    # then write a line for each data_dict

    delete_obsid(conn, obsid, table)

    for data_dict in data_dicts:
        write_new_obsid_line(conn, obsid, data_dict=data_dict, table=table)


def write_new_obsid_line(conn, obsid, data_dict=None, table="SwiftQL"):
    data_dict["obsid"] = obsid
    col_names = []
    values = []
    for k, val in list(data_dict.items()):
        col_names.append(k)
        values.append(val)

    sql = (
        "INSERT INTO "
        + table
        + " ("
        + ",".join(col_names)
        + ") values("
        + ",".join(["?"] * len(values))
        + ")"
    )

    conn.execute(sql, values)
    conn.commit()


def update_obsid_line(conn, obsid, data_dict, table="SwiftQL"):
    sql = "UPDATE %s SET " % (table)
    col_names = []
    values = []
    for k, val in list(data_dict.items()):
        col_names.append(k)
        values.append(val)

    for i in range(len(col_names)):
        if i < (len(col_names) - 1):
            sql += "%s = ?," % (col_names[i])
        else:
            sql += "%s = ? " % (col_names[i])
    sql += 'WHERE obsid="%s"' % (obsid)
    logging.debug("sql update command: ")
    logging.debug(sql)
    logging.debug("values: ")
    logging.debug(values)

    conn.execute(sql, values)
    conn.commit()


def get_db_tab(conn, table_name):
    df = pd.read_sql("select * from %s" % (table_name), conn)
    return df


def get_ql_db_tab(conn):
    df = pd.read_sql("select * from SwiftQL", conn)
    return df


def get_qlevent_db_tab(conn):
    df = pd.read_sql("select * from SwiftQLevent", conn)
    return df


def get_gainoff_fname(conn, met):
    ql_tab = get_ql_db_tab(conn)
    bl = (ql_tab["METstart"] < (met + 30.0)) & (ql_tab["METstop"] > (met - 30.0))
    fnames = []
    if np.sum(bl) > 0:
        qlts = ql_tab[bl].groupby("obsid")
        for obsid, qlt in qlts:
            fnames.append(qlt.loc[qlt["ver"].argmax()]["GainOffFname"])
    else:
        return

    min_dts = np.zeros(len(fnames))
    for i, fname in enumerate(fnames):
        f = fits.open(fname)[1].data
        min_dts[i] = np.min(np.abs(met - f["TIME"]))
    fname = fnames[np.argmin(min_dts)]

    return fname
