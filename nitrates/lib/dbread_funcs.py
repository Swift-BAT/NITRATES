import numpy as np
import os
import sqlite3
from astropy.io import fits
import pandas as pd


def guess_dbfname(dname="."):
    fnames = os.listdir(dname)
    db_fnames = [fname for fname in fnames if ".db" in fname]
    if len(db_fnames) == 1:
        return db_fnames[0]
    elif len(db_fnames) > 1:
        print("More than one db fname to pick from")
        return db_fnames
    else:
        print("Can't find a db fname")
        return


def get_full_sqlite_table_as_df(conn, table_name):
    """
    Gets a full sqlite table and returns it as a Pandas DataFrame

    Should be just one row
    Parameters:
    conn: sqlite connection object

    Returns:
    df DataFrame: The sqlite table as a pandas DataFrame

    """
    sql = "select * from %s" % (table_name)
    df = pd.read_sql(sql, conn)
    return df


def get_info_tab(conn):
    """
    Gets the full DataInfo table

    Should be just one row
    Parameters:
    conn: sqlite connection object

    Returns:
    DataInfo DataFrame: The DataInfo table in a pandas DataFrame

    """
    df = pd.read_sql("select * from DataInfo", conn)
    return df


def get_files_tab(conn):
    """
    Gets the full Files table

    Should be just one row
    Parameters:
    conn: sqlite connection object

    Returns:
    Files DataFrame: The Files table in a pandas DataFrame

    """
    df = pd.read_sql("select * from Files", conn)
    return df


def get_rate_fits_tab(conn):
    """
    Gets the full RateFits table

    Parameters:
    conn: sqlite connection object

    Returns:
    RateFits DataFrame: The RateFits table in a pandas DataFrame

    """

    df = pd.read_sql("select * from RateFits", conn)
    return df


def get_twinds_tab(conn):
    """
    Gets the full TimeWindows table

    Parameters:
    conn: sqlite connection object

    Returns:
    TimeWindow DataFrame: The TimeWindows table in a pandas DataFrame

    """
    df = pd.read_sql("select * from TimeWindows", conn)
    return df


def get_rates_tab(conn):
    """
    Gets the full TimeWindows table

    Parameters:
    conn: sqlite connection object

    Returns:
    TimeWindow DataFrame: The TimeWindows table in a pandas DataFrame

    """
    df = pd.read_sql("select * from Rates", conn)
    return df


def get_top_rate_timeIDs(conn, N=10, TScut=None, ret_rate_tab=False, ret_TSs=False):
    df = get_rates_tab(conn)

    rates_twind_groups = df.groupby("timeID")
    twind_maxs = rates_twind_groups.TS.aggregate("max")
    if TScut is not None:
        bl = twind_maxs >= TScut
        top_twind_TSs = twind_maxs[bl].sort_values()
    else:
        top_twind_TSs = twind_maxs.sort_values().tail(N)
    timeIDs = top_twind_TSs.index.values[::-1]
    TSs = top_twind_TSs.values[::-1]

    if ret_rate_tab:
        return timeIDs, df
    elif ret_TSs:
        return timeIDs, TSs

    return timeIDs


def get_blips_tab(conn):
    """
    Gets the full Blips table

    Parameters:
    conn: sqlite connection object

    Returns:
    Blips DataFrame: The Blips table in a pandas DataFrame

    """
    df = pd.read_sql("select * from Blips", conn)
    return df


def query_blips_tab(
    conn, timeID=None, snr_min=-1e4, snr_max=1e4, duration=None, pc_min=0.0
):
    """
    Gets the Blips table with custom constrains

    Parameters:
    conn: sqlite connection object

    timeID: int (optional) get blips with this timeID
    snr_min: float (optional) get only the blips with >= this snr
    snr_max: float (optional) get only the blips with <= this snr
    duration: float (optional) only get blips with this duration (s)
    pc_min: float (optional) get only the blips with >= this partial coding



    Returns:
    Blips DataFrame: The Blips table in a pandas DataFrame

    """
    sql = "select * from Blips where "
    if timeID is not None:
        sql += "timeID=%d and " % (timeID)
    if duration is not None:
        sql += "duration between %.3f and %.3f and " % (
            duration - 1e-3,
            duration + 1e-3,
        )
    sql += "pc>=%.2f and snr between %.2f and %.2f" % (pc_min, snr_min, snr_max)

    df = pd.read_sql(sql, conn)
    return df


def get_twind_status_tab(conn):
    """
    Gets the full TimeWindowStatus table

    Parameters:
    conn: sqlite connection object

    Returns:
    TimeWindowStatus DataFrame: The TimeWindowStatus table in a pandas DataFrame

    """
    df = pd.read_sql("select * from TimeWindowStatus", conn)
    return df


def get_seeds_tab(conn, proc_group=None):
    """
    Gets the Seeds table

    Parameters:
    conn: sqlite connection object
    proc_group: int (optional) what proc_group to get the seeds for

    Returns:
    Seeds DataFrame: The Seeds table in a pandas DataFrame

    """
    sql = "select * from Seeds"
    if proc_group is not None:
        sql += " where proc_group=%d" % (proc_group)
    df = pd.read_sql(sql, conn)
    return df


def get_square_tab(conn, proc_group=None):
    """
    Gets the JobSquare table

    Parameters:
    conn: sqlite connection object
    proc_group: int (optional) what proc_group to get the seeds for

    Returns:
    JobSquare DataFrame: The JobSquare table in a pandas DataFrame

    """
    sql = "select * from JobSquare"
    if proc_group is not None:
        sql += " where proc_group=%d" % (proc_group)
    df = pd.read_sql(sql, conn)
    return df


def get_imgsig_tab(conn, proc_group=None):
    """
    Gets the Seeds table

    Parameters:
    conn: sqlite connection object
    proc_group: int (optional) what proc_group to get the seeds for

    Returns:
    Seeds DataFrame: The Seeds table in a pandas DataFrame

    """
    sql = "select * from ImageSigs"
    if proc_group is not None:
        sql += " where proc_group=%d" % (proc_group)
    df = pd.read_sql(sql, conn)
    return df
