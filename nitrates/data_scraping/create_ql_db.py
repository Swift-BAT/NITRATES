import sqlite3
import logging, traceback, argparse


def get_sql_tab_list():
    sql_tab_qlevent = """CREATE TABLE SwiftQLevent
            (obsid TEXT,
            ver INTEGER,
            METstart REAL,
            METstop REAL,
            UTCstart TEXT,
            UTCstop TEXT,
            eventURL TEXT,
            eventFname TEXT,
            obs_mode TEXT,
            FOREIGN KEY(obsid) REFERENCES SwiftQLeventOBS(obsid)
            );"""

    sql_tab_qleventOBS = """CREATE TABLE SwiftQLeventOBS
            (obsid TEXT PRIMARY KEY NOT NULL,
            ver INTEGER,
            obsDname TEXT
            );"""

    sql_tab_ql = """CREATE TABLE SwiftQL
            (obsid TEXT PRIMARY KEY NOT NULL,
            ver INTEGER,
            METstart REAL,
            METstop REAL,
            UTCstart TEXT,
            UTCstop TEXT,
            obsURL TEXT,
            obsDname TEXT,
            DetEnbFname TEXT,
            DetQualFname TEXT,
            GainOffFname TEXT,
            patFname TEXT,
            satFname TEXT
            );"""

    sql_tab_qltdrss = """CREATE TABLE SwiftQLtdrss
            (obsid TEXT PRIMARY KEY NOT NULL,
            METstart REAL,
            METstop REAL,
            Duration REAL,
            UTCstart TEXT,
            UTCstop TEXT,
            DateTrig TEXT,
            TrigTime REAL,
            eventURL TEXT,
            eventFname TEXT,
            );"""

    sql_tab_list = [
        sql_tab_qleventOBS,
        sql_tab_qlevent,
        sql_tab_ql,
    ]  # , sql_tab_qltdrss]

    return sql_tab_list


def create_tables(conn, sql_tab_list):
    for sql_tab in sql_tab_list:
        conn.execute(sql_tab)

    conn.commit()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_fname",
        type=str,
        help="file name to create the database at",
        default="BATQL.db",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()

    conn = sqlite3.connect(args.db_fname)

    sql_tab_list = get_sql_tab_list()

    create_tables(conn, sql_tab_list)
