import sqlite3

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
            eventFname TEXT
            );"""

db_fname = "BATQL.db"

conn = sqlite3.connect(db_fname)

conn.execute(sql_tab_qltdrss)
conn.commit()
