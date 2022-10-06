from get_FRB_emails4cron import get_conn, mk_frb_db
db_fname = '/storage/home/gzr5209/work/realtime_workdir_NITRATES/FRB.db'
conn = get_conn(db_fname)
mk_frb_db(conn)
