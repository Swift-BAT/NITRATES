from get_FRB_emails4cron import get_conn, mk_frb_db

db_fname = "/gpfs/group/jak51/default/realtime_workdir/FRB.db"
conn = get_conn(db_fname)
mk_frb_db(conn)
