from astropy.time import Time
import os
import subprocess
import pandas as pd
import email, imaplib, base64
import logging, traceback, argparse
import sqlite3
import numpy as np
import healpy as hp

from hp_funcs import err_circle2prob_map


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_analysis',\
            help="Start analysis if there's a new event",\
            action='store_true')
    parser.add_argument('--dbfname', type=str,\
            help="database file name",
            default="/storage/home/gzr5209/work/realtime_workdir_NITRATES/GECAM.db")
    parser.add_argument('--logfname', type=str,\
            help="log file name",
            default="/storage/home/gzr5209/work/realtime_workdir_NITRATES/GECAMemail_parse.log")
    parser.add_argument('--user', type=str,\
            help="email receiving emails",
            default="amon.bat.psu@gmail.com")
    parser.add_argument('--pas', type=str,\
            help="password for email receiving emails",
            default='wgiymjqcyapafqfp')
    parser.add_argument('--imap_server', type=str,\
            help="server for email receiving emails",
            default="imap.gmail.com")
    parser.add_argument('--script_path', type=str,\
            help="bash script to run analysis",
            default="/storage/home/gzr5209/work/BatML_code_work/NITRATES/run_stuff_grb2_vc_realtime.sh")
    parser.add_argument('--workdir', type=str,\
            help="bash script to run analysis",
            default="/storage/home/gzr5209/work/realtime_workdir_NITRATES/")
    parser.add_argument('--htmldir', type=str,\
            help="bash script to run analysis",
            default="/storage/home/gzr5209/work/realtime_workdir_NITRATES/GECAM_BAT/")

    args = parser.parse_args()
    return args


def mk_frb_db(conn):

    sql_command = '''CREATE TABLE FRB
            (FRBid TEXT PRIMARY KEY NOT NULL,
            name TEXT,
            timeUTC TEXT,
            ra REAL,
            dec REAL,
            error REAL,
            dm REAL,
            snr REAL,
            flux REAL,
            observatory TEXT,
            EventType TEXT,
            importance REAL);'''

    conn.execute(sql_command)

    conn.commit()

def get_conn(db_fname):

    conn = sqlite3.connect(db_fname)

    return conn



def get_table_as_df(conn, table_name):

    sql = "select * from %s" %(table_name)
    df = pd.read_sql(sql, conn)
    return df


def get_imap_conn(user, pas, server):

    M = imaplib.IMAP4_SSL(server)
    M.login(user, pas)
    return M


def get_LVC_email_nums(M, box='INBOX'):

    rv, _ = M.select(box)

    rv, data = M.search(None, '(SUBJECT "LVC Subthreshold")', '(UNSEEN)')

    return data[0].split()


def get_FRB_email_nums(M, box='INBOX'):

    rv, _ = M.select(box)

    rv, data = M.search(None, '(SUBJECT "NEW FRB TRIGGER")', '(UNSEEN)')

    return data[0].split()


def get_GECAM_email_nums(M, box='INBOX'):

    rv, _ = M.select(box)

    rv, data = M.search(None, '(SUBJECT "GCN/GECAM_FLIGHT_NOTICE")', '(UNSEEN)')

    return data[0].split()


def fetch_parse_GECAM_emails(M, nums):

    data_dicts = []

    for num in nums:
        rv, data = M.fetch(num, '(RFC822)')
        print(data[0][1])
        msg = email.message_from_string(data[0][1])
        print(msg)
        m = msg.get_payload()#[0]
        data_dict = {}
        print(m)
        # for line in m.get_payload().split('\n'):
        for line in m.split('\n'):
            if len(line.split(":")) < 2:
                continue
            if line.split()[0] in ['From:', 'Date:', 'To:', 'Subject:']:
                continue
            k = line.split(":")[0].strip("<td>").strip("</td>").strip()
            if 'TRIGGER_DATE' in k:
                val0 = line.split(":")[1].strip().strip("<td>").strip("</td>").strip()
                val0 += ":" + line.split(":")[2].strip().strip("<td>").strip("</td>").strip()
                val0 += ":" + line.split(":")[3].strip().strip("<td>").strip("</td>").strip()
            elif 'TRIG_RA' in k or 'TRIG_DEC' in k or 'TRIG_ERROR' in k or 'TRIGGER_DUR' in k:
                val0 = line.split(":")[1].strip().strip("<td>").strip("</td>").strip()
                val0 = val0.split()[0]
            else:
                val0 = line.split(":")[1].strip().strip("<td>").strip("</td>").strip()
            try:
                val = float(val0)
                if 'TRIGGER_DUR' in k:
                    val /= 1000.0
            except:
                val = val0
            data_dict[k] = val

        # if 'GPSTime' in data_dict.keys():
        data_dicts.append(data_dict)
        logging.debug("data_dict")
        logging.debug(data_dict)
        logging.debug('%d of %d emails parsed'%(len(data_dicts),len(nums)))

    return data_dicts



def fetch_parse_FRB_emails(M, nums):

    data_dicts = []

    for num in nums:
        rv, data = M.fetch(num, '(RFC822)')
        print(data[0][1])
        msg = email.message_from_string(data[0][1])
        print(msg)
        m = msg.get_payload()#[0]
        data_dict = {}
        print(m)
        # for line in m.get_payload().split('\n'):
        for line in m.split('\n'):
            if len(line.split()) != 2:
                continue
            if line.split()[0] in ['From:', 'Date:', 'To:', 'Subject:']:
                continue
            k = line.split()[0][:-1]
            val0 = line.split()[1]
            try:
                val = float(val0)
            except:
                val = val0
            data_dict[k] = val
            # if 'time' in k:
            #     apyt = Time(val, format='isot')
            #     mjd_id = "FRB"+str(int(apyt.mjd*1000))
            #     data_dict['FRBid'] = mjd_id

        # if 'GPSTime' in data_dict.keys():
        data_dicts.append(data_dict)
        logging.debug("data_dict")
        logging.debug(data_dict)
        logging.debug('%d of %d emails parsed'%(len(data_dicts),len(nums)))

    return data_dicts


def fetch_parse_LVC_emails(M, nums):

    data_dicts = []

    for num in nums:
        rv, data = M.fetch(num, '(RFC822)')
        msg = email.message_from_string(data[0][1])
        m = msg.get_payload()[0]
        data_dict = {}
        for line in m.get_payload().split('\n'):
            if len(line.split()) != 2:
                continue
            if line.split()[0] in ['From:', 'Date:', 'To:', 'Subject:']:
                continue
            k = line.split()[0][:-1]
            val0 = line.split()[1]
            try:
                val = float(val0)
            except:
                val = val0
            if k == 't_0':
                k = 'GPSTime'
                apyt = Time(val, format='gps', scale='utc')
                data_dict['isot'] = apyt.isot
            data_dict[k] = val

        if 'GPSTime' in data_dict.keys():
            data_dicts.append(data_dict)
            if len(data_dicts)%25 == 0:
                logging.debug('%d of %d emails parsed'%(len(data_dicts),len(nums)))

    return data_dicts


def get_LVC_skymap_email_nums(M, box='INBOX'):

    rv, _ = M.select(box)

    rv, data = M.search(None, '(SUBJECT "LVC Skymap")', '(UNSEEN)')

    return data[0].split()


def fetch_parse_LVC_skymap_emails(M, nums, dname):

    data_dicts = []

    for num in nums:
        rv, data = M.fetch(num, '(RFC822)')
        msg = email.message_from_string(data[0][1])
        m = msg.get_payload()[0]
        data_dict = {}
        for line in m.get_payload().split('\n'):
#             if len(line.split()) != 2:
#                 continue
            if line.split()[0] in ['From:', 'Date:', 'To:', 'Subject:']:
                continue
            k = line.split()[0][:-1]
            val0 = line.split()[-1]
#             if line.split()[0] == 'Event':
#                 k = line.split()[0] + line.split()[1][:-1]

            try:
                val = float(val0)
            except:
                val = val0
            if 'Eve' in k:
                k = 'UnixTime'
#                 apyt = Time(val, format='gps', scale='utc')
                apyt = Time(val, format='unix', scale='utc')
                data_dict['isot'] = apyt.isot
            data_dict[k] = val

        if 'UnixTime' in data_dict.keys():
            data_dicts.append(data_dict)

        direc = os.path.join(dname, data_dict['SID'])


        for part in msg.walk():
            # this part comes from the snipped I don't understand yet...
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            fileName = part.get_filename()
            fname = os.path.join(direc, fileName)
            if os.path.exists(fname):
                continue
            if not os.path.exists(direc):
                os.mkdir(direc)
            logging.info("Downloading file")
            fp = open(fname, 'wb')
        #     skmap = fits.open(part.get_payload(decode=True))
        #     print skmap
            fp.write(part.get_payload(decode=True))
            fp.close()
            logging.info("File saved to: ")
            logging.info(fname)

    return data_dicts



def main(args):

    logging.basicConfig(filename=args.logfname, level=logging.DEBUG,\
                    format='%(asctime)s-' '%(levelname)s- %(message)s')

    table_name = "GECAM"

    logging.info("Connecting to " + args.imap_server)
    M = get_imap_conn(args.user, args.pas, args.imap_server)

    logging.info("Getting email IDs")
    nums = get_GECAM_email_nums(M)
    logging.debug("Got %d email IDs" %(len(nums)))

    logging.info("Now fetchings and parsing emails")
    email_error = False

    try:
        data_dicts = fetch_parse_GECAM_emails(M, nums)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        logging.warning("Error fetching GECAM emails")
        email_error = True

    db_exists = os.path.exists(args.dbfname)
    html_file = os.path.join(args.htmldir, table_name+".html")
    html_exists = os.path.exists(html_file)

    # logging.info("Now fetchings and parsing skymap emails")
    # nums = get_LVC_skymap_email_nums(M)
    # logging.debug("Got %d email IDs for skymaps" %(len(nums)))
    # try:
    #     sky_map_data_dicts = fetch_parse_LVC_skymap_emails(M,nums,args.workdir)
    # except Exception as E:
    #     logging.error(E)
    #     logging.error(traceback.format_exc())
    #     logging.warning("Error fetching LVC skymaps")

    M.close()
    M.logout()

    logging.info("Done with email stuff")

    if email_error:
        logging.info("There was an error reading emails, so exiting now")
        return


    conn = get_conn(args.dbfname)

    df_new = pd.DataFrame(data_dicts)
    print("df_new:")
    print(df_new)

    if not html_exists:
        df_new.to_html(html_file)


    if db_exists:
        logging.info("Reached here 1")
        try:
	    df_old = get_table_as_df(conn, table_name)
	except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.warning("Error reading db table")
            db_exists=False
	

    New_Trigger = False

    if db_exists:
        logging.info("Reached here 3")
        ID_bl = ~np.isin(df_new['TRIGGER_UID'], df_old['TRIGGER_UID'])
        # df_new = pd.concat([df_old, df_new])

    new_trig_ids = []
    new_trig_times = []
    ras = []
    decs = []
    pos_errors = []
    trig_durs = []

    if db_exists:
        if np.sum(ID_bl) > 0:
            New_Trigger = True
            NewTrigger_df = df_new[ID_bl]
            # if len(NewTrigger_df) > 1:
            # idx = NewTrigger_df.groupby(['SID'])['far'].transform(min) == NewTrigger_df['far']
            for index, row in NewTrigger_df.iterrows():
                new_trig_ids.append(row['TRIGGER_UID'])
                new_trig_times.append(row['TRIGGER_DATE'])
                ras.append(row['TRIG_RA'])
                decs.append(row['TRIG_DEC'])
                pos_errors.append(row['TRIG_ERROR'])
                trig_durs.append(row['TRIGGER_DUR'])

        Nnewtrigs = len(new_trig_ids)

        logging.info(str(Nnewtrigs) + " new events")


        if Nnewtrigs == 0:# and len(df_new) == len(df_old):

            return
    else:
        if len(df_new) > 0:
            New_Trigger = True
            new_trig_ids = df_new['TRIGGER_UID']
            new_trig_times = df_new['TRIGGER_DATE']
            ras = df_new['TRIG_RA']
            decs = df_new['TRIG_DEC']
            pos_errors = df_new['TRIG_ERROR']
            trig_durs = df_new['TRIGGER_DUR']
            Nnewtrigs = len(new_trig_ids)
            logging.info(str(Nnewtrigs) + " new events")



    logging.info("Appending the sql table")
    try:
        df_new.to_sql(table_name, conn, if_exists='append', index=False)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        logging.warning("Trouble writing to DB, trying again")
        conn.close()
        conn = get_conn(args.dbfname)
        try:
            df_new.to_sql(table_name, conn, if_exists='append', index=False)
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.warning("Trouble writing to DB, done trying")
            conn.close()
            #New_Trigger = False
            #logging.warning("Won't run analysis until it can be written")
    try:
        df_tot = pd.concat([df_old, df_new])
        df_tot.to_html(html_file)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        logging.warning("Trouble writing HTML file")




    if args.run_analysis and New_Trigger:
        for i, trigID in enumerate(new_trig_ids):
            trid_id_str = 'G' + str(trigID)
            logging.info("Running analysis for " + str(trigID))
            direc = os.path.join(args.workdir, trid_id_str)
            if not os.path.exists(direc):
                os.mkdir(direc)
            script_out_path= os.path.join(direc,'run_stuff_out.log')

            sky_map = err_circle2prob_map(ras[i], decs[i], pos_errors[i], sys_err=4.0)
            sk_fname = os.path.join(direc, 'skymap.fits')
            hp.write_map(sk_fname, sky_map, nest=True, overwrite=True)


            with open(script_out_path, 'w') as f:
                logging.info("process args: ")
                process_args = [args.script_path, new_trig_times[i], trid_id_str]
               # if trig_durs[i] < 0.3:
               #     process_args = [args.script_path, new_trig_times[i], trid_id_str, 0.128]
               # else:
               #     process_args = [args.script_path, new_trig_times[i], trid_id_str,0.256]
                logging.info(process_args)
                subprocess.Popen(process_args, stdout=f, stderr=f)







if __name__ == "__main__":

    args = cli()

    main(args)
