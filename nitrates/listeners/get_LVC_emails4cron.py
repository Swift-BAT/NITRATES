from astropy.time import Time
import os
import subprocess
import pandas as pd
import email, imaplib, base64
import logging, traceback, argparse
import sqlite3
import numpy as np


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_analysis',\
            help="Start analysis if there's a new event",\
            action='store_true')
    parser.add_argument('--dbfname', type=str,\
            help="database file name",
            default="/storage/work/jjd330/local/bat_data/realtime_workdir/LVCsub.db")
    parser.add_argument('--logfname', type=str,\
            help="log file name",
            default="/storage/work/jjd330/local/bat_data/realtime_workdir/LVCemail_parse.log")
    parser.add_argument('--user', type=str,\
            help="email receiving LVC emails",
            default="amon.bat.psu@gmail.com")
    parser.add_argument('--pas', type=str,\
            help="password for email receiving LVC emails",
            default=None)
    parser.add_argument('--imap_server', type=str,\
            help="server for email receiving LVC emails",
            default="imap.gmail.com")
    parser.add_argument('--script_path', type=str,\
            help="bash script to run analysis",
            default="/storage/work/j/jjd330/local/bat_data/BatML/run_stuff_O3b.sh")
    parser.add_argument('--workdir', type=str,\
            help="bash script to run analysis",
            default="/storage/work/j/jjd330/local/bat_data/realtime_workdir/")
    parser.add_argument('--htmldir', type=str,\
            help="bash script to run analysis",
            default="/storage/work/j/jjd330/local/bat_data/realtime_workdir/LVC_BAT/")

    args = parser.parse_args()
    return args


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

        if 'GPSTime' in list(data_dict.keys()):
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

        if 'UnixTime' in list(data_dict.keys()):
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

    table_name = "LVCsub"

    logging.info("Connecting to " + args.imap_server)
    M = get_imap_conn(args.user, args.pas, args.imap_server)

    logging.info("Getting email IDs")
    nums = get_LVC_email_nums(M)
    logging.debug("Got %d email IDs" %(len(nums)))

    logging.info("Now fetchings and parsing emails")
    email_error = False

    try:
        data_dicts = fetch_parse_LVC_emails(M, nums)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        logging.warning("Error fetching LVC emails")
        logging.warning("Will try to get skymaps but won't try to update DB or run analysis")
        email_error = True

    db_exists = os.path.exists(args.dbfname)
    html_file = os.path.join(args.htmldir, table_name+".html")
    html_exists = os.path.exists(html_file)

    logging.info("Now fetchings and parsing skymap emails")
    nums = get_LVC_skymap_email_nums(M)
    logging.debug("Got %d email IDs for skymaps" %(len(nums)))
    try:
        sky_map_data_dicts = fetch_parse_LVC_skymap_emails(M,nums,args.workdir)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        logging.warning("Error fetching LVC skymaps")

    M.close()
    M.logout()

    logging.info("Done with email stuff")

    if email_error:
        logging.info("There was an error reading emails, so exiting now")
        return


    conn = get_conn(args.dbfname)

    df_new = pd.DataFrame(data_dicts)

    if not html_exists:
        df_new.to_html(html_file)


    if db_exists:
        df_old = get_table_as_df(conn, table_name)

    New_Trigger = False

    if db_exists:
        ID_bl = ~np.isin(df_new['SID'], df_old['SID'])
        # df_new = pd.concat([df_old, df_new])

    new_trig_ids = []
    new_trig_times = []

    if db_exists:
        if np.sum(ID_bl) > 0:
            New_Trigger = True
            NewTrigger_df = df_new[ID_bl]
            # if len(NewTrigger_df) > 1:
            idx = NewTrigger_df.groupby(['SID'])['far'].transform(min) == NewTrigger_df['far']
            for index, row in NewTrigger_df[idx].iterrows():
                new_trig_ids.append(row['SID'])
                new_trig_times.append(row['isot'])

        Nnewtrigs = len(new_trig_ids)

        logging.info(str(Nnewtrigs) + " new events")


        if Nnewtrigs == 0:# and len(df_new) == len(df_old):

            return



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
            New_Trigger = False
            logging.warning("Won't run analysis until it can be written")
    df_tot = pd.concat([df_old, df_new])
    df_tot.to_html(html_file)



    if args.run_analysis and New_Trigger:
        for i, trigID in enumerate(new_trig_ids):
            logging.info("Running analysis for " + trigID)
            direc = os.path.join(args.workdir, trigID)
            if not os.path.exists(direc):
                os.mkdir(direc)
            script_out_path= os.path.join(direc,'run_stuff_out.log')

            with open(script_out_path, 'w') as f:
                logging.info("process args: ")
                process_args = [args.script_path, new_trig_times[i], trigID]
                logging.info(process_args)
                subprocess.Popen(process_args, stdout=f, stderr=f)







if __name__ == "__main__":

    args = cli()

    main(args)
