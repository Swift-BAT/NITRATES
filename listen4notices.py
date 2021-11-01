import gcn
import numpy as np
import healpy as hp
import traceback
import socket
import os
import logging
import sys
import subprocess
import voeventparse as vp
from datetime import datetime

from helper_funcs import send_error_email, send_email
from hp_funcs import err_circle2prob_map
# sys.path.append('/var/lib/mysql_drbd/amon/monitor_scripts/check_events/')
# from functions import slack_message

# run as nohup python $batml_path'listen4notices.py' > /storage/work/jjd330/local/bat_data/realtime_workdir/listen4gcns_out.log 2>&1 &

workdir='/storage/work/j/jjd330/local/bat_data/realtime_workdir/'
script_path='/storage/work/j/jjd330/local/bat_data/BatML/run_stuff_grb2.sh'

INTEGRAL = [gcn.notice_types.INTEGRAL_SPIACS,
            gcn.notice_types.INTEGRAL_WAKEUP,
            gcn.notice_types.INTEGRAL_WEAK]

FERMI = [gcn.notice_types.FERMI_GBM_FLT_POS,
        gcn.notice_types.FERMI_GBM_GND_POS,
        gcn.notice_types.FERMI_GBM_FIN_POS,
        gcn.notice_types.FERMI_GBM_GND_POS,
        ]

MAXI = [gcn.notice_types.MAXI_UNKNOWN]

CALET = [gcn.notice_types.CALET_GBM_FLT_LC]

HAWC = [171]#gcn.notice_types.HAWC_BURST_MONITOR]

IC = [173, 174]#gcn.notice_types.ICECUBE_ASTROTRACK_GOLD,
        # gcn.notice_types.ICECUBE_ASTROTRACK_BRONZE]

# Function to call every time a GCN is received.
# Run only for notices of type
# LVC_PRELIMINARY, LVC_INITIAL, or LVC_UPDATE.
@gcn.handlers.include_notice_types(
    gcn.notice_types.INTEGRAL_SPIACS,
    gcn.notice_types.INTEGRAL_WAKEUP,
    gcn.notice_types.INTEGRAL_WEAK,
    gcn.notice_types.FERMI_GBM_FLT_POS,
    gcn.notice_types.FERMI_GBM_GND_POS,
    gcn.notice_types.FERMI_GBM_FIN_POS,
    gcn.notice_types.FERMI_GBM_GND_POS,
    gcn.notice_types.MAXI_UNKNOWN,
    gcn.notice_types.CALET_GBM_FLT_LC,
    171, 173, 174,
    # gcn.notice_types.HAWC_BURST_MONITOR,
    # gcn.notice_types.ICECUBE_ASTROTRACK_GOLD,
    # gcn.notice_types.ICECUBE_ASTROTRACK_BRONZE,
    )
def process_gcn(payload, root):

    print root.attrib['role']

    role = root.attrib['role']

    notice_type = gcn.handlers.get_notice_type(root)

    try:

        eventtime = root.find('.//ISOTime').text
        print eventtime


    except Exception as E:

        body = str(E)
        body += '\n' + traceback.format_exc()
        subject = 'listen4notices.py error on ' + socket.gethostname()
        send_error_email(subject, body)

    # Read all of the VOEvent parameters from the "What" section.
    params = {elem.attrib['name']:
              elem.attrib['value']
              for elem in root.iterfind('.//Param')}

    # rev = params['Pkt_Ser_Num']
    new_alert = False

    min_tbin = "0.256"

    try:
        name = ''
        if notice_type in MAXI:
            name = 'M'+params['EVENT_ID_NUM']
        elif notice_type in HAWC:
            try:
                name = 'H' + params['run_id'] + params['event_id']
            except Exception as E:
                logger.error(E)
                logger.warning("Couldn't get a position")
                logger.info(params)
                name = 'H' + eventtime
        elif notice_type in IC:
            try:
                name = 'IC' + params['run_id'] + params['event_id']
            except Exception as E:
                logger.error(E)
                logger.warning("Couldn't get a position")
                logger.info(params)
                name = 'IC' + eventtime
        elif notice_type in INTEGRAL:
            name = 'I' + params['TrigID']
        elif notice_type in FERMI:
            name = 'F' + params['TrigID']
        elif notice_type in CALET:
            name = 'C' + params['TrigID']
        else:
            name = eventtime

        direc = os.path.join(workdir, name)
        if not os.path.exists(direc):
            os.mkdir(direc)
            new_alert = True
        rev = 0
        fname = os.path.join(direc, name + '_' + str(rev) + '.xml')
        while True:
            if os.path.exists(fname):
                rev += 1
                fname = os.path.join(direc, name + '_' + str(rev) + '.xml')
            else:
                break
        with open(fname, 'wb') as f:
            f.write(payload)
        with open(fname, 'rb') as f:
            v = vp.load(f)

    except Exception as E:

        body = str(E)
        body += '\n' + traceback.format_exc()
        logger.error(body)
        subject = 'listen4notices.py error on ' + socket.gethostname()
        send_error_email(subject, body)

    has_pos = False
    try:
        pos2d=vp.convenience.get_event_position(v)
        ra = pos2d.ra
        dec = pos2d.dec
        pos_error = pos2d.err
        logger.info('ra: %.3f, dec: %.3f, pos_error: %.3f'%(ra,dec,pos_error))
        if np.isclose(ra,0) and np.isclose(dec,0) and np.isclose(pos_error,0):
            # do something
            has_pos = False
        elif notice_type in CALET:
            has_pos = False
        else:
            has_pos = True
            if notice_type in INTEGRAL:
                sky_map = err_circle2prob_map(ra, dec, pos_error, sys_err=1.0)
            else:
                sky_map = err_circle2prob_map(ra, dec, pos_error)
            sk_fname = os.path.join(direc, 'skymap.fits')
            hp.write_map(sk_fname, sky_map, nest=True, overwrite=True)

    except Exception as E:
        logger.error(E)
        logger.warning("Couldn't get a position")


    try:

        if role == 'test':

            # eventtime = "2019-06-16T0"
            # eventtime += str(np.random.randint(0,high=9))
            # eventtime += ":33:27.059"
            try:
                subj = "Test GCN Notice: Alert Type " + params['NOTICE_TYPE']
            except:
                subj = "Test GCN Notice"
            body = "Test GCN event\n"
            # channel = 'test-alerts'
            mail_fname = 'mailing_list.txt'
        else:
            try:
                subj = "GCN Notice: Alert Type " + params['NOTICE_TYPE']
            except:
                subj = "GCN Notice"
            # subj = "GW event: Maybe Real Alert Type " + params['AlertType']
            body = "GW Notice\n"
            # channel = 'alerts'
            mail_fname = 'mailing_list.txt'
        body += "Starting analysis\n"
        body += "role: " + role + '\n'
        body += "IVORN: " + str(root.attrib['ivorn']) + '\n'
        body += "IsoTime: " + eventtime + '\n'

        for key, value in params.items():
            body += key
            body += '='
            body += value
            body += '\n'

        # GraceID = str(params['GraceID'])

        to = []

        f = open(mail_fname, 'r')
        for line in f:
            to.append(line.split('\n')[0])
        f.close()
        print to

        send_email(subj, body, to)

        # slack_message(':gw: ' +body, channel ,attachment=None)

    except Exception as E:

        body = str(E)
        body += '\n' + traceback.format_exc()
        # print body
        logger.error(body)
        # subject = 'listen4notices.py error on ' + socket.gethostname()
        # send_error_email(subject, body)

    try:
        if 'Time_Scale' in params.keys():
            tscale = float(params['Time_Scale'])
            if tscale < 0.4:
                min_tbin = '0.128'
            if tscale > 2.0:
                min_tbin = '0.512'
        elif 'Data_Timescale' in params.keys():
            tscale = float(params['Data_Timescale'])
            if tscale < 0.3:
                min_tbin = '0.128'
            if tscale > 2.0:
                min_tbin = '0.512'
        elif 'Trig_Timescale' in params.keys():
            tscale = float(params['Trig_Timescale'])
            if tscale < 0.3:
                min_tbin = '0.128'
            if tscale > 2.0:
                min_tbin = '0.512'
    except Exception as E:
        logger.error(E)
        logger.warning("Coulnd't find a time scale")


    try:


        script_out_path= os.path.join(direc,'run_stuff_out.log')

        if new_alert and not (role == "test"):

            with open(script_out_path, 'w') as f:
                process_args = [script_path, eventtime, name, min_tbin]
                subprocess.Popen(process_args, stdout=f, stderr=f)


    except Exception as E:
        body = str(E)
        body += '\n' + traceback.format_exc()
        try:
            graceID = params['GraceID']
            body += '\nGraceID: ' + graceID
        except:
            logger.warn("error and doesn't know GraceID")
        logger.error(body)
        subject = 'listen4notices.py error on ' + socket.gethostname()
        send_error_email(subject, body)



if __name__ == "__main__":

    logger = logging.getLogger('gcn_notice_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    log_fname = os.path.join(workdir, 'gcn_listner.log')
    pid_fname = os.path.join(workdir, 'gcn_listner.pid')
    with open(pid_fname, 'wb') as f:
        f.write(str(os.getpid()))
    fh = logging.FileHandler(filename=log_fname)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    she = logging.StreamHandler(sys.stderr)
    she.setLevel(logging.ERROR)
    she.setFormatter(formatter)
    logger.addHandler(she)

    print 'PID: ', os.getpid()
    gcn.listen(handler=process_gcn, log=logger)
