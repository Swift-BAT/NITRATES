import time
import numpy as np
import pandas as pd
import healpy as hp
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from scipy import interpolate
import os, socket, subprocess, shlex
import argparse
import logging, traceback
import paramiko

from ..lib.helper_funcs import (
    send_email,
    send_error_email,
    send_email_attach,
    send_email_wHTML,
)

from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_files_tab, get_info_tab, guess_dbfname
from ..lib.coord_conv_funcs import convert_radec2imxy, convert_imxy2radec


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument(
        "--fp_dir",
        type=str,
        help="Directory where the detector footprints are",
        default="/storage/work/jjd330/local/bat_data/rtfp_dir_npy/",
    )
    parser.add_argument(
        "--Nrate_jobs", type=int, help="Total number of jobs", default=16
    )
    parser.add_argument(
        "--TSscan", type=float, help="Min TS needed to do a full FoV scan", default=6.25
    )
    parser.add_argument(
        "--pix_fname",
        type=str,
        help="Name of the file with good imx/y coordinates",
        default="good_pix2scan.npy",
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--GWname", type=str, help="Name of the event to submit jobs as", default=""
    )
    parser.add_argument(
        "--queue",
        type=str,
        help="Name of the queue to submit jobs to",
        default="cyberlamp",
    )
    parser.add_argument(
        "--qos", type=str, help="Name of the qos to submit jobs to", default=None
    )
    parser.add_argument(
        "--pcfname",
        type=str,
        help="Name of the partial coding image",
        default="pc_2.img",
    )
    parser.add_argument(
        "--BKGpyscript",
        type=str,
        help="Name of python script for Bkg Estimation",
        default="do_bkg_estimation_wPSs_mp.py",
    )
    parser.add_argument(
        "--RATEpyscript",
        type=str,
        help="Name of python script for Rates analysis",
        default="do_rates_mle_wPSs.py",
    )
    parser.add_argument(
        "--LLHpyscript",
        type=str,
        help="Name of python script for LLH analysis",
        default="do_llh_wPSs_uncoded_realtime.py",
    )
    parser.add_argument(
        "--SCANpyscript",
        type=str,
        help="Name of python script for FoV scan",
        default="do_llh_scan_uncoded.py",
    )
    parser.add_argument(
        "--PEAKpyscript",
        type=str,
        help="Name of python script for FoV scan",
        default="do_intLLH_forPeaks.py",
    )
    parser.add_argument(
        "--do_bkg", help="Submit the BKG estimation script", action="store_true"
    )
    parser.add_argument("--do_rates", help="Submit the Rate jobs", action="store_true")
    parser.add_argument("--do_llh", help="Submit the llh jobs", action="store_true")
    parser.add_argument("--do_scan", help="Submit the scan jobs", action="store_true")
    parser.add_argument(
        "--skip_waiting",
        help="Skip waiting for the stuff to finish and use what's there now",
        action="store_true",
    )
    parser.add_argument(
        "--archive", help="Run in archive mode, not realtime mode", action="store_true"
    )
    parser.add_argument("--rhel7", help="Submit to a rhel7 node", action="store_true")
    parser.add_argument(
        "--pbs_fname",
        type=str,
        help="Name of pbs script",
        default="/storage/work/jjd330/local/bat_data/BatML/submission_scripts/pyscript_template.pbs",
    )
    parser.add_argument(
        "--min_pc", type=float, help="Min partical coding fraction to use", default=0.1
    )
    args = parser.parse_args()
    return args


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot(imx0 - imx1, imy0 - imy1)


def get_rate_res_fnames(direc="."):
    rate_fnames = [
        fname
        for fname in os.listdir(direc)
        if ("rates" in fname) and (fname[-4:] == ".csv")
    ]
    return rate_fnames


def get_res_fnames(direc="."):
    res_fnames = [
        fname
        for fname in os.listdir(direc)
        if (fname[-4:] == ".csv") and (fname[:4] == "res_")
    ]
    return res_fnames


def get_scan_res_fnames(direc="."):
    res_fnames = [
        fname
        for fname in os.listdir(direc)
        if (fname[-4:] == ".csv") and ("scan_res_" in fname)
    ]
    return res_fnames


def get_peak_res_fnames(direc="."):
    res_fnames = [
        fname
        for fname in os.listdir(direc)
        if (fname[-4:] == ".csv") and ("peak_scan_" in fname)
    ]
    return res_fnames


def get_merged_csv_df(csv_fnames):
    dfs = []
    for csv_fname in csv_fnames:
        try:
            dfs.append(pd.read_csv(csv_fname, dtype={"timeID": np.int64}))
        except Exception as E:
            logging.error(E)
            continue
    df = pd.concat(dfs)
    return df


def probm2perc(pmap):
    bl = pmap > 0
    p_map = np.copy(pmap)
    inds_sort = np.argsort(p_map)[::-1]
    perc_map = np.zeros_like(p_map)
    perc_map[inds_sort] = np.cumsum(p_map[inds_sort])  # *\
    perc_map[~bl] = 1.0
    return perc_map


def get_merged_csv_df_wpos(csv_fnames, attfile, perc_map=None, direc=None):
    dfs = []
    for csv_fname in csv_fnames:
        try:
            if direc is None:
                tab = pd.read_csv(csv_fname, dtype={"timeID": np.int64})
            else:
                tab = pd.read_csv(
                    os.path.join(direc, csv_fname), dtype={"timeID": np.int64}
                )
            if len(tab) > 0:
                # att_ind = np.argmin(np.abs(attfile['TIME'] - trigger_time))
                # att_quat = attfile['QPARAM'][att_ind]
                # ras = np.zeros(len(tab))
                # decs = np.zeros(len(tab))
                # for i in xrange(len(ras)):
                # #     print np.shape(res_tab['time'][i]), np.shape(attfile['TIME'])
                #     att_ind0 = np.argmin(np.abs(tab['time'][i] + tab['duration'][i]/2. - attfile['TIME']))
                #     att_quat0 = attfile['QPARAM'][att_ind0]
                #     ras[i], decs[i] = convert_imxy2radec(tab['imx'][i],\
                #                                          tab['imy'][i],\
                #                                         att_quat0)
                t0_ = np.nanmean(tab["time"] + tab["duration"] / 2.0)
                att_ind0 = np.argmin(np.abs(t0_ - attfile["TIME"]))
                att_quat0 = attfile["QPARAM"][att_ind0]
                ras, decs = convert_imxy2radec(tab["imx"], tab["imy"], att_quat0)
                tab["ra"] = ras
                tab["dec"] = decs
                if not perc_map is None:
                    Nside = hp.npix2nside(len(perc_map))
                    hp_inds = hp.ang2pix(Nside, ras, decs, lonlat=True, nest=True)
                    cred_lvl = perc_map[hp_inds]
                    tab["cls"] = cred_lvl
                dfs.append(tab)
        except Exception as E:
            logging.warning(E)
            continue
    df = pd.concat(dfs)
    return df


def mk_seed_tab4scans(
    res_tab, pc_fname, rate_seed_tab, TS_min=6.5, im_steps=20, pc_min=0.1
):
    PC = fits.open(pc_fname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key="T")

    pcbl = pc >= (pc_min * 0.99)
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    imxax = np.linspace(-2, 2, im_steps * 4 + 1)
    imyax = np.linspace(-1, 1, im_steps * 2 + 1)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]

    h = np.histogram2d(pc_imxs, pc_imys, bins=bins)[0]
    inds = np.where(h >= 10)
    squareIDs_all = np.ravel_multi_index(inds, h.shape)

    df_twinds = res_tab.groupby("timeID")
    seed_tabs = []
    for twind, dft in df_twinds:
        if np.max(dft["TS"]) >= TS_min:
            seed_dict = {}
            seed_dict["timeID"] = twind
            seed_dict["dur"] = dft["duration"].values[0]
            seed_dict["time"] = dft["time"].values[0]

            bl_rate_seed = rate_seed_tab["timeID"] == twind
            squareIDs_done = rate_seed_tab["squareID"][bl_rate_seed]
            squareIDs = squareIDs_all[~np.isin(squareIDs_all, squareIDs_done)]

            seed_dict["squareID"] = squareIDs
            seed_tabs.append(pd.DataFrame(seed_dict))

    seed_tab = pd.concat(seed_tabs)

    return seed_tab


def mk_seed_tab(rates_res, TS_min=3.75, im_steps=20):
    imxax = np.linspace(-2, 2, im_steps * 4 + 1)
    imyax = np.linspace(-1, 1, im_steps * 2 + 1)
    # imyg, imxg = np.meshgrid((imyax[1:]+imyax[:-1])/2., (imxax[1:]+imxax[:-1])/2.)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]

    df_twinds = rates_res.groupby("timeID")
    seed_tabs = []
    for twind, dft in df_twinds:
        maxTS = np.nanmax(dft["TS"])
        if maxTS >= TS_min:
            TS_min2_ = min(maxTS - 2.5, 0.8 * maxTS)
            TS_min2 = max(TS_min2_, np.nanmedian(dft["TS"]), TS_min - 0.5)
            seed_dict = {}
            seed_dict["timeID"] = twind
            seed_dict["dur"] = dft["dur"].values[0]
            seed_dict["time"] = dft["time"].values[0]

            pnts = np.vstack([dft["imx"], dft["imy"]]).T
            TSintp = interpolate.LinearNDInterpolator(pnts, dft["TS"])
            imxax = np.linspace(-1.8, 1.8, 8 * 36 + 1)
            imyax = np.linspace(-1.0, 1.0, 8 * 20 + 1)
            xgrid, ygrid = np.meshgrid(imxax, imyax)
            pnts = np.vstack([xgrid.ravel(), ygrid.ravel()]).T
            TSgrid = TSintp(pnts)
            bl = TSgrid >= (TS_min2)
            xs = xgrid.ravel()[bl]
            ys = ygrid.ravel()[bl]

            h = np.histogram2d(xs, ys, bins=bins)[0]
            inds = np.where(h > 0)

            squareIDs = np.ravel_multi_index(inds, h.shape)
            seed_dict["squareID"] = squareIDs
            seed_tabs.append(pd.DataFrame(seed_dict))

    seed_tab = pd.concat(seed_tabs)

    return seed_tab


# def mk_seed_tab(rates_res, TS_min=3.5, im_steps=20):
#
#     imxax = np.linspace(-2,2,im_steps*4+1)
#     imyax = np.linspace(-1,1,im_steps*2+1)
#     im_step = imxax[1] - imxax[0]
#     bins = [imxax, imyax]
#
#     df_twinds = rates_res.groupby('timeID')
#     seed_tabs = []
#     for twind, dft in df_twinds:
#         if np.max(dft['TS']) >= TS_min:
#             seed_dict = {}
#             seed_dict['timeID'] = twind
#             seed_dict['dur'] = dft['dur'].values[0]
#             seed_dict['time'] = dft['time'].values[0]
#
#             pnts = np.vstack([dft['imx'],dft['imy']]).T
#             TSintp = interpolate.LinearNDInterpolator(pnts, dft['TS'])
#             imxax = np.linspace(-1.5, 1.5, 8*30+1)
#             imyax = np.linspace(-.85, .85, 8*17+1)
#             xgrid, ygrid = np.meshgrid(imxax, imyax)
#             pnts = np.vstack([xgrid.ravel(),ygrid.ravel()]).T
#             TSgrid = TSintp(pnts)
#             bl = (TSgrid>=(TS_min-.1))
#             xs = xgrid.ravel()[bl]
#             ys = ygrid.ravel()[bl]
#
#             h = np.histogram2d(xs, ys, bins=bins)[0]
#             inds = np.where(h>0)
#             squareIDs = np.ravel_multi_index(inds, h.shape)
#             seed_dict['squareID'] = squareIDs
#             seed_tabs.append(pd.DataFrame(seed_dict))
#
#     seed_tab = pd.concat(seed_tabs)
#
#     return seed_tab


def mk_job_tab(seed_tab, Njobs, im_steps=20):
    imxax = np.linspace(-2, 2, im_steps * 4 + 1)
    imyax = np.linspace(-1, 1, im_steps * 2 + 1)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]
    squareIDs = np.unique(seed_tab["squareID"])
    shp = (len(imxax) - 1, len(imyax) - 1)
    data_dicts = []
    for i, squareID in enumerate(squareIDs):
        data_dict = {}
        data_dict["proc_group"] = i % Njobs
        indx, indy = np.unravel_index(squareID, shp)
        data_dict["imx0"] = bins[0][indx]
        data_dict["imx1"] = bins[0][indx + 1]
        data_dict["imy0"] = bins[1][indy]
        data_dict["imy1"] = bins[1][indy + 1]
        data_dict["squareID"] = squareID
        data_dicts.append(data_dict)

    job_tab = pd.DataFrame(data_dicts)

    return job_tab


def execute_ssh_cmd(client, cmd, server, retries=5):
    tries = 0
    while tries < retries:
        try:
            stdin, stdout, stderr = client.exec_command(cmd)
            logging.info("stdout: ")
            sto = stdout.read()
            logging.info(sto)
            return sto
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.error("Messed up with ")
            logging.error(cmd)
            client.close()
            client = get_ssh_client(server)
            tries += 1
            logging.debug("retry %d of %d" % (tries, retries))
    return


def get_ssh_client(server, retries=5):
    tries = 0
    try:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        return

    while tries < retries:
        try:
            client.connect(server)
            return client
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            tries += 1
    return


def sub_jobs(
    njobs,
    name,
    pyscript,
    pbs_fname,
    queue="open",
    workdir=None,
    qos=None,
    ssh=True,
    extra_args=None,
    ppn=1,
):
    hostname = socket.gethostname()

    if len(name) > 15:
        name = name[:15]

    if "aci.ics" in hostname and "amon" not in hostname:
        ssh = False

    if ssh:
        ssh_cmd = 'ssh aci-b.aci.ics.psu.edu "'
        server = "aci-b.aci.ics.psu.edu"
        # client = paramiko.SSHClient()
        # client.load_system_host_keys()
        # client.connect(server)
        client = get_ssh_client(server)
        # base_sub_cmd = 'qsub %s -A %s -N %s -v '\
        #             %(args.pbs_fname, args.queue, args.name)
        if qos is not None:
            base_sub_cmd = "qsub %s -A %s -N %s -l nodes=1:ppn=%d -l qos=%s -v " % (
                pbs_fname,
                queue,
                name,
                ppn,
                qos,
            )
        else:
            base_sub_cmd = "qsub %s -A %s -N %s -l nodes=1:ppn=%d -v " % (
                pbs_fname,
                queue,
                name,
                ppn,
            )

    else:
        if qos is not None:
            base_sub_cmd = "qsub %s -A %s -N %s -l nodes=1:ppn=%d -l qos=%s -v " % (
                pbs_fname,
                queue,
                name,
                ppn,
                qos,
            )
        else:
            base_sub_cmd = "qsub %s -A %s -N %s -l nodes=1:ppn=%d -v " % (
                pbs_fname,
                queue,
                name,
                ppn,
            )

    if workdir is None:
        workdir = os.getcwd()
    if extra_args is None:
        extra_args = ""

    cmd = ""
    jobids = []

    for i in range(njobs):
        # cmd_ = 'jobid=%d,workdir=%s,njobs=%d,pyscript=%s' %(i,workdir,njobs,pyscript)
        cmd_ = 'jobid=%d,workdir=%s,njobs=%d,pyscript=%s,extra_args="%s"' % (
            i,
            workdir,
            njobs,
            pyscript,
            extra_args,
        )
        if ssh:
            cmd += base_sub_cmd + cmd_
            if i < (njobs - 1):
                cmd += " | "
            # cmd = base_sub_cmd + cmd_
            # jbid = execute_ssh_cmd(client, cmd, server)
            # jobids.append(jbid)
            # try:
            #     stdin, stdout, stderr = client.exec_command(cmd)
            #     logging.info("stdout: ")
            #     sto = stdout.read()
            #     logging.info(sto)
            #     jobids.append(sto)
            # except Exception as E:
            #     logging.error(E)
            #     logging.error(traceback.format_exc())
            #     logging.error("Messed up with ")
            #     logging.error(cmd)
        else:
            cmd = base_sub_cmd + cmd_
            logging.info("Trying to submit: ")
            logging.info(cmd)

            try:
                os.system(cmd)
                # subprocess.check_call(cmd, shell=True)
            except Exception as E:
                logging.error(E)
                logging.error("Messed up with ")
                logging.error(cmd)

            time.sleep(0.1)
    if ssh:
        # ssh_cmd = 'ssh aci-b.aci.ics.psu.edu "'
        # cmd = ssh_cmd + cmd + '"'
        logging.info("Full cmd to run:")
        logging.info(cmd)
        try:
            jobids = execute_ssh_cmd(client, cmd, server)
            # os.system(cmd)
            # subprocess.check_call(cmd, shell=True)
            # cmd_list = ['ssh', 'aci-b.aci.ics.psu.edu', '"'+cmd+'"']
            # cmd_list = shlex.split(cmd)
            # logging.info(cmd_list)
            # subprocess.check_call(cmd_list)
        except Exception as E:
            logging.error(E)
            logging.error("Messed up with ")
            logging.error(cmd)
    if ssh:
        client.close()
    return jobids


def find_peaks2scan(
    res_df, max_dv=10.0, min_sep=8e-3, max_Npeaks=48, min_Npeaks=2, minTS=6.0
):
    tgrps = res_df.groupby("timeID")

    peak_dfs = []

    for timeID, df_ in tgrps:
        if np.nanmax(df_["TS"]) < minTS:
            continue

        df = df_.sort_values("sig_nllh")
        vals = df["sig_nllh"]
        #         ind_sort = np.argsort(vals)
        min_val = np.nanmin(df["sig_nllh"])

        peak_dict = {
            "timeID": int(timeID),
            "time": np.nanmean(df["time"]),
            "duration": np.nanmean(df["duration"]),
        }

        imxs_ = np.empty(0)
        imys_ = np.empty_like(imxs_)
        As_ = np.empty_like(imxs_)
        Gs_ = np.empty_like(imxs_)

        for row_ind, row in df.iterrows():
            if row["sig_nllh"] > (min_val + max_dv) and len(imxs_) >= min_Npeaks:
                break
            if len(imxs_) >= max_Npeaks:
                break

            if len(imxs_) > 0:
                imdist = np.min(im_dist(row["imx"], row["imy"], imxs_, imys_))
                if imdist <= min_sep:
                    continue

            imxs_ = np.append(imxs_, [row["imx"]])
            imys_ = np.append(imys_, [row["imy"]])
            As_ = np.append(As_, [row["A"]])
            Gs_ = np.append(Gs_, [row["ind"]])

        peak_dict["imx"] = imxs_
        peak_dict["imy"] = imys_
        peak_dict["Signal_A"] = As_
        peak_dict["Signal_gamma"] = Gs_
        peak_dfs.append(pd.DataFrame(peak_dict))

    peaks_df = pd.concat(peak_dfs, ignore_index=True)

    return peaks_df


def main(args):
    fname = "manager"

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    f = open(fname + ".pid", "w")
    f.write(str(os.getpid()))
    f.close()

    logging.info("Wrote pid: %d" % (os.getpid()))

    to = ["delauj2@gmail.com", "aaron.tohu@gmail.com"]
    subject = "BATML " + args.GWname
    body = "Got data and starting analysis"
    try:
        send_email(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    t_0 = time.time()

    has_sky_map = False
    try:
        sky_map_fnames = [
            fname
            for fname in os.listdir(".")
            if "cWB.fits.gz" in fname or "bayestar" in fname or "skymap" in fname
        ]
        sky_map = hp.read_map(sky_map_fnames[0], field=(0,), nest=True)
        logging.info("Opened sky map")
        perc_map = probm2perc(sky_map)
        logging.info("Made perc map")
        has_sky_map = True
    except Exception as E:
        logging.warning("problem reading skymap")
        logging.error(E)
        logging.error(traceback.format_exc())

    try:
        logging.info("Connecting to DB")
        if args.dbfname is None:
            db_fname = guess_dbfname()
            if isinstance(db_fname, list):
                db_fname = db_fname[0]
        else:
            db_fname = args.dbfname
        conn = get_conn(db_fname)
        info_tab = get_info_tab(conn)
        logging.info("Got info table")
        trigtime = info_tab["trigtimeMET"][0]
        files_tab = get_files_tab(conn)
        logging.info("Got files table")
        attfname = files_tab["attfname"][0]
        evfname = files_tab["evfname"][0]

    except Exception as E:
        logging.warning("problem getting files tab from DB")
        logging.error(E)
        logging.error(traceback.format_exc())
        attfname = "attitude.fits"
        evfname = "filter_evdata.fits"

    try:
        attfile = Table.read(attfname)
        logging.info("Opened att file")
    except Exception as E:
        logging.warning("Trouble openning attitude file")
        logging.error(E)
        logging.error(traceback.format_exc())

    try:
        GTI_pnt = Table.read(evfname, hdu="GTI_POINTING")
        logging.info("Opened GTI_pnt")
        logging.info(GTI_pnt)
        tot_exp = 0.0
        for row in GTI_pnt:
            tot_exp += row["STOP"] - row["START"]
        logging.info("Total Exposure Time is %.3f seconds" % (tot_exp))
        if tot_exp < 1.0:
            logging.info("Total Pointing time is <1s")
            logging.info("Exiting now")
            body = (
                "Total Pointing time is < 1s, only %.3f seconds. Exiting analysis."
                % (tot_exp)
            )
            try:
                send_email(subject, body, to)
            except Exception as E:
                logging.error(E)
                logging.error("Trouble sending email")

            return

    except Exception as E:
        logging.warning("Trouble openning GTI file")
        logging.error(E)
        logging.error(traceback.format_exc())

    try:
        good_pix = np.load(args.pix_fname)
        Ngood_pix = len(good_pix)
        if Ngood_pix < 1:
            # stop here
            logging.info("Completely out of FoV")
            logging.info("Exiting now")
            body = "Completely out of FoV. Exiting analysis."
            try:
                send_email(subject, body, to)
            except Exception as E:
                logging.error(E)
                logging.error("Trouble sending email")

            return
        Nratejobs = 4
        if Ngood_pix > 5e4:
            Nratejobs = 16
        if Ngood_pix > 1e5:
            Nratejobs = 32
        if Ngood_pix > 2.5e5:
            Nratejobs = 48
        if Ngood_pix > 5e5:
            Nratejobs = 64
    except Exception as E:
        logging.warn("Trouble reading good pix file")
        Nratejobs = 64
        if args.archive:
            Nratejobs = 108

    if args.do_bkg:
        logging.info("Submitting bkg estimation job now")
        # try:
        if args.archive:
            sub_jobs(
                1,
                "BKG_" + args.GWname,
                args.BKGpyscript,
                args.pbs_fname,
                queue=args.queue,
                ppn=4,
                extra_args="--archive",
                qos=None,
            )
        else:
            sub_jobs(
                1,
                "BKG_" + args.GWname,
                args.BKGpyscript,
                args.pbs_fname,
                queue="open",  # args.queue,\
                ppn=4,
                qos=None,
            )
        logging.info("Job submitted")
        # except Exception as E:
        #     logging.warn(E)
        #     logging.warn("Might have been a problem submitting")

    #  Wait for bkg job to finish before submitting rates jobs
    dt = 0.0
    t_0 = time.time()
    bkg_fname = "bkg_estimation.csv"
    while dt < 16 * 3600.0:
        if os.path.exists(bkg_fname):
            break
        else:
            time.sleep(10.0)
            dt = time.time() - t_0

    if not os.path.exists(bkg_fname):
        logging.info("Didn't do BKG for some reason")
        logging.info("Exiting now")
        body = "Didn't do BKG for some reason. Exiting analysis."
        try:
            send_email(subject, body, to)
        except Exception as E:
            logging.error(E)
            logging.error("Trouble sending email")
        return

    extra_args = "--min_pc %.4f" % (args.min_pc)

    if args.do_rates:
        logging.info("Submitting %d rates jobs now" % (Nratejobs))
        # try:
        sub_jobs(
            Nratejobs,
            "RATES_" + args.GWname,
            args.RATEpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
            extra_args=extra_args,
        )
        logging.info("Jobs submitted")
        # except Exception as E:
        #     logging.warn(E)
        #     logging.warn("Might have been a problem submitting")

    dt = 0.0
    t_0 = time.time()

    while dt < 3600.0 * 36.0:
        rate_res_fnames = get_rate_res_fnames()
        logging.info("%d of %d rate jobs done" % (len(rate_res_fnames), Nratejobs))

        if args.skip_waiting:
            rate_res = get_merged_csv_df(rate_res_fnames)
            try:
                rate_res["dt"] = rate_res["time"] - trigtime
            except Exception:
                pass
            break

        elif len(rate_res_fnames) < Nratejobs:
            time.sleep(30.0)
            dt = time.time() - t_0

        else:
            rate_res = get_merged_csv_df(rate_res_fnames)
            try:
                rate_res["dt"] = rate_res["time"] - trigtime
            except Exception:
                pass
            break

    try:
        body = "Done with rates analysis\n"
        body += "Max TS is %.3f" % (np.max(rate_res["TS"]))
        logging.info(body)
        send_email(subject, body, to)
        rate_res_tab_top = rate_res.sort_values("TS").tail(16)
        body = rate_res_tab_top.to_html()
        logging.info(body)
        # send_email(subject, body, to)
        send_email_wHTML(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    if args.archive:
        seed_tab = mk_seed_tab(rate_res, TS_min=4.15)
    else:
        seed_tab = mk_seed_tab(rate_res)

    seed_tab.to_csv("rate_seeds.csv", index=False)

    Nsquares = len(np.unique(seed_tab["squareID"]))
    Nseeds = len(seed_tab)

    if Nseeds < 1:
        body = "No seeds. Exiting analysis."
        try:
            send_email(subject, body, to)
        except Exception as E:
            logging.error(E)
            logging.error("Trouble sending email")
        return

    if Nsquares > 512:
        Njobs = 96
    elif Nsquares > 128:
        Njobs = 64
    else:
        Njobs = Nsquares / 2
    if (1.0 * Nseeds) / Nsquares >= 8:
        Njobs += Njobs / 2
    if (1.0 * Nseeds) / Nsquares >= 16:
        Njobs += Njobs / 4
    if (1.0 * Nseeds) / Nsquares >= 32:
        Njobs += Njobs / 4

    Njobs = max(Njobs, 1)

    job_tab = mk_job_tab(seed_tab, Njobs)

    job_tab.to_csv("job_table.csv", index=False)

    # Now need to launch those jobs

    # then start the next while loop, curogating the results
    # get the best TSs or TSs > 6 and do whatever with them

    # maybe mk imgs and do batcelldetect on the good time windows
    # or if there's a decent TS from the full analysis

    # also maybe add some emails in here for progress and info and errors
    if args.do_llh:
        logging.info("Submitting %d Jobs now" % (Njobs))
        sub_jobs(
            Njobs,
            "LLH_" + args.GWname,
            args.LLHpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
            extra_args=extra_args,
        )
        logging.info("Jobs submitted, now going to monitor progress")

    t_0 = time.time()
    dt = 0.0
    Ndone = 0

    while dt < 3600.0 * 40.0:
        res_fnames = get_res_fnames()

        if args.skip_waiting:
            try:
                if has_sky_map:
                    res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
                else:
                    res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
                logging.info("Got merged results with RA Decs")
            except Exception as E:
                logging.error(E)
                res_tab = get_merged_csv_df(res_fnames)
                logging.info("Got merged results without RA Decs")
            logging.info("Max TS: %.3f" % (np.max(res_tab["TS"])))
            break

        if len(res_fnames) == Ndone:
            time.sleep(30.0)
            dt = time.time() - t_0

        else:
            Ndone = len(res_fnames)
            logging.info("%d of %d squares scanned" % (Ndone, Nsquares))

            if Ndone < Nsquares:
                res_tab = get_merged_csv_df(res_fnames)

                time.sleep(30.0)
                dt = time.time() - t_0

            else:
                logging.info("Got all of the results now")
                try:
                    if has_sky_map:
                        res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
                    else:
                        res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
                    logging.info("Got merged results with RA Decs")
                except Exception as E:
                    logging.error(E)
                    res_tab = get_merged_csv_df(res_fnames)
                    logging.info("Got merged results without RA Decs")
                logging.info("Max TS: %.3f" % (np.max(res_tab["TS"])))
                break

    try:
        res_tab["dt"] = res_tab["time"] - trigtime
    except Exception:
        pass

    # logging.info("Saving full result table to: ")
    # save_fname = 'full_res_tab.csv'
    # logging.info(save_fname)
    # res_tab.to_csv(save_fname)

    try:
        # body = "Done with LLH analysis\n"
        # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
        res_tab_top = res_tab.sort_values("TS").tail(16)
        body = "LLH analysis results\n"
        body += res_tab_top.to_html()
        logging.info(body)
        # send_email(subject, body, to)
        send_email_wHTML(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    # Now need to find anything interesting and investigate it further
    # probably find each time bin with a TS>6 and scan around each
    # blip with a nllh that's within 5-10 or so

    # Should also probably do submit jobs for a full FoV scan
    # if a TS is found above something border line alert, like
    # TS ~>7-8

    if np.nanmax(res_tab["TS"]) < args.TSscan:
        return

    scan_seed_tab = mk_seed_tab4scans(
        res_tab, args.pcfname, seed_tab, TS_min=args.TSscan, pc_min=args.min_pc
    )

    Nscan_seeds = len(scan_seed_tab)
    logging.info("%d scan seeds" % (Nscan_seeds))
    Nscan_squares = len(np.unique(scan_seed_tab["squareID"]))

    Njobs = 64
    if Nscan_squares < 64:
        Njobs = Nscan_squares / 2
    if Nscan_seeds > 1e3:
        Njobs = 72
    if Nscan_seeds > 2.5e3:
        Njobs = 96
    if Nscan_seeds > 5e3:
        Njobs = 128
    if Nscan_seeds > 1e4:
        Njobs = 160

    scan_job_tab = mk_job_tab(scan_seed_tab, Njobs)

    scan_seed_tab.to_csv("scan_seeds.csv", index=False)
    scan_job_tab.to_csv("scan_job_table.csv", index=False)

    if args.do_scan:
        logging.info("Submitting %d Scan Jobs now" % (Njobs))
        sub_jobs(
            Njobs,
            "SCAN_" + args.GWname,
            args.SCANpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
            extra_args=extra_args,
        )
        logging.info("Jobs submitted, now going to monitor progress")

    t_0 = time.time()
    dt = 0.0
    Ndone = 0

    while dt < 3600.0 * 40.0:
        res_fnames = get_scan_res_fnames()

        if args.skip_waiting:
            if len(res_fnames) < 1:
                scan_res_tab = pd.DataFrame()
                break
            try:
                if has_sky_map:
                    scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
                else:
                    scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
                logging.info("Got merged scan results with RA Decs")
            except Exception as E:
                logging.error(E)
                scan_res_tab = get_merged_csv_df(res_fnames)
                logging.info("Got merged scan results without RA Decs")
            logging.info("Max TS: %.3f" % (np.max(scan_res_tab["TS"])))
            break

        if len(res_fnames) == Ndone:
            time.sleep(30.0)
            dt = time.time() - t_0

        else:
            Ndone = len(res_fnames)
            logging.info("%d of %d squares scanned" % (Ndone, Nscan_squares))

            if Ndone < Nscan_squares:
                scan_res_tab = get_merged_csv_df(res_fnames)

                time.sleep(30.0)
                dt = time.time() - t_0

            else:
                logging.info("Got all of the scan results now")
                try:
                    if has_sky_map:
                        scan_res_tab = get_merged_csv_df_wpos(
                            res_fnames, attfile, perc_map
                        )
                    else:
                        scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
                    logging.info("Got merged scan results with RA Decs")
                except Exception as E:
                    logging.error(E)
                    scan_res_tab = get_merged_csv_df(res_fnames)
                    logging.info("Got merged scan results without RA Decs")
                logging.info("Max TS: %.3f" % (np.max(scan_res_tab["TS"])))
                break

    try:
        scan_res_tab["dt"] = scan_res_tab["time"] - trigtime
    except Exception:
        pass

    # logging.info("Saving full result table to: ")
    # save_fname = 'full_scanRes_tab.csv'
    # logging.info(save_fname)
    # scan_res_tab.to_csv(save_fname)

    full_res_tab = pd.concat([res_tab, scan_res_tab], ignore_index=True)

    try:
        # body = "Done with LLH analysis\n"
        # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
        body = "All LLH analysis results\n"
        full_res_tab_top = full_res_tab.sort_values("TS").tail(16)
        body += full_res_tab_top.to_html()
        logging.info(body)
        # send_email(subject, body, to)
        send_email_wHTML(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    if has_sky_map:
        if np.all(full_res_tab_top["cls"] > 0.995):
            logging.info("None of the top 16 TSs are in the 0.995 credible region.")

    # Now need to put in the part where I find good candidates
    # then do the integrated LLH

    logging.info("Making Peaks Table now")
    peaks_tab = find_peaks2scan(full_res_tab, minTS=args.TSscan)

    Npeaks = len(peaks_tab)
    logging.info("Found %d Peaks to scan" % (Npeaks))
    Njobs = 96

    if Npeaks < Njobs:
        Njobs = Npeaks
        peaks_tab["jobID"] = np.arange(Njobs, dtype=np.int64)
    else:
        jobids = np.array([i % Njobs for i in range(Npeaks)])
        peaks_tab["jobID"] = jobids

    peaks_fname = "peaks.csv"
    peaks_tab.to_csv(peaks_fname)

    logging.info("Submitting %d Jobs now" % (Njobs))
    sub_jobs(
        Njobs,
        "Peak_" + args.GWname,
        args.PEAKpyscript,
        args.pbs_fname,
        queue=args.queue,
        qos=args.qos,
    )
    logging.info("Jobs submitted, now going to monitor progress")

    t_0 = time.time()
    dt = 0.0
    Ndone = 0

    while dt < 3600.0 * 20.0:
        res_fnames = get_peak_res_fnames()

        if len(res_fnames) == Ndone:
            time.sleep(30.0)
            dt = time.time() - t_0

        else:
            Ndone = len(res_fnames)
            logging.info("%d of %d peaks scanned" % (Ndone, Npeaks))

            if Ndone < Npeaks:
                peak_res_tab = get_merged_csv_df(res_fnames)

                time.sleep(30.0)
                dt = time.time() - t_0

            else:
                logging.info("Got all of the peak results now")
                try:
                    if has_sky_map:
                        peak_res_tab = get_merged_csv_df_wpos(
                            res_fnames, attfile, perc_map
                        )
                    else:
                        peak_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
                    logging.info("Got merged peak results with RA Decs")
                except Exception as E:
                    logging.error(E)
                    peak_res_tab = get_merged_csv_df(res_fnames)
                    logging.info("Got merged peak results without RA Decs")
                logging.info("Max TS: %.3f" % (np.max(peak_res_tab["TS"])))
                break

    try:
        peak_res_tab["dt"] = peak_res_tab["time"] - trigtime
    except Exception:
        pass

    idx = peak_res_tab.groupby(["timeID"])["TS"].transform(max) == peak_res_tab["TS"]
    peak_res_max_tab = peak_res_tab[idx]

    maxTS = np.max(peak_res_max_tab["TS"])
    for timeID, df in peak_res_tab.groupby("timeID"):
        if has_sky_map:
            if (np.max(df["TS"]) > 7.0) and (
                df["cls"].iloc[np.argmax(df["TS"])] < 0.995
            ):
                try:
                    subject2 = subject + " Possible Signal"
                    peak_res_tab_top = df.sort_values("TS").tail(16)
                    body = peak_res_tab_top.to_html()
                    send_email_wHTML(subject2, body, to)
                except Exception as E:
                    logging.error(E)
                    logging.error("Trouble sending email")
        else:
            if np.max(df["TS"]) > 7.0:
                try:
                    subject2 = subject + " Possible Signal"
                    peak_res_tab_top = df.sort_values("TS").tail(16)
                    body = peak_res_tab_top.to_html()
                    send_email_wHTML(subject2, body, to)
                except Exception as E:
                    logging.error(E)
                    logging.error("Trouble sending email")

    try:
        # body = "Done with LLH analysis\n"
        # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
        peak_res_tab_top = peak_res_tab.sort_values("TS").tail(16)
        body = peak_res_tab_top.to_html()
        logging.info(body)
        # send_email(subject, body, to)
        send_email_wHTML(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    if has_sky_map:
        if np.all(peak_res_tab_top["cls"] > 0.995):
            logging.info("None of the top 16 TSs are in the 0.995 credible region.")


if __name__ == "__main__":
    args = cli()

    main(args)
