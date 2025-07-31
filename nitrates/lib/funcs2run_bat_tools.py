import os
import subprocess
import sys

# sys.path.append('/storage/work/jjd330/local/bat_data/BatML/HeasoftTools')
from ..HeasoftTools.gen_tools import run_ftool
from ..HeasoftTools.bat_tool_funcs import (
    ev2dpi,
    mk_pc_img,
    mk_sky_img,
    run_batcelldetect,
)
import time
import numpy as np
import multiprocessing as mp
import pandas as pd
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from ..lib.sqlite_funcs import get_conn, write_cats2db, write_sigimg_line
import logging, traceback

from ..config import dir as direc

sys.path.append(os.path.join(direc, "HeasoftTools"))


def do_bkg(bkg_tstart, bkg_tstop, ev_fname, dmask, savedir, e0=14.0, e1=194.9):
    # bkg_tstart = args.bkgt0
    # bkg_tstop = args.bkgt0 + args.bkgdt

    dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (e0, e1, bkg_tstart, bkg_tstop)
    dpi_bkg_fname = os.path.join(savedir, dpif)

    ev2dpi(ev_fname, dpi_bkg_fname, bkg_tstart, bkg_tstop, e0, e1, dmask)

    return dpi_bkg_fname


def do_pc(dmask, att_fname, work_dir, ovrsmp=4, detapp=False):
    pc_fname = os.path.join(work_dir, "pc_%d.img" % (ovrsmp))
    if not os.path.exists(pc_fname):
        mk_pc_img(dmask, pc_fname, dmask, att_fname, ovrsmp=ovrsmp, detapp=detapp)
    return pc_fname


def mk_sig_imgs_pix(
    tstarts,
    dts,
    evf,
    dpi_bkg_fname,
    pc_fname,
    attfile,
    dmask,
    savedir,
    work_dir,
    trig_time,
    conn,
    db_fname,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    snr_cuts=None,
):
    arg_dict_keys = [
        "tstart",
        "dt",
        "ev_fname",
        "bkg_dpi",
        "pc_fname",
        "att_fname",
        "dmask",
        "savedir",
        "pc_fname",
        "e0",
        "e1",
        "oversamp",
    ]
    args_dict_list = []

    for i in range(len(tstarts)):
        arg_dict = {
            "tstart": tstarts[i],
            "dt": dts[i],
            "ev_fname": evf,
            "bkg_dpi": dpi_bkg_fname,
            "att_fname": attfile,
            "pc_fname": pc_fname,
            "dmask": dmask,
            "savedir": savedir,
            "e0": e0,
            "e1": e1,
            "oversamp": oversamp,
        }
        args_dict_list.append(arg_dict)

    t0 = time.time()

    logging.info("%d images to make" % (len(args_dict_list)))

    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key="T")
    pc = PC.data
    pc_bl = pc >= 0.1
    dtp = [("snr", np.float64), ("imx", np.float64), ("imy", np.float64)]

    Nimgs = len(args_dict_list)
    for i in range(Nimgs):
        img_fname, sig_img_fname = mk_sky_sig_img4mp(args_dict_list[i])
        logging.debug("Made img %s" % (sig_img_fname))
        sig_img = fits.open(sig_img_fname)[0].data
        logging.debug("Opened fits file")
        if snr_cuts is None:
            snr_cut = 3.0
        else:
            snr_cut = snr_cuts[i]
        snr_bl = sig_img >= snr_cut
        img_bl = pc_bl & snr_bl
        sig_arr = np.empty(np.sum(img_bl), dtype=dtp)
        sig_arr["snr"] = sig_img[img_bl]
        logging.info("%d pix pass cut" % (np.sum(img_bl)))
        inds = np.where(img_bl)
        imxys = w_t.all_pix2world(inds[1], inds[0], 0)
        sig_arr["imx"] = imxys[0]
        sig_arr["imy"] = imxys[1]
        sig_pix_fname = os.path.join(work_dir, os.path.basename(sig_img_fname)[:-4])
        np.save(sig_pix_fname, sig_arr)
        logging.info("saved to %s" % (sig_pix_fname + ".npy"))
        try:
            write_sigimg_line(
                conn,
                args_dict_list[i]["tstart"],
                args_dict_list[i]["dt"],
                trig_time,
                sig_pix_fname + ".npy",
                np.sum(img_bl),
            )
            logging.info("written to DB")
        except:
            conn.close()
            conn = get_conn(db_fname)
            try:
                write_sigimg_line(
                    conn,
                    args_dict_list[i]["tstart"],
                    args_dict_list[i]["dt"],
                    trig_time,
                    sig_pix_fname + ".npy",
                    np.sum(img_bl),
                )
                logging.info("written to DB")
            except Exception as E:
                logging.error(str(E))
                logging.warn("Failed to write to DB")

        try:
            os.remove(sig_img_fname)
            os.remove(img_fname)
            logging.info("Deleted img files")
        except:
            logging.info("Failed to delete a file")
            pass

    logging.info("Done with all images")
    logging.info(
        "Took %.2f seconds, %.2f minutes"
        % (time.time() - t0, (time.time() - t0) / 60.0)
    )

    return


def mk_sig_imgs_mp(
    nproc,
    tstarts,
    dts,
    evf,
    dpi_bkg_fname,
    pc_fname,
    attfile,
    dmask,
    savedir,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    detapp=False,
    rebal=True,
):
    arg_dict_keys = [
        "tstart",
        "dt",
        "ev_fname",
        "bkg_dpi",
        "pc_fname",
        "att_fname",
        "dmask",
        "savedir",
        "pc_fname",
        "e0",
        "e1",
        "oversamp",
    ]
    args_dict_list = []

    for i in range(len(tstarts)):
        arg_dict = {
            "tstart": tstarts[i],
            "dt": dts[i],
            "ev_fname": evf,
            "bkg_dpi": dpi_bkg_fname,
            "att_fname": attfile,
            "pc_fname": pc_fname,
            "dmask": dmask,
            "savedir": savedir,
            "e0": e0,
            "e1": e1,
            "oversamp": oversamp,
            "detapp": detapp,
            "rebal": rebal,
        }
        args_dict_list.append(arg_dict)

    t0 = time.time()

    logging.info("%d images to make" % (len(args_dict_list)))

    if nproc > 1:
        p = mp.Pool(nproc)
        logging.info("Starting %d procs" % (nproc))
        sig_img_fnames = p.map(mk_sky_sig_img4mp, args_dict_list)
        p.close()
        p.join()
    else:
        sig_img_fnames = list(map(mk_sky_sig_img4mp, args_dict_list))

    logging.info("Done with all images")
    logging.info(
        "Took %.2f seconds, %.2f minutes"
        % (time.time() - t0, (time.time() - t0) / 60.0)
    )

    return


def mk_sky_sig_img4mp(arg_dict):
    img_fname, sig_img_fname = mk_sky_sig_img(
        arg_dict["tstart"],
        arg_dict["dt"],
        arg_dict["ev_fname"],
        arg_dict["bkg_dpi"],
        arg_dict["att_fname"],
        arg_dict["dmask"],
        arg_dict["savedir"],
        e0=arg_dict["e0"],
        e1=arg_dict["e1"],
        oversamp=arg_dict["oversamp"],
        detapp=arg_dict["detapp"],
        rebal=arg_dict["rebal"],
    )

    return img_fname, sig_img_fname


def get_sig_pix_mp(
    nproc,
    tstarts,
    dts,
    evf,
    dpi_bkg_fname,
    pc_fname,
    attfile,
    dmask,
    savedir,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    db_fname=None,
    timeIDs=None,
    RateTSs=None,
):
    arg_dict_keys = [
        "tstart",
        "dt",
        "ev_fname",
        "bkg_dpi",
        "pc_fname",
        "att_fname",
        "dmask",
        "savedir",
        "pc_fname",
        "e0",
        "e1",
        "oversamp",
        "db_fname",
    ]
    args_dict_list = []

    exp_bins = [0.2, 0.3, 0.6, 2]
    TS_cuts = [2.25, 2.0, 1.8, 1.7, 1.65]
    exp_bins = np.digitize(dts, bins=exp_bins)

    for i in range(len(tstarts)):
        if RateTSs is not None:
            if RateTSs[i] < TS_cuts[exp_bins[i]]:
                continue

        arg_dict = {
            "tstart": tstarts[i],
            "dt": dts[i],
            "ev_fname": evf,
            "bkg_dpi": dpi_bkg_fname,
            "att_fname": attfile,
            "pc_fname": pc_fname,
            "dmask": dmask,
            "savedir": savedir,
            "e0": e0,
            "e1": e1,
            "oversamp": oversamp,
            "db_fname": db_fname,
        }
        if timeIDs is not None:
            arg_dict["timeID"] = timeIDs[i]
            arg_dict["RateTS"] = RateTSs[i]
        args_dict_list.append(arg_dict)

    t0 = time.time()

    p = mp.Pool(nproc)

    logging.info("%d images to make" % (len(args_dict_list)))
    logging.info("Starting %d procs" % (nproc))

    sig_img_fnames = p.map(get_sig_pix, args_dict_list)

    p.close()
    p.join()

    logging.info("Done with all images")
    logging.info(
        "Took %.2f seconds, %.2f minutes"
        % (time.time() - t0, (time.time() - t0) / 60.0)
    )


def get_sig_pix(arg_dict):
    img_fname, sig_img_fname = mk_sky_sig_img(
        arg_dict["tstart"],
        arg_dict["dt"],
        arg_dict["ev_fname"],
        arg_dict["bkg_dpi"],
        arg_dict["att_fname"],
        arg_dict["dmask"],
        arg_dict["savedir"],
        e0=arg_dict["e0"],
        e1=arg_dict["e1"],
        oversamp=arg_dict["oversamp"],
    )

    PC = fits.open(arg_dict["pc_fname"])[0]
    sig_img = fits.open(sig_img_fname)[0]
    w_t = WCS(sig_img.header, key="T")

    exp_bins = [0.2, 0.3, 0.6, 2]
    TSaims = [5.25, 5.0, 4.8, 4.6, 4.5]
    exp_bin = np.digitize(arg_dict["dt"], bins=exp_bins)

    snr_cut = max(2.0 * (TSaims[exp_bin] - arg_dict["RateTS"]), 2.0)

    bl = PC.data >= 0.1
    bl_snr = (sig_img.data > snr_cut) & bl
    if np.sum(bl_snr) < 1:
        return
    bl_snr_inds = np.where(bl_snr)
    imxys = w_t.all_pix2world(bl_snr_inds[1], bl_snr_inds[0], 0)
    SNRs = sig_img.data[bl_snr]

    bins = [np.linspace(-2, 2, 10 * 4 + 1), np.linspace(-1, 1, 10 * 2 + 1)]
    imx_inds = np.digitize(imxys[0], bins=bins[0]) - 1
    imy_inds = np.digitize(imxys[1], bins=bins[1]) - 1

    job_inds = np.arange(19)
    job_ids = -1 * np.ones(len(imx_inds), dtype=np.int64)

    imx_bins0 = [
        -1.2,
        -1.2,
        -0.8,
        -0.8,
        -0.4,
        -0.4,
        0.0,
        0.0,
        0.4,
        0.4,
        0.8,
        0.8,
        -1.5,
        0.0,
        -1.5,
        0.0,
        -2.0,
        1.2,
        -2.0,
    ]
    imx_bins1 = [
        -0.8,
        -0.8,
        -0.4,
        -0.4,
        0.0,
        0.0,
        0.4,
        0.4,
        0.8,
        0.8,
        1.2,
        1.2,
        0.0,
        1.5,
        0.0,
        1.5,
        2.0,
        2.0,
        -1.2,
    ]
    imy_bins0 = [
        -0.3,
        0.2,
        -0.3,
        0.2,
        -0.3,
        0.2,
        -0.3,
        0.2,
        -0.3,
        0.2,
        -0.3,
        0.2,
        -0.5,
        -0.5,
        -1.0,
        -1.0,
        0.7,
        -0.3,
        -0.3,
    ]
    imy_bins1 = [
        0.2,
        0.7,
        0.2,
        0.7,
        0.2,
        0.7,
        0.2,
        0.7,
        0.2,
        0.7,
        0.2,
        0.7,
        -0.3,
        -0.3,
        -0.5,
        -0.5,
        1,
        0.7,
        0.7,
    ]
    for i in job_inds:
        bl_bin = (
            (imxys[0] >= imx_bins0[i])
            & (imxys[0] < imx_bins1[i])
            & (imxys[1] >= imy_bins0[i])
            & (imxys[1] < imy_bins1[i])
        )
        job_ids[bl_bin] = i

    if arg_dict["db_fname"] is not None:
        df_dict = {}
        df_dict["timeID"] = arg_dict["timeID"]
        df_dict["imx_ind"] = imx_inds
        df_dict["imy_ind"] = imy_inds
        df_dict["imx"] = imxys[0]
        df_dict["imy"] = imxys[1]
        df_dict["snr"] = SNRs
        df_dict["proc_group"] = job_ids
        df_dict["done"] = 0

        df = pd.DataFrame(df_dict)

        conn = get_conn(arg_dict["db_fname"])

        df.to_sql("ImageSigs", conn, if_exists="append", index=False)

    return sig_img_fname


def mk_sky_sig_img(
    tstart,
    dt,
    evf,
    dpi_bkg_fname,
    attfile,
    dmask,
    savedir,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    detapp=False,
    rebal=True,
):
    tstop = tstart + dt

    dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (e0, e1, tstart, tstop)
    dpi_fname = os.path.join(savedir, dpif)
    if not os.path.exists(dpi_fname):
        ev2dpi(evf, dpi_fname, tstart, tstop, e0, e1, dmask)

    img_fname = os.path.join(
        savedir, "sky_%.1f_%.1f_%.3f_%.3f_os%d_.img" % (e0, e1, tstart, tstop, oversamp)
    )
    sig_img_fname = os.path.join(
        savedir, "sig_%.1f_%.1f_%.3f_%.3f_os%d_.img" % (e0, e1, tstart, tstop, oversamp)
    )

    mk_sky_img(
        dpi_fname,
        img_fname,
        dmask,
        attfile,
        bkg_file=dpi_bkg_fname,
        ovrsmp=oversamp,
        sig_map=sig_img_fname,
        detapp=detapp,
        rebal=rebal,
    )

    return img_fname, sig_img_fname


def do_bkg_ebins(args, ebins):
    bkg_tstart = args.bkgt0
    bkg_tstop = args.bkgt0 + args.bkgdt

    dpif = "dpi_bkg_ebins_%.3f_%.3f_.dpi" % (bkg_tstart, bkg_tstop)
    dpi_bkg_fname = os.path.join(args.savedir, args.obsid, dpif)

    ev2dpi_ebins(args.evf, dpi_bkg_fname, bkg_tstart, bkg_tstop, ebins, args.dmask)

    return dpi_bkg_fname


def std_grb(
    tstart,
    dt,
    evf,
    dpi_bkg_fnames,
    attfile,
    dmask,
    savedir,
    pc="NONE",
    e0=14.0,
    e1=194.9,
    oversamp=4,
    sigmap=False,
    bkgvar=False,
    detapp=False,
):
    # 1 make dpi e0-e1 for sig and bkg times
    # 2 make bkg subtracted sky image
    # 3 run batcelldetect

    tstop = tstart + dt

    # 1
    # obsid_dir = args.obsid
    # aux_dir = os.path.join(obsid_dir, 'auxil')
    # attfile = os.path.join(aux_dir, [fname for fname in os.listdir(aux_dir) if 'pat' in fname][0])

    if np.isscalar(e0):
        e0 = [e0]
        e1 = [e1]

    cat_fnames = []

    for i in range(len(e0)):
        dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (e0[i], e1[i], tstart, tstop)
        dpi_fname = os.path.join(savedir, dpif)

        ev2dpi(evf, dpi_fname, tstart, tstop, e0[i], e1[i], dmask)

        # 2

        img_fname = os.path.join(
            savedir,
            "sky_%.1f_%.1f_%.3f_%.3f_os%d_.img"
            % (e0[i], e1[i], tstart, tstop, oversamp),
        )

        mk_sky_img(
            dpi_fname,
            img_fname,
            dmask,
            attfile,
            bkg_file=dpi_bkg_fnames[i],
            ovrsmp=oversamp,
            detapp=detapp,
        )

        # 3

        cat_fname = os.path.join(
            savedir,
            "cat_%.1f_%.1f_%.3f_%.3f_os%d_.fits"
            % (e0[i], e1[i], tstart, tstop, oversamp),
        )
        cat_fnames.append(cat_fname)

        if sigmap:
            sig_fname = os.path.join(
                savedir,
                "sig_%.1f_%.1f_%.3f_%.3f_os%d_.img"
                % (e0[i], e1[i], tstart, tstop, oversamp),
            )
        else:
            sig_fname = None

        if bkgvar:
            bkgvar_fname = os.path.join(
                savedir,
                "bkgvar_%.1f_%.1f_%.3f_%.3f_os%d_.img"
                % (e0[i], e1[i], tstart, tstop, oversamp),
            )
        else:
            bkgvar_fname = None

        run_batcelldetect(
            img_fname,
            cat_fname,
            ovrsmp=oversamp,
            pcode=pc,
            sigmap=sig_fname,
            bkgvar=bkgvar_fname,
        )

    return cat_fnames


def mk_sky_imgs4time_list(
    tstarts,
    dts,
    evf,
    attfile,
    dmask,
    savedir,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    sigmap=False,
    bkgvar=False,
    detapp=False,
    bkg_dpi="None",
):
    if np.isscalar(tstarts):
        tstarts = [tstarts]
        dts = [dts]
    Ntimes = len(tstarts)

    for i in range(Ntimes):
        tstart = tstarts[i]
        tstop = tstart + dts[i]

        dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (e0, e1, tstart, tstop)
        dpi_fname = os.path.join(savedir, dpif)

        ev2dpi(evf, dpi_fname, tstart, tstop, e0, e1, dmask)

        # 2

        img_fname = os.path.join(
            savedir,
            "sky_%.1f_%.1f_%.3f_%.3f_os%d_.img" % (e0, e1, tstart, tstop, oversamp),
        )
        if bkgvar:
            bkgvar_fname = os.path.join(
                savedir,
                "bkgvar_%.1f_%.1f_%.3f_%.3f_os%d_.img"
                % (e0, e1, tstart, tstop, oversamp),
            )
        else:
            bkgvar_fname = "NONE"

        mk_sky_img(
            dpi_fname,
            img_fname,
            dmask,
            attfile,
            bkg_file=bkg_dpi,
            ovrsmp=oversamp,
            detapp=detapp,
            bkgvar_map=bkgvar_fname,
        )


def do_search(arg_dict):
    cat_fnames = std_grb(
        arg_dict["tstart"],
        arg_dict["dt"],
        arg_dict["ev_fname"],
        arg_dict["bkg_dpis"],
        arg_dict["att_fname"],
        arg_dict["dmask"],
        arg_dict["savedir"],
        pc=arg_dict["pc_fname"],
        e0=arg_dict["e0"],
        e1=arg_dict["e1"],
        oversamp=arg_dict["oversamp"],
    )

    if arg_dict["db_fname"] is not None:
        conn = get_conn(arg_dict["db_fname"], timeout=30.0)
        try:
            write_cats2db(conn, cat_fnames, arg_dict["timeID"])
            logging.info(
                "Wrote results from timeID " + str(arg_dict["timeID"]) + " into DB"
            )
        except Exception as E:
            logging.error(str(E))
            logging.error(traceback.format_exc())
            logging.warning(
                "Failed to write results from timeID "
                + str(arg_dict["timeID"])
                + " into DB"
            )
            logging.info("Trying again")
            conn.close()
            time.sleep(1.0)
            conn = get_conn(arg_dict["db_fname"], timeout=60.0)
            try:
                write_cats2db(conn, cat_fnames, arg_dict["timeID"])
                logging.info(
                    "Wrote results from timeID " + str(arg_dict["timeID"]) + " into DB"
                )
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error(
                    "Failed to write results from timeID "
                    + str(arg_dict["timeID"])
                    + " into DB"
                )
                logging.error("And not trying again")
        conn.close()

    return cat_fnames


def do_search_mp(
    nproc,
    tstarts,
    dts,
    ev_fname,
    bkg_dpis,
    pc_fname,
    att_fname,
    dmask,
    savedir,
    e0=14.0,
    e1=194.9,
    oversamp=4,
    db_fname=None,
    timeIDs=None,
):
    arg_dict_keys = [
        "tstart",
        "dt",
        "ev_fname",
        "bkg_dpis",
        "pc_fname",
        "att_fname",
        "dmask",
        "savedir",
        "pc_fname",
        "e0",
        "e1",
        "oversamp",
        "db_fname",
    ]
    args_dict_list = []
    for i in range(len(tstarts)):
        arg_dict = {
            "tstart": tstarts[i],
            "dt": dts[i],
            "ev_fname": ev_fname,
            "bkg_dpis": bkg_dpis,
            "pc_fname": pc_fname,
            "att_fname": att_fname,
            "dmask": dmask,
            "savedir": savedir,
            "e0": e0,
            "e1": e1,
            "oversamp": oversamp,
            "db_fname": db_fname,
        }
        if timeIDs is not None:
            arg_dict["timeID"] = timeIDs[i]
        args_dict_list.append(arg_dict)

    t0 = time.time()

    if nproc == 1:
        for i in range(len(args_dict_list)):
            do_search(args_dict_list[i])
    else:
        p = mp.Pool(nproc)

        logging.info("Starting %d procs" % (nproc))

        cat_fnames = p.map_async(do_search, args_dict_list).get()

        p.close()
        p.join()

    logging.info("Done with all searches")
    logging.info(
        "Took %.2f seconds, %.2f minutes"
        % (time.time() - t0, (time.time() - t0) / 60.0)
    )
