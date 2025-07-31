import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os
import argparse
import logging, traceback, time

from ..lib.sqlite_funcs import get_conn, write_JobSquare_line, setup_square_stat
from ..lib.dbread_funcs import (
    get_full_sqlite_table_as_df,
    get_files_tab,
    get_info_tab,
    guess_dbfname,
)


def setup_job_tables(
    Njobs, conn, pc_fname, im_steps=10, pc_min=0.1, sig_pix_tab=None, do_stat=False
):
    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key="T")
    pc = PC.data

    pcbl = pc >= pc_min
    pc_inds = np.where(pcbl)
    imxs, imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    bins = [np.linspace(-2, 2, im_steps * 4 + 1), np.linspace(-1, 1, im_steps * 2 + 1)]

    h = np.histogram2d(imxs, imys, bins=bins)[0]

    square_bl = h > 0
    logging.info("%d squareIDs" % (np.sum(square_bl)))
    inds = np.where(square_bl)
    squareIDs = np.ravel_multi_index(inds, h.shape)

    for i, squareID in enumerate(squareIDs):
        data_dict = {}
        data_dict["proc_group"] = i % Njobs
        indx, indy = np.unravel_index(squareID, h.shape)
        data_dict["imx0"] = bins[0][indx]
        data_dict["imx1"] = bins[0][indx + 1]
        data_dict["imy0"] = bins[1][indy]
        data_dict["imy1"] = bins[1][indy + 1]
        data_dict["squareID"] = squareID
        write_JobSquare_line(conn, data_dict)

    if do_stat:
        if sig_pix_tab is None:
            sig_pix_tab = get_full_sqlite_table_as_df(conn, "SigImages")

        timeIDs = sig_pix_tab.timeID.unique()
        logging.info("%d timeIDs" % (len(timeIDs)))

        setup_square_stat(conn, timeIDs, squareIDs)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Njobs", type=int, help="Number of jobs to run for LLH analysis", default=24
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--pcfname",
        type=str,
        help="Name of partial coding image fits file",
        default=None,
    )
    parser.add_argument("--dostat", help="Setup status table", action="store_true")
    parser.add_argument("--imsteps", type=int, help="Steps per 1 imx/y", default=10)
    args = parser.parse_args()
    return args


def main(args):
    fname = "setup_square_db_tables"

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    f = open(fname + ".pid", "w")
    f.write(str(os.getpid()))
    f.close()

    logging.info("Wrote pid: %d" % (os.getpid()))

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB: " + db_fname)
    conn = get_conn(db_fname)

    if args.pcfname is None:
        files_tab = get_files_tab(conn)
        work_dir = files_tab["workDir"][0]
        logging.info("Got files table")
        try:
            if "img" in files_tab["pcodeFname"][0]:
                pcode_fname = files_tab["pcodeFname"][0]
        except:
            fnames = os.listdir(".")
            pc_fnames = [
                fname
                for fname in fnames
                if ("img" in fname) and (("pc" in fname) or ("PC" in fname))
            ]
            if len(pc_fnames) < 1:
                fnames = os.listdir(work_dir)
                pc_fnames = [
                    fname
                    for fname in fnames
                    if ("img" in fname) and (("pc" in fname) or ("PC" in fname))
                ]

            if len(pc_fnames) > 1:
                pc_fname = [fname for fname in pc_fnames if "2" in fname][0]
            else:
                pc_fname = pc_fnames[0]
    else:
        pc_fname = args.pcfname

    setup_job_tables(
        args.Njobs, conn, pc_fname, do_stat=args.dostat, im_steps=args.imsteps
    )

    logging.info("Done setting up square tables, now exiting")


if __name__ == "__main__":
    args = cli()

    main(args)
