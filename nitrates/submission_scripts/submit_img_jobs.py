import os
import numpy as np
import time
import argparse
import logging


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssh", help="Do we need to ssh in to submit?", action="store_true"
    )
    parser.add_argument(
        "--Njobs", type=int, help="Number of jobs to submit", default=16
    )
    parser.add_argument(
        "--dt0",
        type=float,
        help="Time relative to trigger time to start at",
        default=-16.0,
    )
    parser.add_argument(
        "--dt1",
        type=float,
        help="Time relative to trigger time to end at",
        default=16.0,
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default="none"
    )
    parser.add_argument(
        "--workdir", type=str, help="directory to work in", default=None
    )
    parser.add_argument(
        "--name", type=str, help="directory to work in", default="SigImgs"
    )
    parser.add_argument(
        "--queue", type=str, help="what queue to submit this to", default="Open"
    )
    parser.add_argument(
        "--pbs_fname",
        type=str,
        help="file name of the pbs script to submit",
        default="/storage/work/jjd330/local/bat_data/BatML/submission_scripts/sub_sigimgs.pbs",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.ssh:
        ssh_cmd = 'ssh aci-b.aci.ics.psu.edu "'
        base_sub_cmd = "qsub %s -A %s -N %s -v " % (
            args.pbs_fname,
            args.queue,
            args.name,
        )
    else:
        base_sub_cmd = "qsub %s -A %s -N %s -v " % (
            args.pbs_fname,
            args.queue,
            args.name,
        )

    njobs = args.Njobs
    if args.workdir is None:
        workdir = os.getcwd()
    else:
        workdir = args.workdir

    cmd = ""

    dts = np.linspace(args.dt0, args.dt1, njobs + 1)
    for i in range(njobs):
        cmd_ = "workdir=%s,dt0=%.3f,dt1=%.3f,dbfname=%s" % (
            workdir,
            dts[i],
            dts[i + 1],
            args.dbfname,
        )
        if args.ssh:
            cmd += base_sub_cmd + cmd_
            if i < (njobs - 1):
                cmd += " | "
        else:
            cmd = base_sub_cmd + cmd_
            logging.info("Trying to submit: ")
            logging.info(cmd)

            try:
                os.system(cmd)
            except Exception as E:
                logging.error(E)
                logging.error("Messed up with ")
                logging.error(cmd)

            time.sleep(1.0)
    if args.ssh:
        cmd = ssh_cmd + cmd + '"'
        logging.info("Full cmd to run:")
        logging.info(cmd)
        try:
            os.system(cmd)
        except Exception as E:
            logging.error(E)
            logging.error("Messed up with ")
            logging.error(cmd)


if __name__ == "__main__":
    logging.basicConfig(
        filename="submit_jobs_log.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    args = cli()

    main(args)
