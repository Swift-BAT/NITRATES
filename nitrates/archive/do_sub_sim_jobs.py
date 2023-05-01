import time
import numpy as np

# import pandas as pd
# import healpy as hp
# from astropy.table import Table
# from astropy.io import fits
# from astropy.wcs import WCS
# from scipy import interpolate
# import os, socket, subprocess, shlex
import os
import argparse
import logging, traceback

# import paramiko

# from ..lib.helper_funcs import send_email, send_error_email, send_email_attach, send_email_wHTML

from .do_manage import sub_jobs


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nratejobs", type=int, help="Total number of jobs", default=64
    )
    parser.add_argument(
        "--Nllhjobs", type=int, help="Total number of jobs", default=128
    )
    parser.add_argument(
        "--Name", type=str, help="Name of the event to submit jobs as", default=""
    )
    parser.add_argument(
        "--queue",
        type=str,
        help="Name of the queue to submit jobs to",
        default="cyberlamp",
    )
    parser.add_argument("--qos", type=str, help="qos option", default=None)
    parser.add_argument(
        "--BKGpyscript",
        type=str,
        help="Name of python script for Bkg Estimation",
        default="do_bkg_estimation_wPSs.py",
    )
    parser.add_argument(
        "--RATEpyscript",
        type=str,
        help="Name of python script for Rates analysis",
        default="do_rates_mle_wPSs_4sims.py",
    )
    parser.add_argument(
        "--LLHpyscript",
        type=str,
        help="Name of python script for LLH analysis",
        default="do_llh_forSims.py",
    )
    parser.add_argument(
        "--SCANpyscript",
        type=str,
        help="Name of python script for FoV scan",
        default="do_llh_scan_wPSs_realtime.py",
    )
    parser.add_argument(
        "--do_bkg", help="Submit the BKG estimation script", action="store_true"
    )
    parser.add_argument("--do_rates", help="Submit the Rate jobs", action="store_true")
    parser.add_argument("--do_llh", help="Submit the llh jobs", action="store_true")
    parser.add_argument(
        "--do_scan", help="Submit the llh scan jobs", action="store_true"
    )
    parser.add_argument(
        "--pbs_fname",
        type=str,
        help="Name of pbs script",
        default="/storage/work/jjd330/local/bat_data/BatML/submission_scripts/pyscript_template.pbs",
    )
    parser.add_argument(
        "--sim_dir", type=str, help="Name of simulation directory", default=None
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.do_rates:
        extra_args = "--sim_dir " + args.sim_dir
        sub_jobs(
            args.Nratejobs,
            "RATES_" + args.Name,
            args.RATEpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
            extra_args=extra_args,
        )
    if args.do_llh:
        extra_args = "--sim_dir " + args.sim_dir
        sub_jobs(
            args.Nllhjobs,
            "LLH_" + args.Name,
            args.LLHpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
            extra_args=extra_args,
        )
    if args.do_scan:
        sub_jobs(
            args.Nllhjobs,
            "SCAN_" + args.Name,
            args.SCANpyscript,
            args.pbs_fname,
            queue=args.queue,
            qos=args.qos,
        )


if __name__ == "__main__":
    args = cli()

    main(args)
