import numpy as np
import argparse

from ..lib.funcs2run_bat_tools import mk_sky_imgs4time_list


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t0", type=float, help="Min image start time to use (MET (s))", default=None
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Total time in secs to create images during",
        default=8.0,
    )
    parser.add_argument(
        "--dur", type=float, help="Exposure time for each image", default=0.2
    )
    parser.add_argument("--e0", type=float, help="Min energy to use", default=15.0)
    parser.add_argument("--e1", type=float, help="Max energy to use", default=350.0)
    parser.add_argument(
        "--pc_fname", type=str, help="partial coding file name", default="pc_4.img"
    )
    parser.add_argument(
        "--dname", type=str, help="Directory to make images in", default="."
    )
    parser.add_argument(
        "--evfname", type=str, help="Event file name", default="filter_evdata.fits"
    )
    parser.add_argument(
        "--attfname", type=str, help="Attitude file name", default="attitude.fits"
    )
    parser.add_argument(
        "--dmask", type=str, help="Detmask file name", default="detmask.fits"
    )

    args = parser.parse_args()
    return args


def main(args):
    t1 = args.t0 + args.dt
    tstarts = np.arange(args.t0, t1, args.dur)
    dts = args.dur * np.ones_like(tstarts)

    mk_sky_imgs4time_list(
        tstarts,
        dts,
        args.evfname,
        args.attfname,
        args.dmask,
        args.dname,
        e0=args.e0,
        e1=args.e1,
        bkgvar=True,
        detapp=True,
    )


if __name__ == "__main__":
    args = cli()

    main(args)
