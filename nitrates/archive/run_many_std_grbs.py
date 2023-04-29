import numpy as np
import logging, traceback
import argparse


from ..lib.funcs2run_bat_tools import do_bkg, std_grb, do_pc


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evfname", type=str, help="Event data file", default="filter_evdata.fits"
    )
    parser.add_argument(
        "--dmask", type=str, help="detmask file name", default="detmask.fits"
    )
    parser.add_argument(
        "--att_fname", type=str, help="Fname for that att file", default="attitude.fits"
    )
    parser.add_argument(
        "--tstart", type=float, help="Time to start first exp at in MET"
    )
    parser.add_argument(
        "--search_twind",
        type=float,
        help="Search ends at tstart + search_twind",
        default=100.0,
    )
    parser.add_argument("--tstep", type=float, help="Time step in seconds", default=1.0)
    parser.add_argument("--exp", type=float, help="Image Exposure to use", default=1.0)
    parser.add_argument(
        "--bkg_dur", type=float, help="bkg exposure to use", default=20.0
    )
    parser.add_argument(
        "--bkg_offset",
        type=float,
        help="offset of the end of the bkg duration from the exposure start",
        default=-20.0,
    )
    parser.add_argument("--e0", type=float, help="Min energy to use", default=15.0)
    parser.add_argument("--e1", type=float, help="Max energy to use", default=350.0)
    parser.add_argument("--oversamp", type=int, help="Oversampling to use", default=2)
    args = parser.parse_args()
    return args


def main(args):
    start_time = args.tstart

    tstarts = np.arange(start_time, start_time + args.search_twind, args.tstep)

    tends = tstarts + args.exp

    bkg_tstarts = tstarts + args.bkg_offset - args.bkg_dur
    bkg_tends = bkg_tstarts + args.bkg_dur

    Nexps2do = len(tstarts)

    print(("Nexps2do: ", Nexps2do))

    pcfname = do_pc(args.dmask, args.att_fname, ".", ovrsmp=args.oversamp, detapp=True)

    for i in range(Nexps2do):
        print("**********************************")

        bkg_dpi = [
            do_bkg(
                bkg_tstarts[i],
                bkg_tends[i],
                args.evfname,
                args.dmask,
                ".",
                e0=args.e0,
                e1=args.e1,
            )
        ]

        std_grb(
            tstarts[i],
            args.exp,
            args.evfname,
            bkg_dpi,
            args.att_fname,
            args.dmask,
            ".",
            pc=pcfname,
            e0=args.e0,
            e1=args.e1,
            oversamp=args.oversamp,
            detapp=True,
        )

        print("**********************************")
        print(
            (
                "************* Done with image %d of %d *********************"
                % (i + 1, Nexps2do)
            )
        )
        print("**********************************")


if __name__ == "__main__":
    args = cli()

    main(args)
