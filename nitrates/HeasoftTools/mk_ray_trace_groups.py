import os
import subprocess
from ..HeasoftTools.gen_tools import run_ftool, ftool_mp
import argparse
import numpy as np
import time
from astropy.table import Table
import pandas as pd


def run_ftjoin_mp(dname, dname2, fnames, nproc):
    ftool = "ftjoin"

    arg_lists = []

    for fname in fnames:
        arg_list = [ftool]
        fnew = dname2 + fname[:21]
        f_name = dname + fname
        arg_list += [f_name + "[ATTITUDE]", f_name + "[ACS_DATA]"]
        arg_list.append(fnew)
        arg_list += [
            "TIME==TIME_",
            "leftnameprefix=NONE",
            "rightnameprefix=NONE",
            "rightnamesuffix=_",
        ]

        arg_lists.append(arg_list)

    print("Opening pool of %d workers" % (nproc))
    t0 = time.time()

    p = mp.Pool(nproc, init_worker)

    print(os.getpid())
    print("active children: ", mp.active_children())

    try:
        p.map(run_ftool_mp, arg_lists, chunksize=10)
    except KeyboardInterrupt:
        print("active children: ", mp.active_children())
        p.terminate()
        p.join()
        print("terminate, join")
        print("active children: ", mp.active_children())
        sys.exit()

    print("active children: ", mp.active_children())
    p.close()
    p.join()
    print("close, join")
    print("active children: ", mp.active_children())

    print("Finished in %.3f seconds" % (time.time() - t0))


def do_ray_trace(out_fname, att_fname, ra, dec, time, detmask, infile):
    ftool = "batmaskwtimg"
    arg_list = [out_fname, att_fname, str(ra), str(dec)]
    arg_list += [
        "time=%.2f" % (time),
        "rebalance=NO",
        "corrections=forward,unbalanced,flatfield",
        "detmask=" + detmask,
        "infile=" + infile,
    ]
    run_ftool(ftool, arg_list)


def do_ray_trace_ra_dec_list(out_fname, att_fname, ras, decs, time, detmask, infile):
    ftool = "batmaskwtimg"
    for i in range(len(ras)):
        outf = out_fname + "_%.2f_%.2f.img" % (ras[i], decs[i])
        arg_list = [outf, att_fname, str(ras[i]), str(decs[i])]
        arg_list += [
            "time=%.2f" % (time),
            "rebalance=NO",
            "corrections=forward,unbalanced,flatfield",
            "detmask=" + detmask,
            "infile=" + infile,
        ]
        # arg_list += ["time=%.2f" %(time), "rebalance=NO",
        #            "corrections=forward,unbalanced,flatfield",
        #            "infile="+infile]

        run_ftool(ftool, arg_list)


def do_ray_trace_imxy_list(out_fname, att_fname, imxs, imys, time, detmask, infile):
    ftool = "batmaskwtimg"
    for i in range(len(imxs)):
        outf = out_fname + "_%.5f_%.5f.img" % (imxs[i], imys[i])
        arg_list = [outf, att_fname, str(imxs[i]), str(imys[i])]
        arg_list += [
            "time=%.2f" % (time),
            "rebalance=NO",
            "corrections=forward,unbalanced,flatfield",
            "detmask=" + detmask,
            "infile=" + infile,
            "coord_type=tanxy",
        ]

        run_ftool(ftool, arg_list)


def do_footprint_imxy_tab(
    out_fname, att_fname, imxs, imys, detmask, infile, incat, detapp=False
):
    ftool = "batmaskwtimg"
    # for i in xrange(len(imxs)):
    outf = out_fname + "_%.5f_%.5f_%.5f_%.5f_.img" % (
        np.min(imxs),
        np.min(imys),
        np.max(imxs),
        np.max(imys),
    )
    if os.path.isfile(outf):
        print("already made")
        return
    arg_list = [outf, att_fname, "0.0", "0.0"]
    arg_list += [
        "outtype=NONZERO",
        "detmask=" + detmask,
        "infile=" + infile,
        "coord_type=tanxy",
        "incatalog=" + incat,
        "racol=IMX",
        "deccol=IMY",
        "catnumcol=NONE",
        "chatter=1",
        "distfile=CALDB",
    ]
    if detapp:
        arg_list.append("aperture=CALDB:DETECTION")
    run_ftool(ftool, arg_list)


def do_ray_trace_imxy_tab(
    out_fname, att_fname, imxs, imys, detmask, infile, incat, detapp=False
):
    ftool = "batmaskwtimg"
    # for i in xrange(len(imxs)):
    outf = out_fname + "_%.5f_%.5f_%.5f_%.5f_.img" % (
        np.min(imxs),
        np.min(imys),
        np.max(imxs),
        np.max(imys),
    )
    if os.path.isfile(outf):
        print("already made")
        return
    arg_list = [outf, att_fname, "0.0", "0.0"]
    arg_list += [
        "rebalance=NO",
        "corrections=forward,unbalanced,flatfield,subpixelate",
        "detmask=" + detmask,
        "infile=" + infile,
        "coord_type=tanxy",
        "incatalog=" + incat,
        "racol=IMX",
        "deccol=IMY",
        "catnumcol=NONE",
        "chatter=1",
        "distfile=CALDB",
    ]
    if detapp:
        arg_list.append("aperture=CALDB:DETECTION")
    run_ftool(ftool, arg_list)


def mk_imxy_tab(imxs, imys, fname):
    names = ["IMX", "IMY", "NAME"]
    grid_x, grid_y = np.meshgrid(imxs, imys, indexing="ij")
    tab = Table()
    tab["IMX"] = grid_x.ravel()
    tab["IMY"] = grid_y.ravel()
    names = np.array(
        ["%.5f %.5f" % (tab["IMX"][i], tab["IMY"][i]) for i in range(len(tab))]
    )
    tab["NAME"] = names
    print(len(tab), " positions to do")
    tab.write(fname, overwrite=True)


def ev2pha(infile, outfile, tstart, tstop, ebins, detmask):
    ftool = "batbinevt"
    arg_list = [infile, outfile, "PHA", "0", "uniform", ebins]
    arg_list += ["tstart=" + str(tstart), "tstop=" + str(tstop), "detmask=" + detmask]
    run_ftool(ftool, arg_list)


def pha_sys_err(infile, auxfile):
    ftool = "batupdatephakw"
    arg_list = [infile, auxfile]
    run_ftool(ftool, arg_list)

    ftool = "batphasyserr"
    arg_list = [infile, "CALDB"]
    run_ftool(ftool, arg_list)


def mk_small_evt(infile, outfile):
    ftool = "fextract-events"
    arg_list = [infile + "[pha=100:101]", outfile, "gti=GTI"]
    run_ftool(ftool, arg_list)


def mk_rt_aux_file(infile, outfile, imx, imy, dmask, attfile, ra, dec):
    ftool = "batmaskwtevt"
    arg_list = [infile, attfile, str(ra), str(dec)]
    arg_list += [
        "coord_type=sky",
        "auxfile=" + outfile,
        "clobber=True",
        "detmask=" + dmask,
    ]
    run_ftool(ftool, arg_list)


def mk_drm(pha, outfile, dapfile):
    ftool = "batdrmgen"
    arg_list = [pha, outfile, dapfile, "method=TABLE"]
    run_ftool(ftool, arg_list)


def bateconvert(infile, calfile):
    ftool = "bateconvert"
    arg_list = ["infile=" + infile, "calfile=" + calfile, "residfile=CALDB"]
    run_ftool(ftool, arg_list)


def detmask(infile, outfile, dmask):
    ftool = "batdetmask"
    arg_list = [infile, outfile, "detmask=" + dmask]
    run_ftool(ftool, arg_list)


def mk_bkg_mod(infile, outfile, detmask):
    ftool = "batclean"
    arg_list = [infile, outfile]
    arg_list += [
        "incatalog=NONE",
        "detmask=" + detmask,
        "srcclean=NO",
        "outversion=bkgfit",
    ]
    run_ftool(ftool, arg_list)


def mk_pc_img(infile, outfile, detmask, attfile):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += ["detmask=" + detmask, "attitude=" + attfile, "pcodemap=YES"]
    run_ftool(ftool, arg_list)


def cli():
    # default_ebins = '15-40, 25-60, 50-80, 70-100, 90-135, 120-165, 150-195'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        type=str,
        help="In File Name needed for batmaskwtimg",
        default="/storage/work/jjd330/local/bat_data/pha.pha",
    )
    parser.add_argument(
        "--t0", type=float, help="Start time in MET seconds", default=4e8
    )
    parser.add_argument("--imx0", type=float, help="imx low value", default=0.0)
    parser.add_argument("--imy0", type=float, help="imy low value", default=0.0)
    parser.add_argument("--imx1", type=float, help="imx high value", default=0.1)
    parser.add_argument("--imy1", type=float, help="imy high value", default=0.1)
    parser.add_argument(
        "--rtstep", type=float, help="step size in imx/y for ray tracing", default=0.002
    )
    parser.add_argument(
        "--pcmin", type=float, help="Min Partial coding used", default=1e-2
    )
    parser.add_argument(
        "--imrng",
        type=float,
        help="range for imx/y around center point or all",
        default=0.02,
    )
    parser.add_argument(
        "--rtdir",
        type=str,
        help="Directory to save ray traces to",
        default="/storage/home/jjd330/scratch/bat_data/ray_traces/",
    )
    parser.add_argument(
        "--imxy_file", type=str, help="file with imxys to do", default=None
    )
    parser.add_argument("--Njobs", type=int, help="Total number of jobs", default=1)
    parser.add_argument("--job_id", type=int, help="Job ID", default=-1)
    parser.add_argument(
        "--detapp", help="Use the detecion aperture", action="store_true"
    )
    parser.add_argument(
        "--footprint", help="Do footprints instead of maskwts", action="store_true"
    )
    args = parser.parse_args()
    return args


def main(args):
    t_0 = time.time()

    rng = args.imrng

    if args.imxy_file is not None:
        df_imxy = pd.read_csv(args.imxy_file)
        Npnts = len(df_imxy)
        Npnts2do = 1 + Npnts / args.Njobs
        i0 = args.job_id * Npnts2do
        i1 = i0 + Npnts2do
        if args.job_id < 0:
            i0 = 0
            i1 = Npnts
            Npnts2do = Npnts
        print("%d total to do" % (Npnts))
        print("doing %d here" % (Npnts2do))
        df = df_imxy[i0:i1]
        i = 0

        for ind, row in df.iterrows():
            imx0 = row["imx0"]
            imx1 = row["imx1"]
            imy0 = row["imy0"]
            imy1 = row["imy1"]

            imxs = np.linspace(imx0, imx1, int(rng / args.rtstep) + 1)
            imys = np.linspace(imy0, imy1, int(rng / args.rtstep) + 1)

            imxs = np.arange(imx0, imx1, args.rtstep)
            if not np.isclose(imxs[-1], imx1):
                imxs = np.append(imxs, [imx1])
            imys = np.arange(imy0, imy1, args.rtstep)
            if not np.isclose(imys[-1], imy1):
                imys = np.append(imys, [imy1])

            print("imxs")
            print(imxs)
            print("imys")
            print(imys)

            tab_fn = "tab_%.5f_%.5f_%.5f_%.5f_.fits" % (
                np.min(imxs),
                np.min(imys),
                np.max(imxs),
                np.max(imys),
            )

            # make a catalog table to pass to batmaskwtimg

            tab_fname = os.path.join(args.rtdir, tab_fn)

            mk_imxy_tab(imxs, imys, tab_fname)

            if args.footprint:
                out_fname = os.path.join(args.rtdir, "footprint")
                do_footprint_imxy_tab(
                    out_fname,
                    "NONE",
                    imxs,
                    imys,
                    "NONE",
                    args.infile,
                    tab_fname,
                    detapp=args.detapp,
                )
            else:
                out_fname = os.path.join(args.rtdir, "fwd_ray_trace")
                do_ray_trace_imxy_tab(
                    out_fname,
                    "NONE",
                    imxs,
                    imys,
                    "NONE",
                    args.infile,
                    tab_fname,
                    detapp=args.detapp,
                )

            print(
                "Took %.2f seconds, %.2f minutes so far, done with %d of %d"
                % (time.time() - t_0, (time.time() - t_0) / 60.0, i + 1, Npnts2do)
            )
            i += 1

    else:
        nx_steps = int((args.imx1 - args.imx0) / rng) + 1
        ny_steps = int((args.imy1 - args.imy0) / rng) + 1

        print(nx_steps * ny_steps, " ray traces to make")

        if not os.path.exists(args.rtdir):
            os.makedirs(args.rtdir)

        for i in range(nx_steps):
            imx0 = args.imx0 + i * rng
            imx1 = imx0 + rng

            for j in range(ny_steps):
                imy0 = args.imy0 + j * rng
                imy1 = imy0 + rng

                imxs = np.linspace(imx0, imx1, int(rng / args.rtstep) + 1)
                imys = np.linspace(imy0, imy1, int(rng / args.rtstep) + 1)

                imxs = np.arange(imx0, imx1, args.rtstep)
                if not np.isclose(imxs[-1], imx1):
                    imxs = np.append(imxs, [imx1])
                imys = np.arange(imy0, imy1, args.rtstep)
                if not np.isclose(imys[-1], imy1):
                    imys = np.append(imys, [imy1])

                print("imxs")
                print(imxs)
                print("imys")
                print(imys)

                tab_fn = "tab_%.5f_%.5f_%.5f_%.5f_.fits" % (
                    np.min(imxs),
                    np.min(imys),
                    np.max(imxs),
                    np.max(imys),
                )

                # make a catalog table to pass to batmaskwtimg

                tab_fname = os.path.join(args.rtdir, tab_fn)

                mk_imxy_tab(imxs, imys, tab_fname)

                if args.footprint:
                    out_fname = os.path.join(args.rtdir, "footprint")
                    do_footprint_imxy_tab(
                        out_fname,
                        "NONE",
                        imxs,
                        imys,
                        "NONE",
                        args.infile,
                        tab_fname,
                        detapp=args.detapp,
                    )
                else:
                    out_fname = os.path.join(args.rtdir, "fwd_ray_trace")
                    do_ray_trace_imxy_tab(
                        out_fname,
                        "NONE",
                        imxs,
                        imys,
                        "NONE",
                        args.infile,
                        tab_fname,
                        detapp=args.detapp,
                    )

                print(
                    "Took %.2f seconds, %.2f minutes so far, done with %d of %d"
                    % (
                        time.time() - t_0,
                        (time.time() - t_0) / 60.0,
                        (i * ny_steps + j + 1),
                        (nx_steps * ny_steps),
                    )
                )

    print(
        "Took %.2f seconds, %.2f minutes to do everything"
        % (time.time() - t_0, (time.time() - t_0) / 60.0)
    )


if __name__ == "__main__":
    args = cli()

    main(args)
