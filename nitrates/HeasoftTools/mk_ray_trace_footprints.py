import os
import subprocess
from ..HeasoftTools.gen_tools import run_ftool, ftool_mp
import argparse
import numpy as np
import time
from astropy.table import Table


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


def do_ray_trace_imxy_list(out_fname, att_fname, imxs, imys, detmask, infile):
    ftool = "batmaskwtimg"
    for i in range(len(imxs)):
        outf = out_fname + "_%.4f_%.4f.img" % (imxs[i], imys[i])
        arg_list = [outf, att_fname, str(imxs[i]), str(imys[i])]
        arg_list += [
            "outtype=NONZERO",
            "detmask=" + detmask,
            "infile=" + infile,
            "coord_type=tanxy",
            "aperture=CALDB:DETECTION",
        ]

        run_ftool(ftool, arg_list)


def do_footprint_imxy_list(out_fname, att_fname, imxs, imys, time, detmask, infile):
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
            "aperture=CALDB:DETECTION",
        ]

        run_ftool(ftool, arg_list)


def do_ray_trace_imxy_tab(out_fname, att_fname, imxs, imys, detmask, infile, incat):
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
    parser.add_argument("--imx0", type=float, help="imx low value", default=-1.5)
    parser.add_argument("--imy0", type=float, help="imy low value", default=-0.9)
    parser.add_argument("--imx1", type=float, help="imx high value", default=1.5)
    parser.add_argument("--imy1", type=float, help="imy high value", default=0.9)
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
    parser.add_argument("--Njobs", type=int, help="Number of jobs running", default=16)
    parser.add_argument("--job_id", type=int, help="Which job is this", default=-1)
    parser.add_argument(
        "--rtdir",
        type=str,
        help="Directory to save foot prints to",
        default="/gpfs/scratch/jjd330/bat_data/footprint_dir/",
    )
    args = parser.parse_args()
    return args


def main(args):
    t_0 = time.time()

    rng = args.imrng

    imx_ax = np.linspace(-1.8, 1.8, 40 * 36 + 1)
    imy_ax = np.linspace(-1.0, 1.0, 40 * 20 + 1)
    print(imx_ax)
    print(imy_ax)

    imx_grid, imy_grid = np.meshgrid(imx_ax, imy_ax, indexing="ij")
    imxs = imx_grid.ravel()
    imys = imy_grid.ravel()
    Npnts = len(imxs)

    print(Npnts, " total points to make")

    if args.job_id >= 0:
        Nper_job = 1 + Npnts / args.Njobs
        i0 = args.job_id * Nper_job
        i1 = i0 + Nper_job
        imxs = imxs[i0:i1]
        imys = imys[i0:i1]

    Npnts = len(imxs)
    print(Npnts, " points to do in this job")

    if not os.path.exists(args.rtdir):
        os.makedirs(args.rtdir)

    out_fname = os.path.join(args.rtdir, "footprint")
    do_ray_trace_imxy_list(out_fname, "NONE", imxs, imys, "NONE", args.infile)

    print(
        "Took %.2f seconds, %.2f minutes to do everything"
        % (time.time() - t_0, (time.time() - t_0) / 60.0)
    )


if __name__ == "__main__":
    args = cli()

    main(args)
