import os
import subprocess
from ..HeasoftTools.gen_tools import run_ftool
import argparse
import numpy as np
import time
from ..config import dir as direc


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


def mk_pc_img(infile, outfile, attfile, detmask, t=None):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += ["detmask=" + detmask, "attitude=" + attfile, "pcodemap=YES"]
    if time is not None:
        arg_list.append("time=" + str(t))
    run_ftool(ftool, arg_list)


def cli():
    # default_ebins = '15-40, 25-60, 50-80, 70-100, 90-135, 120-165, 150-195'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obsid",
        type=str,
        help="Obsid as a string, as it appears in file names",
        default="00851855000",
    )
    parser.add_argument(
        "--evf",
        type=str,
        help="Event File Name",
        default="/storage/work/jjd330/local/bat_data/00851855000/bat/event/sw00851855000bevshsp_uf.evt",
    )
    parser.add_argument("--dmask", type=str, help="Detmask File Name")
    parser.add_argument(
        "--ebins",
        type=str,
        help="Energy Bins for DRM, Fits file name or CALDB",
        default=os.path.join(direc, "data", "drm_new_ebins.fits"),
    )
    parser.add_argument(
        "--t0", type=float, help="Start time in MET seconds", default=555166977.856
    )
    parser.add_argument(
        "--dt", type=float, help="Duration time in seconds", default=1.0
    )
    parser.add_argument(
        "--pcmin", type=float, help="Min Partial coding used", default=0.0
    )
    parser.add_argument(
        "--workdir",
        type=str,
        help="Directory to work in",
        default="/gpfs/scratch/jjd330/bat_data/rand_work_files/",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        help="Directory to save files to",
        default="/gpfs/scratch/jjd330/bat_data/drms/",
    )
    parser.add_argument("--imx0", type=float, help="imx low value", default=-2.0)
    parser.add_argument("--imy0", type=float, help="imy low value", default=-1.0)
    parser.add_argument("--imx1", type=float, help="imx high value", default=2.0)
    parser.add_argument("--imy1", type=float, help="imy high value", default=1.0)
    parser.add_argument("--imstep", type=float, help="step size in imx/y", default=0.05)
    args = parser.parse_args()
    return args


def main(args):
    # 1 make pha
    # 2 make small event file to keep updating
    # do 3-5 for each imx/y position requested
    # 3 make aux ray trace file for updating the PHA file
    # 4 correct pha with given position and ray tracing for it
    # 5 make DRM
    # 6 make ray tracing dpis for all requested positions

    t_0 = time.time()

    tstart = args.t0
    tstop = args.t0 + args.dt

    imxs = np.linspace(-2.0, 2.0, 40 * 4 + 1)[1:-1]
    imys = np.linspace(-1, 1, 40 * 2 + 1)  # [1:-1]

    imxs = np.arange(args.imx0, args.imx1, args.imstep)
    if not np.isclose(imxs[-1], args.imx1):
        imxs = np.append(imxs, [args.imx1])
    imys = np.arange(args.imy0, args.imy1, args.imstep)
    if not np.isclose(imys[-1], args.imy1):
        imys = np.append(imys, [args.imy1])

    print("imxs: ")
    print(imxs)
    print("imys: ")
    print(imys)

    grids = np.meshgrid(imxs, imys, indexing="ij")

    imxs = grids[0].ravel()
    imys = grids[1].ravel()

    print(len(imxs), " DRMs to make")

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # imxs = np.array([1., 0.8, -.75, -1.1, 1., 0., -1., 0.])
    # imys = np.array([.5, -.4, -.45, .5, .15, -.5, .15, .45])

    # 1
    obsid_dir = os.path.join("/storage/work/jjd330/local/bat_data", args.obsid)
    hk_dir = os.path.join(obsid_dir, "bat", "hk")
    on_off_map = os.path.join(
        hk_dir, [fname for fname in os.listdir(hk_dir) if "bdecb" in fname][0]
    )
    # dap_fname = os.path.join(hk_dir, [fn for fn in os.listdir(hk_dir) if 'bdp.hk' in fn][0])
    aux_dir = os.path.join(obsid_dir, "auxil")
    attfile = os.path.join(
        aux_dir, [fname for fname in os.listdir(aux_dir) if "pat" in fname][0]
    )

    pid = os.getpid()

    pha_fname = os.path.join(
        args.workdir, str(pid) + "_pha_%.3f_%.3f_.pha" % (tstart, tstop)
    )

    ev2pha(args.evf, pha_fname, tstart, tstop, args.ebins, "NONE")

    new_event = os.path.join(args.workdir, str(pid) + "_small.evt")

    mk_small_evt(args.evf, new_event)

    t_1 = time.time()

    from astropy.io import fits

    pc_img_fname = os.path.join(args.workdir, "PC.img")
    if not os.path.exists(pc_img_fname):
        mk_pc_img(on_off_map, pc_img_fname, attfile, "NONE", t=args.t0)
    pc0 = fits.open(pc_img_fname)[0]
    from astropy.wcs import WCS

    w = WCS(pc0.header, key="T")
    w2 = WCS(pc0.header)

    inds = w.all_world2pix(imxs.ravel(), imys.ravel(), 0)

    inds[0] = inds[0].astype(np.int64)
    inds[1] = inds[1].astype(np.int64)

    pcs = pc0.data[inds[1], inds[0]]
    ras, decs = w2.all_pix2world(inds[0], inds[1], 0)

    bl_good = pcs >= args.pcmin

    imxs = imxs[bl_good]
    imys = imys[bl_good]
    ras = ras[bl_good]
    decs = decs[bl_good]

    for i in range(len(imxs)):
        aux_fname = os.path.join(args.workdir, str(pid) + "_aux.fits")
        mk_rt_aux_file(
            new_event, aux_fname, imxs[i], imys[i], "NONE", attfile, ras[i], decs[i]
        )

        pha_sys_err(pha_fname, aux_fname)

        drm_fname = os.path.join(
            args.savedir, "drm_%.3f_%.3f_.fits" % (imxs[i], imys[i])
        )

        # mk_drm(pha_fname, drm_fname, dap_fname)
        mk_drm(pha_fname, drm_fname, "NONE")

    print(
        "Took %.2f seconds, %.2f minutes to do %d DRMs"
        % (time.time() - t_1, (time.time() - t_1) / 60.0, len(imxs))
    )


if __name__ == "__main__":
    args = cli()

    main(args)
