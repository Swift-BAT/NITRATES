import os
import subprocess
import sys
from ..HeasoftTools.gen_tools import run_ftool
import argparse
import time
import numpy as np
import multiprocessing as mp
from astropy.table import Table

# import ..config

# sys.path.append(os.path.join(config.dir, 'HeasoftTools'))


def ev2dpi(infile, outfile, tstart, tstop, e0, e1, detmask):
    ftool = "batbinevt"
    arg_list = [infile, outfile, "DPI", "0", "uniform", str(e0) + "-" + str(e1)]
    arg_list += ["tstart=" + str(tstart), "tstop=" + str(tstop), "detmask=" + detmask]
    run_ftool(ftool, arg_list)


def ev2dpi_ebins(infile, outfile, tstart, tstop, ebins, detmask):
    ftool = "batbinevt"
    arg_list = [infile, outfile, "DPI", "0", "uniform", ebins]
    arg_list += ["tstart=" + str(tstart), "tstop=" + str(tstop), "detmask=" + detmask]
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


def mk_pc_img(infile, outfile, detmask, attfile, ovrsmp=None):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += ["detmask=" + detmask, "attitude=" + attfile, "pcodemap=YES"]
    if ovrsmp is not None:
        arg_list += ["oversampx=" + str(ovrsmp), "oversampy=" + str(ovrsmp)]
    run_ftool(ftool, arg_list)


def mk_sky_img(infile, outfile, detmask, attfile, bkg_file=None, ovrsmp=2):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += [
        "detmask=" + detmask,
        "attitude=" + attfile,
        "oversampx=" + str(ovrsmp),
        "oversampy=" + str(ovrsmp),
    ]
    if bkg_file is not None:
        arg_list += ["bkgfile=" + bkg_file]
    run_ftool(ftool, arg_list)


def run_batcelldetect(
    infile, cat_fname, snr_thresh=3.5, sigmap=None, ovrsmp=2, incat="NONE", pcode="NONE"
):
    ftool = "batcelldetect"
    bkgradius = 15 * ovrsmp
    srcradius = 6 * ovrsmp
    arg_list = [
        "infile=" + infile,
        "outfile=" + cat_fname,
        "snrthresh=" + str(snr_thresh),
    ]
    arg_list += [
        "bkgradius=" + str(bkgradius),
        "srcradius=" + str(srcradius),
        "nadjpix=2",
        "vectorflux=YES",
        "incatalog=" + incat,
        "niter=4",
        "pcodefile=" + pcode,
        "chatter=1",
    ]
    if sigmap is not None:
        arg_list.append("signifmap=" + sigmap)
    print(arg_list)
    run_ftool(ftool, arg_list)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nproc", type=int, help="Number of procesors to use", default=1
    )
    parser.add_argument(
        "--savedir",
        type=str,
        help="Directory to save files",
        default="/gpfs/scratch/jjd330/bat_data/",
    )
    parser.add_argument(
        "--obsid", type=str, help="Obsid as a string, as it appears in file names"
    )
    parser.add_argument("--evf", type=str, help="Event File Name")
    parser.add_argument("--e0", type=float, help="Min energy", default=14.0)
    parser.add_argument("--e1", type=float, help="Max energy", default=194.9)
    parser.add_argument("--dmask", type=str, help="Detmask fname")
    parser.add_argument(
        "--tabfname", type=str, help="Table with seed times from rate pvalues"
    )
    parser.add_argument(
        "--pcut", type=float, help="pvalue to cut on for seed times", default=1e-2
    )
    parser.add_argument("--bkgt0", type=float, help="Bkg start time in MET seconds")
    parser.add_argument(
        "--bkgdt", type=float, help="Bkg duration time in seconds", default=20.0
    )
    parser.add_argument("--nobkg", help="Don't do bkg subtract", action="store_true")
    parser.add_argument("--oversamp", type=int, help="Image Oversampling", default=4)
    args = parser.parse_args()
    return args


def get_seed_times(args):
    tab = Table.read(args.tabfname)
    bl = (tab["pval"] <= args.pcut) & (tab["Nsig"] > 0.0)
    ts = np.vstack({tuple([row["tstart"], row["tstop"]]) for row in tab[bl]})
    # print np.sum(bl), ' seed times to search'
    print(len(ts), " seed times to search")
    # tstart = tab['tstart'][bl]
    # tstop = tab['tstop'][bl]
    tstart = ts[:, 0]
    tstop = ts[:, 1]
    return tstart, tstop


def get_times_all(tmin, tmax):
    tstep = 0.064
    tsize = 0.256

    tstarts = np.arange(tmin, tmax - tstep, tstep)
    tstops = tstarts + tsize

    for i in range(3):
        tstep *= 2
        tsize *= 2

        tstarts_ = np.arange(tmin, tmax - tstep, tstep)
        tstops_ = tstarts_ + tsize
        tstops = np.append(tstops, tstops_)
        tstarts = np.append(tstarts, tstarts_)

    return tstarts, tstops


def do_bkg(args):
    bkg_tstart = args.bkgt0
    bkg_tstop = args.bkgt0 + args.bkgdt

    dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (args.e0, args.e1, bkg_tstart, bkg_tstop)
    dpi_bkg_fname = os.path.join(args.savedir, args.obsid, dpif)

    ev2dpi(args.evf, dpi_bkg_fname, bkg_tstart, bkg_tstop, args.e0, args.e1, args.dmask)

    return dpi_bkg_fname


def do_bkg_ebins(args, ebins):
    bkg_tstart = args.bkgt0
    bkg_tstop = args.bkgt0 + args.bkgdt

    dpif = "dpi_bkg_ebins_%.3f_%.3f_.dpi" % (bkg_tstart, bkg_tstop)
    dpi_bkg_fname = os.path.join(args.savedir, args.obsid, dpif)

    ev2dpi_ebins(args.evf, dpi_bkg_fname, bkg_tstart, bkg_tstop, ebins, args.dmask)

    return dpi_bkg_fname


def std_grb(args, tstart, dt, dpi_bkg_fname, pc="NONE"):
    # 1 make dpi e0-e1 for sig and bkg times
    # 2 make bkg subtracted sky image
    # 3 run batcelldetect

    # tstart = args.t0
    tstop = tstart + dt

    # 1
    obsid_dir = args.obsid
    aux_dir = os.path.join(obsid_dir, "auxil")
    attfile = os.path.join(
        aux_dir, [fname for fname in os.listdir(aux_dir) if "pat" in fname][0]
    )

    dpif = "dpi_%.1f_%.1f_%.3f_%.3f_.dpi" % (args.e0, args.e1, tstart, tstop)
    dpi_fname = os.path.join(args.savedir, obsid_dir, dpif)

    ev2dpi(args.evf, dpi_fname, tstart, tstop, args.e0, args.e1, args.dmask)

    # 2

    img_fname = os.path.join(
        args.savedir,
        obsid_dir,
        "sky_%.1f_%.1f_%.3f_%.3f_os%d_.img"
        % (args.e0, args.e1, tstart, tstop, args.oversamp),
    )

    mk_sky_img(
        dpi_fname,
        img_fname,
        args.dmask,
        attfile,
        bkg_file=dpi_bkg_fname,
        ovrsmp=args.oversamp,
    )

    # 3

    cat_fname = os.path.join(
        args.savedir,
        obsid_dir,
        "cat_%.1f_%.1f_%.3f_%.3f_os%d_.fits"
        % (args.e0, args.e1, tstart, tstop, args.oversamp),
    )

    # sig_fname = os.path.join(args.savedir, obsid_dir, 'sig_%.1f_%.1f_%.3f_%.3f_os%d_.img' %(args.e0, args.e1, tstart,\
    #                                                                           tstop, args.oversamp))

    run_batcelldetect(img_fname, cat_fname, ovrsmp=args.oversamp, pcode=pc)

    return cat_fname


def do_grb_webins(args, ebins, tstart, dt, bkg_dpi, incat, pc="NONE"):
    # same as above func but
    # don't need to make the bkg
    # and use ev2dpi_ebins
    # add args to batcelldetect to take in a cat
    # and do carry sources

    aux_dir = os.path.join(args.obsid, "auxil")
    attfile = os.path.join(
        aux_dir, [fname for fname in os.listdir(aux_dir) if "pat" in fname][0]
    )
    obsid_dir = args.obsid

    tstop = tstart + dt

    dpif = "dpi_ebins_%.3f_%.3f_.dpi" % (tstart, tstop)
    dpi_fname = os.path.join(args.savedir, args.obsid, dpif)

    ev2dpi_ebins(args.evf, dpi_fname, tstart, tstop, ebins, args.dmask)

    img_fname = os.path.join(
        args.savedir, obsid_dir, "sky_ebins_%.3f_%.3f_.img" % (tstart, tstop)
    )

    mk_sky_img(
        dpi_fname,
        img_fname,
        args.dmask,
        attfile,
        bkg_file=bkg_dpi,
        ovrsmp=args.oversamp,
    )

    cat_fname = os.path.join(
        args.savedir, obsid_dir, "cat_ebins_%.3f_%.3f_.fits" % (tstart, tstop)
    )

    # sig_fname = os.path.join(args.savedir, obsid_dir, 'sig_ebins_%.3f_%.3f_.img' %(tstart, tstop))

    run_batcelldetect(img_fname, cat_fname, ovrsmp=args.oversamp, incat=incat, pcode=pc)


def do_search(arg_dict):
    full_cat = std_grb(
        arg_dict["args"],
        arg_dict["tstart"],
        arg_dict["dt"],
        arg_dict["bkg_full_dpi"],
        pc=arg_dict["pc_fname"],
    )

    # do_grb_webins(arg_dict['args'], arg_dict['ebins'], arg_dict['tstart'], arg_dict['dt'],\
    #                arg_dict['bkg_ebins_dpi'], full_cat, pc=arg_dict['pc_fname'])

    return


def do_search_mp(tstarts, dts, args, bkg_full_dpi, pc_fname):
    arg_dict_keys = [
        "tstart",
        "dt",
        "args",
        "bkg_full_dpi",
        "bkg_ebins_dpi",
        "ebins",
        "pc_fname",
    ]
    args_dict_list = []
    for i in range(len(tstarts)):
        arg_dict = {
            "tstart": tstarts[i],
            "dt": dts[i],
            "args": args,
            "bkg_full_dpi": bkg_full_dpi,
            "pc_fname": pc_fname,
        }
        args_dict_list.append(arg_dict)

    t0 = time.time()

    if args.nproc == 1:
        for i in range(len(args_dict_list)):
            do_search(args_dict_list[i])
    else:
        p = mp.Pool(args.nproc)

        print("Starting %d procs" % (args.nproc))

        p.map_async(do_search, args_dict_list).get()

        p.close()
        p.join()

    print("Done with all searches")
    print(
        "Took %.2f seconds, %.2f minutes"
        % (time.time() - t0, (time.time() - t0) / 60.0)
    )


def main(args):
    t_0 = time.time()

    # ebins = "25-100,15-25,100-150"

    obs_save_dir = os.path.join(args.savedir, args.obsid)

    if not os.path.isdir(obs_save_dir):
        print("Directory, %s doesn't exist" % (obs_save_dir))
        print("So making it")
        os.makedirs(obs_save_dir)

    bkg_full_dpi = "NONE"

    if not args.nobkg:
        print("Making full energy bkg dpi")
        bkg_full_dpi = do_bkg(args)
        print("Made ", bkg_full_dpi)

    pc_fname = os.path.join(obs_save_dir, "pc.img")

    aux_dir = os.path.join(args.obsid, "auxil")
    try:
        attfile = os.path.join(
            aux_dir, [fname for fname in os.listdir(aux_dir) if "pat" in fname][0]
        )
    except:
        attfile = os.path.join(
            aux_dir, [fname for fname in os.listdir(aux_dir) if "sat" in fname][0]
        )

    mk_pc_img(args.dmask, pc_fname, args.dmask, attfile, ovrsmp=args.oversamp)

    ev_tab = Table.read(args.evf)

    min_time = np.min(ev_tab["TIME"])
    max_time = np.max(ev_tab["TIME"])

    print("min time: ", min_time)
    print("max time: ", max_time)
    print("total time: ", max_time - min_time)

    if args.tabfname == "all":
        tstarts, tstops = get_times_all(min_time, max_time)
    elif "full" in args.tabfname:
        tstarts, tstops = np.array([min_time]), np.array([max_time])
    else:
        tstarts, tstops = get_seed_times(args)
    dts = tstops - tstarts

    # t0 = args.t0
    # t_bins = np.arange(-15.36, 15.36, args.dt) + t0
    # t_bins = np.arange(-1.5, 1.5, .5) + t0

    ntbins = len(tstarts)

    print("Done with doing bkg")
    print("Took %.3f seconds" % (time.time() - t_0))

    print(ntbins, " time bins to search")
    print("Now setting up search")

    do_search_mp(tstarts, dts, args, bkg_full_dpi, pc_fname)


if __name__ == "__main__":
    args = cli()

    main(args)
