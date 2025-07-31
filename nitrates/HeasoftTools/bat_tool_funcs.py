import os
import subprocess
from ..HeasoftTools.gen_tools import run_ftool
import numpy as np


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


def bateconvert(infile, outfile, calfile):
    ftool = "bateconvert"
    arg_list = [
        "infile=" + infile,
        "calfile=" + calfile,
        "residfile=CALDB",
        "outfile=" + outfile,
        "clobber=YES",
        "pulserfile=CALDB",
        "fltpulserfile=CALDB",
    ]
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


def mk_pc_img(infile, outfile, detmask, attfile, ovrsmp=4, detapp=False):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += [
        "detmask=" + detmask,
        "attitude=" + attfile,
        "pcodemap=YES",
        "clobber=YES",
    ]
    if ovrsmp is not None:
        arg_list += ["oversampx=" + str(ovrsmp), "oversampy=" + str(ovrsmp)]
    if detapp:
        arg_list.append("aperture=CALDB:DETECTION")
    run_ftool(ftool, arg_list)


def mk_sky_img(
    infile,
    outfile,
    detmask,
    attfile,
    bkg_file=None,
    ovrsmp=2,
    sig_map=None,
    bkgvar_map=None,
    detapp=False,
    rebal=True,
):
    ftool = "batfftimage"
    arg_list = [infile, outfile]
    arg_list += [
        "detmask=" + detmask,
        "attitude=" + attfile,
        "oversampx=" + str(ovrsmp),
        "oversampy=" + str(ovrsmp),
        "clobber=YES",
    ]
    if bkg_file is not None:
        arg_list += ["bkgfile=" + bkg_file]
    if bkgvar_map is not None:
        arg_list += ["bkgvarmap=" + bkgvar_map]
    if sig_map is not None:
        arg_list += ["signifmap=" + sig_map]
    if detapp:
        arg_list.append("aperture=CALDB:DETECTION")
    if rebal:
        arg_list.append("rebalance=YES")
    else:
        arg_list.append("rebalance=NO")
    run_ftool(ftool, arg_list)


def run_batcelldetect(
    infile,
    cat_fname,
    snr_thresh=3.5,
    sigmap=None,
    bkgvar=None,
    ovrsmp=2,
    incat="NONE",
    pcode="NONE",
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
    if bkgvar is not None:
        arg_list.append("bkgvarmap=" + bkgvar)
    print(arg_list)
    run_ftool(ftool, arg_list)
