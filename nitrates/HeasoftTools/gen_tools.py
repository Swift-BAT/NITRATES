import os
import sys

# import argparse
import subprocess

# import multiprocessing as mp
import traceback
import signal
import time
import sys
import shutil
import urllib.request, urllib.error, urllib.parse
from astropy.table import Table
import os

sys.path.append("../")
from ..config import ftool_wrap

ftool_sh = ftool_wrap


def run_ftool(ftool, arg_list):
    cmd_list = [ftool_sh]
    # cmd_list = []
    cmd_list.append(ftool)
    # cmd_list.append(arg_list)
    cmd_list += arg_list
    print(cmd_list)
    subprocess.call(cmd_list)


# def run_ftool2(ftool, arg_list):
#
#    ftool_path = os.path.join(sw_bat_heasoft_dir, ftool, ftool)
#
#    cmd_list = [ftool_sh]
#    cmd_list.append(ftool_path)
#    #cmd_list.append(arg_list)
#    cmd_list += arg_list
#    print(cmd_list)
#    subprocess.call(cmd_list)


def run_ftool_mp(arg_list):
    cmd_list = [ftool_sh]
    # cmd_list.append(ftool)
    # cmd_list.append(arg_list)
    cmd_list += arg_list
    # print cmd_list
    subprocess.call(cmd_list)
    # except Exception as E:
    #    print E
    #    print traceback.format_exc()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class cd:
    """
    Context manager for changing the current
    working directory
    """

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
        print("changed to dir: ", self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
        print("changed to dir: ", self.savedPath)


def ftool_mp(nproc, arg_lists, chunksize=10):
    print("Opening pool of %d workers" % (nproc))
    t0 = time.time()

    p = mp.Pool(nproc, init_worker)

    print(os.getpid())
    print("active children: ", mp.active_children())

    try:
        p.map(run_ftool_mp, arg_lists, chunksize=chunksize)
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


def mk_yr_mon(year, month):
    if month > 12:
        year += 1
        month -= 12
    if month < 1:
        year -= 1
        month += 12
    if month < 10:
        mon = "0" + str(month)
    else:
        mon = str(month)
    yr = str(year)
    yr_mon = yr + "_" + mon
    return yr_mon


def t_poly(Tstart, C0, C1, C2, t):
    T1 = (t - Tstart) / 86400.0
    Tcorr = (C0 + C1 * T1 + C2 * T1 * T1) * 1e-6
    return Tcorr


def get_t_offset(tab, t):
    bl = (t > tab["TSTART"]) & (t < tab["TSTOP"])
    t_off = -1.0 * t_poly(
        tab["TSTART"][bl], tab["C0"][bl], tab["C1"][bl], tab["C2"][bl], t
    )
    return t_off


def write_fnames(dname, ext, ident):
    fnames = [fname for fname in os.listdir(dname) if ident in fname]

    list_name = "files.list"

    f = open(dname + list_name, "w")

    for fname in fnames:
        f.write(fname + ext + "\n")

    f.close()


def mk_url(yr_mon, obsid, kind):
    if kind in ["sat", "pat"]:
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/"
            + yr_mon
            + "/"
            + obsid
            + "/auxil/sw"
            + obsid
            + kind
            + ".fits.gz"
        )
    elif kind == "mkf":
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/swift/data/trend/"
            + yr_mon
            + "/misc/mkfilter/"
            + "sw"
            + obsid
            + "s.mkf.gz"
        )
    elif kind == "bittb":
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/swift/data/trend/"
            + yr_mon
            + "/bat/btbimgtr/"
            + "sw"
            + obsid
            + "bittb.fits.gz"
        )
    elif kind == "brt":
        url = (
            "http://heasarc.gsfc.nasa.gov/FTP/swift/data/trend/"
            + yr_mon
            + "/bat/btbratetr/"
            + "sw"
            + obsid
            + "brttbrp.fits.gz"
        )
    return url


def down_file(yr_mon, obsid, kind, m_dir):
    kinds = ["pat", "sat", "bittb", "mkf", "brt"]
    if kind not in kinds:
        print("bad kind")
        return
    url = mk_url(yr_mon, obsid, kind)
    fn = url.split("/")[-1]
    fname = os.path.join(m_dir, yr_mon, obsid, fn)

    try:
        resp = urllib.request.urlopen(url)
        data = resp.read()
        print("Downloading " + url + " to " + fname)
        with open(fname, "w") as f:
            f.write(data)
        print("Download success")
    except Exception as e:
        print(e)
        print(obsid, fname)
        if kind == "pat":
            try:
                print("Trying sat")
                kind = "sat"
                url = mk_url(yr_mon, obsid, kind)
                fn = url.split("/")[-1]
                fname = os.path.join(m_dir, yr_mon, obsid, fn)
                resp = urllib.request.urlopen(url)
                data = resp.read()
                print("Downloading " + url + " to " + fname)
                with open(fname, "w") as f:
                    f.write(data)
                print("Download success")
            except Exception as e:
                print(e)
                print(obsid, fname)

    return


def open_tables(fnames, dname, ks="all", k_nos=None):
    kinds = [
        "att",
        "acs",
        "mkf",
        "bit",
        "brt",
        "F1_gti",
        "F2_gti",
        "saa_gti",
        "val_gti",
    ]
    tab_dict = {}

    if not ks == "all":
        if type(ks) == list:
            kinds = [k for k in kinds if k in ks]
        else:
            kinds = [ks]

    for kind in kinds:
        if k_nos is not None:
            if kind in k_nos:
                continue

        if kind == "acs":
            hdu = 2
            k = "at."
        elif kind == "att":
            hdu = 1
            k = "at."
        else:
            hdu = 1
            k = kind

        fname = [fname for fname in fnames if k in fname]

        if len(fname) < 1:
            continue
        elif len(fname) > 1:
            if kind == "att" or kind == "acs":
                f_n = [fn for fn in fname if "sat" in fn][0]
                fname = os.path.join(dname, f_n)
        else:
            fname = os.path.join(dname, fname[0])

        try:
            if kind == "mkf":
                tab = fits.open(fname)[1].data
            else:
                tab = Table.read(fname, hdu=hdu)
            tab_dict[kind] = tab
        except Exception as E:
            print(E)
            print(kind)

    return tab_dict
