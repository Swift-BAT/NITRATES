import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os
import healpy as hp
import logging, traceback
import sys
import multiprocessing as mp
import argparse

from ..lib.wcs_funcs import world2val


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nproc",
        type=int,
        help="Number of processors to use, 1 for no multi-proc",
        default=1,
    )
    parser.add_argument(
        "--Nside", type=int, help="Nside for mosaiced healpix map", default=2**10
    )
    parser.add_argument(
        "--t0", type=float, help="Min image start time to use (MET (s))", default=0.0
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Duration in secs to use, picks images from t0 to t0+dt",
        default=2.5,
    )
    parser.add_argument(
        "--pc_fname", type=str, help="partial coding file name", default="pc_4.img"
    )
    parser.add_argument(
        "--dname", type=str, help="Directory to look for images", default="."
    )
    args = parser.parse_args()
    return args


def get_sky_and_bkgvar_fnames(dname, t0, t1):
    """
    Finds image and bkgvar image file names in directory, dname and with start time_seeds
    >= t0 and end times <= t1.
    Images need to have specific file names
    "sky_e0_e1_t0_t1_osN_.img"
    and for bkgvar fnames
    "bkgvar_e0_e1_t0_t1_osN_.img"
    """

    sky_fnames = np.array([fname for fname in os.listdir(dname) if fname[:4] == "sky_"])

    t0s = np.array([float(fname.split("_")[3]) for fname in sky_fnames])
    t1s = np.array([float(fname.split("_")[4]) for fname in sky_fnames])

    bkgvar_fnames = np.array(["bkgvar" + fname[3:] for fname in sky_fnames])

    dtp = [
        ("sky_fname", sky_fnames.dtype),
        ("t0", np.float64),
        ("t1", np.float64),
        ("bkgvar_fname", bkgvar_fnames.dtype),
    ]

    bl = (t0s >= t0) & (t1s <= t1)
    arr = np.empty(np.sum(bl), dtype=dtp)

    arr["sky_fname"] = sky_fnames[bl]
    arr["t0"] = t0s[bl]
    arr["t1"] = t1s[bl]
    arr["bkgvar_fname"] = bkgvar_fnames[bl]

    return np.sort(arr, order="t0")


def get_and_add_skymaps(Nside, dname, fname_arr, pc_fname, pc_min=0.01):
    """
    This takes the images in fname_arr and mosaics them onto a healpix
    by summing the images weighted the pixels' bkgvar
    """

    pc_img = fits.open(pc_fname)[0].data

    Nimgs = len(fname_arr)

    sky_maps = np.zeros((Nimgs, hp.nside2npix(Nside))) + np.nan
    bkgvar_maps = np.zeros_like(sky_maps) + np.nan
    pc_maps = np.zeros_like(sky_maps)

    # making the skymaps for each image
    for i in range(Nimgs):
        print("opening files")
        sky_file = fits.open(
            os.path.join(dname, fname_arr[i]["sky_fname"]), lazy_load_hdus=False
        )
        sky_img = np.copy(sky_file[0].data)
        print("making wcs obj")
        w_sky = WCS(sky_file[0].header)
        print("wcs obj made")
        sky_file.close()
        print("closed sky file")
        bkgvar_file = fits.open(os.path.join(dname, fname_arr[i]["bkgvar_fname"]))
        print("opened bkgvar file")
        bkgvar_img = np.copy(bkgvar_file[0].data)
        bkgvar_file.close()
        print("closed bkgvar file")
        ra_mid, dec_mid = w_sky.wcs.crval
        vec_mid = hp.ang2vec(ra_mid, dec_mid, lonlat=True)
        print(vec_mid)
        hp_inds2use = hp.query_disc(Nside, vec_mid, np.deg2rad(70.0))
        hp_ras, hp_decs = hp.pix2ang(Nside, hp_inds2use, lonlat=True)

        print("making maps")
        #     sky_map = np.zeros(hp.nside2npix(Nside)) + np.nan
        sky_maps[i, hp_inds2use] = world2val(w_sky, sky_img, hp_ras, hp_decs)
        print("made sky map")

        #     bkgvar_map = np.zeros(hp.nside2npix(Nside)) + np.nan
        bkgvar_maps[i, hp_inds2use] = world2val(w_sky, bkgvar_img, hp_ras, hp_decs)
        print("made bkgvar map")

        #     pc_map = np.zeros(hp.nside2npix(Nside))
        pc_maps[i, hp_inds2use] = world2val(w_sky, pc_img, hp_ras, hp_decs)
        print("made pc map")

        pcbl = pc_maps[i] < pc_min
        bkgvar_maps[i, pcbl] = np.nan

    #         sky_maps.append(sky_map)
    #         bkgvar_maps.append(bkgvar_map)
    #         pc_maps.append(pc_maps)

    #     sky_maps = np.array(sky_maps)
    #     print "skymaps 2 array"
    #     pc_maps = np.array(pc_maps)
    #     print "pcmaps 2 array"
    #     bkgvar_maps = np.array(bkgvar_maps)
    #     print "bkgvar maps 2 array"

    Nmaps = np.sum(pc_maps >= pc_min, axis=0)
    print(np.shape(Nmaps))
    print(np.min(Nmaps), np.max(Nmaps))

    pc_tot = np.nansum(pc_maps, axis=0)

    # Making the normalized bkgvar weights
    print((np.shape(np.sum(1.0 / bkgvar_maps, axis=0))))
    wt_norm = Nmaps * np.nanmean(1.0 / bkgvar_maps, axis=0)
    wt_maps = (1.0 / bkgvar_maps) / wt_norm
    print((np.shape(wt_maps)))
    wt_sums = np.nansum(wt_maps, axis=0)
    print((wt_sums[(wt_sums > 0)]))

    # Summing the maps with the normalized weights
    summed_sky_map = np.nansum(sky_maps * wt_maps, axis=0)
    # Ignore the pixels with no pc > pc_min
    summed_sky_map[(Nmaps < 1)] = np.nan

    return summed_sky_map, Nmaps, pc_tot


def sky_map2bkg_maps(sky_map, pc_map, pc_min=1e-1, sig_rad=0.5, bkg_rad=2.5):
    """
    Create bkg and bkg_var maps using the same type of sliding anulus
    that batcelldetect uses
    """

    Npix = len(sky_map)
    nside = hp.npix2nside(Npix)
    bl_good = pc_map >= pc_min
    good_hp_inds = np.where(bl_good)
    print(np.shape(good_hp_inds))
    good_hp_inds = good_hp_inds[0]
    print(np.shape(good_hp_inds))
    Npix2use = np.sum(bl_good)
    print(Npix2use)
    all_hp_inds = np.arange(hp.nside2npix(nside), dtype=np.int64)
    hp_map_vecs = hp.pix2vec(nside, all_hp_inds)
    hp_map_vecs = np.swapaxes(np.array(hp_map_vecs), 0, 1)

    bkg_map = np.zeros(Npix)
    bkg_std_map = np.zeros(Npix)

    for hp_ind in good_hp_inds:
        vec = hp_map_vecs[hp_ind]
        sig_pix = hp.query_disc(nside, vec, np.radians(sig_rad))
        bkg_pix = hp.query_disc(nside, vec, np.radians(bkg_rad))
        bkg_bl = ~np.isin(bkg_pix, sig_pix)
        bkg_pix = bkg_pix[bkg_bl]

        bkg_map[hp_ind] = np.nanmean(sky_map[bkg_pix])
        bkg_std_map[hp_ind] = np.nanstd(sky_map[bkg_pix])

    return bkg_map, bkg_std_map


class Worker(mp.Process):
    """
    The worker class for sky_map2bkg_maps_mp
    """

    def __init__(
        self, result_queue, inds2do, hp_map_vecs, sky_map, sig_rad=0.5, bkg_rad=2.5
    ):
        mp.Process.__init__(self)
        self.inds2do = inds2do
        self.hp_map_vecs = hp_map_vecs
        self.sky_map = sky_map
        self.sig_rad = sig_rad
        self.bkg_rad = bkg_rad
        self.Npix = len(sky_map)
        self.Nside = hp.npix2nside(self.Npix)
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name

        bkg_map = np.zeros(self.Npix)
        bkg_std_map = np.zeros(self.Npix)

        for hp_ind in self.inds2do:
            vec = self.hp_map_vecs[hp_ind]
            sig_pix = hp.query_disc(self.Nside, vec, np.radians(self.sig_rad))
            bkg_pix = hp.query_disc(self.Nside, vec, np.radians(self.bkg_rad))
            bkg_bl = ~np.isin(bkg_pix, sig_pix)
            bkg_pix = bkg_pix[bkg_bl]

            bkg_map[hp_ind] = np.nanmean(self.sky_map[bkg_pix])
            bkg_std_map[hp_ind] = np.nanstd(self.sky_map[bkg_pix])

        self.result_queue.put({"bkg_map": bkg_map, "bkg_std_map": bkg_std_map})
        self.result_queue.put(None)

        return


def sky_map2bkg_maps_mp(
    sky_map, pc_map, pc_min=1e-1, sig_rad=0.5, bkg_rad=2.5, Nprocs=4
):
    """
    A multiprocessing version of sky_map2bkg_maps
    """

    Npix = len(sky_map)
    nside = hp.npix2nside(Npix)
    bl_good = pc_map >= pc_min
    good_hp_inds = np.where(bl_good)
    print(np.shape(good_hp_inds))
    good_hp_inds = good_hp_inds[0]
    print(np.shape(good_hp_inds))
    Npix2use = np.sum(bl_good)
    print(Npix2use)
    all_hp_inds = np.arange(hp.nside2npix(nside), dtype=np.int64)
    hp_map_vecs = hp.pix2vec(nside, all_hp_inds)
    hp_map_vecs = np.swapaxes(np.array(hp_map_vecs), 0, 1)

    bkg_map = np.zeros(Npix)
    bkg_std_map = np.zeros(Npix)

    inds_per_proc = 1 + int(float(len(good_hp_inds)) / Nprocs)

    inds_list = []

    for i in range(Nprocs):
        i0 = i * inds_per_proc
        i1 = i0 + inds_per_proc
        inds_list.append(good_hp_inds[i0:i1])

    res_q = mp.Queue()

    Workers = []
    for i in range(Nprocs):
        Workers.append(Worker(res_q, inds_list[i], hp_map_vecs, sky_map))

    for w in Workers:
        w.start()

    res_dicts = []
    Ndone = 0
    while True:
        res = res_q.get()
        if res is None:
            Ndone += 1
            logging.info("%d of %d done" % (Ndone, Nprocs))
            if Ndone >= Nprocs:
                break
        else:
            res_dicts.append(res)
    for w in Workers:
        w.join()

    for res_dict in res_dicts:
        bkg_map += res_dict["bkg_map"]
        bkg_std_map += res_dict["bkg_std_map"]

    return bkg_map, bkg_std_map


def mk_sig_map(sky_map, bkg_map, bkgvar_map):
    """
    snr map from counts/rates map, bkg_map, and bkgvar_map
    """
    return (sky_map - bkg_map) / bkgvar_map


def main(args):
    t1 = args.t0 + args.dt
    # get image file names
    fname_arr = get_sky_and_bkgvar_fnames(args.dname, args.t0, t1)

    # open, mosaic, and sum images
    summed_sky_map, Nmaps, pc_tot = get_and_add_skymaps(
        args.Nside, args.dname, fname_arr, args.pc_fname
    )

    # make bkg and bkg_var/noise maps
    if args.Nproc > 1:
        bkg_map, bkgvar_map = sky_map2bkg_maps_mp(
            summed_sky_map, pc_tot, Nprocs=args.Nproc
        )
    else:
        bkg_map, bkgvar_map = sky_map2bkg_maps(summed_sky_map, pc_tot)

    # make S/N skymap
    sig_map = mk_sig_map(summed_sky_map, bkg_map, bkgvar_map)

    # print out S/N, summed partial coding, and RA, Dec for each
    # pixel with S/N > 6
    print("Summed Partial Coding, S/N, RA, Dec")
    inds_ = np.where((sig_map > 6) & (np.isfinite(sig_map) & (pc_tot > 0.4)))
    for ind in inds_[0]:
        print((pc_tot[ind], sig_map[ind], hp.pix2ang(args.Nside, ind, lonlat=True)))


if __name__ == "__main__":
    args = cli()

    main(args)
