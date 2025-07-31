# import ..config

import numpy as np
import pandas as pd
from scipy import optimize, stats, interpolate
from astropy.io import fits
from astropy.wcs import WCS
import healpy as hp
import os
import argparse
import logging, traceback, time

from ..config import (
    EBINS0,
    EBINS1,
    solid_angle_dpi_fname,
    fp_dir,
    rt_dir,
    rates_resp_out_dir,
    rates_resp_dir,
)
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import (
    get_info_tab,
    guess_dbfname,
    get_files_tab,
    get_twinds_tab,
    get_rate_fits_tab,
)
from ..lib.wcs_funcs import world2val
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..models.models import (
    Bkg_Model_wSA,
    Bkg_Model_wFlatA,
    CompoundModel,
    Point_Source_Model_Binned_Rates,
)
from ..response.ray_trace_funcs import RayTraces, FootPrints


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--Njobs", type=int, help="Total number of jobs", default=16)
    parser.add_argument("--job_id", type=int, help="Which job this is", default=-1)
    parser.add_argument(
        "--pix_fname",
        type=str,
        help="Name of the file with good imx/y coordinates",
        default="good_pix2scan.npy",
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--pcfname",
        type=str,
        help="Name of the partial coding image",
        default="pc_2.img",
    )
    parser.add_argument(
        "--time_seed_fname",
        type=str,
        help="Name of the time seed file",
        default="time_seeds.csv",
    )
    parser.add_argument(
        "--min_pc", type=float, help="Min partical coding fraction to use", default=0.1
    )
    args = parser.parse_args()
    return args


def parse_bkg_csv(bkg_fname, solid_angle_dpi, ebins0, ebins1, bl_dmask, rt_dir):
    bkg_df = pd.read_csv(bkg_fname)
    col_names = bkg_df.columns
    nebins = len(ebins0)

    PSnames = []
    for name in col_names:
        if "_imx" in name:
            PSnames.append(name.split("_")[0])
    print(PSnames)
    Nsrcs = len(PSnames)
    if Nsrcs > 0:
        bkg_name = "Background_"
    else:
        bkg_name = ""

    bkg_mod = Bkg_Model_wFlatA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)

    ps_mods = []

    if Nsrcs > 0:
        rt_obj = RayTraces(rt_dir)
        for i in range(Nsrcs):
            name = PSnames[i]
            imx = bkg_df[name + "_imx"][0]
            imy = bkg_df[name + "_imy"][0]
            mod = Point_Source_Model_Binned_Rates(
                imx,
                imy,
                0.1,
                [ebins0, ebins1],
                rt_obj,
                bl_dmask,
                use_deriv=True,
                name=name,
            )
            ps_mods.append(mod)

    return bkg_df, bkg_name, PSnames, bkg_mod, ps_mods


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot((imx1 - imx0), (imy1 - imy0))


def get_rates_resp_arr(drm_dir):
    fnames = np.array([fn for fn in os.listdir(drm_dir) if "resp_" in fn])
    imxs = np.array([float(fn.split("_")[2]) for fn in fnames])
    imys = np.array([float(fn.split("_")[4]) for fn in fnames])

    dtp = [("imx", np.float64), ("imy", np.float64), ("fname", fnames.dtype)]
    drm_arr = np.empty(len(imxs), dtype=dtp)
    drm_arr["imx"] = imxs
    drm_arr["imy"] = imys
    drm_arr["fname"] = fnames
    return drm_arr


def get_rates_resp_arr_outFoV(drm_dir):
    fnames = np.array([fn for fn in os.listdir(drm_dir) if "resp_" in fn])
    hp_inds = np.array([int(fn.split("_")[2]) for fn in fnames])
    Nside = 2**2
    phis, lats = hp.pix2ang(Nside, hp_inds, nest=True, lonlat=True)
    thetas = 90.0 - lats

    dtp = [
        ("hp_ind", np.int64),
        ("theta", np.float64),
        ("phi", np.float64),
        ("fname", fnames.dtype),
    ]
    drm_arr = np.empty(len(thetas), dtype=dtp)
    drm_arr["theta"] = thetas
    drm_arr["phi"] = phis
    drm_arr["hp_ind"] = hp_inds
    drm_arr["fname"] = fnames
    return drm_arr


class Rates_Resp(object):
    def __init__(self, fname, bl_dmask):
        self.fname = fname
        self.bl_dmask = bl_dmask
        self.read_rates_npz()

    def read_rates_npz(self):
        self.npz_file = np.load(self.fname)
        self.mask_in = self.npz_file["mask_in"]
        self.mask_out = self.npz_file["mask_out"]
        self.ndets_in = np.sum(self.mask_in & self.bl_dmask)
        self.ndets_out = np.sum(self.mask_out & self.bl_dmask)

        print(self.npz_file.files)
        print(self.npz_file["RatesIn"].shape)
        self.nebins = self.npz_file["RatesIn"].shape[1]

        self.rates_in_intps = []
        self.rates_out_intps = []
        for j in range(self.nebins):
            try:
                self.rates_in_intps.append(
                    interpolate.RectBivariateSpline(
                        np.unique(self.npz_file["gamma"]),
                        np.unique(np.log10(self.npz_file["Epeak"])),
                        np.log10(self.npz_file["RatesIn"][:, j]).reshape(21, 23),
                        s=1e-3,
                    )
                )
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                self.rates_in_intps.append(
                    interpolate.RectBivariateSpline(
                        np.unique(self.npz_file["gamma"]),
                        np.unique(np.log10(self.npz_file["Epeak"])),
                        np.log10(self.npz_file["RatesIn"][:, j]).reshape(21, 23),
                        s=0.0,
                    )
                )
            if np.all(self.npz_file["RatesOut"][:, j] > 0.0):
                s = 1e-3
            else:
                s = 0.0
            try:
                self.rates_out_intps.append(
                    interpolate.RectBivariateSpline(
                        np.unique(self.npz_file["gamma"]),
                        np.unique(np.log10(self.npz_file["Epeak"])),
                        np.log10(self.npz_file["RatesOut"][:, j]).reshape(21, 23),
                        s=s,
                    )
                )
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                try:
                    self.rates_out_intps.append(
                        interpolate.RectBivariateSpline(
                            np.unique(self.npz_file["gamma"]),
                            np.unique(np.log10(self.npz_file["Epeak"])),
                            np.log10(self.npz_file["RatesOut"][:, j]).reshape(21, 23),
                            s=0.0,
                        )
                    )
                except Exception as E:
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    self.rates_out_intps.append(
                        interpolate.RectBivariateSpline(
                            np.unique(self.npz_file["gamma"]),
                            np.unique(np.log10(self.npz_file["Epeak"])),
                            np.log10(self.npz_file["RatesOut"][:, j]).reshape(11, 17),
                            s=0.0,
                        )
                    )

    def get_rates_in_out(self, A, Epeak, gamma):
        rates_in = np.zeros(self.nebins)
        rates_out = np.zeros(self.nebins)
        l10Ep = np.log10(Epeak)

        for j in range(self.nebins):
            rates_in[j] = self.rates_in_intps[j](gamma, l10Ep)[0, 0]
            rates_out[j] = self.rates_out_intps[j](gamma, l10Ep)[0, 0]

        return A * self.ndets_in * (10.0**rates_in), A * self.ndets_out * (
            10.0**rates_out
        )


def gauss_sig_bkg_nllh(cnts, nsig, nbkg, bkg_err, sys_err=0.0):
    sigma2 = nbkg + nsig + bkg_err**2 + (sys_err * nsig) ** 2

    N_sub_bkg = cnts - nbkg

    nllh = -1 * np.sum(stats.norm.logpdf(N_sub_bkg - nsig, scale=np.sqrt(sigma2)))

    return nllh


def gauss_nllh2min(theta, data_counts, bkg_counts, bkg_err, cnts_intp):
    Nsig = 10.0 ** theta[0]
    gamma = theta[1]

    Nsigs = Nsig * cnts_intp(gamma)

    return gauss_sig_bkg_nllh(data_counts, Nsigs, bkg_counts, bkg_err)


def gauss_nllh2min_2regs(
    theta,
    data_counts,
    bkg_counts,
    bkg_err,
    data_counts2,
    bkg_counts2,
    bkg_err2,
    rates_resp_obj,
    dt,
    Ndets_in,
    Ndets_out,
):
    A = 10.0 ** theta[0]
    Epeak = 10.0 ** theta[1]
    gamma = theta[2]

    Rates_in, Rates_out = rates_resp_obj.get_rates_in_out(A, Epeak, gamma)

    sys_err = 0.05 * np.ones_like(bkg_counts)

    nllh0 = gauss_sig_bkg_nllh(
        data_counts, Rates_in * dt, bkg_counts, bkg_err, sys_err=sys_err
    )

    if Ndets_out < 100:
        return nllh0

    #     sys_err2 = np.sqrt(sys_err**2 + (Atrans/4.**2))
    sys_err2 = 0.1 * np.ones_like(bkg_counts)

    nllh1 = gauss_sig_bkg_nllh(
        data_counts2, Rates_out * dt, bkg_counts2, bkg_err2, sys_err=sys_err2
    )

    if Ndets_in < 100:
        return nllh1

    return nllh0 + nllh1


def min_det_in_out_nllh(
    data_counts_in,
    data_counts_out,
    model,
    fp_bl,
    params,
    PSnames,
    bkg_name,
    dt,
    rates_resp_obj,
    Ndets_in,
    Ndets_out,
    solid_ang_dmean0,
    solid_ang_dmean1,
    rt_sums0,
    rt_sums1,
    get_bkg_nllh=False,
):
    # bkg_rate, bkg_rate_err = bkg_obj.get_rate(t)

    nebins = model.nebins
    Nsrcs = len(PSnames)

    bkg_dpis = model.get_rate_dpis(params)
    bkg_cnts0 = np.array([np.sum(dpi[fp_bl]) * dt for dpi in bkg_dpis])
    bkg_cnts1 = np.array([np.sum(dpi[~fp_bl]) * dt for dpi in bkg_dpis])

    bkg_rate_errs = np.array(
        [params["err_" + bkg_name + "bkg_rate_" + str(j)] for j in range(nebins)]
    )

    bkg_flatAs = np.array([params[bkg_name + "flat_" + str(j)] for j in range(nebins)])
    bkg_diffAs = 1.0 - bkg_flatAs
    bkg_err0 = (
        bkg_rate_errs * (bkg_flatAs * Ndets_in + bkg_diffAs * solid_ang_dmean0) * dt
    )
    bkg_err1 = (
        bkg_rate_errs * (bkg_flatAs * Ndets_out + bkg_diffAs * solid_ang_dmean1) * dt
    )

    tot_err02 = np.zeros(nebins)
    tot_err02 += bkg_err0**2
    tot_err12 = np.zeros(nebins)
    tot_err12 += bkg_err1**2

    for i in range(Nsrcs):
        ps_rate_errs = np.array(
            [params["err_" + PSnames[i] + "_rate_" + str(j)] for j in range(nebins)]
        )
        ps_err0 = ps_rate_errs * rt_sums0[i] * dt
        tot_err02 += ps_err0**2
        ps_err1 = ps_rate_errs * rt_sums1[i] * dt
        tot_err12 += ps_err1**2

    # could probably just make these zeros later
    bkg_err0 = np.sqrt(tot_err02 / 100.0)
    bkg_err1 = np.sqrt(tot_err12 / 100.0)

    #     args = (data_counts, bkg_cnts, bkg_err, cnts_intp)
    #     Ndet_frac = Ndets0/float(Ndets0+Ndets1)
    args = (
        data_counts_in,
        bkg_cnts0,
        bkg_err0,
        data_counts_out,
        bkg_cnts1,
        bkg_err1,
        rates_resp_obj,
        dt,
        Ndets_in,
        Ndets_out,
    )

    lowers = [-10.0, 1.05, -0.1]
    uppers = [10.0, 3.0, 2.1]
    bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

    # x0s = [[1.5, .725], [2., 1.105],
    #        [2.25, 1.605], [2.5, 1.995]]
    #     x0s = [[2.2, .725], [3.7, 1.605]]
    x0s = [[-1.5, 2.0, 1.0], [-2.0, 2.5, 0.25], [-1.0, 1.5, 1.6], [-3.0, 2.7, 1.3]]

    #     x0 = [1., 1.]
    ress = []
    nlogls = np.zeros(len(x0s))

    for j, x0 in enumerate(x0s):
        res = optimize.minimize(
            gauss_nllh2min_2regs, x0, args=args, method="L-BFGS-B", bounds=bounds
        )

        #         print res
        ress.append(res)
        nlogls[j] = res.fun

    if np.all(np.isnan(nlogls)):
        best_ind = 0
    else:
        best_ind = np.nanargmin(nlogls)

    bf_A = 10.0 ** ress[best_ind].x[0]
    bf_Epeak = 10.0 ** ress[best_ind].x[1]
    bf_ind = ress[best_ind].x[2]

    if get_bkg_nllh:
        #         bkg_nllh0 = gauss_sig_bkg_nllh(data_counts0, 0., bkg_cnts0, bkg_err0)
        #         bkg_nllh1 = gauss_sig_bkg_nllh(data_counts1, 0., bkg_cnts1, bkg_err1)
        bkg_nllh = gauss_nllh2min_2regs([-10.0, 2.0, 1.5], *args)
        return bf_A, bf_Epeak, bf_ind, nlogls[best_ind], bkg_nllh

    return bf_A, bf_Epeak, bf_ind, nlogls[best_ind]


class rates_fp_llh(object):
    def __init__(
        self,
        imxs,
        imys,
        ev_data,
        twind_tab,
        ebins0,
        ebins1,
        bl_dmask,
        rates_resp_dir,
        bkg_fname,
        solid_ang_dpi,
        rt_dir,
    ):
        (
            self.bkg_df,
            self.bkg_name,
            self.PSnames,
            self.bkg_mod,
            self.ps_mods,
        ) = parse_bkg_csv(bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir)
        self.Nsrcs = len(self.PSnames)
        if self.Nsrcs < 1:
            self.model = self.bkg_mod
            self.ray_traces = []
        else:
            self.model_list = [self.bkg_mod]
            self.model_list += self.ps_mods
            self.model = CompoundModel(self.model_list)
            self.ray_traces = [
                self.ps_mods[j].get_rt(self.ps_mods[j].imx, self.ps_mods[j].imy)
                for j in range(self.Nsrcs)
            ]

        #         self.Nfps = len(imxs)
        #         self.fp_dir = fp_dir
        #         self.fp_obj = FootPrints(fp_dir)

        self.Nim_pnts = len(imxs)

        self.imxs = imxs
        self.imys = imys
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(self.ebins0)
        self.twind_tab = twind_tab
        self.exp_groups = self.twind_tab.groupby("duration")
        self.Ndurs = len(self.exp_groups)

        # self.t_bins0 = t_bins0
        # self.t_bins1 = t_bins1
        self.ev_data = ev_data
        self.bl_dmask = bl_dmask
        #         self.ind_ax = np.linspace(-.5, 2.5, 20*3+1)
        #         self.drm_obj = DRMs(drm_dir)
        self.Ndets_tot = np.sum(bl_dmask)
        # self.bkg_mod = bkg_mod
        # self.bkg_df = bkg_df

        self.solid_ang_dpi = solid_ang_dpi
        self.solid_angle_tot = np.sum(solid_ang_dpi[bl_dmask])
        self.solid_angle_mean = np.mean(solid_ang_dpi[bl_dmask])
        self.solid_angs_dmean = self.solid_ang_dpi[bl_dmask] / self.solid_angle_mean
        self.solid_angs_dmean_sum = np.sum(self.solid_angs_dmean)

        self.rates_resp_dir = rates_resp_dir
        self.resp_arr = get_rates_resp_arr(self.rates_resp_dir)

        # self.Atrans = np.array([.05, .06, .07, .08, .09, .1])

    #         self.Atrans = np.array([.05, .09, .125, .25, .4, .65])

    def get_resp_fname(self, imx, imy):
        im_dists = np.hypot((imx - self.resp_arr["imx"]), (imy - self.resp_arr["imy"]))
        min_ind = np.argmin(im_dists)
        return os.path.join(self.rates_resp_dir, self.resp_arr[min_ind]["fname"])

    def set_rates_resp(self, imx, imy):
        fname = self.get_resp_fname(imx, imy)
        self.resp_obj = Rates_Resp(fname, self.bl_dmask)

    def get_fp_vals(self):
        self.fp_bls = []
        self.fpbls = []
        self.ndets = []
        self.solid_angs = []
        self.rt_sums = []
        self.rt_sums1 = []
        for i in range(self.Nfps):
            #             fp_ind = np.argmin(im_dist(self.fp_arr['imx'], self.fp_arr['imy'],
            #                                self.imxs[i], self.imys[i]))
            #             fp = np.load(os.path.join(self.fp_dir,\
            #                     self.fp_arr[fp_ind]['fname']))
            fp = self.fp_obj.get_fp(self.imxs[i], self.imys[i])
            self.fp_bls.append(mask_detxy(fp, self.ev_data))
            self.ndets.append(np.sum(self.bl_dmask & (fp == 1)))
            #             fpbl = (self.bl_dmask&(fp==1))
            fpbl = fp[self.bl_dmask] == 1
            self.fpbls.append(fpbl)
            self.solid_angs.append(np.sum(self.solid_angs_dmean[fpbl]))
            rtsums = []
            rtsums1 = []
            for j in range(self.Nsrcs):
                rtsums.append(np.sum(self.ray_traces[j][fpbl]))
                rtsums1.append(np.sum(self.ray_traces[j][~fpbl]))
            self.rt_sums.append(rtsums)
            self.rt_sums1.append(rtsums1)

    #             self.ndets.append(np.sum(self.bl_dmask&(fp==0)))

    def get_cnts_tbins_ebins_fps(self, dur, ev_data_in, ev_data_out):
        # gaps in twinds might mess this up

        df_twind = self.exp_groups.get_group(dur)
        tbins0 = df_twind["time"].values
        tbins1 = df_twind["time_end"].values

        # tbins0 = self.t_bins0[dur_ind]
        # tbins1 = self.t_bins1[dur_ind]
        ntbins = len(tbins0)
        #         tbin_size = tbins1[0] - tbins0[0]
        #         tstep = tbins0[1] - tbins0[0]
        #         tfreq = int(np.rint(tbin_size/tstep))
        #         t_add = [tbins0[-1] + (i+1)*tstep for i in range(tfreq)]
        #         tbins = np.append(tbins0, t_add)
        #         ebins = np.append(self.ebins0, [self.ebins1[-1]])

        self.cnts_tot = get_cnts_from_tbins_ebins(
            self.ev_data, tbins0, tbins1, self.ebins0, self.ebins1
        )

        logging.debug("Done getting cnts_tot")

        self.cnts_fpte = np.zeros((self.Nfps, ntbins, self.nebins))

        for ii in range(self.Nfps):
            fp_bl = self.fp_bls[ii]
            self.cnts_fpte[ii] = get_cnts_from_tbins_ebins(
                self.ev_data[fp_bl], tbins0, tbins1, self.ebins0, self.ebins1
            )
            logging.debug("Done with %d of %d cnts_fptes" % (ii + 1, self.Nfps))

    def get_drm_stuff(self):
        self.cnts_intps = []
        for i in range(self.Nfps):
            imx = self.imxs[i]
            imy = self.imys[i]
            drm = self.drm_obj.get_drm(imx, imy)
            ebin_ind_edges = get_ebin_ind_edges(drm, self.ebins0, self.ebins1)
            abs_cor = get_abs_cor_rates(imx, imy, drm)
            self.cnts_intps.append(
                get_cnts_intp_obj(self.ind_ax, drm, ebin_ind_edges, abs_cor)
            )

    def run(self):
        t_0 = time.time()
        #         self.get_fp_vals()
        #         self.get_drm_stuff()
        logging.info("Done setting up footprints and drm stuff")
        logging.info("Took %.3f seconds" % (time.time() - t_0))

        res_dicts = []

        for jj in range(self.Nim_pnts):
            imx = self.imxs[jj]
            imy = self.imys[jj]

            try:
                self.set_rates_resp(imx, imy)
            except Exception as E:
                logging.warn(
                    "problem reading npz file for imx, imy %.3f, %.3f" % (imx, imy)
                )
                logging.error(E)
                continue
            fp_bl = self.resp_obj.mask_in[self.bl_dmask]
            fpbl = mask_detxy(self.resp_obj.mask_in, self.ev_data)
            Ndets_in = self.resp_obj.ndets_in
            Ndets_out = self.resp_obj.ndets_out
            ev_data_in = self.ev_data[fpbl]
            ev_data_out = self.ev_data[~fpbl]

            solid_ang = np.sum(self.solid_angs_dmean[fp_bl])
            rt_sums = []
            rt_sums1 = []
            for j in range(self.Nsrcs):
                rt_sums.append(np.sum(self.ray_traces[j][fp_bl]))
                rt_sums1.append(np.sum(self.ray_traces[j][~fp_bl]))

            for ii, exp_group in enumerate(self.exp_groups):
                logging.info("Starting duration size %d of %d" % (ii + 1, self.Ndurs))

                dur = exp_group[0]
                df_twind = exp_group[1]

                tbins0 = df_twind["time"].values
                tbins1 = df_twind["time_end"].values
                timeIDs = df_twind["timeID"].values

                ntbins = len(tbins0)

                logging.debug("ntbins: %d" % (ntbins))
                logging.debug("tbin_size: %.3f" % (dur))

                t_0 = time.time()

                for kk in range(ntbins):
                    ebins = np.append(self.ebins0, [self.ebins1[-1]])
                    tbl = (ev_data_in["TIME"] >= tbins0[kk]) & (
                        ev_data_in["TIME"] < tbins1[kk]
                    )
                    cnts_in = np.histogram(ev_data_in[tbl]["ENERGY"], bins=ebins)[0]
                    tbl = (ev_data_out["TIME"] >= tbins0[kk]) & (
                        ev_data_out["TIME"] < tbins1[kk]
                    )
                    cnts_out = np.histogram(ev_data_out[tbl]["ENERGY"], bins=ebins)[0]

                    res_dict = {
                        "dur": dur,
                        "imx": imx,
                        "imy": imy,
                        "ndets": Ndets_in,
                        "solid_angle": solid_ang,
                        "timeID": timeIDs[kk],
                    }
                    for Nps in range(self.Nsrcs):
                        res_dict[self.PSnames[Nps] + "_rt_sum"] = rt_sums[Nps]

                    res_dict["time"] = tbins0[kk]

                    bkg_ind = np.argmin(
                        np.abs((tbins0[kk] + dur / 2.0) - self.bkg_df["time"])
                    )
                    bkg_row = self.bkg_df.iloc[bkg_ind]

                    (
                        res_dict["A"],
                        res_dict["Epeak"],
                        res_dict["gamma"],
                        res_dict["nllh"],
                        res_dict["bkg_nllh"],
                    ) = min_det_in_out_nllh(
                        cnts_in,
                        cnts_out,
                        self.model,
                        fp_bl,
                        bkg_row,
                        self.PSnames,
                        self.bkg_name,
                        dur,
                        self.resp_obj,
                        Ndets_in,
                        Ndets_out,
                        solid_ang,
                        self.solid_angs_dmean_sum - solid_ang,
                        rt_sums,
                        rt_sums1,
                        get_bkg_nllh=True,
                    )

                    TS = np.sqrt(2.0 * (res_dict["bkg_nllh"] - res_dict["nllh"]))
                    if np.isnan(TS):
                        TS = 0.0
                    res_dict["TS"] = TS
                    res_dicts.append(res_dict)

                logging.info(
                    "Done with %d of %d positions for duration %d of %d"
                    % (jj + 1, self.Nim_pnts, ii + 1, self.Ndurs)
                )

        return res_dicts


class rates_fp_llh_outFoV(object):
    def __init__(
        self,
        hp_inds,
        ev_data,
        twind_tab,
        ebins0,
        ebins1,
        bl_dmask,
        rates_resp_dir,
        bkg_fname,
        solid_ang_dpi,
        rt_dir,
    ):
        (
            self.bkg_df,
            self.bkg_name,
            self.PSnames,
            self.bkg_mod,
            self.ps_mods,
        ) = parse_bkg_csv(bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir)
        self.Nsrcs = len(self.PSnames)
        if self.Nsrcs < 1:
            self.model = self.bkg_mod
            self.ray_traces = []
        else:
            self.model_list = [self.bkg_mod]
            self.model_list += self.ps_mods
            self.model = CompoundModel(self.model_list)
            self.ray_traces = [
                self.ps_mods[j].get_rt(self.ps_mods[j].imx, self.ps_mods[j].imy)
                for j in range(self.Nsrcs)
            ]

        #         self.Nfps = len(imxs)
        #         self.fp_dir = fp_dir
        #         self.fp_obj = FootPrints(fp_dir)

        self.Npnts = len(hp_inds)

        self.hp_inds = hp_inds
        self.Nside = 2**2
        phis, lats = hp.pix2ang(self.Nside, self.hp_inds, lonlat=True, nest=True)
        self.phis = phis
        self.thetas = 90.0 - lats

        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(self.ebins0)
        self.twind_tab = twind_tab
        self.exp_groups = self.twind_tab.groupby("duration")
        self.Ndurs = len(self.exp_groups)

        # self.t_bins0 = t_bins0
        # self.t_bins1 = t_bins1
        self.ev_data = ev_data
        self.bl_dmask = bl_dmask
        #         self.ind_ax = np.linspace(-.5, 2.5, 20*3+1)
        #         self.drm_obj = DRMs(drm_dir)
        self.Ndets_tot = np.sum(bl_dmask)
        # self.bkg_mod = bkg_mod
        # self.bkg_df = bkg_df

        self.solid_ang_dpi = solid_ang_dpi
        self.solid_angle_tot = np.sum(solid_ang_dpi[bl_dmask])
        self.solid_angle_mean = np.mean(solid_ang_dpi[bl_dmask])
        self.solid_angs_dmean = self.solid_ang_dpi[bl_dmask] / self.solid_angle_mean
        self.solid_angs_dmean_sum = np.sum(self.solid_angs_dmean)

        self.rates_resp_dir = rates_resp_dir
        self.resp_arr = get_rates_resp_arr_outFoV(self.rates_resp_dir)

        # self.Atrans = np.array([.05, .06, .07, .08, .09, .1])

    #         self.Atrans = np.array([.05, .09, .125, .25, .4, .65])

    def get_resp_fname(self, hp_ind):
        min_ind = np.where(self.resp_arr["hp_ind"] == hp_ind)[0][0]
        return os.path.join(self.rates_resp_dir, self.resp_arr[min_ind]["fname"])

    def set_rates_resp(self, hp_ind):
        fname = self.get_resp_fname(hp_ind)
        self.resp_obj = Rates_Resp(fname, self.bl_dmask)

    def run(self):
        t_0 = time.time()
        #         self.get_fp_vals()
        #         self.get_drm_stuff()
        logging.info("Done setting up footprints and drm stuff")
        logging.info("Took %.3f seconds" % (time.time() - t_0))

        res_dicts = []

        for jj in range(self.Npnts):
            hp_ind = self.hp_inds[jj]
            theta = self.thetas[jj]
            phi = self.phis[jj]

            try:
                self.set_rates_resp(hp_ind)
            except Exception as E:
                print("problem reading npz file for hp_ind,")
                print(hp_ind)
                logging.error(E)
                logging.error(traceback.format_exc())
                continue
            fp_bl = self.resp_obj.mask_in[self.bl_dmask]
            fpbl = mask_detxy(self.resp_obj.mask_in, self.ev_data)
            Ndets_in = self.resp_obj.ndets_in
            Ndets_out = self.resp_obj.ndets_out
            ev_data_in = self.ev_data[fpbl]
            ev_data_out = self.ev_data[~fpbl]

            solid_ang = np.sum(self.solid_angs_dmean[fp_bl])
            rt_sums = []
            rt_sums1 = []
            for j in range(self.Nsrcs):
                rt_sums.append(np.sum(self.ray_traces[j][fp_bl]))
                rt_sums1.append(np.sum(self.ray_traces[j][~fp_bl]))

            for ii, exp_group in enumerate(self.exp_groups):
                logging.info("Starting duration size %d of %d" % (ii + 1, self.Ndurs))

                dur = exp_group[0]
                df_twind = exp_group[1]

                tbins0 = df_twind["time"].values
                tbins1 = df_twind["time_end"].values
                timeIDs = df_twind["timeID"].values

                ntbins = len(tbins0)
                #                 tbin_size = tbins1[0] - tbins0[0]
                #                 tstep = tbins0[1] - tbins0[0]

                logging.debug("ntbins: %d" % (ntbins))
                logging.debug("tbin_size: %.3f" % (dur))
                #                 logging.debug("tstep: %.3f" %(tstep))

                #                 t_0 = time.time()
                #                 self.get_cnts_tbins_ebins_fps(dur)
                #                 logging.info("Done getting cnts_fpte")
                #                 logging.info("Took %.3f seconds" %(time.time()-t_0))

                t_0 = time.time()

                for kk in range(ntbins):
                    ebins = np.append(self.ebins0, [self.ebins1[-1]])
                    tbl = (ev_data_in["TIME"] >= tbins0[kk]) & (
                        ev_data_in["TIME"] < tbins1[kk]
                    )
                    cnts_in = np.histogram(ev_data_in[tbl]["ENERGY"], bins=ebins)[0]
                    tbl = (ev_data_out["TIME"] >= tbins0[kk]) & (
                        ev_data_out["TIME"] < tbins1[kk]
                    )
                    cnts_out = np.histogram(ev_data_out[tbl]["ENERGY"], bins=ebins)[0]

                    res_dict = {
                        "dur": dur,
                        "theta": theta,
                        "phi": phi,
                        "hp_ind": hp_ind,
                        "ndets": Ndets_in,
                        "solid_angle": solid_ang,
                        "timeID": timeIDs[kk],
                    }
                    for Nps in range(self.Nsrcs):
                        res_dict[self.PSnames[Nps] + "_rt_sum"] = rt_sums[Nps]

                    res_dict["time"] = tbins0[kk]

                    bkg_ind = np.argmin(
                        np.abs((tbins0[kk] + dur / 2.0) - self.bkg_df["time"])
                    )
                    bkg_row = self.bkg_df.iloc[bkg_ind]
                    #                     bkg_diffuse = np.array([bkg_row['diffuse_'+str(i)] for i\
                    #                                     in range(self.nebins)])
                    #                     bkg_flat = np.array([bkg_row['flat_'+str(i)] for i\
                    #                                     in range(self.nebins)])

                    #                     res_dict['Nsig'], res_dict['Plaw_Ind'], res_dict['nllh'],\
                    #                     res_dict['bkg_nllh'] = min_det_fp_nllh(cnts_per_tbin[kk],\
                    #                                self.model, fpbl, bkg_row, self.PSnames, self.bkg_name,\
                    #                                 tbin_size,\
                    #                                 cnts_intp, Ndets, solid_ang, rt_sums,\
                    #                                 get_bkg_nllh=True)

                    (
                        res_dict["A"],
                        res_dict["Epeak"],
                        res_dict["gamma"],
                        res_dict["nllh"],
                        res_dict["bkg_nllh"],
                    ) = min_det_in_out_nllh(
                        cnts_in,
                        cnts_out,
                        self.model,
                        fp_bl,
                        bkg_row,
                        self.PSnames,
                        self.bkg_name,
                        dur,
                        self.resp_obj,
                        Ndets_in,
                        Ndets_out,
                        solid_ang,
                        self.solid_angs_dmean_sum - solid_ang,
                        rt_sums,
                        rt_sums1,
                        get_bkg_nllh=True,
                    )

                    #                     res_dict['Nsig'], res_dict['Plaw_Ind'], res_dict['nllh'], res_dict['bkg_nllh'] =\
                    #                                     min_det_fp_nllh2(cnts_per_tbin[kk], cnts_per_tbin1[kk],\
                    #                                                             self.model, fpbl,\
                    #                                     bkg_row, self.PSnames, self.bkg_name,\
                    #                                     tbin_size, cnts_intp,\
                    #                                     Ndets, self.Ndets_tot-Ndets, solid_ang,\
                    #                                     self.solid_angs_dmean_sum-solid_ang,\
                    #                                     rt_sums, rt_sums1, self.Atrans,\
                    #                                     get_bkg_nllh=True)

                    TS = np.sqrt(2.0 * (res_dict["bkg_nllh"] - res_dict["nllh"]))
                    if np.isnan(TS):
                        TS = 0.0
                    res_dict["TS"] = TS
                    res_dicts.append(res_dict)

                logging.info(
                    "Done with %d of %d positions for duration %d of %d"
                    % (jj + 1, self.Npnts, ii + 1, self.Ndurs)
                )

        return res_dicts


def main(args):
    logging.basicConfig(
        filename="rates_llh_analysis_%d.log" % (args.job_id),
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB")
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    trigtime = info_tab["trigtimeMET"][0]

    drm_dir = files_tab["drmDir"][0]
    rt_dir = files_tab["rtDir"][0]

    evfname = files_tab["evfname"][0]
    dmfname = files_tab["detmask"][0]
    ev_data = fits.open(evfname, memmap=False)[1].data
    logging.debug("Opened up event file")
    dmask = fits.open(dmfname)[0].data
    bl_dmask = dmask == 0
    logging.debug("Opened up dmask file")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # probably get times from twind table

    if args.time_seed_fname is None:
        twind_df = get_twinds_tab(conn)
    else:
        twind_df = pd.read_csv(args.time_seed_fname)
        twind_df["time_end"] = twind_df["time"] + twind_df["duration"]

    logging.info("Got TimeWindows table")

    logging.info("Getting bkg estimation from file")

    # rate_fits_df = get_rate_fits_tab(conn)
    while True:
        try:
            bkg_fits_df = pd.read_csv(args.bkg_fname)
            break
        except:
            time.sleep(10.0)
    # bkg_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)
    logging.info("Got bkg estimation")

    min_bin_size = np.min(twind_df["duration"])

    logging.info("Smallest duration to test is %.3fs" % (min_bin_size))

    exp_groups = twind_df.groupby("duration")

    nexps = len(exp_groups)

    # rates_resp_dir = '/gpfs/scratch/jjd330/bat_data/rates_resps/'
    rate_resp_arr = get_rates_resp_arr(rates_resp_dir)

    imxs = rate_resp_arr["imx"]
    imys = rate_resp_arr["imy"]

    # rates_resp_out_dir = '/gpfs/scratch/jjd330/bat_data/rates_resps_outFoV2/'
    rate_resp_out_arr = get_rates_resp_arr_outFoV(rates_resp_out_dir)

    hp_inds = rate_resp_out_arr["hp_ind"]

    solid_angle_dpi = np.load(solid_angle_dpi_fname)

    try:
        PC = fits.open(args.pcfname)[0]
        pc = PC.data
        w_t = WCS(PC.header, key="T")

        pcs = world2val(w_t, pc, imxs, imys)
        logging.info("min, max pcs: %.4f, %.4f" % (np.min(pcs), np.max(pcs)))
        min_pc = max(args.min_pc - 0.25, 0.00499)
        logging.info("min_pc: %.4f" % (min_pc))
        logging.info("sum(pcs>min_pc): %d" % (np.sum(pcs > min_pc)))
        pc_bl = pcs > min_pc
        imxs = imxs[pc_bl]
        imys = imys[pc_bl]
    except Exception as E:
        logging.warn("Couldn't use PC img")
        logging.error(E)
        logging.error(traceback.format_exc())
    # Should add a thing like this for the out FoV hp_inds

    # try:
    #     good_pix = np.load(args.pix_fname)
    #     im_dists = np.zeros_like(imxs)
    #     if len(good_pix) < 1:
    #         logging.info("pix2scan file is there are 0 pixels to scan")
    #         # logging.info("Exiting")
    #         # return
    #     for i in range(len(imxs)):
    #         im_dists[i] = np.min(im_dist(imxs[i], imys[i],\
    #                         good_pix['imx'], good_pix['imy']))
    #     bl = (im_dists<.3)
    #     imxs = imxs[bl]
    #     imys = imys[bl]
    # except Exception as E:
    #     logging.error(E)
    #     logging.warning("Trouble reading the pix2scan file")
    #     logging.info("Using whole FoV")

    Nim_pnts = len(imxs)
    Nhp_pnts = len(hp_inds)
    Ntot_pnts = Nim_pnts + Nhp_pnts
    logging.info("%d total grid points" % (Ntot_pnts))
    Nper_job = 1 + int(Ntot_pnts / float(args.Njobs))
    if args.job_id > -1:
        i0 = args.job_id * Nper_job
        i1 = i0 + Nper_job
        if i0 < Nim_pnts:
            imxs = imxs[i0:i1]
            imys = imys[i0:i1]
            logging.info("%d grid points to do" % (len(imxs)))
            rates_llh_obj = rates_fp_llh(
                imxs,
                imys,
                ev_data,
                twind_df,
                ebins0,
                ebins1,
                bl_dmask,
                rates_resp_dir,
                args.bkg_fname,
                solid_angle_dpi,
                rt_dir,
            )
            save_fname = "rates_llh_res_%d_.csv" % (args.job_id)

        elif i0 >= Nim_pnts:
            i0_ = Nper_job * (int(i0 - Nim_pnts) // Nper_job)
            i1_ = i0_ + Nper_job
            hp_inds = hp_inds[i0_:i1_]
            logging.info("hp_inds: ")
            logging.info(hp_inds)
            logging.info("%d grid points to do" % (len(hp_inds)))
            rates_llh_obj = rates_fp_llh_outFoV(
                hp_inds,
                ev_data,
                twind_df,
                ebins0,
                ebins1,
                bl_dmask,
                rates_resp_out_dir,
                args.bkg_fname,
                solid_angle_dpi,
                rt_dir,
            )
            save_fname = "rates_llh_out_res_%d_.csv" % (args.job_id)

    # bkg_mod = Bkg_Model_wSA(bl_dmask, solid_angle_dpi, nebins)

    # rate_llh_obj = rates_fp_llh(imxs, imys, ev_data, twind_df,\
    #                             ebins0, ebins1, fp_dir,\
    #                             bl_dmask, drm_dir, args.bkg_fname,\
    #                             solid_angle_dpi, rt_dir)

    res_dicts = rates_llh_obj.run()
    logging.info("Done with analysis")
    logging.info("%d results to write" % (len(res_dicts)))

    # append_rate_tab(conn, df_twind, quad_dict['id'], bkg_llh_tbins, llhs, bf_nsigs, bf_inds)
    #
    # logging.info("Appended rate results to DB")

    df = pd.DataFrame(res_dicts)
    logging.info("Done making results into DataFrame")
    # save_fname = 'rates_llh_res_%d_.csv' %(args.job_id)
    df.to_csv(save_fname, index=False)


if __name__ == "__main__":
    args = cli()

    main(args)
