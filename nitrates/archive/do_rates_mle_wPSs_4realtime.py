import numpy as np
import pandas as pd
from scipy import optimize, stats, interpolate
from astropy.io import fits
from astropy.wcs import WCS
import os
import argparse
import logging, traceback, time

from ..config import quad_dicts, EBINS0, EBINS1, drm_quad_dir, solid_angle_dpi_fname
from ..lib.sqlite_funcs import get_conn, append_rate_tab
from ..lib.dbread_funcs import (
    get_info_tab,
    guess_dbfname,
    get_files_tab,
    get_twinds_tab,
    get_rate_fits_tab,
)
from ..analysis_seeds.bkg_rate_estimation import (
    get_quad_rate_objs_from_db,
    rate_obj_from_sqltab,
)
from ..analysis_seeds.mle_rates_for_realtime import (
    do_rate_mle,
    do_rate_mle_mp,
    get_abs_cor_rates,
    get_cnts_intp_obj,
    get_quad_cnts_tbins,
)
from ..lib.counting_and_quad_funcs import get_quad_cnts_tbins_fast, dmask2ndets_perquad
from ..lib.drm_funcs import get_ebin_ind_edges, DRMs
from ..lib.wcs_funcs import world2val
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..models.models import (
    Bkg_Model_wSA,
    Bkg_Model_wFlatA,
    CompoundModel,
    Point_Source_Model_Binned_Rates,
    Model,
)
from ..response.ray_trace_funcs import RayTraces


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument(
        "--fp_dir",
        type=str,
        help="Directory where the detector footprints are",
        default="/storage/work/jjd330/local/bat_data/rtfp_dir_npy/",
    )
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


def get_fp_arr(fp_dir):
    fnames = np.array(os.listdir(fp_dir))
    imxs = np.array([float(fn.split("_")[1]) for fn in fnames])
    imys = np.array([float(fn.split("_")[2][:-4]) for fn in fnames])
    dtp = [("imx", np.float64), ("imy", np.float64), ("fname", fnames.dtype)]
    fp_arr = np.empty(len(imxs), dtype=dtp)
    fp_arr["imx"] = imxs
    fp_arr["imy"] = imys
    fp_arr["fname"] = fnames
    return fp_arr


def gauss_sig_bkg_nllh(cnts, nsig, nbkg, bkg_err):
    sigma2 = nbkg + nsig + bkg_err**2

    N_sub_bkg = cnts - nbkg

    nllh = -1 * np.sum(stats.norm.logpdf(N_sub_bkg - nsig, scale=np.sqrt(sigma2)))

    return nllh


def gauss_nllh2min(theta, data_counts, bkg_counts, bkg_err, cnts_intp):
    Nsig = 10.0 ** theta[0]
    gamma = theta[1]

    Nsigs = Nsig * cnts_intp(gamma)

    return gauss_sig_bkg_nllh(data_counts, Nsigs, bkg_counts, bkg_err)


def min_det_fp_nllh(
    data_counts,
    model,
    fp_bl,
    params,
    PSnames,
    bkg_name,
    dt,
    cnts_intp,
    Ndets,
    solid_ang_dmean,
    rt_sums,
    get_bkg_nllh=False,
):
    # bkg_rate, bkg_rate_err = bkg_obj.get_rate(t)

    nebins = model.nebins
    Nsrcs = len(PSnames)

    bkg_dpis = model.get_rate_dpis(params)
    bkg_cnts = np.array([np.sum(dpi[fp_bl]) * dt for dpi in bkg_dpis])

    bkg_rate_errs = np.array(
        [params["err_" + bkg_name + "bkg_rate_" + str(j)] for j in range(nebins)]
    )

    bkg_flatAs = np.array([params[bkg_name + "flat_" + str(j)] for j in range(nebins)])
    bkg_diffAs = 1.0 - bkg_flatAs
    bkg_err = bkg_rate_errs * (bkg_flatAs * Ndets + bkg_diffAs * solid_ang_dmean) * dt

    tot_err2 = np.zeros(nebins)
    tot_err2 += bkg_err**2
    for i in range(Nsrcs):
        ps_rate_errs = np.array(
            [params["err_" + PSnames[i] + "_rate_" + str(j)] for j in range(nebins)]
        )
        ps_err = ps_rate_errs * rt_sums[i] * dt
        tot_err2 += ps_err**2

    bkg_err = np.sqrt(tot_err2)

    args = (data_counts, bkg_cnts, bkg_err, cnts_intp)

    lowers = [-2.0, 0.25]
    uppers = [10.0, 2.25]
    bounds = optimize.Bounds(np.array(lowers), np.array(uppers))

    #     x0s = [[1., 1.], [2., 1.],
    #            [1., 2.], [2., 2.]]
    x0s = [[1.0, 0.725], [2.0, 1.105], [1.0, 1.605], [2.0, 1.995]]

    #     x0 = [1., 1.]
    ress = []
    nlogls = np.zeros(len(x0s))

    for j, x0 in enumerate(x0s):
        res = optimize.minimize(
            gauss_nllh2min, x0, args=args, method="L-BFGS-B", bounds=bounds
        )

        #         print res
        ress.append(res)
        nlogls[j] = res.fun

    if np.all(np.isnan(nlogls)):
        best_ind = 0
    else:
        best_ind = np.nanargmin(nlogls)

    bf_nsig = 10.0 ** ress[best_ind].x[0]
    # bf_nsig = ress[best_ind].x[0]
    bf_ind = ress[best_ind].x[1]

    if get_bkg_nllh:
        bkg_nllh = gauss_sig_bkg_nllh(data_counts, 0.0, bkg_cnts, bkg_err)
        return bf_nsig, bf_ind, nlogls[best_ind], bkg_nllh

    return bf_nsig, bf_ind, nlogls[best_ind]


class rates_fp_llh(object):
    def __init__(
        self,
        imxs,
        imys,
        ev_data,
        twind_tab,
        ebins0,
        ebins1,
        fp_dir,
        fp_arr,
        bl_dmask,
        drm_dir,
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

        self.Nfps = len(imxs)
        self.fp_arr = fp_arr
        self.fp_dir = fp_dir
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
        self.ind_ax = np.linspace(-0.5, 2.5, 20 * 3 + 1)
        self.drm_obj = DRMs(drm_dir)
        self.Ndets_tot = np.sum(bl_dmask)
        # self.bkg_mod = bkg_mod
        # self.bkg_df = bkg_df

        self.solid_ang_dpi = solid_ang_dpi
        self.solid_angle_tot = np.sum(solid_ang_dpi[bl_dmask])
        self.solid_angle_mean = np.mean(solid_ang_dpi[bl_dmask])
        self.solid_angs_dmean = self.solid_ang_dpi[bl_dmask] / self.solid_angle_mean

    def get_fp_vals(self):
        self.fp_bls = []
        self.fpbls = []
        self.ndets = []
        self.solid_angs = []
        self.rt_sums = []
        for i in range(self.Nfps):
            fp_ind = np.argmin(
                im_dist(
                    self.fp_arr["imx"], self.fp_arr["imy"], self.imxs[i], self.imys[i]
                )
            )
            fp = np.load(os.path.join(self.fp_dir, self.fp_arr[fp_ind]["fname"]))
            self.fp_bls.append(mask_detxy(fp, self.ev_data))
            self.ndets.append(np.sum(self.bl_dmask & (fp == 1)))
            #             fpbl = (self.bl_dmask&(fp==1))
            fpbl = fp[self.bl_dmask] == 1
            self.fpbls.append(fpbl)
            self.solid_angs.append(np.sum(self.solid_angs_dmean[fpbl]))
            rtsums = []
            for j in range(self.Nsrcs):
                rtsums.append(np.sum(self.ray_traces[j][fpbl]))
            self.rt_sums.append(rtsums)

    #             self.ndets.append(np.sum(self.bl_dmask&(fp==0)))

    def get_cnts_tbins_ebins_fps(self, dur):
        # gaps in twinds might mess this up

        df_twind = self.exp_groups.get_group(dur)
        tbins0 = df_twind["time"].values
        tbins1 = df_twind["time_end"].values

        # tbins0 = self.t_bins0[dur_ind]
        # tbins1 = self.t_bins1[dur_ind]
        ntbins = len(tbins0)
        tbin_size = tbins1[0] - tbins0[0]
        tstep = tbins0[1] - tbins0[0]
        tfreq = int(np.rint(tbin_size / tstep))
        t_add = [tbins0[-1] + (i + 1) * tstep for i in range(tfreq)]
        tbins = np.append(tbins0, t_add)
        ebins = np.append(self.ebins0, [self.ebins1[-1]])

        self.cnts_fpte = np.zeros((self.Nfps, ntbins, self.nebins))

        for ii in range(self.Nfps):
            fp_bl = self.fp_bls[ii]
            h = np.histogramdd(
                [self.ev_data["TIME"][fp_bl], self.ev_data["ENERGY"][fp_bl]],
                bins=[tbins, ebins],
            )[0]

            if tfreq <= 1:
                h2 = h
            else:
                h2 = np.zeros((h.shape[0] - (tfreq - 1), h.shape[1]))
                for i in range(tfreq):
                    i0 = i
                    i1 = -tfreq + 1 + i
                    if i1 < 0:
                        h2 += h[i0:i1]
                    else:
                        h2 += h[i0:]
            self.cnts_fpte[ii] = h2

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
        self.get_fp_vals()
        self.get_drm_stuff()
        logging.info("Done setting up footprints and drm stuff")
        logging.info("Took %.3f seconds" % (time.time() - t_0))

        res_dicts = []

        for ii, exp_group in enumerate(self.exp_groups):
            logging.info("Starting duration size %d of %d" % (ii + 1, self.Ndurs))

            dur = exp_group[0]
            df_twind = exp_group[1]

            tbins0 = df_twind["time"].values
            tbins1 = df_twind["time_end"].values
            timeIDs = df_twind["timeID"].values

            ntbins = len(tbins0)
            tbin_size = tbins1[0] - tbins0[0]
            tstep = tbins0[1] - tbins0[0]

            t_0 = time.time()
            self.get_cnts_tbins_ebins_fps(dur)
            logging.info("Done getting cnts_fpte")
            logging.info("Took %.3f seconds" % (time.time() - t_0))

            for jj in range(self.Nfps):
                cnts_intp = self.cnts_intps[jj]
                cnts_per_tbin = self.cnts_fpte[jj]
                Ndets = self.ndets[jj]
                solid_ang = self.solid_angs[jj]
                rt_sums = self.rt_sums[jj]
                fpbl = self.fpbls[jj]
                imx = self.imxs[jj]
                imy = self.imys[jj]

                bf_nsigs = np.zeros(ntbins)
                bf_inds = np.zeros(ntbins)
                nllhs = np.zeros(ntbins)
                bkg_nllhs = np.zeros(ntbins)

                t_0 = time.time()

                for kk in range(ntbins):
                    res_dict = {
                        "dur": tbin_size,
                        "imx": imx,
                        "imy": imy,
                        "ndets": Ndets,
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

                    (
                        res_dict["Nsig"],
                        res_dict["Plaw_Ind"],
                        res_dict["nllh"],
                        res_dict["bkg_nllh"],
                    ) = min_det_fp_nllh(
                        cnts_per_tbin[kk],
                        self.model,
                        fpbl,
                        bkg_row,
                        self.PSnames,
                        self.bkg_name,
                        tbin_size,
                        cnts_intp,
                        Ndets,
                        solid_ang,
                        rt_sums,
                        get_bkg_nllh=True,
                    )

                    TS = np.sqrt(2.0 * (res_dict["bkg_nllh"] - res_dict["nllh"]))
                    if np.isnan(TS):
                        TS = 0.0
                    res_dict["TS"] = TS
                    res_dicts.append(res_dict)

                #                 print "Done minimizing in loop of tbins"
                #                 print "Took %.3f seconds" %(time.time()-t_0)

                logging.info(
                    "Done with %d of %d positions for duration %d of %d"
                    % (jj + 1, self.Nfps, ii + 1, self.Ndurs)
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
    ev_data = fits.open(evfname)[1].data
    logging.debug("Opened up event file")
    dmask = fits.open(dmfname)[0].data
    bl_dmask = dmask == 0
    logging.debug("Opened up dmask file")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    nebins = len(ebins0)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    # probably get times from twind table

    twind_df = get_twinds_tab(conn)

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

    fp_arr = get_fp_arr(args.fp_dir)

    imxax = np.arange(-1.6, 1.61, 0.1)
    imyax = np.arange(-0.9, 0.91, 0.1)
    imxg, imyg = np.meshgrid(imxax, imyax)
    imxs = imxg.ravel()
    imys = imyg.ravel()

    PC = fits.open(args.pcfname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key="T")

    pcs = world2val(w_t, pc, imxs, imys)
    print(np.min(pcs), np.max(pcs))
    print(np.sum(pcs > 0.08))
    pc_bl = pcs > 0.08
    imxs = imxs[pc_bl]
    imys = imys[pc_bl]

    try:
        good_pix = np.load(args.pix_fname)
        im_dists = np.zeros_like(imxs)
        if len(good_pix) < 1:
            logging.info("pix2scan file is there are 0 pixels to scan")
            logging.info("Exiting")
            return
        for i in range(len(imxs)):
            im_dists[i] = np.min(
                im_dist(imxs[i], imys[i], good_pix["imx"], good_pix["imy"])
            )
        bl = im_dists < 0.1
        imxs = imxs[bl]
        imys = imys[bl]
    except Exception as E:
        logging.error(E)
        logging.warning("Trouble reading the pix2scan file")
        logging.info("Using whole FoV")

    Npnts = len(imxs)
    logging.info("%d total grid points" % (Npnts))
    Nper_job = 1 + int(Npnts / float(args.Njobs))
    if args.job_id > -1:
        i0 = args.job_id * Nper_job
        i1 = i0 + Nper_job
        imxs = imxs[i0:i1]
        imys = imys[i0:i1]
    logging.info("%d grid points to do" % (len(imxs)))

    solid_angle_dpi = np.load(solid_angle_dpi_fname)

    # bkg_mod = Bkg_Model_wSA(bl_dmask, solid_angle_dpi, nebins)

    rate_llh_obj = rates_fp_llh(
        imxs,
        imys,
        ev_data,
        twind_df,
        ebins0,
        ebins1,
        args.fp_dir,
        fp_arr,
        bl_dmask,
        drm_dir,
        args.bkg_fname,
        solid_angle_dpi,
        rt_dir,
    )

    res_dicts = rate_llh_obj.run()
    logging.info("Done with analysis")
    logging.info("%d results to write" % (len(res_dicts)))

    # append_rate_tab(conn, df_twind, quad_dict['id'], bkg_llh_tbins, llhs, bf_nsigs, bf_inds)
    #
    # logging.info("Appended rate results to DB")

    df = pd.DataFrame(res_dicts)
    logging.info("Done making results into DataFrame")
    save_fname = "rates_llh_res_%d_.csv" % (args.job_id)
    df.to_csv(save_fname, index=False)


if __name__ == "__main__":
    args = cli()

    main(args)
