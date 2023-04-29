import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import os
from scipy import optimize, stats
import argparse
import time
import multiprocessing as mp
import logging, traceback

from ..lib.logllh_ebins_funcs import get_cnt_ebins_normed, log_pois_prob
from ..response.ray_trace_funcs import ray_trace_square
from ..lib.drm_funcs import get_ebin_ind_edges, DRMs, get_cnts_intp_obj
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..lib.trans_func import get_pb_absortion
from ..analysis_seeds.bkg_linear_rates import get_lin_rate_obj


def get_abs_cor_rates(imx, imy, drm):
    drm_emids = (drm[1].data["ENERG_LO"] + drm[1].data["ENERG_HI"]) / 2.0
    absorbs = get_pb_absortion(drm_emids, imx, imy)
    abs_cor = (1.0) / (absorbs)
    return abs_cor


class llh_ebins_square(object):
    def __init__(
        self,
        event_data,
        drm_obj,
        rt_obj,
        ebins0,
        ebins1,
        dmask,
        bkg_t0,
        bkg_dt,
        t0,
        dt,
        imx0,
        imx1,
        imy0,
        imy1,
    ):
        self._all_data = event_data

        # self.drm_obj = drm_obj
        self.drm = drm_obj.get_drm((imx0 + imx1) / 2.0, (imy0 + imy1) / 2.0)
        self.rt_obj = rt_obj
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        self.dmask = dmask
        self.bl_dmask = dmask == 0
        self.good_dets = np.where(self.bl_dmask)
        self.ndets = np.sum(self.bl_dmask)
        self.imx0 = imx0
        self.imx1 = imx1
        self.imy0 = imy0
        self.imy1 = imy1

        self.ebin_ind_edges = get_ebin_ind_edges(self.drm, self.ebins0, self.ebins1)
        print("shape(self.ebin_ind_edges): ", np.shape(self.ebin_ind_edges))

        self.abs_cor = get_abs_cor_rates(
            (imx0 + imx1) / 2.0, (imy0 + imy1) / 2.0, self.drm
        )

        self.ind_ax = np.linspace(-1.5, 3.5, 20 * 5 + 1)
        self.cnts_intp = get_cnts_intp_obj(
            self.ind_ax, self.drm, self.ebin_ind_edges, self.abs_cor
        )

        self.bkg_obj = get_lin_rate_obj(
            self._all_data, self.t0, self.ebins0, self.ebins1, trng=4
        )

        self.set_bkg_time(bkg_t0, bkg_dt)

        self.set_sig_time(t0, dt)

    def set_bkg_time(self, t0, dt):
        print("Setting up Bkg calcs")

        self.bkg_t0 = t0
        self.bkg_dt = dt

        print("bkg_t0, bkg_dt", self.bkg_t0, self.bkg_dt)

        # bkg_data = self._all_data
        t_bl = (self._all_data["TIME"] > self.bkg_t0) & (
            self._all_data["TIME"] < (self.bkg_t0 + self.bkg_dt)
        )
        self.bkg_data = self._all_data[t_bl]

        print("bkg sum time: ", np.sum(t_bl))

        self.bkg_data_dpis = det2dpis(self.bkg_data, self.ebins0, self.ebins1)
        self.bkg_cnts = np.array(
            [np.sum(bkg_dpi[self.bl_dmask]) for bkg_dpi in self.bkg_data_dpis]
        )
        print("bkg_cnts: ", self.bkg_cnts)
        self.bkg_rates = self.bkg_cnts / self.bkg_dt
        self.bkg_rate_errs = np.sqrt(self.bkg_cnts) / self.bkg_dt

        print("Done with Bkg calcs")
        print("bkg rates: ")
        print(self.bkg_rates)
        print("bkg rate errors: ")
        print(self.bkg_rate_errs)

    def set_sig_time(self, t0, dt):
        print("Setting up Signal Data")

        self.sig_t0 = t0
        self.sig_dt = dt

        # self.data = np.copy(self._all_data)
        t_bl = (self._all_data["TIME"] > self.sig_t0) & (
            self._all_data["TIME"] < (self.sig_t0 + self.sig_dt)
        )
        self.data = self._all_data[t_bl]

        self.data_dpis = det2dpis(self.data, self.ebins0, self.ebins1)

        self.data_cnts_blm = np.array([dpi[self.bl_dmask] for dpi in self.data_dpis])

        print("Data Counts per Ebins: ")
        print([np.sum(self.data_cnts_blm[i]) for i in xrange(self.nebins)])

        bkg_rate, bkg_err = self.bkg_obj.get_rate(self.sig_t0)

        # self.exp_bkg_cnts = bkg_rate*self.sig_dt
        # self.bkg_cnt_errs = 2.5*bkg_err*self.sig_dt

        self.bkg_cnt_errs = 5.0 * self.bkg_rate_errs * self.sig_dt
        self.exp_bkg_cnts = self.bkg_rates * self.sig_dt

        print("Done setting up Signal Stuff")

    def brt_src_model(self, imx, imy, cnts, index):
        return None  # can this be deleted?

    def model(self, imx, imy, sig_cnts, index, bkg_cnts):
        # return a dpi per ebin of sig_mod + bkg_mod
        # actually dpi[dmask_bl_arr]

        # bkg mod easy
        bkg_mod = bkg_cnts / self.ndets

        sig_ebins_normed = self.cnts_intp(index)

        sig_cnts_per_ebin = sig_cnts * sig_ebins_normed

        # print "Getting raytraces"

        ray_trace = self.rt_obj.get_intp_rt(imx, imy)

        # print "Calculating sig_mod"

        rt_bl = ray_trace[self.bl_dmask]

        # print "Got ray trace, masked"

        rt_bl = rt_bl / np.sum(rt_bl)

        # print np.shape(rt_bl), np.shape(sig_cnts_per_ebin)

        # sig_mod = np.array([rt_bl*sig_cnt for sig_cnt\
        #                    in sig_cnts_per_ebin])

        mod_cnts = np.array(
            [bkg_mod[i] + rt_bl * sig_cnts_per_ebin[i] for i in xrange(self.nebins)]
        )

        # return np.add(bkg_mod, sig_mod)
        return mod_cnts

    def calc_logprior(self, bkg_cnts):
        logprior = stats.norm.logpdf(
            bkg_cnts, loc=self.exp_bkg_cnts, scale=self.bkg_cnt_errs
        )

        return logprior

    def nllh(self, theta):
        imx = theta[0]
        imy = theta[1]
        sig_cnts = 10.0 ** theta[2]
        index = theta[3]
        bkg_cnts = theta[4:] * self.exp_bkg_cnts

        model_cnts = self.model(imx, imy, sig_cnts, index, bkg_cnts)

        nllh = -1.0 * np.sum(log_pois_prob(model_cnts, self.data_cnts_blm))

        nlp = -1.0 * np.sum(self.calc_logprior(bkg_cnts))

        return nllh + nlp

    def unnorm_params(self, theta):
        imx = theta[0] * (self.imx1 - self.imx0) + self.imx0
        imy = theta[1] * (self.imy1 - self.imy0) + self.imy0
        sig_cnts = 10.0 ** (
            theta[2] * (self.uppers[2] - self.lowers[2]) + self.lowers[2]
        )
        index = theta[3] * (self.uppers[3] - self.lowers[3]) + self.lowers[3]
        bkg_cnts = theta[4:] * self.exp_bkg_cnts

        return imx, imy, sig_cnts, index, bkg_cnts

    def nllh_normed_params(self, theta):
        if np.any(np.isnan(theta)):
            return np.inf

        imx, imy, sig_cnts, index, bkg_cnts = self.unnorm_params(theta)

        model_cnts = self.model(imx, imy, sig_cnts, index, bkg_cnts)

        nllh = -1.0 * np.sum(log_pois_prob(model_cnts, self.data_cnts_blm))

        nlp = -1.0 * np.sum(self.calc_logprior(bkg_cnts))

        return nllh + nlp

    def min_nllh(self, meth="L-BFGS-B", x0=None, maxiter=100, seed=None):
        if x0 is None:
            x0 = [
                (self.imx0 + self.imx1) / 2.0,
                (self.imy0 + self.imy1) / 2.0,
                1.0,
                1.5,
                1.0,
                1.0,
                1.0,
                1.0,
            ]

        func2min = self.nllh

        self.lowers = np.append(
            [self.imx0, self.imy0, 0.0, -0.5], 0.5 * np.ones(self.nebins)
        )
        self.uppers = np.append(
            [self.imx1, self.imy1, 4.0, 2.5], 2.0 * np.ones(self.nebins)
        )

        if meth == "dual_annealing":
            lowers = np.append([0.0, 0.0, 0.0, 0.0], self.lowers[4:])
            uppers = np.append([1.0, 1.0, 1.0, 1.0], self.uppers[4:])

            bnds = np.array([lowers, uppers]).T

            print(np.shape(bnds))
            print(bnds)

            func2min = self.nllh_normed_params

            res = optimize.dual_annealing(func2min, bnds, maxiter=maxiter, seed=seed)

        else:
            bnds = optimize.Bounds(lowers, uppers)

            res = optimize.minimize(
                func2min, x0, method=meth, bounds=bnds, maxiter=maxiter
            )

        self.result = res

        return res

    def min_bkg_nllh(self, meth="L-BFGS-B", x0=None):
        if x0 is None:
            x0 = np.zeros(self.nebins)

        lowers = 0.2 * np.ones(self.nebins)
        uppers = 10.0 * np.ones(self.nebins)

        func2min = self.Bkg_nllh

        if meth == "dual_annealing":
            bnds = np.array([lowers, uppers]).T

            print(np.shape(bnds))
            print(bnds)

            res = optimize.dual_annealing(func2min, bnds)

        else:
            bnds = optimize.Bounds(lowers, uppers)

            res = optimize.minimize(func2min, x0, method=meth, bounds=bnds)

        self.bkg_result = res

        self.bkg_nllh = res.fun

        return res

    def Bkg_nllh(self, bkg_factors):
        nllhs = []
        nlps = []

        bkg_cnts = bkg_factors * self.exp_bkg_cnts

        nlogprior = -1.0 * np.sum(self.calc_logprior(bkg_cnts))

        for i in xrange(self.nebins):
            bcnts = bkg_cnts[i] / self.ndets
            nllhs.append(-1.0 * log_pois_prob(bcnts, self.data_cnts_blm[i]))

        bkg_nllh = np.sum(np.array(nllhs)) + nlogprior

        return bkg_nllh


def min_nlogl_from_seed(mp_dict):
    args = mp_dict["args"]
    seed_row = mp_dict["row"]

    logging.info("Starting proc with seed row %d" % (seed_row.index))
    logging.info(str(seed_row))

    res_dict_keys = [
        "bkg_nllh",
        "sig_nllh",
        "nsig",
        "ind",
        "imx",
        "imy",
        "bkg_norms",
        "time",
        "exp",
    ]
    res_dict = {}

    ebins0 = np.array([14.0, 24.0, 36.3, 55.4, 80.0, 120.7])
    ebins1 = np.append(ebins0[1:], [194.9])
    nebins = len(ebins0)

    ev_data = fits.open(args.evfname)[1].data
    dmask = fits.open(args.dmask_fname)[0].data

    trig_time = seed_row["time"]  # 555166977.856
    dts = ev_data["TIME"] - trig_time
    t_end = trig_time + seed_row["exp"]
    mask_vals = mask_detxy(dmask, ev_data)

    bkg_t0 = trig_time - 30.0
    bkg_dt = 20.0

    bl_ev = (
        (ev_data["TIME"] > (bkg_t0 - 1.0))
        & (ev_data["TIME"] < (t_end + 1.0))
        & (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] < 195.0)
        & (ev_data["ENERGY"] >= 14.0)
        & (mask_vals == 0.0)
    )

    ev_data0 = ev_data[bl_ev]

    imx = seed_row["imx"]
    imy = seed_row["imy"]
    dimxy = 0.016

    res_dict["time"] = seed_row["time"]
    res_dict["exp"] = seed_row["exp"]

    imx0 = imx - dimxy / 2.0
    imx1 = imx + dimxy / 2.0

    imy0 = imy - dimxy / 2.0
    imy1 = imy + dimxy / 2.0

    logging.debug("setting up ray traces")

    try:
        rt_obj = ray_trace_square(
            imx0 - 0.0025, imx1 + 0.0025, imy0 - 0.0025, imy1 + 0.0025, args.rt_dir
        )
    except Exception as E:
        logging.error("Trouble with ray tracing")
        logging.error(traceback.format_exs())
        res_dict["imx"] = imx
        res_dict["imy"] = imy
        res_dict["nsig"] = 0.0
        res_dict["ind"] = 0.0
        res_dict["bkg_nllh"] = 0.0
        res_dict["sig_nllh"] = 0.0
        res_dict["bkg_norms"] = np.zeros(nebins)
        return res_dict

    logging.debug("Done with ray traces")

    drm_obj = DRMs(args.drm_dir)

    logging.info("Setting up llh object now")

    llh_obj = llh_ebins_square(
        ev_data0,
        drm_obj,
        rt_obj,
        ebins0,
        ebins1,
        dmask,
        bkg_t0,
        bkg_dt,
        trig_time,
        seed_row["exp"],
        imx0,
        imx1,
        imy0,
        imy1,
    )

    logging.info("Minimizing background nlogl now")

    res_bkg = llh_obj.min_bkg_nllh()
    bkg_nllh = res_bkg.fun
    res_dict["bkg_nllh"] = bkg_nllh

    logging.info("Now doing signal llh")

    seed = 1022

    try:
        res = llh_obj.min_nllh(meth="dual_annealing", maxiter=400, seed=seed)
    except Exception as E:
        logging.error("error while minimizing signal nllh")
        logging.error("problem with imx0: %.3f imy0: %.3f" % (imx0, imy0))
        logging.error(traceback.format_exc())
        res_dict["imx"] = imx
        res_dict["imy"] = imy
        res_dict["nsig"] = 0.0
        res_dict["ind"] = 0.0
        res_dict["bkg_nllh"] = 0.0
        res_dict["sig_nllh"] = 0.0
        res_dict["bkg_norms"] = np.zeros(nebins)
        return res_dict

        # raise E

    res_dict["sig_nllh"] = res.fun
    params = llh_obj.unnorm_params(res.x)
    res_dict["imx"] = params[0]
    res_dict["imy"] = params[1]
    res_dict["nsig"] = params[2]
    res_dict["ind"] = params[3]
    res_dict["bkg_norms"] = res.x[4:]  # params[4]

    logging.info("Done with seed row %d" % (seed_row.index))

    return res_dict


def seeds2mp(seed_tab, args):
    nprocs = args.nproc

    res_dict_keys = [
        "bkg_nllh",
        "sig_nllh",
        "nsig",
        "ind",
        "imx",
        "imy",
        "time",
        "exp",
        "bkg_norms",
    ]

    mp_dict_keys = ["args", "row"]

    nrows = len(seed_tab)

    mp_dict_list = []

    for i in xrange(nrows):
        mpdict = {"args": args, "row": seed_tab[i]}
        mp_dict_list.append(mpdict)

    if nprocs == 1:
        results = []
        for i in xrange(nrows):
            results.append(min_nlogl_from_seed(mp_dict_list[i]))

    else:
        p = mp.Pool(nprocs)

        logging.info("Starting %d procs" % (nprocs))

        t0 = time.time()

        results = p.map_async(min_nlogl_from_seed, mp_dict_list).get()

        p.close()
        p.join()

        logging.info(
            "Took %.2f seconds, %.2f minutes"
            % (time.time() - t0, (time.time() - t0) / 60.0)
        )

    tab = Table(results)

    tab.write(args.fname)


"""
Want to do one square (per script) and iter
over all the time scales

So takes an input trigger time and tests
all time scales (sig_dts) and start times (sig_t0s)
within something like (+/- 30s)
(maybe start with +/- 15s for now)

Will also take the bounds of the square (imx/y_0/1)
And all the relavent data (event data, det_mask,
dmr and ray trace directories)

So first thing to do is to read in all the data
Then initialize the ray trace and drm objects
(might want to have those just read in all files from
the beginning)
Then initilaize llh_obj, with first time window that
will be tested.
Then loop over the time windows doing the minimziation
for each time window
and at each iteration just re-set the sig_times in the
likelihood object (and possibly the bkg_times)

"""


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drm_dir", type=str, help="drm_directory")
    parser.add_argument(
        "--rt_dir",
        type=str,
        help="rt_directory",
        default="/gpfs/scratch/jjd330/bat_data/ray_traces2/",
    )
    parser.add_argument("--evfname", type=str, help="Event data file")
    parser.add_argument("--dmask_fname", type=str, help="Detector mask file")
    parser.add_argument("--tabfname", type=str, help="seed table filename")
    parser.add_argument("--fname", type=str, help="filename to results to")
    parser.add_argument("--nproc", type=int, help="Number of procs to use", default=2)
    parser.add_argument("--snrcut", type=float, help="SNR cut for seeds", default=5.0)
    parser.add_argument(
        "--snrmax", type=float, help="Max SNR from seedsto use", default=None
    )
    parser.add_argument(
        "--pcmin", type=float, help="Partial Coding min for seeds", default=0.1
    )
    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        filename="min_logl_from_seeds3.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    seed_tab = Table.read(args.tabfname)
    if args.snrmax is None:
        bl = (seed_tab["pc"] >= args.pcmin) & (seed_tab["snr"] >= args.snrcut)
    else:
        bl = (
            (seed_tab["pc"] >= args.pcmin)
            & (seed_tab["snr"] >= args.snrcut)
            & (seed_tab["snr"] < args.snrmax)
        )

    print(np.sum(bl), " seeds to minimize at")

    seeds2mp(seed_tab[bl], args)


if __name__ == "__main__":
    args = cli()

    main(args)
