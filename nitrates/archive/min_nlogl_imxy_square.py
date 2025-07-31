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

        self.set_bkg_time(bkg_t0, bkg_dt)

        self.set_sig_time(t0, dt)

        # Solver.__init__(self, **kwargs)

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
        print([np.sum(self.data_cnts_blm[i]) for i in range(self.nebins)])

        self.exp_bkg_cnts = self.bkg_rates * self.sig_dt
        self.bkg_cnt_errs = 5.0 * self.bkg_rate_errs * self.sig_dt

        print("Done setting up Signal Stuff")

    def model(self, imx, imy, sig_cnts, index, bkg_cnts):
        # return a dpi per ebin of sig_mod + bkg_mod
        # actually dpi[dmask_bl_arr]

        # bkg mod easy
        bkg_mod = bkg_cnts / self.ndets

        # sig mod needs to use the DRM to go
        # from sig_cnts, index, imx/y to cnts
        # per ebin
        # then needs imx/y to get the raytracing
        # to go make dpis

        # print "imx/y: ", imx, imy

        # print "getting DRM"

        # drm_f = self.drm_obj.get_drm(imx, imy)

        # print "getting sig cnts per ebin"

        # sig_ebins_normed = get_cnt_ebins_normed(index, drm_f,\
        #                                self.ebin_ind_edges)

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
            [bkg_mod[i] + rt_bl * sig_cnts_per_ebin[i] for i in range(self.nebins)]
        )

        # return np.add(bkg_mod, sig_mod)
        return mod_cnts

    def calc_logprior(self, bkg_cnts):
        logprior = stats.norm.logpdf(
            bkg_cnts, loc=self.exp_bkg_cnts, scale=self.bkg_cnt_errs
        )

        return logprior

    def Prior(self, cube):
        # imx = 2.*(cube[0]) - .5
        # imy = 1.*(cube[1] - .5)
        imx = 1.33 + 0.1 * (cube[0] - 0.5)
        imy = 0.173 + 0.1 * (cube[1] - 0.5)

        sig_cnts = 10 ** (cube[2] * 4)

        index = 2.5 * (cube[3]) - 0.5

        bkg_cnts = self.exp_bkg_cnts + self.bkg_cnt_errs * ndtri(cube[4:])

        return np.append([imx, imy, sig_cnts, index], bkg_cnts)

    def LogLikelihood(self, cube):
        # print "shape(cube), ", np.shape(cube)
        imx = cube[0]
        imy = cube[1]
        sig_cnts = cube[2]
        index = cube[3]
        bkg_cnts = cube[4:]
        # print imx, imy
        # print sig_cnts, index
        # print bkg_cnts

        # should output a dpi per ebins
        # with the sig_mod + bkg_mod
        model_cnts = self.model(imx, imy, sig_cnts, index, bkg_cnts)

        llh = np.sum(log_pois_prob(model_cnts, self.data_cnts_blm))

        print(imx, imy)
        print(sig_cnts, index)
        print(bkg_cnts)
        print(llh)

        return llh

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

        for i in range(self.nebins):
            bcnts = bkg_cnts[i] / self.ndets
            nllhs.append(-1.0 * log_pois_prob(bcnts, self.data_cnts_blm[i]))

        bkg_nllh = np.sum(np.array(nllhs)) + nlogprior

        return bkg_nllh


def min_nlogl(args, imx, imy, ev_data, dmask, rt_obj, drm_obj, t0, dt, dimxy=2.6e-2):
    t_0 = time.time()

    imx0 = imx - dimxy / 2.0
    imx1 = imx + dimxy / 2.0

    imy0 = imy - dimxy / 2.0
    imy1 = imy + dimxy / 2.0

    logging.info("Starting with imx %.3f, imy %.3f" % (imx, imy))

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

    trig_time = t0  # 555166977.856
    t_end = trig_time + dt
    # mask_vals = mask_detxy(dmask, ev_data)

    bkg_t0 = trig_time - 30.0
    bkg_dt = 20.0

    # bl_ev = (ev_data['TIME'] > (bkg_t0 -1.))&(ev_data['TIME']<(t_end+1.))&\
    #     (ev_data['EVENT_FLAGS']<1)&\
    #     (ev_data['ENERGY']<195.)&(ev_data['ENERGY']>=14.)&\
    #     (mask_vals==0.)

    ev_data0 = ev_data

    res_dict["time"] = trig_time
    res_dict["exp"] = dt

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
        dt,
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
        res = llh_obj.min_nllh(meth="dual_annealing", maxiter=150, seed=seed)
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

    logging.info("Done minimizing, took %.3f seconds" % (time.time() - t_0))

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

    for i in range(nrows):
        mpdict = {"args": args, "row": seed_tab[i]}
        mp_dict_list.append(mpdict)

    if nprocs == 1:
        results = []
        for i in range(nrows):
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
    parser.add_argument("--fname", type=str, help="filename to results to")
    parser.add_argument("--nproc", type=int, help="Number of procs to use", default=1)
    parser.add_argument(
        "--imx0", type=float, help="Min value of imx for square", default=-1.0
    )
    parser.add_argument(
        "--imx1", type=float, help="Max value of imx for square", default=-0.8
    )
    parser.add_argument(
        "--imy0", type=float, help="Min value of imy for square", default=-1.0
    )
    parser.add_argument(
        "--imy1", type=float, help="Max value of imy for square", default=-0.8
    )
    parser.add_argument("--t0", type=float, help="Signal min time (MET)")
    parser.add_argument("--dt", type=float, help="Signal time bin size (s)")
    parser.add_argument(
        "--ntbins", type=int, help="Number of time bins to do", default=1
    )
    parser.add_argument("--tfreq", type=int, help="tstep = dt/tfreq", default=2)
    parser.add_argument(
        "--pcmin", type=float, help="Partial Coding min for seeds", default=0.1
    )
    parser.add_argument(
        "--logfname", type=str, help="log file name", default="min_nlogl.log"
    )
    args = parser.parse_args()
    return args


def main(args):
    logfname = args.logfname
    if os.path.exists(logfname):
        for i in range(100):
            logfname = args.logfname + ".%d" % (i)
            if os.path.exists(logfname):
                continue
            else:
                break
    logging.basicConfig(
        filename=logfname,
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    results = []

    t0s = np.linspace(0.0, args.dt * (args.ntbins / args.tfreq), args.ntbins) + args.t0
    t1s = t0s + args.dt

    dimxy = 2.5e-2

    rt_obj = ray_trace_square(
        args.imx0 - dimxy,
        args.imx1 + dimxy,
        args.imy0 - dimxy,
        args.imy1 + dimxy,
        args.rt_dir,
    )

    drm_obj = DRMs(args.drm_dir)

    imxs = np.arange(args.imx0 + dimxy / 2.0, args.imx1, dimxy)
    imys = np.arange(args.imy0 + dimxy / 2.0, args.imy1, dimxy)

    grids = np.meshgrid(imxs, imys, indexing="ij")
    imxs = np.ravel(grids[0])
    imys = np.ravel(grids[1])

    logging.info("%d time bins to do" % (len(t0s)))
    logging.info("%d imx/y positions to do" % (len(imxs)))

    ev_data = fits.open(args.evfname)[1].data
    dmask = fits.open(args.dmask_fname)[0].data

    mask_vals = mask_detxy(dmask, ev_data)

    bl_ev = (
        (ev_data["TIME"] > (t0s[0] - 30.0))
        & (ev_data["TIME"] < (t1s[-1] + 1.0))
        & (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] < 195.0)
        & (ev_data["ENERGY"] >= 14.0)
        & (mask_vals == 0.0)
    )

    ev_data = ev_data[bl_ev]

    for i in range(len(t0s)):
        for j in range(len(imxs)):
            res_dict = min_nlogl(
                args, imxs[j], imys[j], ev_data, dmask, rt_obj, drm_obj, t0s[i], args.dt
            )
            results.append(res_dict)

    tab = Table(results)

    tab.write(args.fname)


if __name__ == "__main__":
    args = cli()

    main(args)
