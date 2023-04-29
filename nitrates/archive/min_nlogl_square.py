import numpy as np
from astropy.io import fits
import os
from scipy import optimize, stats
import argparse
import time

from ..lib.logllh_ebins_funcs import get_cnt_ebins_normed, log_pois_prob
from ..response.ray_trace_funcs import ray_trace_square
from ..lib.drm_funcs import get_ebin_ind_edges, DRMs
from ..lib.event2dpi_funcs import det2dpis


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

        self.drm_obj = drm_obj
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

        self.ebin_ind_edges = get_ebin_ind_edges(
            self.drm_obj.get_drm(0.0, 0.0), self.ebins0, self.ebins1
        )
        print("shape(self.ebin_ind_edges): ", np.shape(self.ebin_ind_edges))

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

        drm_f = self.drm_obj.get_drm(imx, imy)

        # print "getting sig cnts per ebin"

        sig_ebins_normed = get_cnt_ebins_normed(index, drm_f, self.ebin_ind_edges)

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
        imx = theta[0] * (self.imx1 - self.imx0) / 5.0 + self.imx0
        imy = theta[1] * (self.imy1 - self.imy0) / 5.0 + self.imy0
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

        self.lowers = [self.imx0, self.imy0, -0.5, -0.5, 0.2, 0.2, 0.2, 0.2]
        self.uppers = [self.imx1, self.imy1, 4.0, 2.5, 10.0, 10.0, 10.0, 10.0]

        if meth == "dual_annealing":
            lowers = np.append([0.0, 0.0, 0.0, 0.0], self.lowers[-4:])
            uppers = np.append([5.0, 5.0, 1.0, 1.0], self.uppers[-4:])

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
    parser.add_argument(
        "--trig_time", type=float, help="Center time of search in MET seconds"
    )
    parser.add_argument(
        "--imx0", type=float, help="Lower imx value of square", default=0.0
    )
    parser.add_argument(
        "--imx1", type=float, help="Higher imx value of square", default=0.1
    )
    parser.add_argument(
        "--imy0", type=float, help="Lower imy value of square", default=0.0
    )
    parser.add_argument(
        "--imy1", type=float, help="Higher imy value of square", default=0.1
    )
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
    args = parser.parse_args()
    return args


def main(args):
    ebins0 = np.array([14.0, 24.0, 48.9, 98.8])
    ebins1 = np.append(ebins0[1:], [194.9])

    ev_data = fits.open(args.evfname)[1].data
    dmask = fits.open(args.dmask_fname)[0].data

    bkg_twind = (-40.0, -20.0)

    test_twind = (-15.0, 15.0)

    dt_min = 0.128

    test_dts = dt_min * (2 ** np.arange(6))

    test_t0 = args.trig_time + test_twind[0]

    dts = ev_data["TIME"] - args.trig_time

    bl_ev = (
        (dts > -41.0)
        & (dts < 20.0)
        & (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] < 195.0)
    )
    ev_data = ev_data[bl_ev]

    drm_obj = DRMs(args.drm_dir)

    rt_obj = ray_trace_square(
        args.imx0 - 0.01,
        args.imx1 + 0.01,
        args.imy0 - 0.01,
        args.imy1 + 0.01,
        args.rt_dir,
    )

    bkg_t0 = args.trig_time + bkg_twind[0]
    bkg_dt = bkg_twind[1] - bkg_twind[0]

    sig_t0 = args.trig_time + test_twind[0]
    sig_dt = test_dts[0]

    llh_obj = llh_ebins_square(
        ev_data,
        drm_obj,
        rt_obj,
        ebins0,
        ebins1,
        dmask,
        bkg_t0,
        bkg_dt,
        sig_t0,
        sig_dt,
        args.imx0,
        args.imx1,
        args.imy0,
        args.imy1,
    )

    bkg_nllhs = []
    bkg_xs = []

    sig_nllhs = []
    sig_xs = []

    seed = 1234

    t0 = time.time()

    for ii in range(len(test_dts)):
        sig_dt = test_dts[ii]

        sig_t0_ax = np.arange(test_twind[0], test_twind[1], sig_dt / 2.0)

        for jj in range(len(sig_t0_ax)):
            sig_t0 = args.trig_time + sig_t0_ax[jj]

            llh_obj.set_sig_time(sig_t0, sig_dt)

            res_bkg = llh_obj.min_bkg_nllh()
            print(res_bkg)
            bkg_nllh = res_bkg.fun
            bkg_nllhs.append(bkg_nllh)
            bkg_xs.append(res_bkg.x)

            res = llh_obj.min_nllh(meth="dual_annealing", maxfun=5000, seed=seed)
            print("Sig result")
            print(res)
            sig_nllhs.append(res.fun)
            sig_xs.append(res.x)

            print("Done with dt %.3f at t0 %.3f"(sig_dt, sig_t0_ax))
            print(
                "Taken %.2f seconds, %.2f minutes so far"
                % (time.time() - t0, (time.time() - t0) / 60.0)
            )


if __name__ == "__main__":
    args = cli()

    main(args)
