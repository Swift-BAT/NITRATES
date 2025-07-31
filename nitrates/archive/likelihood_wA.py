import numpy as np
from astropy.io import fits
import os
from scipy import optimize, stats
import argparse
import time
import emcee

# from corner import corner

from ..lib.logllh_ebins_funcs import get_cnt_ebins_normed, log_pois_prob
from ..response.ray_trace_funcs import ray_trace_square
from ..lib.drm_funcs import get_ebin_ind_edges, DRMs, get_cnts_intp_obj
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..archive.min_nlogl_from_seeds import llh_ebins_square, get_abs_cor_rates
from ..lib.trans_func import get_pb_absortion


def get_sys_err(sys_err_file, ebin_ind_edges):
    sys_errs = np.array(
        [
            np.mean(
                sys_err_file["SYS_ERR"][0][
                    ebin_ind_edges[i][0] : ebin_ind_edges[i][1] + 1
                ]
            )
            for i in range(len(ebin_ind_edges))
        ]
    )

    return sys_errs


def cnts_pdf(xax, n0, sys_err_frac):
    # sigma = np.sqrt(n0 + (sys_err_frac*n0)**2)
    sigma = sys_err_frac * n0

    return stats.norm.pdf(xax, loc=n0, scale=sigma)


class llh_ebins_square_wsampler(object):
    def __init__(
        self,
        event_data,
        drm_obj,
        rt_obj,
        ebins0,
        ebins1,
        dmask,
        sys_errs,
        bkg_t0,
        bkg_dt,
        t0,
        dt,
        imx0,
        imx1,
        imy0,
        imy1,
        mk_bkg=False,
        E0=50.0,
    ):
        self._all_data = event_data

        # self.drm_obj = drm_obj
        self.drm = drm_obj.get_drm((imx0 + imx1) / 2.0, (imy0 + imy1) / 2.0)
        self.rt_obj = rt_obj
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        self.sys_fracs = sys_errs
        self.dmask = dmask
        self.bl_dmask = dmask == 0
        self.good_dets = np.where(self.bl_dmask)
        self.ndets = np.sum(self.bl_dmask)
        self.imx0 = imx0
        self.imx1 = imx1
        self.imy0 = imy0
        self.imy1 = imy1
        self.calc_bkg = mk_bkg
        self.nwalkers = 100

        self.ebin_ind_edges = get_ebin_ind_edges(self.drm, self.ebins0, self.ebins1)
        print("ebin_ind_edges")
        print(self.ebin_ind_edges)

        self.abs_cor = get_abs_cor_rates(
            (imx0 + imx1) / 2.0, (imy0 + imy1) / 2.0, self.drm
        )

        self.ind_ax = np.linspace(-2.0, 4.0, 20 * 6 + 1)
        # self.cnts_intp = get_cnts_intp_obj(self.ind_ax,\
        #                                self.drm,\
        #                                self.ebin_ind_edges,\
        #                                self.abs_cor,\
        #                                  E0=E0)

        self.rates_intp = get_cnts_intp_obj(
            self.ind_ax,
            self.drm,
            self.ebin_ind_edges,
            self.abs_cor,
            E0=E0,
            normed=False,
        )

        self.bkg_rates = 1e0 * np.ones(self.nebins)
        self.bkg_rate_errs = 0.1 * self.bkg_rates
        # self.bkg_rates = .925*np.array([.98*233.0, 1.04*236., .98*181.0, .525*159.25])/.256
        # self.bkg_rate_errs = 20.*np.array([0.922, 0.937, 0.807, 0.797])/.256

        if self.calc_bkg:
            self.set_bkg_time(bkg_t0, bkg_dt)

        self.set_sig_time(t0, dt)

        #         self.lowers = np.append([self.imx0, self.imy0, -3.0,
        #                                  self.ind_ax[0]+.5],\
        #                                 np.ones(self.nebins))
        #         self.uppers = np.append([self.imx1, self.imy1, 5.0,
        #                                  self.ind_ax[-1]-.5],\
        #                                 6.*np.ones(self.nebins))

        lwrs = np.append(
            [self.imx0, self.imy0, 1e-5, self.ind_ax[0] + 0.5],
            0.25 * np.ones(self.nebins),
        )
        self.lowers = np.append(lwrs, np.ones(self.nebins))

        uprs = np.append(
            [self.imx1, self.imy1, 5e2, self.ind_ax[-1] - 0.5],
            4.0 * np.ones(self.nebins),
        )

        self.uppers = np.append(uprs, 1e5 * np.ones(self.nebins))

        self.ranges = self.uppers - self.lowers

        self.npars = len(self.uppers)

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
        self.bkg_rate_errs = 5.0 * np.sqrt(self.bkg_cnts) / self.bkg_dt

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

        self.tot_data_cnts = np.sum(self.data_cnts_blm)
        print("Total data counts: ", self.tot_data_cnts)

        self.exp_bkg_cnts = self.bkg_rates * self.sig_dt
        self.bkg_cnt_errs = self.bkg_rate_errs * self.sig_dt

        print("Expected bkg cnts and err: ")
        print(self.exp_bkg_cnts)
        print(self.bkg_cnt_errs)

        print("Done setting up Signal Stuff")

    def flux2rate(self, A, ind):
        # rate per ebin per det on-axis equivalent, no geometric corrections
        # to get actual rate, rate per ebin = np.sum(ray_trace)*rate_ebins

        rate_ebins = np.array(
            get_cnt_ebins(
                A, ind, self.drm, self.ebin_ind_edges, E0=self.E0, abs_cor=self.abs_cor
            )
        )

    def model2(self, imx, imy, A, index, bkg_cnts):
        # return a dpi per ebin of sig_mod + bkg_mod
        # actually dpi[dmask_bl_arr]

        # bkg mod easy
        bkg_mod = bkg_cnts / self.ndets

        # cnts per ebin per det (on-axis equivalent)
        sig_cnts_ebins = A * self.rates_intp(index) * self.sig_dt

        # if np.any(~np.isfinite(sig_ebins_normed)):
        #    logging.debug("Problem with cnts_intp")
        #    logging.debug("imx: %.3f, imy: %.3f, sig_cnts: %.1f, index: %.2f"\
        #              %(imx, imy, sig_cnts, index))

        # sig_rate_per_ebin = sig_cnts*sig_ebins_normed

        ray_trace = self.rt_obj.get_intp_rt(imx, imy)
        if ray_trace is None:
            return np.nan
        if np.any(~np.isfinite(ray_trace)):
            logging.debug("Problem with rt_obj.get_intp_rt")
            logging.debug("imx: %.3f, imy: %.3f" % (imx, imy))

        rt_bl = ray_trace[self.bl_dmask]

        # ndets*pcode*geometric_correction
        # (geo_cor is cos(theta) plus some other stuff)
        rt_sum = np.sum(rt_bl)

        sig_cnts_per_ebin = rt_sum * sig_cnts_ebins

        # rt_bl = rt_bl/rt_sum

        mod_cnts = np.array(
            [bkg_mod[i] + rt_bl * sig_cnts_per_ebin[i] for i in range(self.nebins)]
        )

        # mod_cnts = np.array([bkg_mod[i] + rt_bl*sig_cnts_per_ebin[i]\
        #                     for i in xrange(self.nebins)])

        return mod_cnts

    def model(self, imx, imy, sig_cnts, index, bkg_cnts):
        # return a dpi per ebin of sig_mod + bkg_mod
        # actually dpi[dmask_bl_arr]

        # bkg mod easy
        bkg_mod = bkg_cnts / self.ndets

        sig_ebins_normed = self.cnts_intp(index)
        # if np.any(~np.isfinite(sig_ebins_normed)):
        #    logging.debug("Problem with cnts_intp")
        #    logging.debug("imx: %.3f, imy: %.3f, sig_cnts: %.1f, index: %.2f"\
        #              %(imx, imy, sig_cnts, index))

        sig_cnts_per_ebin = sig_cnts * sig_ebins_normed

        ray_trace = self.rt_obj.get_intp_rt(imx, imy)
        if ray_trace is None:
            return np.nan
        if np.any(~np.isfinite(ray_trace)):
            logging.debug("Problem with rt_obj.get_intp_rt")
            logging.debug("imx: %.3f, imy: %.3f" % (imx, imy))

        rt_bl = ray_trace[self.bl_dmask]

        rt_bl = rt_bl / np.sum(rt_bl)

        mod_cnts = np.array(
            [bkg_mod[i] + rt_bl * sig_cnts_per_ebin[i] for i in range(self.nebins)]
        )

        return mod_cnts

    def flux2cnts(self, imx, imy, A, ind, sig_rate_norms):
        sig_cnts_ebins = A * self.rates_intp(ind) * self.sig_dt

        ray_trace = self.rt_obj.get_intp_rt(imx, imy)

        rt_bl = ray_trace[self.bl_dmask]

        # ndets*pcode*geometric_correction
        # (geo_cor is cos(theta) plus some other stuff)
        rt_sum = np.sum(rt_bl)

        sig_cnts_per_ebin0 = rt_sum * sig_cnts_ebins
        sig_cnts_per_ebin = sig_cnts_per_ebin0 * sig_rate_norms

        return sig_cnts_per_ebin

    def like(self, imx, imy, A, ind, sig_rate_norms, bkg_rates):
        bkg_mod = self.sig_dt * bkg_rates / self.ndets

        # cnts per ebin per det (on-axis equivalent)
        sig_cnts_ebins = A * self.rates_intp(ind) * self.sig_dt

        ray_trace = self.rt_obj.get_intp_rt(imx, imy)

        rt_bl = ray_trace[self.bl_dmask]

        # ndets*pcode*geometric_correction
        # (geo_cor is cos(theta) plus some other stuff)
        rt_sum = np.sum(rt_bl)

        sig_cnts_per_ebin0 = rt_sum * sig_cnts_ebins
        sig_cnts_per_ebin = sig_cnts_per_ebin0 * sig_rate_norms

        # rt_bl = rt_bl/rt_sum

        mod_cnts = np.array(
            [
                bkg_mod[i] + rt_bl * sig_cnts_per_ebin[i] / rt_sum
                for i in range(self.nebins)
            ]
        )

        cnt_lprob = np.log(
            cnts_pdf(sig_cnts_per_ebin, sig_cnts_per_ebin0, self.sys_fracs)
        )

        lprob = np.sum(cnt_lprob)

        llh = np.sum(log_pois_prob(mod_cnts, self.data_cnts_blm))

        return llh + lprob

    def calc_logprior(self, bkg_rates):
        logprior = stats.norm.logpdf(
            bkg_rates, loc=self.bkg_rates, scale=self.bkg_rate_errs
        )

        return np.sum(logprior)

    def theta2params(self, theta):
        imx, imy, A, ind = theta[:4]
        sig_rate_norms = theta[4 : 4 + self.nebins]
        bkg_rates = theta[4 + self.nebins :]

        return imx, imy, A, ind, sig_rate_norms, bkg_rates

    def lnprior(self, theta):
        imx, imy, A, ind, sig_rate_norms, bkg_rates = self.theta2params(theta)

        if np.any(np.array(theta) < self.lowers):
            return -np.inf
        elif np.any(np.array(theta) > self.uppers):
            return -np.inf
        else:
            return 0.0
            # lp = self.calc_logprior(bkg_rates)
            # return lp

    def lnlike(self, theta):
        imx, imy, A, ind, sig_rate_norms, bkg_rates = self.theta2params(theta)

        #         model_cnts = self.model(imx, imy, sig_cnts, ind, bkg_cnts)

        #         llh = np.sum(log_pois_prob(model_cnts,\
        #                             self.data_cnts_blm))

        llh = self.like(imx, imy, A, ind, sig_rate_norms, bkg_rates)

        return llh

    def lnprob(self, theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.lnlike(theta)

    def mk_sampler(self, nwalkers=None):
        if nwalkers is not None:
            self.nwalkers = nwalkers

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.npars, self.lnprob)

    def run_sampler(self, starting_guesses, Nsteps, rngs=None):
        if rngs is None:
            rngs = 0.25 * self.ranges

        if starting_guesses is None:
            starts = None
        else:
            starts = [
                starting_guesses + rngs * (np.random.random(size=self.npars) - 0.5)
                for i in range(self.nwalkers)
            ]

        self.sampler.run_mcmc(starts, Nsteps)
