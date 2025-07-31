import numpy as np
from scipy import optimize, stats, interpolate
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import abc
import os
import argparse
import logging, traceback
import time
import pandas as pd
from copy import copy

# import ..config

from ..analysis_seeds.bkg_rate_estimation import rate_obj_from_sqltab
from ..lib.sqlite_funcs import (
    get_conn,
    write_result,
    write_results,
    timeID2time_dur,
    write_results_fromSigImg,
    update_square_stat,
    write_square_res_line,
    write_square_results,
)
from ..lib.dbread_funcs import (
    get_rate_fits_tab,
    guess_dbfname,
    get_seeds_tab,
    get_info_tab,
    get_files_tab,
    get_square_tab,
    get_full_sqlite_table_as_df,
)
from ..config import EBINS0, EBINS1, solid_angle_dpi_fname
from ..models.flux_models import Plaw_Flux
from ..llh_analysis.minimizers import (
    NLLH_ScipyMinimize_Wjacob,
    imxy_grid_miner,
    NLLH_ScipyMinimize,
)
from ..response.ray_trace_funcs import RayTraces, get_rt_arr
from ..llh_analysis.LLH import LLH_webins
from ..models.models import Bkg_Model_wSA, Point_Source_Model, Model, Flux2Rate
from ..lib.drm_funcs import get_ebin_ind_edges, DRMs, get_cnts_intp_obj
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..lib.trans_func import get_pb_absortion, get_pb_mu
from ..response.response import Response
from ..lib.logllh_ebins_funcs import get_gammaln, log_pois_prob

# need to read rate fits from DB
# and read twinds
# and read/get event, dmask, and ebins
# then get bkg_llh_obj and a minimizer
# then loop over all time windows
# minimizing nllh and recording bf params


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--job_id", type=int, help="ID to tell it what seeds to do", default=-1
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--rt_dir", type=str, help="Directory with ray traces", default=None
    )
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default="pc_2.img"
    )
    parser.add_argument(
        "--job_fname",
        type=str,
        help="File name for table with what imx/y square for each job",
        default="job_table.csv",
    )
    parser.add_argument(
        "--seed_fname",
        type=str,
        help="File name of table with seeds",
        default="seed_tab_2.csv",
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--pix_fname",
        type=str,
        help="Name of the file with good imx/y coordinates",
        default="good_pix2scan.npy",
    )
    args = parser.parse_args()
    return args


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot((imx1 - imx0), (imy1 - imy0))


class Flux_Model(object, metaclass=abc.ABCMeta):
    def __init__(self, name, param_names, param_bounds=None, E0=50.0):
        self._name = name
        self._param_names = param_names
        self._E0 = E0
        self._npar = len(param_names)
        if param_bounds is None:
            param_bounds = {}
            for pname in param_names:
                if "A" in pname:
                    param_bounds[pname] = (1e-6, 1e4)
                elif "E" in pname:
                    param_bounds[pname] = (1e-1, 1e4)
                else:
                    param_bounds[pname] = (-1e1, 1e1)
        self._param_bounds = param_bounds

    @property
    def name(self):
        return self._name

    @property
    def E0(self):
        return self._E0

    @property
    def param_names(self):
        return self._param_names

    @property
    def param_bounds(self):
        return self._param_bounds

    @property
    def npar(self):
        return self._npar

    @abc.abstractmethod
    def spec(self, E, params):
        pass

    def get_photon_flux(self, Emin, Emax, params, esteps=128, num=False):
        if hasattr(self, "specIntegral") and not num:
            return self.specIntegral(Emax, params) - self.specIntegral(Emin, params)
        flux = 0.0
        try:
            flux = quad(self.spec, Emin, Emax, args=(params), epsabs=1e-7, epsrel=1e-5)[
                0
            ]
        except:
            Es = np.linspace(Emin, Emax, int(esteps))
            dE = Es[1] - Es[0]
            flux = np.sum(self.spec(Es, params)) * dE
        return flux

    def get_photon_fluxes(self, Ebins, params, esteps=128):
        if hasattr(self, "specIntegral_bins"):
            photon_fluxes = self.specIntegral_bins(Ebins, params)
        else:
            Npnts = len(Ebins) - 1
            photon_fluxes = np.zeros(Npnts)
            for i in range(Npnts):
                photon_fluxes[i] = self.get_photon_flux(
                    Ebins[i], Ebins[i + 1], params, esteps=esteps
                )
        return photon_fluxes


class Plaw_Flux(Flux_Model):
    def __init__(self, **kwds):
        param_names = ["A", "gamma"]
        param_bounds = {"A": (1e-6, 1e1), "gamma": (0.0, 2.5)}
        super(Plaw_Flux, self).__init__(
            "plaw", param_names, param_bounds=param_bounds, **kwds
        )
        self.param_guess = {"A": 1e-2, "gamma": 1.5}

    def spec(self, E, params):
        return params["A"] * (E / self.E0) ** (-params["gamma"])

    def specIntegral(self, E, params):
        #         if np.isclose(params['gamma'], 1):
        if np.abs(params["gamma"] - 1.0) < 1e-6:
            return (params["A"] * self.E0) * np.log(E)

        return ((params["A"] * E) / (1.0 - params["gamma"])) * (E / self.E0) ** (
            -params["gamma"]
        )

    def specIntegral_bins(self, Ebins, params):
        #         if np.isclose(params['gamma'], 1):
        if np.abs(params["gamma"] - 1.0) < 1e-6:
            return (params["A"] * self.E0) * (np.log(Ebins[1:] / Ebins[:-1]))

        Epow = Ebins ** (1.0 - params["gamma"])
        return (
            (params["A"] / (1.0 - params["gamma"]))
            * (self.E0 ** params["gamma"])
            * (Epow[1:] - Epow[:-1])
        )


#         return ((params['A']*E)/(1.-params['gamma']))*\
#                 (E/self.E0)**(-params['gamma'])


class CompoundModel(Model):
    def __init__(self, model_list, name=None):
        self.model_list = model_list

        self.Nmodels = len(model_list)

        self.model_names = [model.name for model in model_list]

        if name is None:
            name = ""
            for mname in self.model_names:
                name += mname + "+"
            name = name[:-1]

        param_names = []

        self.param_name_map = {}
        param_dict = {}

        has_prior = False
        Tdep = False
        self.ntbins = 0

        for model in self.model_list:
            if model.has_prior:
                has_prior = True
            if model.Tdep:
                Tdep = True
                self.ntbins = max(self.ntbins, model.ntbins)

            mname = model.name

            pname_map = {}

            for pname in model.param_names:
                _name = mname + "_" + pname
                param_names.append(_name)
                param_dict[_name] = model.param_dict[pname]
                pname_map[pname] = _name

            self.param_name_map[mname] = pname_map

        bl_dmask = self.model_list[0].bl_dmask

        super(CompoundModel, self).__init__(
            name,
            bl_dmask,
            param_names,
            param_dict,
            self.model_list[0].nebins,
            has_prior=has_prior,
            Tdep=Tdep,
        )

        self._last_params_ebin = [{} for i in range(self.nebins)]
        self._last_rate_dpi = [np.ones(self.ndets) for i in range(self.nebins)]

    def get_model_params(self, params):
        param_list = []

        for model in self.model_list:
            param = {}
            pname_map = self.param_name_map[model.name]
            for k in model.param_names:
                param[k] = params[pname_map[k]]
            param_list.append(param)

        return param_list

    def get_rate_dpis(self, params, **kwargs):
        if self.Tdep:
            #             tbins0 = kwargs['tbins0']
            #             tbins1 = kwargs['tbins1']
            ntbins = self.ntbins
            rate_dpis = np.zeros((ntbins, self.nebins, self.ndets))
        else:
            rate_dpis = np.zeros((self.nebins, self.ndets))

        for model in self.model_list:
            param = {}
            pname_map = self.param_name_map[model.name]
            for k in model.param_names:
                param[k] = params[pname_map[k]]

            if model.Tdep:
                rate_dpis += model.get_rate_dpis(param)
            else:
                if self.Tdep:
                    rate_dpi = model.get_rate_dpis(param)[np.newaxis, :, :]
                    #                     print "rate_dpi shape: ", rate_dpi.shape
                    rate_dpis += np.ones_like(rate_dpis) * rate_dpi
                else:
                    rate_dpis += model.get_rate_dpis(param)

        return rate_dpis

    def get_rate_dpi(self, params, j, **kwargs):
        if params == self._last_params_ebin[j]:
            return self._last_rate_dpi[j]

        if self.Tdep:
            #             tbins0 = kwargs['tbins0']
            #             tbins1 = kwargs['tbins1']
            ntbins = self.ntbins
            rate_dpi = np.zeros((ntbins, self.ndets))
        else:
            rate_dpi = np.zeros(self.ndets)

        for model in self.model_list:
            param = {}
            pname_map = self.param_name_map[model.name]
            for k in model.param_names:
                param[k] = params[pname_map[k]]

            if model.Tdep:
                #                 rate_dpis += model.get_rate_dpis(param, tbins0, tbins1)
                rate_dpi += model.get_rate_dpi(param, j)
            else:
                if self.Tdep:
                    rate_dpi_ = model.get_rate_dpi(param, j)[np.newaxis, :]
                    #                     print "rate_dpi shape: ", rate_dpi.shape
                    rate_dpi += np.ones_like(rate_dpi) * rate_dpi_
                else:
                    try:
                        rate_dpi += model.get_rate_dpi(param, j)
                    except Exception as E:
                        print(E)
                        rate_dpi += model.get_rate_dpis(param)[j]
        self._last_params_ebin[j] = params
        self._last_rate_dpi[j] = rate_dpi

        return rate_dpi

    def get_log_prior(self, params, j=None):
        lp = 0.0

        if self.has_prior:
            param_list = self.get_model_params(params)
            for i, model in enumerate(self.model_list):
                if model.has_prior:
                    try:
                        lp += model.get_log_prior(param_list[i], j=j)
                    except:
                        lp += model.get_log_prior(param_list[i])
        return lp

    def get_dr_dps(self, params):
        # loop through param list and see if it has this function

        dr_dps = []

        for i, model in enumerate(self.model_list):
            param_list = self.get_model_params(params)
            if model.has_deriv:
                dr_dps += model.get_dr_dps(param_list[i])

        return dr_dps

    def get_dr_dp(self, params, j):
        # loop through param list and see if it has this function

        dr_dps = []

        for i, model in enumerate(self.model_list):
            param_list = self.get_model_params(params)
            if model.has_deriv:
                dr_dps += model.get_dr_dp(param_list[i], j)

        return dr_dps

    def get_dnlp_dp(self, params, j):
        dNLP_dp = []

        if self.has_prior:
            param_list = self.get_model_params(params)
            for i, model in enumerate(self.model_list):
                if model.has_prior:
                    dNLP_dp += model.get_dnlp_dp(param_list[i], j)
        return dNLP_dp


HALF_LOG2PI = 0.5 * np.log(2 * np.pi)


def norm_logpdf(x, sig, mu):
    return -np.square((x - mu) / sig) / 2.0 - np.log(sig) - HALF_LOG2PI


class Bkg_Model_wSAfixed(Model):
    def __init__(
        self,
        bl_dmask,
        solid_ang_dpi,
        nebins,
        flat_diff_ratios,
        exp_rates=None,
        bkg_sigs=None,
        use_prior=False,
        use_deriv=False,
    ):
        self.sa_dpi = solid_ang_dpi
        self.solid_angs = solid_ang_dpi[bl_dmask]
        self.solid_ang_mean = np.mean(self.solid_angs)

        self.rate_names = ["bkg_rate_" + str(i) for i in range(nebins)]

        self.ratios = np.array(flat_diff_ratios)
        # 1 = Af + Ad
        # rat = Af/Ad
        # 1 = Ad*rat + Ad
        # Ad = 1 / (1 + rat)
        # self.diff_As = 1. / (1. / self.ratios)
        self.diff_As = 1.0 / (1.0 + self.ratios)
        self.flat_As = 1.0 - self.diff_As

        param_names = self.rate_names
        #         param_names += self.flat_names

        param_dict = {}

        #         if t is None:
        #             rates = bkg_obj.get_rate((bkg_obj.t0+bkg_obj.t1)/2.)[0]
        #         else:
        #             rates = bkg_obj.get_rate(t)[0]

        for i, pname in enumerate(param_names):
            pdict = {}
            pdict["bounds"] = (5e-5, 1e2)
            pdict["nuis"] = True
            pdict["fixed"] = False
            pdict["val"] = 0.05
            param_dict[pname] = pdict

        super(Bkg_Model_wSAfixed, self).__init__(
            "Background", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        self._rate_ones = np.ones(self.ndets)
        self._rate_zeros = np.zeros(self.ndets)
        if use_deriv:
            self.has_deriv = True
        if use_prior:
            if exp_rates is not None and bkg_sigs is not None:
                self.set_prior(exp_rates, bkg_sigs)

    def set_prior(self, exp_rates, bkg_sigs):
        self.exp_rates = exp_rates
        self.bkg_sigs = bkg_sigs

    def get_rate_dpis(self, params):
        #         rate_dpis = []
        rate_dpis = np.zeros((self.nebins, self.ndets))

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j = int(pname[-1])
            rate_dpis[j] += (
                self.diff_As[j] * params[pname] * self.solid_angs
                + self.flat_As[j] * params[pname]
            )

        return rate_dpis

    def get_rate_dpi(self, params, j):
        #         rate_dpis = []
        rate_dpi = np.zeros(self.ndets)

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0:
                continue
            rate_dpi += (
                self.diff_As[j] * params[pname] * self.solid_angs / self.solid_ang_mean
                + self.flat_As[j] * params[pname]
            )
        return rate_dpi

    def get_dr_dps(self, params):
        #         dr_dFlats = np.zeros((self.nebins,self.ndets))
        #         dr_dDifs = np.zeros((self.nebins,self.ndets))
        dr_dps = []

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j = int(pname[-1])
            if "dif" in pname:
                #                 dr_dDifs[j] += params[pname]*self.solid_angs
                dr_dps.append(self.solid_angs)
            else:
                dr_dps.append(self._rate_ones)
        #                 dr_dps[j] += params[pname]*self._rate_ones

        return dr_dps

    def get_dr_dp(self, params, j):
        #         dr_dFlats = np.zeros((self.nebins,self.ndets))
        #         dr_dDifs = np.zeros((self.nebins,self.ndets))
        dr_dps = []

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0:
                continue
            dr_dps.append(
                self.diff_As[j0] * self.solid_angs / self.solid_ang_mean
                + self.flat_As[j0]
            )

        return dr_dps

    def get_log_prior(self, params, j=None):
        lp = 0.0

        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0 and j is not None:
                continue
            lp += norm_logpdf(params[pname], self.bkg_sigs[j0], self.exp_rates[j0])
        #             lp += stats.norm.logpdf(params[pname], loc=self.exp_rates[j0],\
        #                                    scale=self.bkg_sigs[j0])

        return lp

    def get_dnlp_dp(self, params, j):
        dnlp_dps = []
        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0:
                continue
            dnlp_dps.append(
                (params[pname] - self.exp_rates[j0]) / np.square(self.bkg_sigs[j0])
            )

        return dnlp_dps


class Flux2Rate_4PBtrans(object):
    def __init__(self, response, flux_model):
        self.resp = response
        self.fmodel = flux_model
        self.drm_Ebins = np.append(self.resp.drm_e0s, self.resp.drm_e1s[-1])
        self.nebins = self.resp.nebins
        self.get_mus()
        self.cos_theta = 1.0
        self.get_trans(self.cos_theta)

    def get_mus(self):
        self.pb_rhomus = get_pb_mu(self.resp.drm_emids) * 11.35

    def get_trans(self, cos_theta):
        self.trans = np.exp(-0.1 * self.pb_rhomus / cos_theta)
        self.cos_theta = cos_theta
        return self.trans

    def get_rate_ebins(self, flux_params, cos_theta=None):
        photon_fluxes = self.fmodel.get_photon_fluxes(self.drm_Ebins, flux_params)

        if cos_theta is None:
            cos_theta = np.cos(
                np.arctan(np.hypot(flux_params["imx"], flux_params["imy"]))
            )
        if np.abs(self.cos_theta - cos_theta) > 0.05:
            self.get_trans(cos_theta)
        rate_mat = self.resp.get_rates_from_flux(photon_fluxes) * self.trans[:, None]
        rate_ebins = np.sum(rate_mat, axis=0)
        return rate_ebins

    def get_rate_ebin(self, flux_params, ebin, cos_theta=None):
        photon_fluxes = self.fmodel.get_photon_fluxes(self.drm_Ebins, flux_params)

        if cos_theta is None:
            cos_theta = np.cos(
                np.arctan(np.hypot(flux_params["imx"], flux_params["imy"]))
            )
        if np.abs(self.cos_theta - cos_theta) > 0.05:
            self.get_trans(cos_theta)
        rate_mat = self.resp.get_rate_from_flux(photon_fluxes, ebin) * self.trans
        rate_ebins = np.sum(rate_mat)
        return rate_ebins


def lognorm_logpdf(x, sig, mu):
    return (
        -np.log(sig)
        - ((np.log(x) - mu) ** 2) / (2.0 * sig**2)
        - np.log(x)
        - 0.5 * (np.log(2 * np.pi))
    )


def kum_pdf(x, a, b):
    return a * b * x ** (a - 1.0) * (1.0 - x**a) ** (b - 1.0)


def kum_mode(a, b):
    return ((a - 1) / (a * b - 1)) ** (1 / a)


def kum_logpdf(x, a, b):
    return np.log(a * b) + (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x**a)


def kum_deriv_logpdf(x, a, b):
    return (1.0 - a + (x**a) * (-1.0 + a * b)) / (x * (-1.0 + x**a))


def deriv2_kum_logpdf(x, a, b):
    return (
        -a
        + (1.0 - a * b) * x ** (2.0 * a)
        - (a - 1.0) * (a * (b - 1.0) - 2.0) * (x**a)
        + 1.0
    ) / np.square(x * (-1.0 + x**a))


class Point_Source_Model_Wuncoded(Model):
    # should have methods for getting rate/fully illuminated det
    # and for getting the correct ray trace

    def __init__(
        self,
        imx,
        imy,
        dimxy,
        flux_model,
        drm_obj,
        ebins,
        rt_obj,
        fp_obj,
        bl_dmask,
        name="Signal",
        use_deriv=False,
        use_rt_deriv=False,
        use_prior=False,
        prior_type="kum",
    ):
        self.dimxy = dimxy
        self.imx = imx
        self.imy = imy
        self.imx0 = imx - dimxy / 2.0
        self.imx1 = imx + dimxy / 2.0
        self.imy0 = imy - dimxy / 2.0
        self.imy1 = imy + dimxy / 2.0

        self.fmodel = flux_model

        self.drm_obj = drm_obj
        self.drm_im_update = 0.05

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        param_names = ["imx", "imy"]
        param_names += self.fmodel.param_names
        self.frac_names = ["uncoded_frac_" + str(i) for i in range(nebins)]
        param_names += self.frac_names
        #         param_names.append('scat_fact')

        print("param_names: ")
        print(param_names)

        param_bounds = {"imx": (self.imx0, self.imx1), "imy": (self.imy0, self.imy1)}

        for pname in self.fmodel.param_names:
            param_bounds[pname] = self.fmodel.param_bounds[pname]

        mus = np.linspace(-2.4, -1.8, nebins)
        sigs = np.linspace(0.85, 0.6, nebins)

        self.prior_mu = {self.frac_names[i]: mus[i] for i in range(nebins)}
        self.prior_sig = {self.frac_names[i]: sigs[i] for i in range(nebins)}

        kum_as = np.linspace(1.7, 2.2, nebins)
        kum_bs = 50 * np.ones(nebins)

        self.prior_kum_a = {self.frac_names[i]: kum_as[i] for i in range(nebins)}
        self.prior_kum_b = {self.frac_names[i]: kum_bs[i] for i in range(nebins)}

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "imx":
                pdict["bounds"] = (self.imx0, self.imx1)
                pdict["val"] = self.imx
            elif pname == "imy":
                pdict["bounds"] = (self.imy0, self.imy1)
                pdict["val"] = self.imy
            elif pname == "d":
                pdict["bounds"] = (1e-4, 1.0)
                pdict["val"] = 1e-1
            elif "uncoded_frac" in pname:
                pdict["bounds"] = (1e-4, 0.75)
                pdict["val"] = 0.1
            else:
                pdict["bounds"] = self.fmodel.param_bounds[pname]
                if hasattr(self.fmodel, "param_guess"):
                    pdict["val"] = self.fmodel.param_guess[pname]
                else:
                    pdict["val"] = (pdict["bounds"][1] + pdict["bounds"][0]) / 2.0
            pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Point_Source_Model_Wuncoded, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        if use_deriv:
            self.has_deriv = True

        self.drm_obj = drm_obj
        self.set_abs_cor_type()
        self.set_resp(imx, imy)

        self.rt_obj = rt_obj
        self.fp_obj = fp_obj

        self._rt_im_update = 1e-7
        self._rt_imx = imx - 10.0
        self._rt_imy = imy - 10.0
        self._fp_im_update = 1e-4
        self._fp_imx = imx - 10.0
        self._fp_imy = imy - 10.0
        self.use_rt_deriv = use_rt_deriv
        if self.use_rt_deriv:
            self._rt, self._drt_dx, self._drt_dy = self.get_rt_wderiv(imx, imy)
        else:
            self._rt = self.get_rt(imx, imy)
        self._fp = self.get_fp(imx, imy)
        self._rt_imx = imx
        self._rt_imy = imy

        self._last_params_ebin = [{} for i in range(nebins)]
        self._last_rate_dpi = [np.ones(self.ndets) for i in range(nebins)]

        self.uncoded = 1.0 - self._fp
        self.pc = (self.ndets - np.sum(self.uncoded)) / self.ndets
        self.ones = np.ones(self.ndets)
        #         self.ndets_uncoded = np.sum(self.uncoded)
        #         self.ndets_coded = self.ndets - self.ndets_uncoded

        self.prior_type = prior_type
        if prior_type == "log_uniform":
            self.prior_func = self.log_uniform_prior
            self.deriv_prior_func = self.deriv_log_uniform_prior
        elif prior_type == "kum":
            self.prior_func = self.kum_prior
            self.deriv_prior_func = self.deriv_kum_prior
        elif prior_type == "log_norm":
            self.prior_func = self.log_norm_prior
            self.deriv_prior_func = self.deriv_log_norm_prior

    def set_abs_cor_type(self, cor_type="op"):
        self.cor_type = cor_type

    def set_resp(self, imx, imy):
        self.drm = self.drm_obj.get_drm(imx, imy)

        self.resp = Response(self.drm, ebins=self.ebins, cor_type=self.cor_type)

        self.flux2rate = Flux2Rate(self.resp, self.fmodel)
        self.flux2rate_pbtrans = Flux2Rate_4PBtrans(self.resp, self.fmodel)
        #         self.flux2rate_uncoded = Flux2Rate_4UnCoded(self.resp, self.fmodel)
        #         self.flux2rate_scat = Flux2Rate_4Scattered(self.resp, self.fmodel)

        if self.has_deriv:
            self.flux2rate.setup_gamma_deriv()

        # either have drm updating here or in response object

    def get_flux2rate(self, imx, imy):
        if im_dist(imx, imy, self.resp.imx, self.resp.imy) > self.drm_im_update:
            self.set_resp(imx, imy)
        return self.flux2rate

    def get_rates(self, flux_params, imx=None, imy=None):
        if (imx is not None) and (imy is not None):
            if im_dist(imx, imy, self.resp.imx, self.resp.imy) > self.drm_im_update:
                self.set_resp(imx, imy)
        return self.flux2rate.get_rate_ebins(flux_params)

    def get_rates_uncoded(self, flux_params):
        return self.flux2rate_pbtrans.get_rate_ebins(flux_params)

    def get_rate(self, flux_params, ebin, imx=None, imy=None):
        if (imx is not None) and (imy is not None):
            if im_dist(imx, imy, self.resp.imx, self.resp.imy) > self.drm_im_update:
                self.set_resp(imx, imy)
        return self.flux2rate.get_rate_ebin(flux_params, ebin)

    def get_rate_uncoded(self, flux_params, ebin):
        return self.flux2rate_pbtrans.get_rate_ebin(flux_params, ebin)

    def get_rates_scat(self, flux_params, element="pb"):
        return self.flux2rate_scat.get_rate_ebins(
            flux_params, flux_params["scat_fact"], self.pc, element=element
        )

    def get_rt_wderiv(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) > self._rt_im_update:
            rt, drt_dx, drt_dy = self.rt_obj.get_intp_rt(imx, imy, get_deriv=True)
            self._rt = rt[self.bl_dmask]
            self._drt_dx = drt_dx[self.bl_dmask]
            self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy

        return self._rt, self._drt_dx, self._drt_dy

    def get_rt(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) < self._rt_im_update:
            return self._rt
        else:
            rt = self.rt_obj.get_intp_rt(imx, imy, get_deriv=False)
            self._rt = rt[self.bl_dmask]
            self._shadow = 1.0 - self._rt
            fp = self.get_fp(imx, imy)
            self._shadow[self.uncoded] = 0.0
            # self._drt_dx = drt_dx[self.bl_dmask]
            # self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy

        return self._rt

    def get_fp(self, imx, imy):
        if np.hypot(imx - self._fp_imx, imy - self._fp_imy) < self._fp_im_update:
            return self._fp
        else:
            fp = self.fp_obj.get_fp(imx, imy)
            self._fp = fp[self.bl_dmask].astype(np.int64)
            self._unfp = 1 - self._fp
            self.uncoded = self._fp < 0.1
            # self._drt_dx = drt_dx[self.bl_dmask]
            # self._drt_dy = drt_dy[self.bl_dmask]
            self._fp_imx = imx
            self._fp_imy = imy

        return self._fp

    def get_rate_dpis(self, params):
        # need to fix this

        imx = params["imx"]
        imy = params["imy"]

        rate_pdet_ebins = self.get_rates(params, imx=imx, imy=imy)

        rate_pdet_ebins_uncoded = self.get_rates_uncoded(params)
        #         rate_pdet_ebins_scat = self.get_rates_uncoded(params)

        rt = self.get_rt(imx, imy)
        shadow = 1.0 - rt
        shadow[(self.uncoded > 0.1)] = 0.0

        rate_dpis = np.array(
            [
                rt * rate_pdet
                + (shadow) * rate_pdet_ebins_uncoded[i]
                + (self.uncoded) * rate_pdet * params[self.frac_names[i]]
                for i, rate_pdet in enumerate(rate_pdet_ebins)
            ]
        )

        return rate_dpis

    def get_rate_dpi(self, params, j):
        if params == self._last_params_ebin[j]:
            return self._last_rate_dpi[j]

        imx = params["imx"]
        imy = params["imy"]

        # should add way to calc this for only ebin j
        rate_pdet = self.get_rate(params, j, imx=imx, imy=imy)

        rate_pdet_uncoded = self.get_rate_uncoded(params, j)
        #         rate_pdet_ebins_scat = self.get_rates_uncoded(params)

        rt = self.get_rt(imx, imy)
        #         fp = self.get_fp(imx, imy)
        #         shadow = (1. - rt)
        #         shadow[(self.uncoded>.1)] = 0.0

        rate_dpi = (
            rt * rate_pdet
            + self._shadow * rate_pdet_uncoded
            + (self._unfp) * rate_pdet * params[self.frac_names[j]]
        )

        #         rate_dpis = np.array([rt*rate_pdet + (shadow)*rate_pdet_ebins_uncoded[i] +\
        #                               (self.uncoded)*rate_pdet*params[self.frac_names[i]]\
        #                               for i, rate_pdet in enumerate(rate_pdet_ebins)])

        self._last_params_ebin[j] = params
        self._last_rate_dpi[j] = rate_dpi

        return rate_dpi

    def log_uniform_prior(self, params, pname):
        return -np.log(
            (
                params[pname]
                * (
                    np.log(self.param_dict[pname]["bounds"][1])
                    - np.log(self.param_dict[pname]["bounds"][0])
                )
            )
        )

    def deriv_log_uniform_prior(self, params, pname):
        return 1.0 / params[pname]

    def log_norm_prior(self, params, pname):
        return lognorm_logpdf(
            params[pname], self.prior_sig[pname], self.prior_mu[pname]
        )

    def deriv_log_norm_prior(self, params, pname):
        return (
            -self.prior_mu[pname] + self.prior_sig[pname] ** 2 + np.log(params[pname])
        ) / (params[pname] * self.prior_sig[pname] ** 2)

    def kum_prior(self, params, pname):
        return kum_logpdf(
            params[pname], self.prior_kum_a[pname], self.prior_kum_b[pname]
        )

    def deriv_kum_prior(self, params, pname):
        return -kum_deriv_logpdf(
            params[pname], self.prior_kum_a[pname], self.prior_kum_b[pname]
        )

    def get_log_prior(self, params, j=None):
        lp = 0.0
        for pname in self.frac_names:
            if int(pname[-1]) == j or j is None:
                lp += self.prior_func(params, pname)
        #                 lp -= np.log((params[pname]*(np.log(\
        #                         self.param_dict[pname]['bounds'][1]) -\
        #                         np.log(self.param_dict[pname]['bounds'][0]))))
        return lp

    def get_dnlp_dp(self, params, j):
        dnlp_dps = []
        for pname in self.frac_names:
            if int(pname[-1]) == j or j is None:
                #                 dnlp_dps.append( 1./params[pname] )
                dnlp_dps.append(self.deriv_prior_func(params, pname))

        return dnlp_dps

    def get_dr_dp(self, params, j):
        dr_dps = []

        imx = params["imx"]
        imy = params["imy"]

        if self.use_rt_deriv:
            rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
            fp = self.get_fp(imx, imy)
        else:
            rt = self.get_rt(imx, imy)
            fp = self.get_fp(imx, imy)

        rate_pdet = self.get_rate(params, j, imx=imx, imy=imy)

        for pname in self.param_names:
            if self.param_dict[pname]["fixed"]:
                continue
            if pname in self.frac_names:
                if int(pname[-1]) != j:
                    continue

            dr_dps.append(self._unfp * rate_pdet)

        return dr_dps

    def get_dr_dps(self, params):
        imx = params["imx"]
        imy = params["imy"]

        if self.use_rt_deriv:
            rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
        else:
            rt = self.get_rt(imx, imy)

        rate_pdet_ebins = self.get_rates(params, imx=imx, imy=imy)

        dR_dA = rate_pdet_ebins / params["A"]

        dR_dG = params["A"] * self.flux2rate.get_gamma_deriv(params["gamma"])

        dr_da = np.array([rt * drdA for drdA in dR_dA])

        dr_dg = np.array([rt * drdG for drdG in dR_dG])

        #         dr_dx = np.array([drt_dx*rate_pdet for rate_pdet in rate_pdet_ebins])
        #         dr_dy = np.array([drt_dy*rate_pdet for rate_pdet in rate_pdet_ebins])

        if self.param_dict["imx"]["fixed"]:
            return [dr_da, dr_dg]

        dr_dimx = rate_pdet_ebins[:, np.newaxis] * drt_dimx
        dr_dimy = rate_pdet_ebins[:, np.newaxis] * drt_dimy
        return [dr_dimx, dr_dimy, dr_da, dr_dg]


class footprint_file_npy(object):
    # maybe add the update im_dist thing here too
    # and save the last ray trace made

    def __init__(self, fp_arr0, fp_dir, rng=0.02, rtstep=0.002, mmap=False):
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        imx_ax = (imxs[1:] + imxs[:-1]) / 2.0
        imy_ax = (imys[1:] + imys[:-1]) / 2.0
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.nbytes = 0.0

        self.rng = rng
        self.rtstep = rtstep
        self.imx0 = fp_arr0["imx0"]
        self.imy0 = fp_arr0["imy0"]
        self.imx1 = fp_arr0["imx1"]
        self.imy1 = fp_arr0["imy1"]
        self.fname = os.path.join(fp_dir, fp_arr0["fname"])
        logging.debug("opening file: " + fp_arr0["fname"])
        try:
            self.datas = np.load(self.fname)
        except Exception as E:
            logging.warning("couldn't open file " + self.fname)
            logging.error(traceback.format_exc())
            return False

        self.nbytes = self.datas.nbytes

        self.imxs = _imxs_rt_file + self.imx0
        self.imys = _imys_rt_file + self.imy0
        self.grid_shape = grids[0].shape
        self.imx_ax = imx_ax + self.imx0
        self.imy_ax = imy_ax + self.imy0
        #         self.setup_intp()
        #         self.nbytes *= 4
        self.last_used = time.time()

    def get_fp_from_ind(self, ind):
        # return self.file[ind].data
        return self.datas[ind]

    def get_fp(self, imx, imy):
        ind = np.argmin(np.hypot(imx - self.imxs, imy - self.imys))
        # return self.file[ind].data
        return self.datas[ind]

    def get_fp_corners(self, imx, imy):
        x0_ind = np.argmin(np.abs(imx - self.imx_ax))
        x1_ind = x0_ind + 1
        y0_ind = np.argmin(np.abs(imy - self.imy_ax))
        y1_ind = y0_ind + 1
        ravel_ind00 = np.ravel_multi_index((x0_ind, y0_ind), self.grid_shape)
        ravel_ind10 = np.ravel_multi_index((x1_ind, y0_ind), self.grid_shape)
        ravel_ind01 = np.ravel_multi_index((x0_ind, y1_ind), self.grid_shape)
        ravel_ind11 = np.ravel_multi_index((x1_ind, y1_ind), self.grid_shape)

        inds = [ravel_ind00, ravel_ind10, ravel_ind01, ravel_ind11]

        fps = [
            [self.get_fp_from_ind(inds[0]), self.get_fp_from_ind(inds[2])],
            [self.get_fp_from_ind(inds[1]), self.get_fp_from_ind(inds[3])],
        ]
        imxs = []
        imys = []

        for ind in inds:
            #             ray_traces.append(self.get_rt_from_ind(ind))
            imxs.append(self.imxs[ind])
            imys.append(self.imys[ind])

        return fps, imxs, imys

    def get_closest_ind(self, imx, imy):
        return np.argmin(np.hypot(imx - self.imxs, imy - self.imys))

    def get_closest_ax_inds(self, imx, imy):
        xind = np.argmin(np.abs(imx - self.imx_ax))
        yind = np.argmin(np.abs(imy - self.imy_ax))
        return xind, yind


class FootPrints(object):
    def __init__(
        self,
        fp_dir,
        rng=0.02,
        rtstep=0.002,
        mmap=False,
        im_dist_update=1e-4,
        max_nbytes=1e9,
        npy=True,
        ident="footprint",
    ):
        logging.info(rng)
        logging.info(rtstep)
        self.rng = rng
        self.rtstep = rtstep
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.fp_dir = fp_dir
        self.mmap = mmap
        self.fp_arr = get_rt_arr(self.fp_dir, ident=ident)
        self.nbytes = 0.0
        self.max_nbytes = max_nbytes
        self.npy = npy

        # use fnames as the dictionary keys
        self.fp_files = {}

        self._last_imx = 10.0
        self._last_imy = 10.0
        self._last_dimx = 10.0
        self._last_dimy = 10.0
        self._update_im_dist = im_dist_update

    def close_fp_file_obj(self, fp_arr_ind):
        k = self.fp_arr[fp_arr_ind]["fname"]
        logging.debug("nbytes=" + str(self.nbytes))
        self.nbytes -= self.fp_files[k].nbytes
        logging.debug("closing file " + k)
        del self.fp_files[k]
        self.fp_arr["time"][fp_arr_ind] = np.nan
        gc.collect()

    def mem_check(self):
        if self.nbytes > self.max_nbytes:
            ind2close = np.nanargmin(self.fp_arr["time"])
            self.close_fp_file_obj(ind2close)

    def open_fp_file_obj(self, fp_arr0):
        if self.npy:
            fp_file = footprint_file_npy(
                fp_arr0, self.fp_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        else:
            rt_file = ray_trace_file(
                rt_arr0, self.rt_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        self.nbytes += fp_file.nbytes
        logging.debug("nbytes_total=" + str(self.nbytes))
        self.mem_check()
        self.fp_files[fp_arr0["fname"]] = fp_file

    def get_fp_file_obj(self, imx, imy):
        ind = np.argmin(
            np.hypot(
                imx - (self.fp_arr["imx0"] + self.fp_arr["imx1"]) / 2.0,
                imy - (self.fp_arr["imy0"] + self.fp_arr["imy1"]) / 2.0,
            )
        )

        fp_arr0 = self.fp_arr[ind]

        if not (
            (fp_arr0["imx0"] <= imx <= fp_arr0["imx1"])
            and (fp_arr0["imy0"] <= imy <= fp_arr0["imy1"])
        ):
            logging.warning(
                "No footprint files for this imx, imy: %.3f, %.3f" % (imx, imy)
            )

        fname = fp_arr0["fname"]

        if fname not in list(self.fp_files.keys()):
            self.open_fp_file_obj(fp_arr0)
        self.fp_arr["time"][ind] = time.time()
        return self.fp_files[fname]

    def check_im_dist(self, imx, imy, deriv=False):
        return (
            np.hypot(imx - self._last_imx, imy - self._last_imy) < self._update_im_dist
        )

    def get_fp(self, imx, imy):
        if not self.check_im_dist(imx, imy):
            fp_file = self.get_fp_file_obj(imx, imy)
            self._last_footprint = fp_file.get_fp(imx, imy)
            self._last_imx = imx
            self._last_imy = imy

        return self._last_footprint


class LLH_webins(object):
    def __init__(
        self, event_data, ebins0, ebins1, bl_dmask, t0=None, dt=None, model=None
    ):
        self._all_data = event_data
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        self.bl_dmask = bl_dmask
        self.t0 = 0.0
        self.dt = 0.0
        self.ebin = -1

        if t0 is not None and dt is not None:
            self.set_time(t0, dt)

        if model is not None:
            self.set_model(model)

    def set_time(self, t0, t1):
        """
        Sets the start time and duration for the LLH
        analysis.

        Parameters:
        t0: start time in MET seconds
        dt: duration in seconds
        """

        if np.isscalar(t0):
            t0 = np.array([t0])
        if np.isscalar(t1):
            t1 = np.array([t1])

        if np.all(self.t0 == t0) and np.all(self.t1 == t1):
            return
        self.t0 = t0
        self.dt = 0.0
        self.t1 = t1

        t_bl = np.zeros(len(self._all_data), dtype=bool)
        for i in range(len(self.t0)):
            t_bl = np.logical_or(
                (self._all_data["TIME"] >= self.t0[i])
                & (self._all_data["TIME"] < self.t1[i]),
                t_bl,
            )
            self.dt += self.t1[i] - self.t0[i]
        self.data = self._all_data[t_bl]

        self.data_dpis = np.array(
            det2dpis(self.data, self.ebins0, self.ebins1, bl_dmask=self.bl_dmask)
        )
        self.gamma_vals = get_gammaln(self.data_dpis)

    def set_model(self, model):
        self.model = model
        self.nparams = self.model.nparams

    def set_ebin(self, j):
        if "all" in str(j):
            self.ebin = -1
        else:
            self.ebin = j

    def get_llh(self, params):
        if self.ebin < 0:
            mod_cnts = self.model.get_rate_dpis(params) * self.dt
            if np.any(mod_cnts <= 0):
                return -np.inf
            llh = np.sum(
                log_pois_prob(mod_cnts, self.data_dpis, gam_val=self.gamma_vals)
            )

        else:
            mod_cnts = self.model.get_rate_dpi(params, self.ebin) * self.dt
            if np.any(mod_cnts <= 0):
                return -np.inf
            llh = np.sum(
                log_pois_prob(
                    mod_cnts,
                    self.data_dpis[self.ebin],
                    gam_val=self.gamma_vals[self.ebin],
                )
            )

        return llh

    def get_logprior(self, params):
        lp = 0.0
        if self.model.has_prior:
            if self.ebin < 0:
                lp = self.model.get_log_prior(params)
            else:
                lp = self.model.get_log_prior(params, j=self.ebin)
        return lp

    def get_logprob(self, params):
        logp = self.get_logprior(params)

        llh = self.get_llh(params)

        return logp + llh

    def get_logprob_jacob(self, params):
        if self.ebin < 0:
            mod_cnts = self.model.get_rate_dpis(params) * self.dt
            if np.any(np.isclose(mod_cnts, 0)):
                mod_cnts = 1e-6 * np.ones_like(mod_cnts)

            fact = 1.0 - (self.data_dpis / mod_cnts)

            dNs_dparam = self.model.get_dr_dps(params)

            jacob = [
                np.sum(fact * dNs_dparam[i]) * self.dt for i in range(len(dNs_dparam))
            ]

        else:
            mod_cnts = self.model.get_rate_dpi(params, self.ebin) * self.dt
            if np.any(np.isclose(mod_cnts, 0)):
                mod_cnts = 1e-6 * np.ones_like(mod_cnts)

            fact = 1.0 - (self.data_dpis[self.ebin] / mod_cnts)

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            if self.model.has_prior:
                dNLP_dparams = self.model.get_dnlp_dp(params, self.ebin)
            else:
                dNLP_dparams = np.zeros(len(dR_dparams))

            jacob = [
                dNLP_dparams[i] + np.sum(fact * dR_dparams[i]) * self.dt
                for i in range(len(dR_dparams))
            ]

        return jacob


def log_prob2min(frac_bkg_rate, bkg_exp_rate, bkg_err, llh_obj, params, ebin):
    try:
        frac = frac_bkg_rate[0]
        bkg_rate = frac_bkg_rate[1]
        if frac <= 0 or frac >= 1.0 or bkg_rate <= 0:
            return np.inf
        params["Signal_uncoded_frac_" + str(ebin)] = frac
        params["Background_bkg_rate_" + str(ebin)] = bkg_rate
        nllh = -llh_obj.get_llh(params)
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
    return (
        nllh
        + np.log(frac * (np.log(0.5) - np.log(1e-3)))
        - stats.norm.logpdf(bkg_rate, scale=bkg_err, loc=bkg_exp_rate)
    )


def var_around_min(frac_min, bkg_min, bkg_err, llh_obj, params, ebin, sig_mod, bkg_mod):
    params = copy(params)
    params["Signal_uncoded_frac_" + str(ebin)] = frac_min
    params["Background_bkg_rate_" + str(ebin)] = bkg_min
    dt = llh_obj.dt
    ndets = llh_obj.model.ndets
    mod_cnts = llh_obj.model.get_rate_dpi(params, ebin) * dt
    data_cnts = llh_obj.data_dpis[ebin]

    imx, imy = params["Signal_imx"], params["Signal_imy"]
    foot_print = sig_mod.get_fp(imx, imy)
    uncoded = 1.0 - foot_print
    params_ = {}
    params_["A"] = params["Signal_A"]
    params_["gamma"] = params["Signal_gamma"]
    S_rate = dt * sig_mod.get_rates(params_, imx=imx, imy=imy)[ebin]

    dr_df = uncoded * S_rate
    #     dr_db = dt*np.ones(llh_obj.model.ndets)/ndets
    dr_db = dt * bkg_mod.get_dr_dp(params, ebin)[0]

    dnLLH2_df = np.sum(np.square(dr_df) * data_cnts / np.square(mod_cnts))

    dnLLH2_db = np.sum(np.square(dr_db) * data_cnts / np.square(mod_cnts))

    dnLLH2_dfb = np.sum(dr_df * dr_db * data_cnts / np.square(mod_cnts))

    #     dnLP_df2 = 1./frac_min**2

    if sig_mod.prior_type == "log_norm":
        frac_mu = sig_mod.prior_mu[sig_mod.frac_names[ebin]]
        frac_sig = sig_mod.prior_sig[sig_mod.frac_names[ebin]]
        dnLP_df2 = -(-frac_mu + frac_sig**2 + np.log(frac_min) - 1) / np.square(
            frac_sig * frac_min
        )
    elif sig_mod.prior_type == "kum":
        frac_a = sig_mod.prior_kum_a[sig_mod.frac_names[ebin]]
        frac_b = sig_mod.prior_kum_b[sig_mod.frac_names[ebin]]
        dnLP_df2 = -deriv2_kum_logpdf(frac_min, frac_a, frac_b)

    dnLP_db2 = 1.0 / (bkg_err) ** 2

    return 1.0 / (dnLLH2_df + dnLP_df2), 1.0 / (dnLLH2_db + dnLP_db2), 1.0 / dnLLH2_dfb


def integrated_LLH_webins(
    llh_obj, miner, params, bl_dmask, bkg_rates, bkg_errs, sig_mod, bkg_mod
):
    nebins = llh_obj.nebins

    log_prob = 0.0
    global Nfunc_calls
    bf_params = {}

    for j in range(nebins):
        miner.set_fixed_params(list(params.keys()), values=list(params.values()))
        # logging.debug('Params: ')
        # logging.debug(params)
        e0_pnames = []
        for pname in miner.param_names:
            try:
                if int(pname[-1]) == j:
                    e0_pnames.append(pname)
            except:
                pass
        miner.set_fixed_params(e0_pnames, fixed=False)
        # logging.debug('e0_pnames: ')
        # logging.debug(e0_pnames)

        llh_obj.set_ebin(j)

        bf_vals, nllh, res = miner.minimize()
        nllh = nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]
            if "frac" in pname:
                bf_frac = bf_vals[0][ii]
            elif "bkg_rate" in pname:
                bf_bkg_rate = bf_vals[0][ii]

        #         res=optimize.fmin(log_prob2min, [.1, 1e-2], args=(bkg_rates[j],\
        #                                 bkg_errs[j], llh_obj, params, j),\
        #                           full_output=True, disp=0)

        #         Nfunc_calls += res[3]
        frac_var, bkg_var, fb_cov = var_around_min(
            bf_frac, bf_bkg_rate, bkg_errs[j], llh_obj, params, j, sig_mod, bkg_mod
        )

        I_mat = np.matrix(
            [[1.0 / frac_var, 1.0 / fb_cov], [1.0 / fb_cov, 1.0 / bkg_var]]
        )
        gauss_integral = np.log(1.0 / np.sqrt(np.linalg.det(I_mat / (2.0 * (np.pi)))))
        log_prob += -nllh + gauss_integral
    #     print "Nfunc_calls: ", Nfunc_calls
    # print bf_params
    return log_prob


def int_LLH2min(
    A, llh_obj, miner, params, sig_mod, bkg_mod, bl_dmask, bkg_rates, bkg_errs
):
    params["Signal_A"] = A
    miner.set_fixed_params(["Signal_A"], [A])
    return -integrated_LLH_webins(
        llh_obj, miner, params, bl_dmask, bkg_rates, bkg_errs, sig_mod, bkg_mod
    )


def int_LLH2min_Agamma(
    A_gamma, llh_obj, miner, params, sig_mod, bkg_mod, bl_dmask, bkg_rates, bkg_errs
):
    if A_gamma[1] < 0 or A_gamma[0] < 1e-5 or A_gamma[1] > 2.5:
        return np.inf
    params["Signal_A"] = A_gamma[0]
    params["Signal_gamma"] = A_gamma[1]
    miner.set_fixed_params(["Signal_A"], [A_gamma[0]])
    miner.set_fixed_params(["Signal_gamma"], [A_gamma[1]])
    return -integrated_LLH_webins(
        llh_obj, miner, params, bl_dmask, bkg_rates, bkg_errs, sig_mod, bkg_mod
    )


def do_analysis(
    seed_tab,
    pl_flux,
    drm_obj,
    rt_dir,
    fp_dir,
    bkg_llh_obj,
    sig_llh_obj,
    trigger_time,
    work_dir,
    jobid,
):
    bkg_rates = np.array(
        [0.07609196, 0.05611105, 0.04255688, 0.03599852, 0.02635129, 0.02274061]
    )
    bkg_errs = 0.05 * bkg_rates
    bkg_flat_diff_ratios = np.array(
        [0.00065753, 0.00089188, 0.16403959, 0.00139088, 0.00190105, 0.4973408]
    )

    ebins0 = sig_llh_obj.ebins0
    ebins1 = sig_llh_obj.ebins1
    nebins = len(ebins0)
    bl_dmask = sig_llh_obj.bl_dmask

    tgrps = seed_tab.groupby("timeID")
    Ntbins = len(tgrps)
    logging.info("%d times to do" % (Ntbins))

    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    # bkg_miner = NLLH_ScipyMinimize('')
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    rt_obj = RayTraces(rt_dir, max_nbytes=4e9)
    fp_obj = FootPrints(fp_dir)

    for timeID, seeds in tgrps:
        logging.debug("")
        logging.debug("***************************")
        logging.debug("")
        logging.info("Starting timeID: %d" % (timeID))

        Npix = len(seeds)
        logging.debug("%d Pixels to minimize at" % (Npix))
        logging.debug("")
        logging.debug("***************************")
        logging.debug("")

        if Npix < 1:
            fname = os.path.join(work_dir, "int_llh_res_%d_%d_.csv" % (timeID, jobid))
            logging.info("Nothing to write for timeID %d" % (timeID))
            f = open(fname, "w")
            f.write("NONE")
            f.close()
            continue

        imxs = seeds["imx"].values
        imys = seeds["imy"].values

        t0 = seeds["time"].values[0]
        dt = seeds["duration"].values[0]
        t1 = t0 + dt

        logging.debug("Starting time: %.3f, with duration %.3f" % (t0, dt))

        res_dict = {"timeID": timeID}

        tmid = t0 + dt / 2.0
        res_dict["time"] = t0
        res_dict["duration"] = dt

        sig_llh_obj.set_time(t0, t1)

        bkg_mod = Bkg_Model_wSAfixed(
            bl_dmask,
            solid_ang_dpi,
            nebins,
            bkg_flat_diff_ratios,
            exp_rates=bkg_rates,
            bkg_sigs=bkg_errs,
            use_prior=True,
            use_deriv=True,
        )

        imx_, imy_ = np.nanmean(imxs), np.nanmean(imys)

        sig_mod = Point_Source_Model_Wuncoded(
            imx_,
            imy_,
            0.3,
            pl_flux,
            drm_obj,
            [ebins0, ebins1],
            rt_obj,
            fp_obj,
            bl_dmask,
            use_deriv=True,
            use_prior=True,
        )

        sig_mod.drm_im_update = 0.2
        comp_mod = CompoundModel([bkg_mod, sig_mod])
        sig_llh_obj.set_model(comp_mod)
        sig_miner = NLLH_ScipyMinimize_Wjacob("")
        sig_miner.set_llh(sig_llh_obj)

        params_ = {
            bkg_mod.name + "_" + bkg_mod.param_names[j]: bkg_rates[j]
            for j in range(nebins)
        }
        for pname in sig_mod.param_names:
            params_["Signal_" + pname] = sig_mod.param_dict[pname]["val"]

        TSs = np.zeros(Npix)
        sig_nllhs = np.zeros(Npix)
        As = np.zeros(Npix)
        gammas = np.zeros(Npix)
        imxs_ = np.zeros(Npix)
        imys_ = np.zeros(Npix)
        Nfevs = np.zeros(Npix, dtype=np.int64)
        Statuss = np.zeros(Npix, dtype=np.int64)

        for ii in range(Npix):
            params_["Signal_imx"] = imxs[ii]
            params_["Signal_imy"] = imys[ii]
            try:
                logging.debug(
                    "iter: %d, imx: %.4f, imy: %.4f" % (ii, imxs[ii], imys[ii])
                )
                res = optimize.minimize(
                    int_LLH2min_Agamma,
                    [0.025, 0.75],
                    method="Nelder-Mead",
                    args=(
                        sig_llh_obj,
                        sig_miner,
                        params_,
                        sig_mod,
                        bkg_mod,
                        bl_dmask,
                        bkg_rates,
                        bkg_errs,
                    ),
                    options={"fatol": 1e-2, "xatol": 1e-3, "maxfev": 360},
                )

                # res = optimize.fmin(int_LLH2min, [.1], args=(sig_llh_obj, params_,\
                #                 sig_mod, bl_dmask,\
                #                 bkg_rates, bkg_errs), full_output=True)
                sig_nllhs[ii] = res.fun
                As[ii] = res.x[0]
                gammas[ii] = res.x[1]
                imxs_[ii] = imxs[ii]
                imys_[ii] = imys[ii]
                Nfevs[ii] = res.nfev
                Statuss[ii] = res.status
                logging.debug("res: ")
                logging.debug(res)

                if not res.success:
                    logging.warning("Min failed")
                    logging.warning(res)

            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Failed to minimize seed: ")
                logging.error((imxs[ii], imys[ii]))

        try:
            logging.debug("Min NLLH: %.2f" % (np.nanmin(sig_nllhs[(sig_nllhs > 0)])))
        except:
            pass

        res_dict["imx"] = imxs_
        res_dict["imy"] = imys_
        res_dict["A"] = As
        res_dict["ind"] = gammas
        res_dict["sig_nllh"] = sig_nllhs
        res_dict["Nfevs"] = Nfevs
        res_dict["Status"] = Statuss
        res_df = pd.DataFrame(res_dict)

        fname = os.path.join(work_dir, "int_llh_res_%d_%d_.csv" % (timeID, jobid))

        res_df.to_csv(fname)
        logging.info("Saved results to")
        logging.info(fname)


def main(args):
    fname = "int_llh_scan_" + str(args.job_id)

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    t_0 = time.time()

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

    evfname = files_tab["evfname"][0]
    ev_data = fits.open(evfname)[1].data
    dmask_fname = files_tab["detmask"][0]
    dmask = fits.open(dmask_fname)[0].data
    bl_dmask = dmask == 0.0
    logging.debug("Opened up event and detmask files")

    # bkg_fits_df = pd.read_csv(args.bkg_fname)

    # rate_fits_df = get_rate_fits_tab(conn)
    # bkg_rates_obj = rate_obj_from_sqltab(rate_fits_df, 0, 1)

    time_starting = time.time()
    proc_num = args.job_id
    # init classes up here

    drm_dir = files_tab["drmDir"][0]
    if args.rt_dir is None:
        rt_dir = files_tab["rtDir"][0]
    else:
        rt_dir = args.rt_dir
    drm_obj = DRMs(drm_dir)
    # rt_obj = RayTraces(rt_dir, max_nbytes=1e10)
    work_dir = files_tab["workDir"][0]

    pl_flux = Plaw_Flux()

    ebins0 = np.array(config.EBINS0)
    ebins1 = np.array(config.EBINS1)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    bkg_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    try:
        good_pix = np.load(args.pix_fname)
    except Exception as E:
        logging.error(E)
        logging.warning("No pix2scan file")

    PC = fits.open(args.pcfname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key="T")

    # pcbl = (pc>=0.1)
    # pc_inds = np.where(pcbl)
    # pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)
    # logging.debug("Min pc_imx, pc_imy: %.2f, %.2f" %(np.nanmin(pc_imxs), np.nanmin(pc_imys)))
    # logging.debug("Max pc_imx, pc_imy: %.2f, %.2f" %(np.nanmax(pc_imxs), np.nanmax(pc_imys)))

    # conn = get_conn(db_fname)
    # if proc_num >= 0:
    #     square_tab = get_square_tab(conn, proc_group=proc_num)
    # else:
    #     square_tab = get_square_tab(conn)

    square_tab = pd.read_csv(args.job_fname)
    bl = square_tab["proc_group"] == proc_num
    square_tab = square_tab[bl]

    seed_tab = pd.read_csv(args.seed_fname)
    bl = seed_tab["proc_group"] == proc_num
    seed_tab = seed_tab[bl]

    # rate_res_tab = pd.read_csv(args.rate_fname)
    #
    # logging.info("Read in Square and Rates Tables, now to do analysis")

    fp_dir = "/gpfs/scratch/jjd330/bat_data/footprints_npy/"

    do_analysis(
        seed_tab,
        pl_flux,
        drm_obj,
        rt_dir,
        fp_dir,
        bkg_llh_obj,
        sig_llh_obj,
        trigtime,
        work_dir,
        proc_num,
    )
    # do_analysis(square_tab, rate_res_tab, good_pix['imx'], good_pix['imy'], pl_flux,\
    #                 drm_obj, rt_dir,\
    #                 bkg_llh_obj, sig_llh_obj,\
    #                 conn, db_fname, trigtime, work_dir,bkg_fits_df)
    conn.close()


if __name__ == "__main__":
    args = cli()

    main(args)
