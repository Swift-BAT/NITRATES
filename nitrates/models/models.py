import numpy as np
import abc
from scipy import stats, interpolate
import logging, traceback
from copy import copy, deepcopy
import healpy as hp

from ..response.response import Response, ResponseInFoV, ResponseInFoV2
from ..models.flux_models import Plaw_Flux
from ..lib.trans_func import get_pb_absortion, get_pb_mu
from ..lib.stat_funcs import Norm_1D, Norm_2D, Norm_3D
from ..lib.hp_funcs import ang_sep
from ..lib.coord_conv_funcs import imxy2theta_phi, theta_phi2imxy
from ..config import RESP_TAB_DNAME, HP_FLOR_RESP_DNAME, COMP_FLOR_RESP_DNAME

import six


# have model classes for all things that contribute counts
# like signal, bkg, bright sources
# the model class will take the relavent paramters
# and return the expected counts

# The LLH object should be able to contain a list of these
# and get the expected counts from each of them

# So the Model objects never need to touch the actual data

# Should probably make a base class


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


def get_plaw_and_bkg_mod(
    imx,
    imy,
    ebins0,
    ebins1,
    bl_dmask,
    rt_obj,
    drm_obj,
    dimxy=0.01,
    bkg_mod=None,
    bkg_rate_obj=None,
    t=0.0,
):
    pl_flux = Plaw_Flux()

    sig_mod = Point_Source_Model(
        imx, imy, dimxy, pl_flux, drm_obj, [ebins0, ebins1], rt_obj, bl_dmask
    )

    if bkg_mod is None:
        bkg_mod = Bkg_Model(bkg_rate_obj, bl_dmask, t=t)

    comp_mod = CompoundModel([bkg_mod, sig_mod])

    return comp_mod


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot(imx0 - imx1, imy0 - imy1)


class Flux2Rate(object):
    def __init__(self, response, flux_model):
        self.resp = response
        self.drm_Ebins = np.append(self.resp.drm_e0s, self.resp.drm_e1s[-1])
        self.fmodel = flux_model
        self.nebins = self.resp.nebins

    def get_rate_ebins(self, flux_params):
        photon_fluxes = self.fmodel.get_photon_fluxes(self.drm_Ebins, flux_params)
        rate_mat = self.resp.get_rates_from_flux(photon_fluxes)
        rate_ebins = np.sum(rate_mat, axis=0)
        return rate_ebins

    def get_rate_ebin(self, flux_params, ebin):
        photon_fluxes = self.fmodel.get_photon_fluxes(self.drm_Ebins, flux_params)
        rate_mat = self.resp.get_rate_from_flux(photon_fluxes, ebin)
        rate_ebins = np.sum(rate_mat)
        return rate_ebins

    def setup_gamma_deriv(self, A=1.0):
        self.gammas = np.linspace(-1.5, 4.5, 20 * 6 + 1)
        self.dg = self.gammas[1] - self.gammas[0]
        flux_params = {"A": A, "gamma": self.gammas[0]}
        self.rates = np.zeros((len(self.gammas), self.resp.nebins))
        for i in range(len(self.gammas)):
            flux_params["gamma"] = self.gammas[i]
            self.rates[i] = self.get_rate_ebins(flux_params)
        self.dr_dg = np.gradient(self.rates, self.dg, axis=0)
        self.dr_dg_intp = interpolate.interp1d(self.gammas, self.dr_dg.T)

    def get_gamma_deriv(self, gamma):
        return self.dr_dg_intp(gamma)


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

    def setup_gamma_deriv(self, imx, imy, A=1.0):
        self.gammas = np.linspace(-1.5, 4.5, 20 * 6 + 1)
        self.dg = self.gammas[1] - self.gammas[0]
        flux_params = {"A": A, "gamma": self.gammas[0], "imx": imx, "imy": imy}
        self.rates = np.zeros((len(self.gammas), self.resp.nebins))
        for i in range(len(self.gammas)):
            flux_params["gamma"] = self.gammas[i]
            self.rates[i] = self.get_rate_ebins(flux_params)
        self.dr_dg = np.gradient(self.rates, self.dg, axis=0)
        self.dr_dg_intp = interpolate.interp1d(self.gammas, self.dr_dg.T)

    def get_gamma_deriv(self, gamma):
        return self.dr_dg_intp(gamma)


@six.add_metaclass(abc.ABCMeta)
class Model(object):
    # , metaclass=abc.ABCMeta

    def __init__(
        self,
        name,
        bl_dmask,
        param_names,
        param_dict,
        nebins,
        has_prior=False,
        Tdep=False,
    ):
        self._name = name
        self._bl_dmask = bl_dmask
        self._ndets = np.sum(bl_dmask)
        self._param_names = param_names
        self._param_dict = param_dict
        self._nparams = len(param_names)
        self._has_prior = has_prior
        self._nebins = nebins
        self._Tdep = Tdep
        self.has_deriv = False

    @property
    def has_prior(self):
        return self._has_prior

    @property
    def Tdep(self):
        return self._Tdep

    @property
    def name(self):
        return self._name

    @property
    def bl_dmask(self):
        return self._bl_dmask

    @property
    def ndets(self):
        return self._ndets

    @property
    def param_names(self):
        return self._param_names

    @property
    def param_dict(self):
        return self._param_dict

    @property
    def nparams(self):
        return self._nparams

    @property
    def nebins(self):
        return self._nebins

    @abc.abstractmethod
    def get_rate_dpis(self, params):
        pass


class Bkg_Model(Model):
    def __init__(
        self,
        bkg_obj,
        bl_dmask,
        rate_bounds=None,
        t=None,
        bkg_err_fact=2.0,
        use_prior=True,
    ):
        nebins = bkg_obj.nebins
        self.bkg_obj = bkg_obj

        param_names = ["bkg_rate_" + str(i) for i in range(nebins)]

        param_dict = {}

        if t is None:
            rates = bkg_obj.get_rate((bkg_obj.t0 + bkg_obj.t1) / 2.0)[0]
        else:
            rates = bkg_obj.get_rate(t)[0]

        for i, pname in enumerate(param_names):
            pdict = {}
            if rate_bounds is None:
                pdict["bounds"] = (1e0, 1e5)
            else:
                pdict["bounds"] = rate_bounds[pname]
            pdict["nuis"] = True
            pdict["fixed"] = False
            pdict["val"] = rates[i]
            param_dict[pname] = pdict

        super(Bkg_Model, self).__init__(
            "Background", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        self._rate_ones = np.ones(self.ndets) / self.ndets
        self.bkg_err_fact = bkg_err_fact

        if t is not None:
            self.set_time(t)

    def set_time(self, t):
        self._t = t
        self._rates, self._errs = self.get_exp_rates_errs(t=t)
        self._errs *= self.bkg_err_fact

    def get_rate_dpis(self, params):
        rate_dpis = []

        # for k, val in params.iteritems():
        for pname in self.param_names:
            rate_dpis.append(params[pname] * self._rate_ones)

        return np.array(rate_dpis)

    def get_exp_rate_dpis(self, t):
        rates, errs = bkg_obj.get_rate(t)

        rate_dpis = []

        for i in range(len(rates)):
            rate_dpis.append(rates[i] * self._rate_ones)

        return np.array(rate_dpis)

    def get_exp_rates_errs(self, t):
        rates, errs = self.bkg_obj.get_rate(t)

        return rates, errs

    def get_log_prior(self, params, t=None):
        lp = 0.0
        for k, val in params.items():
            lp += stats.norm.logpdf(
                val, loc=self._rates[int(k[-1])], scale=self._errs[int(k[-1])]
            )
        return lp


class Bkg_Model_wSA(Model):
    def __init__(
        self,
        bl_dmask,
        solid_ang_dpi,
        nebins,
        use_prior=False,
        use_deriv=False,
        param_vals=None,
    ):
        self.sa_dpi = solid_ang_dpi
        self.solid_angs = solid_ang_dpi[bl_dmask]

        self.dif_names = ["diffuse_" + str(i) for i in range(nebins)]
        self.flat_names = ["flat_" + str(i) for i in range(nebins)]

        param_names = self.dif_names
        param_names += self.flat_names

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
            if param_vals is not None:
                try:
                    pdict["val"] = param_vals[pname]
                except:
                    pdict["val"] = 0.05
            else:
                pdict["val"] = 0.05
            param_dict[pname] = pdict

        super(Bkg_Model_wSA, self).__init__(
            "Background", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        self._rate_ones = np.ones(self.ndets)
        self._rate_zeros = np.zeros(self.ndets)
        if use_deriv:
            self.has_deriv = True

    def get_rate_dpis(self, params):
        #         rate_dpis = []
        rate_dpis = np.zeros((self.nebins, self.ndets))

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j = int(pname[-1])
            if "dif" in pname:
                rate_dpis[j] += params[pname] * self.solid_angs
            else:
                rate_dpis[j] += params[pname] * self._rate_ones

        return rate_dpis

    def get_rate_dpi(self, params, j):
        #         rate_dpis = []
        rate_dpi = np.zeros(self.ndets)

        # for k, val in params.iteritems():
        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0:
                continue
            if "dif" in pname:
                rate_dpi += params[pname] * self.solid_angs
            else:
                rate_dpi += params[pname] * self._rate_ones

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
            if "dif" in pname:
                #                 dr_dDifs[j] += params[pname]*self.solid_angs
                dr_dps.append(self.solid_angs)
            else:
                dr_dps.append(self._rate_ones)
        #                 dr_dps[j] += params[pname]*self._rate_ones

        return dr_dps


class Bkg_Model_wFlatA(Model):
    def __init__(
        self, bl_dmask, solid_ang_dpi, nebins, use_prior=False, use_deriv=False
    ):
        self.sa_dpi = solid_ang_dpi
        self.solid_angs = solid_ang_dpi[bl_dmask]
        self.solid_ang_mean = np.mean(self.solid_angs)

        self.rate_names = ["bkg_rate_" + str(i) for i in range(nebins)]

        self.flat_names = ["flat_" + str(i) for i in range(nebins)]

        #         self.rat_names = ['diff_flat_' + str(i) for i\
        #                                in xrange(nebins)]
        # 1 = Af + Ad
        # rat = Af/Ad
        # 1 = Ad*rat + Ad
        # Ad = 1 / (1 + rat)
        #         self.diff_As = 1. / (1. + self.ratios)
        #         self.flat_As = 1. - self.diff_As

        param_names = self.rate_names
        param_names += self.flat_names

        param_dict = {}

        #         if t is None:
        #             rates = bkg_obj.get_rate((bkg_obj.t0+bkg_obj.t1)/2.)[0]
        #         else:
        #             rates = bkg_obj.get_rate(t)[0]

        for i, pname in enumerate(param_names):
            pdict = {}
            if "rate" in pname:
                pdict["bounds"] = (5e-5, 1e2)
                pdict["val"] = 0.05
            else:
                pdict["bounds"] = (0.0, 1.0)
                pdict["val"] = 0.25
            pdict["nuis"] = True
            pdict["fixed"] = False
            param_dict[pname] = pdict

        super(Bkg_Model_wFlatA, self).__init__(
            "Background", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        self._rate_ones = np.ones(self.ndets)
        self._rate_zeros = np.zeros(self.ndets)
        if use_deriv:
            self.has_deriv = True

    #         if use_prior:
    #             if exp_rates is not None and bkg_sigs is not None:
    #                 self.set_prior(exp_rates, bkg_sigs)

    def set_bkg_row(self, bkg_row, bkg_name="", fix_flats=True, err_factor=2.0):
        self.bkg_row = bkg_row
        bkg_rates = np.array(
            [bkg_row[bkg_name + "bkg_rate_" + str(j)] for j in range(self.nebins)]
        )
        bkg_rate_errs = np.array(
            [
                bkg_row["err_" + bkg_name + "bkg_rate_" + str(j)]
                for j in range(self.nebins)
            ]
        )
        bkg_flats = np.array(
            [bkg_row[bkg_name + "flat_" + str(j)] for j in range(self.nebins)]
        )
        self.flat_vals = bkg_flats
        for j, pname in enumerate(self.flat_names):
            self.param_dict[pname]["val"] = bkg_flats[j]
            self.param_dict[self.rate_names[j]]["val"] = bkg_rates[j]
            if fix_flats:
                self.param_dict[pname]["fixed"] = True
                self.param_dict[pname]["nuis"] = False
        self.set_prior(bkg_rates, bkg_rate_errs, err_factor=err_factor)

    def set_prior(self, exp_rates, bkg_sigs, err_factor=2.0):
        self.exp_rates = exp_rates
        self.bkg_sigs = bkg_sigs
        self.err_factor = err_factor

        self.log_prior_funcs = []
        for j in range(self.nebins):
            self.log_prior_funcs.append(
                Norm_1D(
                    self.exp_rates[j], np.square(self.err_factor * self.bkg_sigs[j])
                )
            )

    def get_rate_dpis(self, params):
        #         rate_dpis = []
        rate_dpis = np.zeros((self.nebins, self.ndets))

        for j in range(self.nebins):
            rate_dpis[j] += self.get_rate_dpi(params, j)

        # for k, val in params.iteritems():
        #         for pname in self.param_names:

        #             j = int(pname[-1])
        #             rate_dpis[j] += self.diff_As[j]*params[pname]*self.solid_angs +\
        #                                 self.flat_As[j]*params[pname]

        return rate_dpis

    def get_rate_dpi(self, params, j):
        #         rate_dpis = []
        #         rate_dpi = np.zeros(self.ndets)

        rate = params[self.rate_names[j]]
        #         log_rat = params[self.log_rat_names[j]]
        flat_A = params[self.flat_names[j]]
        diff_A = 1.0 - flat_A

        #         ratio = np.exp(log_rat)
        #         diff_A = ratio/(1. + ratio)
        #         flat_A = 1. - diff_A

        rate_dpi = rate * ((diff_A / self.solid_ang_mean) * self.solid_angs + flat_A)

        #         # for k, val in params.iteritems():
        #         for pname in self.param_names:

        #             j0 = int(pname[-1])
        #             if j != j0:
        #                 continue
        #             rate_dpi += self.diff_As[j]*params[pname]*\
        #                         self.solid_angs/self.solid_ang_mean +\
        #                                 self.flat_As[j]*params[pname]
        return rate_dpi

    def get_dr_dps(self, params):
        dr_dbrs = []
        dr_dlrs = []

        for j in range(self.nebins):
            e_zeros = np.zeros((self.nebins, self.ndets))
            e_zeros[j, :] = 1.0
            drdps = self.get_dr_dp(params, j)
            dr_dbrs.append(drdps[0] * e_zeros)
            dr_dlrs.append(drdps[1] * e_zeros)

        dr_dps = dr_dbrs
        dr_dps += dr_dlrs

        return dr_dps

    def get_dr_dp(self, params, j):
        #         dr_dFlats = np.zeros((self.nebins,self.ndets))
        #         dr_dDifs = np.zeros((self.nebins,self.ndets))
        dr_dps = []

        rate = params[self.rate_names[j]]
        #         log_rat = params[self.log_rat_names[j]]

        #         ratio = np.exp(log_rat)
        #         diff_A = ratio/(1. + ratio)
        #         flat_A = 1. - diff_A
        flat_A = params[self.flat_names[j]]
        diff_A = 1.0 - flat_A

        # dr_drate
        if not self.param_dict[self.rate_names[j]]["fixed"]:
            dr_dps.append(diff_A * self.solid_angs / self.solid_ang_mean + flat_A)

        # dr_dlogratio = rate*( dAdiff_d...*solid_angs/solid_ang_mean +
        # dAflat_d...)
        # dAdiff_dlogratio = ratio / (ratio+1)^2
        # dAflat_dlogratio = -ratio / (ratio+1)^2
        #         dr_dps.append( (rate*ratio/np.square(1.+ratio))*(\
        #                         (self.solid_angs/self.solid_ang_mean) - 1.))

        # dr_dflat
        if not self.param_dict[self.flat_names[j]]["fixed"]:
            dr_dps.append(rate * (1.0 - (self.solid_angs / self.solid_ang_mean)))

        return dr_dps

    def get_log_prior(self, params, j=None):
        lp = 0.0

        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0 and j is not None:
                continue
            lp += self.log_prior_funcs[j].logpdf(params[self.rate_names[j]])
        #             lp += norm_logpdf(params[pname], self.bkg_sigs[j0], self.exp_rates[j0])
        #             lp += stats.norm.logpdf(params[pname], loc=self.exp_rates[j0],\
        #                                    scale=self.bkg_sigs[j0])

        return lp

    def get_dnlp_dp(self, params, j):
        pname = self.rate_names[j]
        dnlp_dps = -1 * self.log_prior_funcs[j].jacob_log_pdf(
            params[self.rate_names[j]]
        )
        if self.param_dict[pname]["fixed"]:
            return []

        return list(dnlp_dps)

    def get_hess_nlogprior(self, params, j):
        return -1 * self.log_prior_funcs[j].hess_log_pdf


class Bkg_Model_wFlatA(Model):
    def __init__(
        self, bl_dmask, solid_ang_dpi, nebins, use_prior=False, use_deriv=False
    ):
        self.sa_dpi = solid_ang_dpi
        self.solid_angs = solid_ang_dpi[bl_dmask]
        self.solid_ang_mean = np.mean(self.solid_angs)

        self.rate_names = ["bkg_rate_" + str(i) for i in range(nebins)]

        self.flat_names = ["flat_" + str(i) for i in range(nebins)]

        #         self.rat_names = ['diff_flat_' + str(i) for i\
        #                                in xrange(nebins)]
        # 1 = Af + Ad
        # rat = Af/Ad
        # 1 = Ad*rat + Ad
        # Ad = 1 / (1 + rat)
        #         self.diff_As = 1. / (1. + self.ratios)
        #         self.flat_As = 1. - self.diff_As

        param_names = self.rate_names
        param_names += self.flat_names

        param_dict = {}

        #         if t is None:
        #             rates = bkg_obj.get_rate((bkg_obj.t0+bkg_obj.t1)/2.)[0]
        #         else:
        #             rates = bkg_obj.get_rate(t)[0]

        for i, pname in enumerate(param_names):
            pdict = {}
            if "rate" in pname:
                pdict["bounds"] = (5e-5, 1e2)
                pdict["val"] = 0.05
            else:
                pdict["bounds"] = (0.0, 1.0)
                pdict["val"] = 0.25
            pdict["nuis"] = True
            pdict["fixed"] = False
            param_dict[pname] = pdict

        super(Bkg_Model_wFlatA, self).__init__(
            "Background", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        self._rate_ones = np.ones(self.ndets)
        self._rate_zeros = np.zeros(self.ndets)

        self.bkg_sigs = np.zeros(self.nebins)
        self.err_factor = 1.0

        if use_deriv:
            self.has_deriv = True

    #         if use_prior:
    #             if exp_rates is not None and bkg_sigs is not None:
    #                 self.set_prior(exp_rates, bkg_sigs)

    def set_bkg_row(self, bkg_row, bkg_name="", fix_flats=True, err_factor=2.0):
        self.bkg_row = bkg_row
        bkg_rates = np.array(
            [bkg_row[bkg_name + "bkg_rate_" + str(j)] for j in range(self.nebins)]
        )
        bkg_rate_errs = np.array(
            [
                bkg_row["err_" + bkg_name + "bkg_rate_" + str(j)]
                for j in range(self.nebins)
            ]
        )
        bkg_flats = np.array(
            [bkg_row[bkg_name + "flat_" + str(j)] for j in range(self.nebins)]
        )
        self.flat_vals = bkg_flats
        for j, pname in enumerate(self.flat_names):
            self.param_dict[pname]["val"] = bkg_flats[j]
            self.param_dict[self.rate_names[j]]["val"] = bkg_rates[j]
            if fix_flats:
                self.param_dict[pname]["fixed"] = True
                self.param_dict[pname]["nuis"] = False
        self.set_prior(bkg_rates, bkg_rate_errs, err_factor=err_factor)

    def set_prior(self, exp_rates, bkg_sigs, err_factor=2.0):
        self.exp_rates = exp_rates
        self.bkg_sigs = bkg_sigs
        self.err_factor = err_factor

        self.log_prior_funcs = []
        for j in range(self.nebins):
            self.log_prior_funcs.append(
                Norm_1D(
                    self.exp_rates[j], np.square(self.err_factor * self.bkg_sigs[j])
                )
            )

    def get_rate_dpis(self, params):
        #         rate_dpis = []
        rate_dpis = np.zeros((self.nebins, self.ndets))

        for j in range(self.nebins):
            rate_dpis[j] += self.get_rate_dpi(params, j)

        # for k, val in params.iteritems():
        #         for pname in self.param_names:

        #             j = int(pname[-1])
        #             rate_dpis[j] += self.diff_As[j]*params[pname]*self.solid_angs +\
        #                                 self.flat_As[j]*params[pname]

        return rate_dpis

    def get_rate_dpi(self, params, j):
        rate = params[self.rate_names[j]]
        flat_A = params[self.flat_names[j]]
        diff_A = 1.0 - flat_A

        rate_dpi = rate * ((diff_A / self.solid_ang_mean) * self.solid_angs + flat_A)

        return rate_dpi

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        rate_dpis_err = np.zeros((self.nebins, self.ndets))
        rate_dpis = np.zeros((self.nebins, self.ndets))

        for j in range(self.nebins):
            rate_dpi, rate_dpi_err = self.get_rate_dpi_err(
                params, j, ret_rate_dpis=True
            )
            rate_dpis[j] += rate_dpi
            rate_dpis_err[j] += rate_dpi_err

        if ret_rate_dpis:
            return rate_dpis, rate_dpis_err
        return rate_dpis_err

    def get_rate_dpi_err(self, params, j, ret_rate_dpis=False):
        #         rate = params[self.rate_names[j]]
        #         flat_A = params[self.flat_names[j]]
        #         diff_A = 1. - flat_A

        # make this a flat error for now
        # so the dets with lower solid angle
        # will have a larger fractional error for now
        bkg_sig = self.bkg_sigs[j] * self.err_factor

        rate_dpi = self.get_rate_dpi(params, j)
        eff_err = 0.04

        rate_dpi_err = np.sqrt(bkg_sig**2 + (eff_err * rate_dpi) ** 2)

        if ret_rate_dpis:
            return rate_dpi, rate_dpi_err
        else:
            return rate_dpi_err

    def get_dr_dps(self, params):
        dr_dbrs = []
        dr_dlrs = []

        for j in range(self.nebins):
            if (
                self.param_dict[self.rate_names[j]]["fixed"]
                and self.param_dict[self.flat_names[j]]["fixed"]
            ):
                continue
            e_zeros = np.zeros((self.nebins, self.ndets))
            e_zeros[j, :] = 1.0
            drdps = self.get_dr_dp(params, j)
            dr_dbrs.append(drdps[0] * e_zeros)
            dr_dlrs.append(drdps[1] * e_zeros)

        dr_dps = dr_dbrs
        dr_dps += dr_dlrs

        return dr_dps

    def get_dr_dp(self, params, j):
        #         dr_dFlats = np.zeros((self.nebins,self.ndets))
        #         dr_dDifs = np.zeros((self.nebins,self.ndets))
        dr_dps = []

        rate = params[self.rate_names[j]]
        #         log_rat = params[self.log_rat_names[j]]

        #         ratio = np.exp(log_rat)
        #         diff_A = ratio/(1. + ratio)
        #         flat_A = 1. - diff_A
        flat_A = params[self.flat_names[j]]
        diff_A = 1.0 - flat_A

        # dr_drate
        if not self.param_dict[self.rate_names[j]]["fixed"]:
            dr_dps.append(diff_A * self.solid_angs / self.solid_ang_mean + flat_A)

        # dr_dlogratio = rate*( dAdiff_d...*solid_angs/solid_ang_mean +
        # dAflat_d...)
        # dAdiff_dlogratio = ratio / (ratio+1)^2
        # dAflat_dlogratio = -ratio / (ratio+1)^2
        #         dr_dps.append( (rate*ratio/np.square(1.+ratio))*(\
        #                         (self.solid_angs/self.solid_ang_mean) - 1.))

        # dr_dflat
        if not self.param_dict[self.flat_names[j]]["fixed"]:
            dr_dps.append(rate * (1.0 - (self.solid_angs / self.solid_ang_mean)))

        return dr_dps

    def get_log_prior(self, params, j=None):
        lp = 0.0

        for pname in self.param_names:
            j0 = int(pname[-1])
            if j != j0 and j is not None:
                continue
            lp += self.log_prior_funcs[j].logpdf(params[self.rate_names[j]])
        #             lp += norm_logpdf(params[pname], self.bkg_sigs[j0], self.exp_rates[j0])
        #             lp += stats.norm.logpdf(params[pname], loc=self.exp_rates[j0],\
        #                                    scale=self.bkg_sigs[j0])

        return lp

    def get_dnlp_dp(self, params, j):
        pname = self.rate_names[j]
        dnlp_dps = -1 * self.log_prior_funcs[j].jacob_log_pdf(
            params[self.rate_names[j]]
        )
        if self.param_dict[pname]["fixed"]:
            return []

        return list(dnlp_dps)

    def get_hess_nlogprior(self, params, j):
        return -1 * self.log_prior_funcs[j].hess_log_pdf


# class Known_Source_Model():

#     # should also be at set imx, imy with a rng (but probably default
#     # to smaller).
#     # Also while I'm thinking about it it's not a give that
#     # this needs to be a different class than ImBox_Source

#     def __init__(self, )


class Point_Source_Model(Model):
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
        bl_dmask,
        name="Signal",
        use_deriv=False,
        use_rt_deriv=False,
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

        param_bounds = {"imx": (self.imx0, self.imx1), "imy": (self.imy0, self.imy1)}

        for pname in self.fmodel.param_names:
            param_bounds[pname] = self.fmodel.param_bounds[pname]

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "imx":
                pdict["bounds"] = (self.imx0, self.imx1)
                pdict["val"] = self.imx
            elif pname == "imy":
                pdict["bounds"] = (self.imy0, self.imy1)
                pdict["val"] = self.imy
            else:
                pdict["bounds"] = self.fmodel.param_bounds[pname]
                if hasattr(self.fmodel, "param_guess"):
                    pdict["val"] = self.fmodel.param_guess[pname]
                else:
                    pdict["val"] = (pdict["bounds"][1] + pdict["bounds"][0]) / 2.0
            pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Point_Source_Model, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins
        )

        if use_deriv:
            self.has_deriv = True

        self.drm_obj = drm_obj
        self.set_abs_cor_type()
        self.set_resp(imx, imy)

        self.rt_obj = rt_obj

        self._rt_im_update = 1e-7
        self._rt_imx = imx - 10.0
        self._rt_imy = imy - 10.0
        self.use_rt_deriv = use_rt_deriv
        if self.use_rt_deriv:
            self._rt, self._drt_dx, self._drt_dy = self.get_rt_wderiv(imx, imy)
        else:
            self._rt = self.get_rt(imx, imy)
        self._rt_imx = imx
        self._rt_imy = imy

    def set_abs_cor_type(self, cor_type="op"):
        self.cor_type = cor_type

    def set_resp(self, imx, imy):
        self.drm = self.drm_obj.get_drm(imx, imy)

        self.resp = Response(self.drm, ebins=self.ebins, cor_type=self.cor_type)

        self.flux2rate = Flux2Rate(self.resp, self.fmodel)

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
            # self._drt_dx = drt_dx[self.bl_dmask]
            # self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy

        return self._rt

    #     def handle_params(self, params):

    #         flux_params = {}

    #         for k, val in params.iteritems():
    #             if k == 'imx':
    #                 imx = val
    #             elif k == 'imy':
    #                 imy = val
    #             else:
    #                 flux_params[k] = val

    #         return

    def get_rate_dpis(self, params):
        imx = params["imx"]
        imy = params["imy"]

        rate_pdet_ebins = self.get_rates(params, imx=imx, imy=imy)

        rt = self.get_rt(imx, imy)

        rate_dpis = np.array([rt * rate_pdet for rate_pdet in rate_pdet_ebins])

        return rate_dpis

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
                pdict["nuis"] = False
            elif pname == "imy":
                pdict["bounds"] = (self.imy0, self.imy1)
                pdict["val"] = self.imy
                pdict["nuis"] = False
            elif pname == "d":
                pdict["bounds"] = (1e-4, 1.0)
                pdict["val"] = 1e-1
                pdict["nuis"] = False
            elif "uncoded_frac" in pname:
                pdict["bounds"] = (1e-4, 0.75)
                pdict["val"] = kum_mode(
                    self.prior_kum_a[pname], self.prior_kum_b[pname]
                )
                pdict["nuis"] = True
            #                 pdict['val'] = 0.1
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
            self.deriv2_prior_func = self.deriv2_kum_prior
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
            self.flux2rate_pbtrans.setup_gamma_deriv(imx, imy)

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
            self.max_rt = np.max(self._rt)
            # self._shadow = (1. - self._rt)
            self._shadow = self.max_rt - self._rt
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
            self._fp[(self._rt > 1e-2)] = 1
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
        # shadow = (1. - rt)
        # shadow[(self.uncoded>.1)] = 0.0

        rate_dpis = np.array(
            [
                rt * rate_pdet
                + (self._shadow) * rate_pdet_ebins_uncoded[i]
                + self.max_rt * (self._unfp) * rate_pdet * params[self.frac_names[i]]
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
            + self.max_rt * (self._unfp) * rate_pdet * params[self.frac_names[j]]
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

    def deriv2_kum_prior(self, params, pname):
        return -deriv2_kum_logpdf(
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

    def get_hess_nlogprior(self, params, j):
        return np.array([[self.deriv2_prior_func(params, self.frac_names[j])]])

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

            dr_dps.append(self.max_rt * self._unfp * rate_pdet)

        return dr_dps

    def get_dr_dgamma(self, params):
        rt = self.get_rt(params["imx"], params["imy"])

        drdgs = params["A"] * self.flux2rate.get_gamma_deriv(params["gamma"])
        drdgs_trans = params["A"] * self.flux2rate_pbtrans.get_gamma_deriv(
            params["gamma"]
        )

        dr_dgs = np.array(
            [
                rt * drdg
                + (self._shadow) * drdgs_trans[i]
                + self.max_rt * (self._unfp) * drdg * params[self.frac_names[i]]
                for i, drdg in enumerate(drdgs)
            ]
        )

        return dr_dgs

    def get_dr_dps(self, params):
        # dr_dp = np.zeros((self.nebins,self.ndets))

        #         imx = params['imx']
        #         imy = params['imy']

        #         if self.use_rt_deriv:
        #             rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
        #         else:
        #             rt = self.get_rt(imx, imy)

        dr_dps = []

        for pname in self.param_names:
            if self.param_dict[pname]["fixed"]:
                continue
            if pname == "A":
                dr_dps.append(self.get_rate_dpis(params) / params["A"])
            elif pname == "gamma":
                dr_dps.append(self.get_dr_dgamma(params))

        return dr_dps


class Point_Source_Model_Binned_Rates(Model):
    # should have methods for getting rate/fully illuminated det
    # and for getting the correct ray trace

    # Counts_per_full_illum_det_for_equivalent_onaxis = Counts*(sum(rt_onaxis)/sum(rt))
    # rate param will be tot_rate/sum(rt)

    def __init__(
        self,
        imx,
        imy,
        dimxy,
        ebins,
        rt_obj,
        bl_dmask,
        name="Point_Source",
        err_fact=2.0,
        use_prior=False,
        rates=None,
        errs=None,
        use_deriv=False,
    ):
        self.dimxy = dimxy
        self.imx = imx
        self.imy = imy
        self.imx0 = imx - dimxy / 2.0
        self.imx1 = imx + dimxy / 2.0
        self.imy0 = imy - dimxy / 2.0
        self.imy1 = imy + dimxy / 2.0

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        param_names = ["imx", "imy"]
        self.rate_names = ["rate_" + str(i) for i in range(nebins)]
        param_names += self.rate_names

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "imx":
                pdict["bounds"] = (self.imx0, self.imx1)
                pdict["val"] = self.imx
            elif pname == "imy":
                pdict["bounds"] = (self.imy0, self.imy1)
                pdict["val"] = self.imy
            else:
                if rates is None:
                    pdict["val"] = 1e-1
                else:
                    j = str(pname[-1])
                    pdict["val"] = rates[j]
                pdict["bounds"] = (5e-8, 1e2)
            pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Point_Source_Model_Binned_Rates, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        #         if use_prior:
        #             self.set_rate_prior(rates, errs)
        if use_deriv:
            self.has_deriv = True

        self.rt_obj = rt_obj

        self._rt_im_update = 1e-7
        self._rt_imx = imx - 1
        self._rt_imy = imy - 1
        self._rt = self.get_rt(imx, imy)
        #         self._rt, self._drt_dx, self._drt_dy = self.get_rt_wderiv(imx, imy)
        self._rt_imx = imx
        self._rt_imy = imy

    def set_rate_prior(self, rates, errs):
        self._rates = rates
        self._errs = errs

    def get_rt_wderiv(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) > self._rt_im_update:
            rt, drt_dx, drt_dy = self.rt_obj.get_intp_rt(imx, imy, get_deriv=True)
            self._rt = rt[self.bl_dmask]
            self._drt_dx = drt_dx[self.bl_dmask]
            self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy
            self._rt_sum = np.sum(self._rt)

        return self._rt, self._drt_dx, self._drt_dy

    def get_rt(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) > self._rt_im_update:
            rt = self.rt_obj.get_intp_rt(imx, imy)
            self._rt = rt[self.bl_dmask]
            #             self._drt_dx = drt_dx[self.bl_dmask]
            #             self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy
            self._rt_sum = np.sum(self._rt)

        return self._rt

    def get_rate_dpis(self, params):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpis = np.array([rt * params[pname] for pname in self.rate_names])

        return rate_dpis

    def get_rate_dpi(self, params, j):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpi = rt * params[self.rate_names[j]]

        return rate_dpi

    def get_log_prior(self, params):
        lp = 0.0
        for k, val in params.items():
            lp += stats.norm.logpdf(
                val, loc=self._rates[int(k[-1])], scale=self._errs[int(k[-1])]
            )
        return lp

    def get_dr_dps(self, params):
        imx = params["imx"]
        imy = params["imy"]

        rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)

        dr_dps = [rt for i in range(self.nebins)]
        dr_dps = []
        for i in range(self.nebins):
            one = np.zeros(self.nebins)
            one[i] = 1.0
            dr_dps.append([rt * one[ii] for ii in range(self.nebins)])

        if self.param_dict["imx"]["fixed"]:
            return dr_dps

        dr_dimx = rate_pdet_ebins[:, np.newaxis] * drt_dimx
        dr_dimy = rate_pdet_ebins[:, np.newaxis] * drt_dimy
        dr_dps = [dr_dimx, dr_dimy] + dr_dps

        return dr_dps

    def get_dr_dp(self, params, j):
        dr_dps = []
        imx = params["imx"]
        imy = params["imy"]

        if self.param_dict[self.rate_names[j]]["fixed"]:
            return []

        rt = self.get_rt(imx, imy)

        dr_dps = [rt]

        return dr_dps


class Point_Source_Model_Binned_Rates(Model):
    # should have methods for getting rate/fully illuminated det
    # and for getting the correct ray trace

    # Counts_per_full_illum_det_for_equivalent_onaxis = Counts*(sum(rt_onaxis)/sum(rt))
    # rate param will be tot_rate/sum(rt)

    def __init__(
        self,
        imx,
        imy,
        dimxy,
        ebins,
        rt_obj,
        bl_dmask,
        name="Point_Source",
        err_fact=2.0,
        use_prior=False,
        rates=None,
        errs=None,
        use_deriv=False,
    ):
        self.dimxy = dimxy
        self.imx = imx
        self.imy = imy
        self.imx0 = imx - dimxy / 2.0
        self.imx1 = imx + dimxy / 2.0
        self.imy0 = imy - dimxy / 2.0
        self.imy1 = imy + dimxy / 2.0

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        param_names = ["imx", "imy"]
        self.rate_names = ["rate_" + str(i) for i in range(nebins)]
        param_names += self.rate_names

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "imx":
                pdict["bounds"] = (self.imx0, self.imx1)
                pdict["val"] = self.imx
            elif pname == "imy":
                pdict["bounds"] = (self.imy0, self.imy1)
                pdict["val"] = self.imy
            else:
                if rates is None:
                    pdict["val"] = 1e-1
                else:
                    j = str(pname[-1])
                    pdict["val"] = rates[j]
                pdict["bounds"] = (5e-8, 1e2)
            pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Point_Source_Model_Binned_Rates, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        #         if use_prior:
        #             self.set_rate_prior(rates, errs)
        if use_deriv:
            self.has_deriv = True

        self.rt_obj = rt_obj

        self._rt_im_update = 1e-7
        self._rt_imx = imx - 1
        self._rt_imy = imy - 1
        self._rt = self.get_rt(imx, imy)
        #         self._rt, self._drt_dx, self._drt_dy = self.get_rt_wderiv(imx, imy)
        self._rt_imx = imx
        self._rt_imy = imy

    def set_rate_prior(self, rates, errs):
        self._rates = rates
        self._errs = errs

    def get_rt_wderiv(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) > self._rt_im_update:
            rt, drt_dx, drt_dy = self.rt_obj.get_intp_rt(imx, imy, get_deriv=True)
            self._rt = rt[self.bl_dmask]
            self._drt_dx = drt_dx[self.bl_dmask]
            self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy
            self._rt_sum = np.sum(self._rt)

        return self._rt, self._drt_dx, self._drt_dy

    def get_rt(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) > self._rt_im_update:
            rt = self.rt_obj.get_intp_rt(imx, imy)
            self._rt = rt[self.bl_dmask]
            #             self._drt_dx = drt_dx[self.bl_dmask]
            #             self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy
            self._rt_sum = np.sum(self._rt)

        return self._rt

    def get_rate_dpis(self, params):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpis = np.array([rt * params[pname] for pname in self.rate_names])

        return rate_dpis

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpis = np.array([rt * params[pname] for pname in self.rate_names])
        rate_dpis_err = 0.04 * rate_dpis

        if ret_rate_dpis:
            return rate_dpis, rate_dpis_err
        return rate_dpis_err

    def get_rate_dpi(self, params, j):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpi = rt * params[self.rate_names[j]]

        return rate_dpi

    def get_rate_dpi_err(self, params, j, ret_rate_dpis=False):
        imx = params["imx"]
        imy = params["imy"]

        rt = self.get_rt(imx, imy)

        rate_dpi = rt * params[self.rate_names[j]]
        rate_dpi_err = 0.04 * rate_dpi

        if ret_rate_dpis:
            return rate_dpi, rate_dpi_err
        return rate_dpi_err

    def get_log_prior(self, params):
        lp = 0.0
        for k, val in params.items():
            lp += stats.norm.logpdf(
                val, loc=self._rates[int(k[-1])], scale=self._errs[int(k[-1])]
            )
        return lp

    def get_dr_dps(self, params):
        imx = params["imx"]
        imy = params["imy"]

        #         rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
        rt = self.get_rt(imx, imy)

        dr_dps = [rt for i in range(self.nebins)]
        dr_dps = []
        for i in range(self.nebins):
            one = np.zeros(self.nebins)
            one[i] = 1.0
            dr_dps.append([rt * one[ii] for ii in range(self.nebins)])

        if self.param_dict["imx"]["fixed"]:
            return dr_dps

        dr_dimx = rate_pdet_ebins[:, np.newaxis] * drt_dimx
        dr_dimy = rate_pdet_ebins[:, np.newaxis] * drt_dimy
        dr_dps = [dr_dimx, dr_dimy] + dr_dps

        return dr_dps

    def get_dr_dp(self, params, j):
        dr_dps = []
        imx = params["imx"]
        imy = params["imy"]

        if self.param_dict[self.rate_names[j]]["fixed"]:
            return []

        rt = self.get_rt(imx, imy)

        dr_dps = [rt]

        return dr_dps


class Bkg_and_Point_Source_Model(Model):
    # should have methods for getting rate/fully illuminated det
    # and for getting the correct ray trace

    # Counts_per_full_illum_det_for_equivalent_onaxis = Counts*(sum(rt_onaxis)/sum(rt))
    # rate param will be tot_rate/sum(rt)

    # Possibly do this as having the Bkg Model and PS binned Model just in here
    # and have things like get_rate_dpis is just
    # bkg_mod.get_rate_dpis + ps_mod.get_rate_dpis
    # Then the prior can be different

    def __init__(
        self,
        solid_ang_dpi,
        ebins,
        rt_obj,
        bl_dmask,
        ps_names,
        bkg_row=None,
        name="",
        use_prior=True,
        use_deriv=True,
        dimxy=0.1,
        min_snr2prior=3.5,
        bkg_err_fact=1.0,
    ):
        self.bkg_mod = Bkg_Model_wFlatA(
            bl_dmask,
            solid_ang_dpi,
            len(ebins[0]),
            use_prior=use_prior,
            use_deriv=use_deriv,
        )
        self.Nsrcs = len(ps_names)
        self.ps_mods = []
        self.min_snr2prior = min_snr2prior
        self.bkg_err_fact = bkg_err_fact
        nebins = len(ebins[0])
        for i in range(self.Nsrcs):
            imx = bkg_row[ps_names[i] + "_imx"]
            imy = bkg_row[ps_names[i] + "_imy"]
            self.ps_mods.append(
                Point_Source_Model_Binned_Rates(
                    imx,
                    imy,
                    dimxy,
                    ebins,
                    rt_obj,
                    bl_dmask,
                    use_prior=use_prior,
                    use_deriv=use_deriv,
                    name=ps_names[i],
                )
            )

        self.mod_list = [self.bkg_mod]
        self.mod_list += self.ps_mods
        self.comp_mod = CompoundModel(self.mod_list)
        self.max_Ndim_allowed = 3

        super(Bkg_and_Point_Source_Model, self).__init__(
            name,
            bl_dmask,
            self.comp_mod.param_names,
            self.comp_mod.param_dict,
            nebins,
            has_prior=use_prior,
        )

        if use_deriv:
            self.has_deriv = True
        #         if use_prior and (cov_mats is not None) and (param_mus is not None):
        if use_prior and (bkg_row is not None):
            self.set_bkg_row(bkg_row)

    #             self.set_prior(param_mus, cov_mats)

    #         self.pnames_by_ebin = []
    #         for j in range(nebins):
    #             pnames = []
    #             for pname in self.param_names:
    #                 try:
    #                     if int(pname[-1]) == j:
    #                         pnames.append(pname)
    #                 except:
    #                     pass
    #             self.pnames_by_ebin.append(pnames)

    def set_bkg_row(self, bkg_row):
        col_names = list(bkg_row.keys())

        PSnames = []
        for name in col_names:
            if "_imx" in name:
                PSnames.append(name.split("_")[0])
        Nsrcs = len(PSnames)
        if Nsrcs > 0:
            bkg_name = "Background_"
        else:
            bkg_name = ""

        PS_params = []
        bkg_params = {}
        PS_rates = {}
        PS_errs = {}
        PS_snrs = {}

        all_mod_names = [bkg_name]
        all_mod_names += PSnames

        bkg_rates = np.array(
            [bkg_row[bkg_name + "bkg_rate_" + str(j)] for j in range(self.nebins)]
        )
        bkg_rate_errs = self.bkg_err_fact * np.array(
            [
                bkg_row["err_" + bkg_name + "bkg_rate_" + str(j)]
                for j in range(self.nebins)
            ]
        )
        bkg_flats = np.array(
            [bkg_row[bkg_name + "flat_" + str(j)] for j in range(self.nebins)]
        )

        self.bkg_mod.set_bkg_row(bkg_row, bkg_name=bkg_name)

        for i in range(Nsrcs):
            PS_rates[PSnames[i]] = np.array(
                [bkg_row[PSnames[i] + "_rate_" + str(j)] for j in range(self.nebins)]
            )
            PS_errs[PSnames[i]] = np.array(
                [
                    bkg_row["err_" + PSnames[i] + "_rate_" + str(j)]
                    for j in range(self.nebins)
                ]
            )
            PS_snrs[PSnames[i]] = PS_rates[PSnames[i]] / PS_errs[PSnames[i]]
            for j in range(self.nebins):
                self.param_dict[PSnames[i] + "_rate_" + str(j)]["val"] = PS_rates[
                    PSnames[i]
                ][j]

        Ndim = 1 + Nsrcs
        self.Ndim_prior_max = Ndim
        self.Ndim_priors = []
        corr_coefs = []

        cov_mats = []

        pnames_by_ebin = []
        err_pnames_by_ebin = []
        corr_pnames_by_ebin = []

        param_mus_by_ebin = []

        for j in range(self.nebins):
            pnames = [bkg_name + "bkg_rate_" + str(j)]
            self.comp_mod.param_dict[pnames[0]]["nuis"] = True
            err_pnames = ["err_" + bkg_name + "bkg_rate_" + str(j)]
            #             param_mus = np.zeros(Ndim)
            #             param_mus[0] = bkg_row[pnames[0]]
            param_mus = [bkg_row[pnames[0]]]
            Ndim = 1
            PSs_included = []
            for i in range(Nsrcs):
                pname = PSnames[i] + "_rate_" + str(j)
                if PS_snrs[PSnames[i]][j] < self.min_snr2prior:
                    self.comp_mod.param_dict[pname]["nuis"] = False
                    self.comp_mod.param_dict[pname]["fixed"] = True
                    continue
                if Ndim >= self.max_Ndim_allowed:
                    snrs = [PS_snrs[PSname][j] for PSname in PSs_included]
                    logging.debug("Ndim >= max_Ndim_allowed")
                    logging.debug("snrs: ")
                    logging.debug(snrs)
                    if PS_snrs[PSnames[i]][j] <= min(snrs):
                        self.comp_mod.param_dict[pname]["nuis"] = False
                        self.comp_mod.param_dict[pname]["fixed"] = True
                        continue
                    else:
                        pname_min = pnames[np.argmin(snrs) + 1]
                        pnames.remove(pname_min)
                        PSs_included.remove(PSs_included[np.argmin(snrs)])
                        self.comp_mod.param_dict[pname_min]["nuis"] = False
                        self.comp_mod.param_dict[pname_min]["fixed"] = True
                        Ndim -= 1
                Ndim += 1
                pnames.append(pname)
                PSs_included.append(PSnames[i])
            for ii in range(1, len(pnames)):
                err_pnames.append("err_" + pnames[ii])
                param_mus.append(bkg_row[pnames[ii]])
                self.comp_mod.param_dict[pnames[ii]]["nuis"] = True

            self.Ndim_priors.append(Ndim)
            pnames_by_ebin.append(pnames)
            err_pnames_by_ebin.append(err_pnames)
            param_mus_by_ebin.append(np.array(param_mus))
            logging.debug("Setting priors for ebin " + str(j))
            logging.debug("Ndim: " + str(Ndim))
            logging.debug("err_pnames_by_ebin[j]: ")
            logging.debug(err_pnames)

        for j in range(self.nebins):
            Ndim = self.Ndim_priors[j]
            cov_mat = np.zeros((Ndim, Ndim))

            for ii in range(Ndim):
                cov_mat[ii, ii] = (bkg_row[err_pnames_by_ebin[j][ii]]) ** 2
            cov_mat[0, 0] *= self.bkg_err_fact

            for ii in range(Ndim - 1):
                pname0 = pnames_by_ebin[j][ii]
                for jj in range(ii + 1, Ndim):
                    pname1 = pnames_by_ebin[j][jj]
                    try:
                        corr_name = "corr_" + pname0 + "_" + pname1
                        cov_mat[ii, jj] = bkg_row[corr_name]
                    except:
                        corr_name = "corr_" + pname1 + "_" + pname0
                        cov_mat[ii, jj] = bkg_row[corr_name]
                    cov_mat[jj, ii] = bkg_row[corr_name]
                    cov_mat[ii, jj] *= np.sqrt(cov_mat[ii, ii] * cov_mat[jj, jj])
                    cov_mat[jj, ii] *= np.sqrt(cov_mat[ii, ii] * cov_mat[jj, jj])

            cov_mats.append(cov_mat)

        self.bkg_flats = bkg_flats
        self.bkg_rate_errs = bkg_rate_errs
        self.PS_errs = PS_errs
        self.PS_rates = PS_rates
        self.bkg_rates = bkg_rates
        self.pnames_by_ebin = pnames_by_ebin

        self.set_prior(param_mus_by_ebin, cov_mats)

    def set_prior(self, param_mus, cov_mats):
        # list of cov_mats with 1 per ebin
        self.cov_mats = cov_mats
        # list of array of param_mus with len of nebins and
        self.param_mus = param_mus

        self.hess_nlpriors = []

        self.log_prior_funcs = []

        for j in range(self.nebins):
            if self.Ndim_priors[j] == 3:
                self.log_prior_funcs.append(
                    Norm_3D(self.param_mus[j], self.cov_mats[j])
                )
            elif self.Ndim_priors[j] == 2:
                self.log_prior_funcs.append(
                    Norm_2D(self.param_mus[j], self.cov_mats[j])
                )
            elif self.Ndim_priors[j] == 1:
                # Need to make Norm_1D object
                self.log_prior_funcs.append(
                    Norm_1D(self.param_mus[j][0], self.cov_mats[j][0, 0])
                )
            elif self.Ndim_priors[j] > 3:
                print("Not supported yet")

        logging.debug("set prior")
        logging.debug("Ndim_priors: ")
        logging.debug(self.Ndim_priors)
        logging.debug("cov_mats: ")
        logging.debug(self.cov_mats)
        logging.debug("param_mus: ")
        logging.debug(self.param_mus)
        logging.debug("param_dict: ")
        logging.debug(self.param_dict)

    def get_rate_dpis(self, params):
        return self.comp_mod.get_rate_dpis(params)

    def get_rate_dpi(self, params, j):
        return self.comp_mod.get_rate_dpi(params, j)

    def get_dr_dp(self, params, j):
        return self.comp_mod.get_dr_dp(params, j)

    def get_dr_dps(self, params):
        return self.comp_mod.get_dr_dps(params)

    def get_log_prior(self, params, j=None):
        lp = 0.0

        if j is None:
            for j in range(self.nebins):
                pnames = self.pnames_by_ebin[j]
                #                 x, y, z = (params[pname] for pname in pnames)
                params_ = (params[pname] for pname in pnames)
                lp += self.log_prior_funcs[j].logpdf(*params_)

        #                 covmat = self.cov_mats[j]
        #                 mus = self.param_mus[j]
        #                 pnames = self.pnames_by_ebin[j]
        #                 xs = np.zeros_like(mus)
        #                 for i, pname in enumerate(pnames):
        #                     xs[i] = params[pname] - mus[i]
        #                 lp += tri_norm_log_pdf_from_covmat(xs, self.cov_mats[j])

        else:
            pnames = self.pnames_by_ebin[j]
            #             x, y, z = (params[pname] for pname in pnames)
            params_ = (params[pname] for pname in pnames)
            lp += self.log_prior_funcs[j].logpdf(*params_)

        #             covmat = self.cov_mats[j]
        #             mus = self.param_mus[j]
        #             pnames = self.pnames_by_ebin[j]
        #             xs = np.zeros_like(mus)
        #             for i, pname in enumerate(pnames):
        #                 xs[i] = params[pname] - mus[i]
        #             print "ebin: ", j
        #             print "covmat: ", covmat
        #             print "pnames: ", pnames
        #             print "xs: ", xs
        #             lp += tri_norm_log_pdf_from_covmat(xs, covmat)

        return lp

    def get_dnlp_dp(self, params, j):
        #         covmat = self.cov_mats[j]
        #         mus = self.param_mus[j]
        #         pnames = self.pnames_by_ebin[j]
        #         xs = np.zeros_like(mus)
        #         for i, pname in enumerate(pnames):
        #             xs[i] = params[pname] - mus[i]

        #         dnlp_dps = -1*jacob_log_tri_norm_log_pdf_from_covmat(xs, covmat)

        pnames = self.pnames_by_ebin[j]
        #         x, y, z = (params[pname] for pname in pnames)
        params_ = (params[pname] for pname in pnames)
        dnlp_dps = -1 * self.log_prior_funcs[j].jacob_log_pdf(*params_)
        for pname in pnames:
            if self.param_dict[pname]["fixed"]:
                return []

        return list(dnlp_dps)

    def get_dnlp_dps(self, params):
        dnlp_dps = []

        res_dict = {}

        for j in range(self.nebins):
            #             covmat = self.cov_mats[j]
            #             mus = self.param_mus[j]
            #             pnames = self.pnames_by_ebin[j]
            #             xs = np.zeros_like(mus)
            #             for i, pname in enumerate(pnames):
            #                 xs[i] = params[pname] - mus[i]

            #             dnlpdps = -1*jacob_log_tri_norm_log_pdf_from_covmat(xs, covmat)

            pnames = self.pnames_by_ebin[j]
            params_ = (params[pname] for pname in pnames)
            dnlpdps = -1 * self.log_prior_funcs[j].jacob_log_pdf(*params_)

            for i, pname in enumerate(pnames):
                res_dict[pname] = dnlpdps[i]

        for pname in self.param_names:
            if pname in list(res_dict.keys()):
                if self.param_dict[pname]["fixed"]:
                    continue
                dnlp_dps.append(res_dict[pname])

        return dnlp_dps

    def get_hess_nlogprior(self, params, j):
        return -1 * self.log_prior_funcs[j].hess_log_pdf


# class CompoundModel(Model):
#
#     def __init__(self, model_list, name=None):
#
#         self.model_list = model_list
#
#         self.Nmodels = len(model_list)
#
#         self.model_names = [model.name for model in model_list]
#
#         if name is None:
#             name = ''
#             for mname in self.model_names:
#                 name += mname + '+'
#             name = name[:-1]
#
#         param_names = []
#
#         self.param_name_map = {}
#         param_dict = {}
#
#         has_prior = False
#         Tdep = False
#         self.ntbins = 0
#
#         for model in self.model_list:
#
#             if model.has_prior:
#                 has_prior = True
#             if model.Tdep:
#                 Tdep = True
#                 self.ntbins = max(self.ntbins, model.ntbins)
#
#             mname = model.name
#
#             pname_map = {}
#
#             for pname in model.param_names:
#
#                 if mname == '':
#                     _name = pname
#                 else:
#                     _name = mname + '_' + pname
#                 param_names.append(_name)
#                 param_dict[_name] = model.param_dict[pname]
#                 pname_map[pname] = _name
#
#             self.param_name_map[mname] = pname_map
#
#         bl_dmask = self.model_list[0].bl_dmask
#
#         super(CompoundModel, self).__init__(name, bl_dmask,\
#                                 param_names, param_dict,\
#                                 self.model_list[0].nebins,\
#                                 has_prior=has_prior, Tdep=Tdep)
#
#
#         self._last_params_ebin = [{} for i in range(self.nebins)]
#         self._last_rate_dpi = [np.ones(self.ndets) for i in range(self.nebins)]
#
#
#
#     def get_model_params(self, params):
#
#         param_list = []
#
#         for model in self.model_list:
#             param = {}
#             pname_map = self.param_name_map[model.name]
#             for k in model.param_names:
#                 param[k] = params[pname_map[k]]
#             param_list.append(param)
#
#         return param_list
#
#
#     def get_rate_dpis(self, params, **kwargs):
#
#         if self.Tdep:
# #             tbins0 = kwargs['tbins0']
# #             tbins1 = kwargs['tbins1']
#             ntbins = self.ntbins
#             rate_dpis = np.zeros((ntbins,self.nebins,self.ndets))
#         else:
#             rate_dpis = np.zeros((self.nebins,self.ndets))
#
#         for model in self.model_list:
#
#             param = {}
#             pname_map = self.param_name_map[model.name]
#             for k in model.param_names:
#                 param[k] = params[pname_map[k]]
#
#             if model.Tdep:
#                 rate_dpis += model.get_rate_dpis(param)
#             else:
#                 if self.Tdep:
#                     rate_dpi = (model.get_rate_dpis(param)[np.newaxis,:,:])
# #                     print "rate_dpi shape: ", rate_dpi.shape
#                     rate_dpis += np.ones_like(rate_dpis)*rate_dpi
#                 else:
#                     rate_dpis += model.get_rate_dpis(param)
#
#
#         return rate_dpis
#
#
#     def get_rate_dpi(self, params, j, **kwargs):
#
#         if params == self._last_params_ebin[j]:
#             return self._last_rate_dpi[j]
#
#         if self.Tdep:
# #             tbins0 = kwargs['tbins0']
# #             tbins1 = kwargs['tbins1']
#             ntbins = self.ntbins
#             rate_dpi = np.zeros((ntbins,self.ndets))
#         else:
#             rate_dpi = np.zeros(self.ndets)
#
#         for model in self.model_list:
#
#             param = {}
#             pname_map = self.param_name_map[model.name]
#             for k in model.param_names:
#                 param[k] = params[pname_map[k]]
#
#             if model.Tdep:
# #                 rate_dpis += model.get_rate_dpis(param, tbins0, tbins1)
#                 rate_dpi += model.get_rate_dpi(param, j)
#             else:
#                 if self.Tdep:
#                     rate_dpi_ = model.get_rate_dpi(param, j)[np.newaxis,:]
# #                     print "rate_dpi shape: ", rate_dpi.shape
#                     rate_dpi += np.ones_like(rate_dpi)*rate_dpi_
#                 else:
#                     try:
#                         rate_dpi += model.get_rate_dpi(param, j)
#                     except Exception as E:
#                         print(E)
#                         rate_dpi += model.get_rate_dpis(param)[j]
#         self._last_params_ebin[j] = params
#         self._last_rate_dpi[j] = rate_dpi
#
#         return rate_dpi
#
#
#
#     def get_log_prior(self, params, j=None):
#
#         lp = 0.0
#
#         if self.has_prior:
#             param_list = self.get_model_params(params)
#             for i, model in enumerate(self.model_list):
#                 if model.has_prior:
#                     try:
#                         lp += model.get_log_prior(param_list[i], j=j)
#                     except:
#                         lp += model.get_log_prior(param_list[i])
#         return lp
#
#
#     def get_dr_dps(self, params):
#
#         # loop through param list and see if it has this function
#
#         dr_dps = []
#
#         for i, model in enumerate(self.model_list):
#             param_list = self.get_model_params(params)
#             if model.has_deriv:
#                 dr_dps += model.get_dr_dps(param_list[i])
#
#         return dr_dps
#
#
#     def get_dr_dp(self, params, j):
#
#         # loop through param list and see if it has this function
#
#         dr_dps = []
#
#         for i, model in enumerate(self.model_list):
#             param_list = self.get_model_params(params)
#             if model.has_deriv:
#                 dr_dps += model.get_dr_dp(param_list[i], j)
#
#         return dr_dps
#
#
#     def get_dnlp_dp(self, params, j):
#
#         dNLP_dp = []
#
#         if self.has_prior:
#             param_list = self.get_model_params(params)
#             for i, model in enumerate(self.model_list):
#                 if model.has_prior:
#                     dNLP_dp += model.get_dnlp_dp(param_list[i], j)
#         return dNLP_dp
#
#
#     def get_hess_nlogprior(self, params, j):
#
#         Ndim = 0
#         hess_list = []
#         if self.has_prior:
#             param_list = self.get_model_params(params)
#             for i, model in enumerate(self.model_list):
#                 if model.has_prior:
#                     hess = model.get_hess_nlogprior(param_list[i], j)
#                     hess_list.append(hess)
#                     Ndim += hess.shape[0]
#
#         hess_nlogprior = np.zeros((Ndim,Ndim))
#         i0 = 0
#         for hess in hess_list:
#             Nd = hess.shape[0]
#             i1 = i0 + Nd
#             hess_nlogprior[i0:i1,i0:i1] += hess
#             i0 = i1
#
#         return hess_nlogprior


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
                if mname == "":
                    _name = pname
                else:
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

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        rate_dpis = np.zeros((self.nebins, self.ndets))
        err_dpis2 = np.zeros_like(rate_dpis)

        for model in self.model_list:
            param = {}
            pname_map = self.param_name_map[model.name]
            for k in model.param_names:
                param[k] = params[pname_map[k]]

            rate_dpi, err_dpi = model.get_rate_dpis_err(param, ret_rate_dpis=True)
            rate_dpis += rate_dpi
            err_dpis2 += err_dpi**2

        if ret_rate_dpis:
            return rate_dpis, np.sqrt(err_dpis2)
        return np.sqrt(err_dpis2)

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

    def get_rate_dpi_err(self, params, j, ret_rate_dpis=False):
        rate_dpis = np.zeros(self.ndets)
        err_dpis2 = np.zeros_like(rate_dpis)

        for model in self.model_list:
            param = {}
            pname_map = self.param_name_map[model.name]
            for k in model.param_names:
                param[k] = params[pname_map[k]]

            rate_dpi, err_dpi = model.get_rate_dpi_err(param, j, ret_rate_dpis=True)
            rate_dpis += rate_dpi
            err_dpis2 += err_dpi**2

        if ret_rate_dpis:
            return rate_dpis, np.sqrt(err_dpis2)
        return np.sqrt(err_dpis2)

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

    def get_hess_nlogprior(self, params, j):
        Ndim = 0
        hess_list = []
        if self.has_prior:
            param_list = self.get_model_params(params)
            for i, model in enumerate(self.model_list):
                if model.has_prior:
                    hess = model.get_hess_nlogprior(param_list[i], j)
                    hess_list.append(hess)
                    Ndim += hess.shape[0]

        hess_nlogprior = np.zeros((Ndim, Ndim))
        i0 = 0
        for hess in hess_list:
            Nd = hess.shape[0]
            i1 = i0 + Nd
            hess_nlogprior[i0:i1, i0:i1] += hess
            i0 = i1

        return hess_nlogprior


def detxy2batxy(detx, dety):
    batx = 0.42 * detx - (285 * 0.42) / 2
    baty = 0.42 * dety - (172 * 0.42) / 2
    return batx, baty


def batxy2detxy(batx, baty):
    detx = (batx + (285 * 0.42) / 2) / 0.42
    dety = (baty + (172 * 0.42) / 2) / 0.42
    return detx, dety


def bldmask2batxys(bl_dmask):
    detys, detxs = np.where(bl_dmask)
    return detxy2batxy(detxs, detys)


class Source_Model_InFoV(Model):
    def __init__(
        self,
        flux_model,
        ebins,
        bl_dmask,
        rt_obj,
        name="Signal",
        use_deriv=False,
        use_prior=False,
    ):
        self.fmodel = flux_model

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        self.resp_dname = "/storage/work/jjd330/local/bat_data/resp_tabs_ebins/"
        self.flor_resp_dname = (
            "/gpfs/scratch/jjd330/bat_data/flor_resps_ebins_wRatCorr/"
        )

        param_names = ["theta", "phi"]
        param_names += self.fmodel.param_names

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "theta":
                pdict["bounds"] = (0.0, 180.0)
                pdict["val"] = 180.0
                pdict["nuis"] = False
            elif pname == "phi":
                pdict["bounds"] = (0.0, 360.0)
                pdict["val"] = 0.0
                pdict["nuis"] = False
            #             elif pname == 'd':
            #                 pdict['bounds'] = (1e-4, 1.)
            #                 pdict['val'] = 1e-1
            #                 pdict['nuis'] = False
            #             elif 'uncoded_frac' in pname:
            #                 pdict['bounds'] = (1e-4, .75)
            #                 pdict['val'] = kum_mode(self.prior_kum_a[pname], self.prior_kum_b[pname])
            #                 pdict['nuis'] = True
            # #                 pdict['val'] = 0.1
            else:
                pdict["bounds"] = self.fmodel.param_bounds[pname]
                if hasattr(self.fmodel, "param_guess"):
                    pdict["val"] = self.fmodel.param_guess[pname]
                else:
                    pdict["val"] = (pdict["bounds"][1] + pdict["bounds"][0]) / 2.0
                pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Source_Model_InFoV, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        if use_deriv:
            self.has_deriv = True

        self.get_batxys()
        self.flor_err = 0.2
        self.non_flor_err = 0.12
        self.coded_err = 0.05

        self.rt_obj = rt_obj
        #         self.fp_obj = fp_obj

        self._rt_im_update = 1e-7
        self._rt_imx = -10.0
        self._rt_imy = -10.0

        self._fp_im_update = 1e-4
        self._fp_imx = -10.0
        self._fp_imy = -10.0

        self._resp_update = 5.0
        self._resp_phi = np.nan
        self._resp_theta = np.nan

        self._trans_update = 5e-3
        self._trans_phi = np.nan
        self._trans_theta = np.nan

        self.ones = np.ones(self.ndets)

    def get_batxys(self):
        yinds, xinds = np.where(self.bl_dmask)
        self.batxs, self.batys = detxy2batxy(xinds, yinds)

    def set_theta_phi(self, theta, phi):
        if (
            ang_sep(phi, 90.0 - theta, self._resp_phi, 90.0 - self._resp_theta)
            > self._resp_update
        ) or np.isnan(self._resp_phi):
            logging.info("Making new response object")
            self.resp_obj = ResponseInFoV(
                self.resp_dname,
                self.flor_resp_dname,
                self.ebins0,
                self.ebins1,
                self.bl_dmask,
                self.rt_obj,
            )
            self._resp_theta = theta
            self._resp_phi = phi
            self._trans_theta = theta
            self._trans_phi = phi
            self.resp_obj.set_theta_phi(theta, phi)

        elif (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            logging.info("Updating transmission")
            self._trans_theta = theta
            self._trans_phi = phi
            self.resp_obj.update_trans(theta, phi)

        self.theta = theta
        self.phi = phi

    #         imx, imy = theta_phi2imxy(theta, phi)

    #         rt = self.get_rt(imx, imy)

    def set_flux_params(self, flux_params):
        self.flux_params = deepcopy(flux_params)
        resp_ebins = np.append(
            self.resp_obj.PhotonEmins, [self.resp_obj.PhotonEmaxs[-1]]
        )
        self.flux_params["A"] = 1.0
        self.normed_photon_fluxes = self.fmodel.get_photon_fluxes(
            resp_ebins, self.flux_params
        )

        #         self.normed_rate_dpis = np.swapaxes(self.resp_obj.get_rate_dpis_from_photon_fluxes(\
        #                                            self.normed_photon_fluxes),0,1)
        #         self.normed_err_rate_dpis = np.swapaxes(np.sqrt((self.flor_err*self.resp_obj.\
        #                         get_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes))**2 +\
        #                         (self.non_flor_err*self.resp_obj.\
        #                         get_non_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes))**2),0,1)

        self.normed_flor_rate_dpis = np.swapaxes(
            self.resp_obj.get_flor_rate_dpis_from_photon_fluxes(
                self.normed_photon_fluxes
            ),
            0,
            1,
        )

        self.normed_non_flor_rate_dpis = np.swapaxes(
            self.resp_obj.get_non_flor_rate_dpis_from_photon_fluxes(
                self.normed_photon_fluxes
            ),
            0,
            1,
        )

        self.normed_rate_dpis = (
            self.normed_flor_rate_dpis + self.normed_non_flor_rate_dpis
        )

        self.normed_err_rate_dpis = np.sqrt(
            (self.flor_err * self.normed_flor_rate_dpis) ** 2
            + (self.non_flor_err * self.normed_non_flor_rate_dpis) ** 2
        )

    def get_rate_dpis(self, params):
        theta = params["theta"]
        phi = params["phi"]
        A = params["A"]
        if (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            self.set_theta_phi(theta, phi)
            self.set_flux_params(self.flux_params)
        imx, imy = theta_phi2imxy(theta, phi)

        #         trans_dpi0 = self.resp_obj.lines_trans_dpis[:,0]
        #         coded = np.isclose(trans_dpi0, 1.0)

        #         rt = self.get_rt(imx, imy)
        #         rt[~coded] = 1.0
        #         rt[self.uncoded] = 1.0

        #         rate_dpis = A*self.normed_flor_rate_dpis
        rate_dpis = A * self.normed_rate_dpis

        #         for j in range(self.nebins):
        #             rate_dpis[j] += A*rt*self.normed_non_flor_rate_dpis[j]

        return rate_dpis

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        theta = params["theta"]
        phi = params["phi"]
        A = params["A"]
        if (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            self.set_theta_phi(theta, phi)
            self.set_flux_params(self.flux_params)
        #         imx, imy = theta_phi2imxy(theta, phi)
        #         rt = self.get_rt(imx, imy)
        #         rt[self.uncoded] = 1.0

        #         trans_dpi0 = self.resp_obj.lines_trans_dpis[:,0]
        #         coded = np.isclose(trans_dpi0, 1.0)

        #         rt = self.get_rt(imx, imy)
        #         rt[~coded] = 1.0

        rate_dpis = A * self.normed_flor_rate_dpis
        err_rate_dpis2 = np.square(A * self.normed_flor_rate_dpis * self.flor_err)

        for j in range(self.nebins):
            rate_dpi = A * self.normed_non_flor_rate_dpis[j]
            rate_dpis[j] += rate_dpi
            err_rate_dpis2[j] += np.square(rate_dpi * self.coded_err)
        #             err_rate_dpis2[j][~coded] += np.square(rate_dpi[~coded]*self.non_flor_err)
        #             err_rate_dpis2[j][coded] += np.square(rate_dpi[coded]*self.coded_err)

        if ret_rate_dpis:
            return rate_dpis, np.sqrt(err_rate_dpis2)
        return np.sqrt(err_rate_dpis)

    def get_rate_dpi(self, params, j):
        rate_dpis = self.get_rate_dpis(params)
        return rate_dpis[j]

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

    def get_hess_nlogprior(self, params, j):
        return np.array([[self.deriv2_prior_func(params, self.frac_names[j])]])

    def get_dr_dgamma(self, params):
        rt = self.get_rt(params["imx"], params["imy"])

        drdgs = params["A"] * self.flux2rate.get_gamma_deriv(params["gamma"])
        drdgs_trans = params["A"] * self.flux2rate_pbtrans.get_gamma_deriv(
            params["gamma"]
        )

        dr_dgs = np.array(
            [
                rt * drdg
                + (self._shadow) * drdgs_trans[i]
                + self.max_rt * (self._unfp) * drdg * params[self.frac_names[i]]
                for i, drdg in enumerate(drdgs)
            ]
        )

        return dr_dgs

    def get_dr_dps(self, params):
        # dr_dp = np.zeros((self.nebins,self.ndets))

        #         imx = params['imx']
        #         imy = params['imy']

        #         if self.use_rt_deriv:
        #             rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
        #         else:
        #             rt = self.get_rt(imx, imy)

        dr_dps = []

        for pname in self.param_names:
            if self.param_dict[pname]["fixed"]:
                continue
            if pname == "A":
                dr_dps.append(self.get_rate_dpis(params) / params["A"])
            elif pname == "gamma":
                dr_dps.append(self.get_dr_dgamma(params))

        return dr_dps


class Source_Model_InOutFoV(Model):
    def __init__(
        self,
        flux_model,
        ebins,
        bl_dmask,
        rt_obj,
        name="Signal",
        use_deriv=False,
        use_prior=False,
        resp_tab_dname=None,
        hp_flor_resp_dname=None,
        comp_flor_resp_dname=None,
        use_tube_corr=False,
        use_under_corr=False
    ):
        self.fmodel = flux_model

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        if resp_tab_dname is None:
            # from ..config import RESP_TAB_DNAME
            self.resp_dname = RESP_TAB_DNAME
        else:
            self.resp_dname = resp_tab_dname

        if hp_flor_resp_dname is None:
            # from ..config import HP_FLOR_RESP_DNAME
            self.flor_resp_dname = HP_FLOR_RESP_DNAME
        else:
            self.flor_resp_dname = hp_flor_resp_dname

        if comp_flor_resp_dname is None:
            # from ..config import COMP_FLOR_RESP_DNAME
            self.comp_flor_resp_dname = COMP_FLOR_RESP_DNAME
        else:
            self.comp_flor_resp_dname = comp_flor_resp_dname

        param_names = ["theta", "phi"]
        param_names += self.fmodel.param_names

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "theta":
                pdict["bounds"] = (0.0, 180.0)
                pdict["val"] = 180.0
                pdict["nuis"] = False
            elif pname == "phi":
                pdict["bounds"] = (0.0, 360.0)
                pdict["val"] = 0.0
                pdict["nuis"] = False
            #             elif pname == 'd':
            #                 pdict['bounds'] = (1e-4, 1.)
            #                 pdict['val'] = 1e-1
            #                 pdict['nuis'] = False
            #             elif 'uncoded_frac' in pname:
            #                 pdict['bounds'] = (1e-4, .75)
            #                 pdict['val'] = kum_mode(self.prior_kum_a[pname], self.prior_kum_b[pname])
            #                 pdict['nuis'] = True
            # #                 pdict['val'] = 0.1
            else:
                pdict["bounds"] = self.fmodel.param_bounds[pname]
                if hasattr(self.fmodel, "param_guess"):
                    pdict["val"] = self.fmodel.param_guess[pname]
                else:
                    pdict["val"] = (pdict["bounds"][1] + pdict["bounds"][0]) / 2.0
                pdict["nuis"] = False
            pdict["fixed"] = False

            param_dict[pname] = pdict

        super(Source_Model_InOutFoV, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        if use_deriv:
            self.has_deriv = True

        self.get_batxys()
        self.flor_err = 0.2
        self.comp_flor_err = 0.16
        self.non_flor_err = 0.12
        self.non_coded_err = 0.1
        self.coded_err = 0.05

        self.rt_obj = rt_obj
        #         self.fp_obj = fp_obj

        self._rt_im_update = 1e-7
        self._rt_imx = -10.0
        self._rt_imy = -10.0

        self._fp_im_update = 1e-4
        self._fp_imx = -10.0
        self._fp_imy = -10.0

        self._resp_update = 5.0
        self._resp_phi = np.nan
        self._resp_theta = np.nan

        self._trans_update = 5e-3
        self._trans_phi = np.nan
        self._trans_theta = np.nan

        self.ones = np.ones(self.ndets)

        self.norm_photon_flux_dict = {}

        # whether to apply ofov corrections to responses 
        self.use_tube_corr = use_tube_corr
        self.tube_corr = False
        self.use_under_corr = use_under_corr
        self.under_corr = False


    def get_batxys(self):
        yinds, xinds = np.where(self.bl_dmask)
        self.batxs, self.batys = detxy2batxy(xinds, yinds)

    def set_theta_phi(self, theta, phi):
        if (
            ang_sep(phi, 90.0 - theta, self._resp_phi, 90.0 - self._resp_theta)
            > self._resp_update
        ) or np.isnan(self._resp_phi):
            logging.info("Making new response object")
            self.resp_obj = ResponseInFoV2(
                self.resp_dname,
                self.flor_resp_dname,
                self.comp_flor_resp_dname,
                self.ebins0,
                self.ebins1,
                self.bl_dmask,
                self.rt_obj,
            )
            self._resp_theta = theta
            self._resp_phi = phi
            self._trans_theta = theta
            self._trans_phi = phi
            self.resp_obj.set_theta_phi(theta, phi)

            if (theta < 90.0 - hp.pix2ang(2**2, 56, lonlat=True)[1]) and (phi > 220.0) and (phi < 335.0) and self.use_tube_corr:
                self.resp_obj2 = ResponseInFoV2(
                    self.resp_dname,
                    self.flor_resp_dname,
                    self.comp_flor_resp_dname,
                    self.ebins0,
                    self.ebins1,
                    self.bl_dmask,
                    self.rt_obj,
                )

                self.resp_obj2.set_theta_phi(71.0, phi)

                self.tube_corr = True
            elif theta > 93.0 and self.use_under_corr:
                self.resp_obj2 = ResponseInFoV2(
                    self.resp_dname,
                    self.flor_resp_dname,
                    self.comp_flor_resp_dname,
                    self.ebins0,
                    self.ebins1,
                    self.bl_dmask,
                    self.rt_obj,
                )

                self.resp_obj2.set_theta_phi(93.0, phi)

                hp_inds = np.array(list(self.resp_obj2.comp_flor_resp_obj.resp_dict.keys()))
                phis, lats = hp.pix2ang(2**2, hp_inds, lonlat=True)
                thetas = 90 - lats
                for i in range(len(hp_inds)):
                    if phis[i] > 230 and phis[i] < 271:
                        for j in range(5):
                            self.resp_obj2.comp_flor_resp_obj.resp_dict[hp_inds[i]][15,j,:,:] = self.resp_obj2.comp_flor_resp_obj.resp_dict[hp_inds[i]][15,5,:,:]

                self.under_corr = True
            else:
                self.tube_corr = False
                self.under_corr = False


        elif (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            logging.info("Updating transmission")
            self._trans_theta = theta
            self._trans_phi = phi
            self.resp_obj.update_trans(theta, phi)

        self.theta = theta
        self.phi = phi

    #         imx, imy = theta_phi2imxy(theta, phi)

    #         rt = self.get_rt(imx, imy)

    def set_flux_params(self, flux_params):
        self.flux_params = deepcopy(flux_params)
        self.flux_params["A"] = 1.0
        if tuple(self.flux_params.values()) in self.norm_photon_flux_dict.keys():
            self.normed_photon_fluxes = self.norm_photon_flux_dict[
                tuple(self.flux_params.values())
            ]
        else:
            resp_ebins = np.append(
                self.resp_obj.PhotonEmins, [self.resp_obj.PhotonEmaxs[-1]]
            )
            self.normed_photon_fluxes = self.fmodel.get_photon_fluxes(
                resp_ebins, self.flux_params
            )
            self.norm_photon_flux_dict[
                tuple(self.flux_params.values())
            ] = self.normed_photon_fluxes

        #         self.normed_rate_dpis = np.swapaxes(self.resp_obj.get_rate_dpis_from_photon_fluxes(\
        #                                            self.normed_photon_fluxes),0,1)
        #         self.normed_err_rate_dpis = np.swapaxes(np.sqrt((self.flor_err*self.resp_obj.\
        #                         get_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes))**2 +\
        #                         (self.non_flor_err*self.resp_obj.\
        #                         get_non_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes))**2),0,1)

        #         self.normed_flor_rate_dpis = np.swapaxes(self.resp_obj.\
        #                         get_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes),0,1)

        #         self.normed_non_flor_rate_dpis = np.swapaxes(self.resp_obj.\
        #                         get_non_flor_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes),0,1)

        #         self.normed_rate_dpis = self.normed_flor_rate_dpis + self.normed_non_flor_rate_dpis

        #         self.normed_err_rate_dpis = np.sqrt((self.flor_err*self.normed_flor_rate_dpis)**2 +\
        #                                             (self.non_flor_err*self.normed_non_flor_rate_dpis)**2)

        if self.tube_corr or self.under_corr:
            self.normed_comp_flor_rate_dpis = np.swapaxes(
                self.resp_obj.get_comp_flor_rate_dpis_from_photon_fluxes(
                    self.normed_photon_fluxes
                ) + self.resp_obj2.get_comp_flor_rate_dpis_from_photon_fluxes(
                    self.normed_photon_fluxes)
,
                0,
                1,
            )
        else:
            self.normed_comp_flor_rate_dpis = np.swapaxes(
                self.resp_obj.get_comp_flor_rate_dpis_from_photon_fluxes(
                    self.normed_photon_fluxes
                ),
                0,
                1,
            )

        self.normed_photoe_rate_dpis = np.swapaxes(
            self.resp_obj.get_photoe_rate_dpis_from_photon_fluxes(
                self.normed_photon_fluxes
            ),
            0,
            1,
        )

        self.normed_rate_dpis = (
            self.normed_comp_flor_rate_dpis + self.normed_photoe_rate_dpis
        )

        self.normed_err_rate_dpis = np.sqrt(
            (self.comp_flor_err * self.normed_comp_flor_rate_dpis) ** 2
            + (self.non_coded_err * self.normed_photoe_rate_dpis) ** 2
        )

    def get_rate_dpis(self, params):
        theta = params["theta"]
        phi = params["phi"]
        A = params["A"]
        if (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            self.set_theta_phi(theta, phi)
            self.set_flux_params(self.flux_params)
        #        imx, imy = theta_phi2imxy(theta, phi)

        #         trans_dpi0 = self.resp_obj.lines_trans_dpis[:,0]
        #         coded = np.isclose(trans_dpi0, 1.0)

        #         rt = self.get_rt(imx, imy)
        #         rt[~coded] = 1.0
        #         rt[self.uncoded] = 1.0

        #         rate_dpis = A*self.normed_flor_rate_dpis
        rate_dpis = A * self.normed_rate_dpis

        #         for j in range(self.nebins):
        #             rate_dpis[j] += A*rt*self.normed_non_flor_rate_dpis[j]

        return rate_dpis

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        # TODO: THIS DEPENDS OF THETA BEING IN PARAMS BUT MAY HAVE SET THETA OUTSIDE OF THIS AND SELF.THETA MAY EXIST: DONE
        try:
            theta = params["theta"]
        except KeyError:
            theta = self.theta

        try:
            phi = params["phi"]
        except KeyError:
            phi = self.phi

        A = params["A"]
        if (
            ang_sep(phi, 90.0 - theta, self._trans_phi, 90.0 - self._trans_theta)
            > self._trans_update
        ):
            self.set_theta_phi(theta, phi)
            self.set_flux_params(self.flux_params)
        #         imx, imy = theta_phi2imxy(theta, phi)
        #         rt = self.get_rt(imx, imy)
        #         rt[self.uncoded] = 1.0

        #         trans_dpi0 = self.resp_obj.lines_trans_dpis[:,0]
        #         coded = np.isclose(trans_dpi0, 1.0)

        #         rt = self.get_rt(imx, imy)
        #         rt[~coded] = 1.0

        #         rate_dpis = A*self.normed_flor_rate_dpis
        #         err_rate_dpis2 = np.square(A*self.normed_flor_rate_dpis*self.flor_err)

        #         for j in range(self.nebins):
        #             rate_dpi = A*self.normed_non_flor_rate_dpis[j]
        #             rate_dpis[j] += rate_dpi
        #             err_rate_dpis2[j] += np.square(rate_dpi*self.coded_err)
        # #             err_rate_dpis2[j][~coded] += np.square(rate_dpi[~coded]*self.non_flor_err)
        # #             err_rate_dpis2[j][coded] += np.square(rate_dpi[coded]*self.coded_err)
        #         if ret_rate_dpis:
        #             return rate_dpis, np.sqrt(err_rate_dpis2)
        #         return np.sqrt(err_rate_dpis)

        rate_dpis = A * self.normed_rate_dpis
        err_rate_dpis = A * self.normed_err_rate_dpis

        if ret_rate_dpis:
            return rate_dpis, err_rate_dpis
        return err_rate_dpis

    def get_rate_dpi(self, params, j):
        rate_dpis = self.get_rate_dpis(params)
        return rate_dpis[j]

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

    def get_hess_nlogprior(self, params, j):
        return np.array([[self.deriv2_prior_func(params, self.frac_names[j])]])

    def get_dr_dgamma(self, params):
        rt = self.get_rt(params["imx"], params["imy"])

        drdgs = params["A"] * self.flux2rate.get_gamma_deriv(params["gamma"])
        drdgs_trans = params["A"] * self.flux2rate_pbtrans.get_gamma_deriv(
            params["gamma"]
        )

        dr_dgs = np.array(
            [
                rt * drdg
                + (self._shadow) * drdgs_trans[i]
                + self.max_rt * (self._unfp) * drdg * params[self.frac_names[i]]
                for i, drdg in enumerate(drdgs)
            ]
        )

        return dr_dgs

    def get_dr_dps(self, params):
        # dr_dp = np.zeros((self.nebins,self.ndets))

        #         imx = params['imx']
        #         imy = params['imy']

        #         if self.use_rt_deriv:
        #             rt, drt_dimx, drt_dimy = self.get_rt_wderiv(imx, imy)
        #         else:
        #             rt = self.get_rt(imx, imy)

        dr_dps = []

        for pname in self.param_names:
            if self.param_dict[pname]["fixed"]:
                continue
            if pname == "A":
                dr_dps.append(self.get_rate_dpis(params) / params["A"])
            elif pname == "gamma":
                dr_dps.append(self.get_dr_dgamma(params))

        return dr_dps


class Sig_Bkg_Model(Model):
    def __init__(self, bl_dmask, sig_mod, bkg_mod, use_prior=False, use_deriv=False):
        param_names = ["A"]

        param_dict = {}
        for i, pname in enumerate(param_names):
            pdict = {}
            pdict["bounds"] = (0, 1e5)
            pdict["val"] = 0.1
            pdict["nuis"] = False
            pdict["fixed"] = False
            param_dict[pname] = pdict

        nebins = sig_mod.nebins

        super(Sig_Bkg_Model, self).__init__(
            "Sig_Bkg", bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        if use_deriv:
            self.has_deriv = True

        self.bkg_mod = bkg_mod
        self.sig_mod = sig_mod
        self.dt = 0.0

    def set_bkg_params(self, bkg_params):
        self.bkg_params = copy(bkg_params)
        self.bkg_rate_dpis, self.bkg_rate_dpis_err = self.bkg_mod.get_rate_dpis_err(
            bkg_params, ret_rate_dpis=True
        )
        self.bkg_rate_dpis_err2 = self.bkg_rate_dpis_err**2
        self.bkg_rate_dpis = np.ravel(self.bkg_rate_dpis)
        self.bkg_rate_dpis_err2 = np.ravel(self.bkg_rate_dpis_err2)

    def set_sig_params(self, sig_params):
        self.sig_params = copy(sig_params)
        self.sig_params["A"] = 1.0
        self.sig_rate_dpis, self.sig_rate_dpis_err = self.sig_mod.get_rate_dpis_err(
            sig_params, ret_rate_dpis=True
        )
        self.sig_rate_dpis_err2 = self.sig_rate_dpis_err**2
        self.sig_rate_dpis = np.ravel(self.sig_rate_dpis)
        self.sig_rate_dpis_err2 = np.ravel(self.sig_rate_dpis_err2)
        self.set_dur(self.dt)

    def set_dur(self, dt):
        self.dt = dt
        self.bkg_cnt_dpis = self.bkg_rate_dpis * self.dt  # .astype(np.float32)
        self.bkg_cnt_dpis_err2 = (
            self.bkg_rate_dpis_err2 * self.dt * self.dt
        )  # .astype(np.float32)
        self.sig_cnt_dpis = self.sig_rate_dpis * self.dt  # .astype(np.float32)
        self.sig_cnt_dpis_err2 = (
            self.sig_rate_dpis_err2 * self.dt * self.dt
        )  # .astype(np.float32)

    def get_rate_dpis(self, params):
        #         rate_dpis = numba_mult_sig_add_bkg(self.sig_rate_dpis, self.bkg_rate_dpis, params['A'])
        rate_dpis = params["A"] * self.sig_rate_dpis + self.bkg_rate_dpis
        return rate_dpis

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        rate_dpis_err = (
            (params["A"] ** 2) * self.sig_rate_dpis_err2 + self.bkg_rate_dpis_err2
        ) ** 0.5
        #         rate_dpis_err = numba_mult_add_errs(self.sig_rate_dpis_err, self.bkg_rate_dpis_err, params['A']**2)
        if ret_rate_dpis:
            rate_dpis = self.get_rate_dpis(params)
            return rate_dpis, rate_dpis_err
        return rate_dpis_err

    def get_cnt_dpis(self, params):
        #         rate_dpis = numba_mult_sig_add_bkg(self.sig_rate_dpis, self.bkg_rate_dpis, params['A'])
        cnt_dpis = params["A"] * self.sig_cnt_dpis + self.bkg_cnt_dpis
        return cnt_dpis

    def get_cnt_dpis_err(self, params, ret_cnt_dpis=False):
        cnt_dpis_err = (
            (params["A"] ** 2) * self.sig_cnt_dpis_err2 + self.bkg_cnt_dpis_err2
        ) ** 0.5
        if ret_cnt_dpis:
            cnt_dpis = self.get_cnt_dpis(params)
            return cnt_dpis, cnt_dpis_err
        return cnt_dpis_err

    def get_cnt_dpis_err2(self, params, ret_cnt_dpis=False):
        cnt_dpis_err2 = (
            params["A"] ** 2
        ) * self.sig_cnt_dpis_err2 + self.bkg_cnt_dpis_err2
        if ret_cnt_dpis:
            cnt_dpis = self.get_cnt_dpis(params)
            return cnt_dpis, cnt_dpis_err2
        return cnt_dpis_err2

    def get_dr_dps(self, params):
        dr_dps = [self.sig_rate_dpis]
        return dr_dps

    def get_dc_dps(self, params):
        dc_dps = [self.sig_cnt_dpis]
        return dc_dps
