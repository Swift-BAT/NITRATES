import numpy as np
import logging, traceback

from models import Bkg_Model, Point_Source_Model
from logllh_ebins_funcs import log_pois_prob, get_gammaln
from event2dpi_funcs import det2dpis
from numba import jit, njit, prange



def get_bkg_llh_obj(ev_data, ebins0, ebins1, bl_dmask,\
                    bkg_obj, t0, dt):

    bkg_mod = Bkg_Model(bkg_obj, bl_dmask, t=t0)

    llh_bkg = LLH_webins(ev_data, ebins0, ebins1, bl_dmask,\
                        t0=t0, dt=dt, model=bkg_mod)

    return llh_bkg

class LLH_webins(object):


    def __init__(self, event_data, ebins0, ebins1,\
                 bl_dmask, t0=None, t1=None, model=None):

        self._all_data = event_data
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        self.bl_dmask = bl_dmask
        self.t0 = 0.0
        self.t1 = 0.0
        self.ebin = -1

        if t0 is not None and t1 is not None:
            self.set_time(t0, t1)

        if model is not None:
            self.set_model(model)


    def set_time(self, t0, t1):
        '''
        Sets the start time and duration for the LLH
        analysis.

        Parameters:
        t0: start time in MET seconds
        dt: duration in seconds
        '''

        if np.isscalar(t0):
            t0 = np.array([t0])
        if np.isscalar(t1):
            t1 = np.array([t1])

        if np.all(self.t0 == t0) and np.all(self.t1 == t1):
            return
        self.t0 = t0
        self.dt = 0.0
        self.t1 = t1

        t_bl = np.zeros(len(self._all_data), dtype=np.bool)
        for i in range(len(self.t0)):
            t_bl = np.logical_or((self._all_data['TIME']>=self.t0[i])&\
                    (self._all_data['TIME']<self.t1[i]),t_bl)
            self.dt += self.t1[i] - self.t0[i]
        self.data = self._all_data[t_bl]

        self.data_dpis = np.array(det2dpis(self.data, self.ebins0,\
                              self.ebins1, bl_dmask=self.bl_dmask))
        self.gamma_vals = get_gammaln(self.data_dpis)

    def set_model(self, model):

        self.model = model
        self.nparams = self.model.nparams

    def set_ebin(self, j):

        if 'all' in str(j):
            self.ebin = -1
        else:
            self.ebin = j


    def get_llh(self, params):

        if self.ebin < 0:
            mod_cnts = self.model.get_rate_dpis(params)*self.dt
            if np.any(mod_cnts <= 0):
                return -np.inf
            llh = np.sum(log_pois_prob(mod_cnts, self.data_dpis,\
                                        gam_val=self.gamma_vals))

        else:
            mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
            if np.any(mod_cnts <= 0):
                return -np.inf
            llh = np.sum(log_pois_prob(mod_cnts, self.data_dpis[self.ebin],\
                                    gam_val=self.gamma_vals[self.ebin]))


        return llh


    def get_logprior(self, params):

        lp = 0.0
        if self.model.has_prior:
            if self.ebin < 0:
                j=None
            else:
                j=self.ebin
            lp = self.model.get_log_prior(params, j=j)
        return lp


    def get_logprob(self, params):

        logp = self.get_logprior(params)

        llh = self.get_llh(params)

        return logp + llh


    def get_logprob_jacob(self, params):

        if self.ebin < 0:
            mod_cnts = self.model.get_rate_dpis(params)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)


            fact = (1. - (self.data_dpis / mod_cnts))

            dNs_dparam = self.model.get_dr_dps(params)

            jacob = [np.sum(fact*dNs_dparam[i])*self.dt for i\
                    in range(len(dNs_dparam))]

        else:

            mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)


            fact = (1. - (self.data_dpis[self.ebin] / mod_cnts))

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            if self.model.has_prior:
                dNLP_dparams = self.model.get_dnlp_dp(params, self.ebin)
            else:
                dNLP_dparams = np.zeros(len(dR_dparams))

            jacob = [dNLP_dparams[i] + np.sum(fact*dR_dparams[i])*self.dt\
                     for i in range(len(dR_dparams))]

        return jacob

    def get_logprob_hess(self, params):

        if self.ebin < 0:

            print("Not supported for multiple ebins yet")
            return 0


        else:

            mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)

            fact = (self.data_dpis[self.ebin])/np.square(mod_cnts)

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            Ndim = len(dR_dparams)

            dNLProb_hess = np.zeros((Ndim,Ndim))

            for i in range(Ndim):
                dNLProb_hess[i,i] = np.sum( np.square(dR_dparams[i]*self.dt)*fact )
                for j in range(i+1,Ndim):
                    dNLProb_hess[i,j] = np.sum((dR_dparams[i]*self.dt)*\
                                             (dR_dparams[j]*self.dt)*fact)
                    dNLProb_hess[j,i] += dNLProb_hess[i,j]


            if self.model.has_prior:

                dNLProb_hess += self.model.get_hess_nlogprior(params, self.ebin)



        return dNLProb_hess


@njit(cache=True, fastmath=True)
def pois_norm_conv_n0(mu, sig):
    sig2 = sig**2
    return np.exp(((sig2-mu)**2 - mu**2)/(2.*sig2))

@njit(cache=True, fastmath=True)
def pois_norm_conv_n1(mu, sig):
    sig2 = sig**2
    return ((mu-sig2))*np.exp((sig2/2.)-mu)

@njit(cache=True, fastmath=True)
def pois_norm_conv_n2(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    return eterm*(-mu*sig2 + .5*(mu**2 + sig2**2 + sig2))

@njit(cache=True, fastmath=True)
def pois_norm_conv_n3(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    return eterm*.5*(((mu-sig2)**3)/3.0 + sig2*(mu-sig2))

@njit(cache=True, fastmath=True)
def pois_norm_conv_n4(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    mu_sig2 = mu - sig2
    return (eterm/24.0)*((mu_sig2)**4 + 6*(sig2*mu_sig2**2) + 3*(sig2**2))

@njit(cache=True, fastmath=True)
def pois_norm_conv_n5(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    mu_sig2 = mu - sig2
    return (eterm/(5*24.0))*((mu_sig2)**5 + 5*2*(sig2*mu_sig2**3) + 5*3*(sig2**2)*mu_sig2)

@njit(cache=True, fastmath=True)
def pois_norm_conv_n6(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    mu_sig2 = mu - sig2
    return (eterm/(6*5*24.0))*((mu_sig2)**6 + 5*3*(sig2*mu_sig2**4) +\
                               5*3*3*(sig2**2)*mu_sig2**2 + 5*3*(sig2**3))

@njit(cache=True, fastmath=True)
def pois_norm_conv_n7(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2/2.) - mu)
    mu_sig2 = mu - sig2
    return (eterm/(7*6*5*24.0))*((mu_sig2)**7 + 7*3*(sig2*mu_sig2**5) +\
               7*5*3*(sig2**2)*mu_sig2**3 + 7*5*3*(sig2**3)*mu_sig2)


@njit(cache=True, fastmath=True)
def num_factorial(N):
    res = 1.0
    for i in range(1,N+1):
        res *= i
    return res

@njit(cache=True, fastmath=True)
def pois_norm_num_conv(mu, sig, N):

    res = 0.0
    Nmu = 256
    dmu = 8.0*sig/Nmu
    norm_A = (1./(2.*np.pi*sig**2))**.5
    fact = num_factorial(N)
    mu0 = mu - dmu*Nmu/2
    if mu0 < 0:
        mu0 = 0.0
    for i in range(Nmu):
        mu_ = mu0 + i*dmu
        norm_prob = norm_A*np.exp(-(mu_ - mu)**2/(2*sig**2))
        pois_prob = ((mu_**N)/fact)*np.exp(-mu_)
        res += norm_prob*pois_prob*dmu
    return res


@njit(cache=True, fastmath=True)
def logl_pois_norm_conv(mu, sig, N, size):
    llh_ = 0.0
    for i in range(size):
        if N[i] == 0:
            llh = np.log(pois_norm_conv_n0(mu[i], sig[i]))
        elif N[i] == 1:
            llh = np.log(pois_norm_conv_n1(mu[i], sig[i]))
        elif N[i] == 2:
            llh = np.log(pois_norm_conv_n2(mu[i], sig[i]))
        elif N[i] == 3:
            llh = np.log(pois_norm_conv_n3(mu[i], sig[i]))
        elif N[i] == 4:
            llh = np.log(pois_norm_conv_n4(mu[i], sig[i]))
        elif N[i] == 5:
            llh = np.log(pois_norm_conv_n5(mu[i], sig[i]))
        elif N[i] == 6:
            llh = np.log(pois_norm_conv_n6(mu[i], sig[i]))
        elif N[i] == 7:
            llh = np.log(pois_norm_conv_n7(mu[i], sig[i]))
        else:
            llh = np.log(pois_norm_num_conv(mu[i], sig[i], N[i]))
        llh_ += llh
    return llh_


class LLH_webins(object):


    def __init__(self, event_data, ebins0, ebins1,\
                 bl_dmask, t0=None, t1=None,\
                 model=None, has_err=False):

        self._all_data = event_data
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        self.bl_dmask = bl_dmask
        self.t0 = 0.0
        self.t1 = 0.0
        self.ebin = -1
        self.set_has_error(has_err)

        if t0 is not None and t1 is not None:
            self.set_time(t0, t1)

        if model is not None:
            self.set_model(model)


    def set_time(self, t0, t1):
        '''
        Sets the start time and duration for the LLH
        analysis.

        Parameters:
        t0: start time in MET seconds
        dt: duration in seconds
        '''

        if np.isscalar(t0):
            t0 = np.array([t0])
        if np.isscalar(t1):
            t1 = np.array([t1])

        if np.all(self.t0 == t0) and np.all(self.t1 == t1):
            return
        self.t0 = t0
        self.dt = 0.0
        self.t1 = t1

        t_bl = np.zeros(len(self._all_data), dtype=np.bool)
        for i in range(len(self.t0)):
            t_bl = np.logical_or((self._all_data['TIME']>=self.t0[i])&\
                    (self._all_data['TIME']<self.t1[i]),t_bl)
            self.dt += self.t1[i] - self.t0[i]
        self.data = self._all_data[t_bl]

        self.data_dpis = det2dpis(self.data, self.ebins0, self.ebins1,\
                                bl_dmask=self.bl_dmask)
        self.data_dpis_flat = np.ravel(self.data_dpis)
        self.gamma_vals = get_gammaln(self.data_dpis)

        self.data_size = self.data_dpis.size

    def set_model(self, model):

        self.model = model
        self.nparams = self.model.nparams

    def set_ebin(self, j):

        if 'all' in str(j):
            self.ebin = -1
        else:
            self.ebin = j


    def set_has_error(self, has_error):

        self.has_error = has_error


    def get_llh(self, params):

        if self.has_error:
#             mod_cnts = self.model.get_rate_dpis(params)*self.dt
#             mod_err = self.model.get_rate_dpis_err(params)*self.dt
            if self.ebin < 0:
                mod_rate, mod_rate_err = self.model.get_rate_dpis_err(params, ret_rate_dpis=True)
                if not np.all(mod_rate > 0):
                    return -np.inf

                llh = logl_pois_norm_conv(np.ravel(mod_rate*self.dt),\
                                          np.ravel(mod_rate_err*self.dt),\
                                          self.data_dpis_flat, self.data_size)
            else:
                mod_rate, mod_rate_err = self.model.get_rate_dpi_err(params, self.ebin,\
                                                                    ret_rate_dpis=True)
                if np.any(mod_rate <= 0):
                    return -np.inf
                llh = logl_pois_norm_conv(mod_rate*self.dt,\
                                          mod_rate_err*self.dt,\
                                          self.data_dpis[self.ebin],\
                                          len(self.data_dpis[self.ebin]))

        else:
            if self.ebin < 0:
                mod_cnts = self.model.get_rate_dpis(params)*self.dt
                if np.any(mod_cnts <= 0):
                    return -np.inf
                llh = np.sum(log_pois_prob(mod_cnts, self.data_dpis,\
                                            gam_val=self.gamma_vals))

            else:
                mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
                if np.any(mod_cnts <= 0):
                    return -np.inf
                llh = np.sum(log_pois_prob(mod_cnts, self.data_dpis[self.ebin],\
                                        gam_val=self.gamma_vals[self.ebin]))


        return llh


    def get_logprior(self, params):

        lp = 0.0
        if self.model.has_prior:
            if self.ebin < 0:
                j=None
            else:
                j=self.ebin
            lp = self.model.get_log_prior(params, j=j)
        return lp


    def get_logprob(self, params):

        logp = self.get_logprior(params)

        llh = self.get_llh(params)

        return logp + llh


    def get_logprob_jacob(self, params):

        if self.ebin < 0:
            mod_cnts = self.model.get_rate_dpis(params)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)


            fact = (1. - (self.data_dpis / mod_cnts))

            dNs_dparam = self.model.get_dr_dps(params)

            jacob = [np.sum(fact*dNs_dparam[i])*self.dt for i\
                    in range(len(dNs_dparam))]

        else:

            mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)


            fact = (1. - (self.data_dpis[self.ebin] / mod_cnts))

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            if self.model.has_prior:
                dNLP_dparams = self.model.get_dnlp_dp(params, self.ebin)
            else:
                dNLP_dparams = np.zeros(len(dR_dparams))

            jacob = [dNLP_dparams[i] + np.sum(fact*dR_dparams[i])*self.dt\
                     for i in range(len(dR_dparams))]

        return jacob

    def get_logprob_hess(self, params):

        if self.ebin < 0:

            print("Not supported for multiple ebins yet")
            return 0


        else:

            mod_cnts = self.model.get_rate_dpi(params, self.ebin)*self.dt
            if np.any(np.isclose(mod_cnts,0)):
                mod_cnts = 1e-6*np.ones_like(mod_cnts)

            fact = (self.data_dpis[self.ebin])/np.square(mod_cnts)

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            Ndim = len(dR_dparams)

            dNLProb_hess = np.zeros((Ndim,Ndim))

            for i in range(Ndim):
                dNLProb_hess[i,i] = np.sum( np.square(dR_dparams[i]*self.dt)*fact )
                for j in range(i+1,Ndim):
                    dNLProb_hess[i,j] = np.sum((dR_dparams[i]*self.dt)*\
                                             (dR_dparams[j]*self.dt)*fact)
                    dNLProb_hess[j,i] += dNLProb_hess[i,j]


            if self.model.has_prior:

                dNLProb_hess += self.model.get_hess_nlogprior(params, self.ebin)



        return dNLProb_hess
