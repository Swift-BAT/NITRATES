import numpy as np
import abc
from scipy.integrate import quad
import six


def get_eflux_from_model(flux_mod, params, E0, E1, esteps=1e4):
    Es = np.linspace(E0, E1, int(esteps))
    dE = Es[1] - Es[0]
    kev2erg = 1.60218e-9
    flux = np.sum(flux_mod.spec(Es, params) * Es) * dE * kev2erg
    return flux


@six.add_metaclass(abc.ABCMeta)
class Flux_Model(object):
    # , metaclass=abc.ABCMeta

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


# class Plaw_Flux(Flux_Model):
#
#     def __init__(self, **kwds):
#
#         param_names = ['A', 'gamma']
#         param_bounds = {'A':(1e-6, 1e1), 'gamma':(0.0, 2.5)}
#         super(Plaw_Flux, self).__init__('plaw', param_names,\
#                                         param_bounds=param_bounds,\
#                                         **kwds)
#         self.param_guess = {'A':1e-2, 'gamma':1.5}
#
#     def spec(self, E, params):
#
#         return params['A']*(E/self.E0)**(-params['gamma'])
#
#     def specIntegral(self, E, params):
#
#         if np.isclose(params['gamma'], 1):
#             return (params['A']*self.E0)*np.log(E)
#
#         return ((params['A']*E)/(1.-params['gamma']))*\
#                 (E/self.E0)**(-params['gamma'])


class Cutoff_Plaw_Flux(Flux_Model):
    def __init__(self, **kwds):
        param_names = ["A", "gamma", "Epeak"]
        super(Cutoff_Plaw_Flux, self).__init__("cut_plaw", param_names, **kwds)
        self.param_guess = {"A": 1e-1, "gamma": 1.0, "Epeak": 5e1}

    def spec(self, E, params):
        return (
            params["A"]
            * ((E / self.E0) ** (-params["gamma"]))
            * np.exp(-E * (2.0 - params["gamma"]) / params["Epeak"])
        )


class Band_Flux(Flux_Model):
    def __init__(self, **kwds):
        param_names = ["A", "alpha", "beta", "Epeak"]
        param_bounds = {
            "A": (1e-6, 1e6),
            "alpha": (-3.0, 1.5),
            "beta": (-10.0, 0.0),
            "Epeak": (1.0, 1e4),
        }
        super(Band_Flux, self).__init__(
            "band", param_names, param_bounds=param_bounds, **kwds
        )
        self.param_guess = {"A": 1e-1, "alpha": -1.0, "beta": -2.5, "Epeak": 5e1}

    def Elow_spec(self, E, params):
        # A*((E/E0)**alpha)*exp[-(alpha+2)*E/Epeak]
        return (
            params["A"]
            * ((E / self.E0) ** (params["alpha"]))
            * np.exp(-(params["alpha"] + 2.0) * E / params["Epeak"])
        )

    def Ehi_spec(self, E, params):
        # A*((E/E0)**beta)*exp[beta-alpha]*((alpha-beta)*(Epeak/E0)/(alpha+2))**(alpha-beta)
        return (
            params["A"]
            * ((E / self.E0) ** (params["beta"]))
            * np.exp(params["beta"] - params["alpha"])
            * (
                (params["alpha"] - params["beta"])
                * (params["Epeak"] / self.E0)
                / (params["alpha"] + 2.0)
            )
            ** (params["alpha"] - params["beta"])
        )

    def spec(self, E, params):
        Ebreak = (
            (params["alpha"] - params["beta"])
            * params["Epeak"]
            / (params["alpha"] + 2.0)
        )
        if np.isscalar(E):
            if E < Ebreak:
                return self.Elow_spec(E, params)
            else:
                return self.Ehi_spec(E, params)
        low_inds = np.where((E < Ebreak))
        hi_inds = np.where((E >= Ebreak))
        spec_arr = np.zeros_like(E)
        spec_arr[low_inds] = self.Elow_spec(E[low_inds], params)
        spec_arr[hi_inds] = self.Ehi_spec(E[hi_inds], params)
        return spec_arr
