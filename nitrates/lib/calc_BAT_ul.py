import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import os

from ..config import resp_dname

#cwd = os.getcwd()


# comptomized and band spectra functions
# E is photon energy, A is normalization, E0 is pivot energy

def band_Elow_spec(E, A, alpha, beta, Epeak, E0):
    # A*((E/E0)**alpha)*exp[-(alpha+2)*E/Epeak]
    return A*((E/E0)**(alpha))*np.exp(-(alpha+2.0)*E/Epeak)

def band_Ehi_spec(E, A, alpha, beta, Epeak, E0):
    # A*((E/E0)**beta)*exp[beta-alpha]*((alpha-beta)*(Epeak/E0)/(alpha+2))**(alpha-beta)
    return A*((E/E0)**(beta))*np.exp(beta-alpha)*\
            ((alpha-beta)*(Epeak/E0)/(alpha+2.0))**(alpha-beta)

def band_spec(E, A, alpha, beta, Epeak, E0=100.0):
    '''
    Band spectra
    '''

    Ebreak = (alpha - beta)*Epeak/(alpha+2.0)
    if np.isscalar(E):
        if E < Ebreak:
            return band_Elow_spec(E, A, alpha, beta, Epeak, E0)
        else:
            return band_Ehi_spec(E, A, alpha, beta, Epeak, E0)
    low_inds = np.where((E<Ebreak))
    hi_inds = np.where((E>=Ebreak))
    spec = np.zeros_like(E)
    spec[low_inds] = band_Elow_spec(E[low_inds], A, alpha, beta, Epeak, E0)
    spec[hi_inds] = band_Ehi_spec(E[hi_inds], A, alpha, beta, Epeak, E0)
    return spec


def comp_spec(E, A, alpha, Epeak, E0=100.0):
    '''
    Comptomized spectra
    '''

    spec = A*((E/E0)**(alpha))*np.exp(-E*(2.+alpha)/Epeak)
    return spec


def get_comp_photon_fluxes(Elos, Ehis, A, alpha, Epeak, E0=100.0, Nsub_bins=10):
    '''
    Gets the photon flux (photons per cm2 per s) in a set of energy bins
    for a Comptomized spectra
    '''


    DeltaEs = Ehis - Elos
    NEs = len(Elos)

    fluxes = np.zeros(NEs)

    # loops over photon energy bins
    for i in range(NEs):

        Es = np.linspace(Elos[i], Ehis[i], Nsub_bins)
        dE = Es[1] - Es[0]
        # integrate spec*dE to get photon flux in bin
        fluxes[i] = np.sum(comp_spec(Es, A, alpha, Epeak, E0=E0))*dE

    return fluxes

def get_band_photon_fluxes(Elos, Ehis, A, alpha, beta, Epeak, E0=100.0, Nsub_bins=10):
    '''
    Gets the photon flux (photons per cm2 per s) in a set of energy bins
    for a Band spectra
    '''

    DeltaEs = Ehis - Elos
    NEs = len(Elos)

    fluxes = np.zeros(NEs)

    # loops over photon energy bins
    for i in range(NEs):

        Es = np.linspace(Elos[i], Ehis[i], Nsub_bins)
        dE = Es[1] - Es[0]
        # integrate spec*dE to get photon flux in bin
        fluxes[i] = np.sum(band_spec(Es, A, alpha, beta, Epeak, E0=E0))*dE

    return fluxes


def get_comp_rates(resp_mat, Elos, Ehis, A, alpha, Epeak, E0=100.0):
    '''
    Gets the expected rate count from a specific Comptomized spectra
    in each energy bin

    resp_mat: the DRM
    Elos: the low side of the Photon energy bins for the DRM
    Ehis: the high side of the Photon energy bins for the DRM
    A: spectral norm
    alpha: alpha
    Epeak: Epeak (keV)

    returns:
    rates: array with a length of the number of energy bins in resp_mat

    '''

    # get photon flux in each photon energy bin
    fluxes = get_comp_photon_fluxes(Elos, Ehis, A, alpha, Epeak, E0=E0)
    # multiply DRM by photon fluxes and sum over photon energy axis
    rates = np.sum(resp_mat*fluxes[:,np.newaxis], axis=0)
    return rates

def get_comp_eflux(Emin, Emax, A, alpha, Epeak, E0=100.0, Nbins=int(1e4)):
    '''
    Gets the energy flux from a specific Comptomized spectra from Emin to Emax

    Emin: the low bound of the energy flux
    Emax: the high bound of the energy flux
    A: spectral norm
    alpha: alpha
    Epeak: Epeak (keV)

    returns:
    flux: float (erg / cm2 / s)

    '''

    Es = np.linspace(Emin, Emax, int(Nbins))
    dE = Es[1] - Es[0]
    kev2erg = 1.60218e-9
    # integrate spectra*E*dE to get energy flux and convert to erg
    flux = np.sum(comp_spec(Es, A, alpha, Epeak, E0=E0)*Es)*dE*kev2erg
    return flux

def rate2comp_eflux(rate, resp_mat, Elos, Ehis, alpha, Epeak, Emin, Emax):
    '''
    Returns the flux from a specific Comptomized spectra that would result in
    an expected rate count (summed over all energy bins) of rate

    rate: total rate counts across energy bins
    resp_mat: the DRM
    Elos: the low side of the Photon energy bins for the DRM
    Ehis: the high side of the Photon energy bins for the DRM
    A: spectral norm
    alpha: alpha
    Epeak: Epeak (keV)
    Emin: the low bound of the energy flux
    Emax: the high bound of the energy flux

    returns:
    flux: float (erg / cm2 / s)

    '''

    # first get what the rate would be given A=1
    rates_norm = get_comp_rates(resp_mat, Elos, Ehis, 1.0, alpha, Epeak)
    rate_norm = np.sum(rates_norm)
    # Then A can be solved from A/A_norm = rate / rate_norm
    # where A is the spec norm for the flux that creates "rate" counts
    A = rate / rate_norm
    flux = get_comp_eflux(Emin, Emax, A, alpha, Epeak)
    return flux


def get_band_rates(resp_mat, Elos, Ehis, A, alpha, beta, Epeak, E0=100.0):
    '''
    Gets the expected rate count from a specific BAND spectra
    in each energy bin

    resp_mat: the DRM
    Elos: the low side of the Photon energy bins for the DRM
    Ehis: the high side of the Photon energy bins for the DRM
    A: spectral norm
    alpha: alpha
    beta: beta
    Epeak: Epeak (keV)

    returns:
    rates: array with a length of the number of energy bins in resp_mat

    '''

    # get photon flux in each photon energy bin
    fluxes = get_band_photon_fluxes(Elos, Ehis, A, alpha, beta, Epeak, E0=E0)
    # multiply DRM by photon fluxes and sum over photon energy axis
    rates = np.sum(resp_mat*fluxes[:,np.newaxis], axis=0)
    return rates

def get_band_eflux(Emin, Emax, A, alpha, beta, Epeak, E0=100.0, Nbins=int(1e4)):
    '''
    Gets the energy flux from a specific Band spectra from Emin to Emax

    Emin: the low bound of the energy flux
    Emax: the high bound of the energy flux
    A: spectral norm
    alpha: alpha
    beta: beta
    Epeak: Epeak (keV)

    returns:
    flux: float (erg / cm2 / s)

    '''

    Es = np.linspace(Emin, Emax, int(Nbins))
    dE = Es[1] - Es[0]
    kev2erg = 1.60218e-9
    # integrate spectra*E*dE to get energy flux and convert to erg
    flux = np.sum(band_spec(Es, A, alpha, beta, Epeak, E0=E0)*Es)*dE*kev2erg
    return flux

def rate2band_eflux(rate, resp_mat, Elos, Ehis, alpha, beta, Epeak, Emin, Emax):
    '''
    Returns the flux from a specific Band spectra that would result in
    an expected rate count (summed over all energy bins) of rate

    rate: total rate counts across energy bins
    resp_mat: the DRM
    Elos: the low side of the Photon energy bins for the DRM
    Ehis: the high side of the Photon energy bins for the DRM
    A: spectral norm
    alpha: alpha
    beta: beta
    Epeak: Epeak (keV)
    Emin: the low bound of the energy flux
    Emax: the high bound of the energy flux

    returns:
    flux: float (erg / cm2 / s)

    '''

    # first get what the rate would be given A=1
    rates_norm = get_band_rates(resp_mat, Elos, Ehis, 1.0, alpha, beta, Epeak)
    rate_norm = np.sum(rates_norm)
    # Then A can be solved from A/A_norm = rate / rate_norm
    # where A is the spec norm for the flux that creates "rate" counts
    A = (rate / rate_norm)
    flux = get_band_eflux(Emin, Emax, A, alpha, beta, Epeak)
    return flux



def get_drm_tab(grid_id, old=False, get_ebounds=False):

    num_str = str(grid_id)
    if len(num_str) < 2:
        num_str = '0' + num_str

    if old:
        drm_fname = os.path.join(drm_dir_old, 'BAT_alldet_grid_%s.rsp'%(num_str))

    else:
        drm_fname = os.path.join(drm_dir_nitrates, 'BAT_alldet_grid_%s.rsp'%(num_str))

    drm_tab = Table.read(drm_fname)
    if get_ebounds:
        ebounds_tab = Table.read(drm_fname, hdu=2)
        return drm_tab, ebounds_tab
    return drm_tab



def get_resp4ul_tab(theta,phi):
    drm_fname = resp_dname + 'NITRATES_alldet_theta_%s_phi_%s_.rsp'%(theta,phi)
    drm_tab = Table.read(drm_fname)
    
    return drm_tab


