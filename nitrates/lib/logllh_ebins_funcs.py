import numpy as np
from astropy.io import fits
import os
from scipy import stats, optimize, special


def plaw_spec(A, ind, E, E0=50.0):
    return A * (E / E0) ** (-ind)


def plaw_flux(A, ind, E0, E1, esteps=10, E_0=50.0):
    Es = np.linspace(E0, E1, esteps)
    dE = Es[1] - Es[0]
    flux = np.sum(plaw_spec(A, ind, Es, E0=E_0)) * dE
    return flux


def get_phot_fluxes(A, ind, drm_e0, drm_e1, E0=50.0):
    photon_fluxes = np.zeros_like(drm_e0)
    for i in range(len(photon_fluxes)):
        photon_fluxes[i] = plaw_flux(A, ind, drm_e0[i], drm_e1[i], E_0=E0)
    return photon_fluxes


def get_eflux(A, ind, E0, E1, esteps=100, E_0=50):
    Es = np.linspace(E0, E1, esteps)
    dE = Es[1] - Es[0]
    kev2erg = 1.60218e-9
    flux = np.sum(plaw_spec(A, ind, Es, E0=E_0) * Es) * dE * kev2erg
    return flux


def get_cnts_mat(A, ind, drm_e0, drm_e1, drm_mat, E0=50.0, abs_cor=None):
    photon_fluxes = get_phot_fluxes(A, ind, drm_e0, drm_e1, E0=E0)
    if abs_cor is not None:
        photon_fluxes = photon_fluxes * abs_cor
    cnts_mat = np.array(
        [drm_mat[:, i] * photon_fluxes for i in range(np.shape(drm_mat)[1])]
    ).T
    return cnts_mat


def get_cnt_ebins(A, ind, drm, ebin_ind_edges, E0=50.0, abs_cor=None):
    drm_mat = drm[1].data["MATRIX"]
    drm_e0 = drm[1].data["ENERG_LO"]
    drm_e1 = drm[1].data["ENERG_HI"]

    cnts_mat = get_cnts_mat(A, ind, drm_e0, drm_e1, drm_mat, E0=E0, abs_cor=abs_cor)
    cnts_imgs = np.sum(cnts_mat, axis=0)

    cnts_ebins = [
        np.sum(cnts_imgs[ebin_ind_edges[i][0] : ebin_ind_edges[i][1] + 1])
        for i in range(len(ebin_ind_edges))
    ]

    return np.array(cnts_ebins)


def get_cnt_ebins_normed(ind, drm, ebin_ind_edges, abs_cor=None, E0=50.0):
    # print "getting stuff from drm file"

    drm_mat = drm[1].data["MATRIX"]
    drm_e0 = drm[1].data["ENERG_LO"]
    drm_e1 = drm[1].data["ENERG_HI"]

    # print "doing get_cnts_mat"

    cnts_mat = get_cnts_mat(1.0, ind, drm_e0, drm_e1, drm_mat, E0=E0, abs_cor=abs_cor)
    cnts_imgs = np.sum(cnts_mat, axis=0)

    # print "shape(ebin_ind_edges): ", np.shape(ebin_ind_edges)

    cnts_ebins = np.array(
        [
            np.sum(cnts_imgs[ebin_ind_edges[i][0] : ebin_ind_edges[i][1] + 1])
            for i in range(len(ebin_ind_edges))
        ]
    )

    normed_cnts_ebins = cnts_ebins / np.sum(cnts_ebins)

    return normed_cnts_ebins


# def log_pois_prob(theta, n):
#     return stats.poisson.logpmf(n, theta)


def get_gammaln(n):
    return special.gammaln(n + 1)


def log_pois_prob(theta, n, gam_val=None):
    if gam_val is None:
        return stats.poisson.logpmf(n, theta)
    else:
        return special.xlogy(n, theta) - gam_val - theta


def get_nllh_ebins(
    datas, bkg_cnts, bkg_errs, ray_trace, index, Ns, cks, drm, ebin_ind_edges, dmask
):
    blm = dmask == 0
    nllhs = []
    # c0s = get_cnt_imgs(A, index, drm_e0, drm_e1, drm_mat)
    c0s = get_cnt_ebins(index, drm, ebin_ind_edges)
    c0s = c0s / np.sum(c0s)
    N_sig_cnts_per_ebin = Ns * c0s

    # N_sig_cnts_per_ebin_det = N_sig_cnts_per_ebin*ray_trace

    n_ebins = len(datas)

    n_dets = 1.0 * np.sum(blm)
    bkg_cnts_per_ebin = cks * bkg_cnts
    bkg_prior_logpdf = stats.norm.logpdf(
        bkg_cnts_per_ebin, loc=bkg_cnts, scale=bkg_errs
    )
    bkg_cnts_per_det_ebin = bkg_cnts_per_ebin / n_dets

    # add in the bkg_errs

    for i in range(n_ebins):
        # nllhs.append(get_nllh(datas[i], bkg_mods[i], ray_trace,\
        #                      N_sig_cnts_per_ebin[i], cks[i]))

        scnts = N_sig_cnts_per_ebin[i] * ray_trace[blm]
        bcnts = bkg_cnts_per_det_ebin[i]

        nllhs.append(-1.0 * (log_pois_prob(bcnts + scnts, datas[i][blm])))

    nllh = np.sum(np.array(nllhs)) - np.sum(bkg_prior_logpdf)
    return nllh
