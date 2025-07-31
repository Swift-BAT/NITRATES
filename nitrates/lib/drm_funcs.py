import numpy as np
from astropy.io import fits
import os
from ..lib.logllh_ebins_funcs import get_cnt_ebins_normed, get_cnt_ebins


def get_drm_arr(drm_dir):
    drm_fnames = np.array([fn for fn in os.listdir(drm_dir) if "drm_" in fn])

    imxs = np.array([float(fn.split("_")[1]) for fn in drm_fnames])
    imys = np.array([float(fn.split("_")[2]) for fn in drm_fnames])

    dtp = [("imx", np.float64), ("imy", np.float64), ("fname", drm_fnames.dtype)]
    drm_arr = np.empty(len(imxs), dtype=dtp)
    drm_arr["imx"] = imxs
    drm_arr["imy"] = imys
    drm_arr["fname"] = drm_fnames

    return drm_arr


def get_ebin_ind_edges(drm, ebins0, ebins1):
    # drm = fits.open(os.path.join(b_dir, drm_arr['fname'][0]))
    drm_ebins0 = drm[2].data["E_MIN"]
    drm_ebins1 = drm[2].data["E_MAX"]
    ebin_ind_edges = [
        (
            np.argmin(np.abs(drm_ebins0 - ebins0[i])),
            np.argmin(np.abs(drm_ebins1 - ebins1[i])),
        )
        for i in range(len(ebins0))
    ]

    return ebin_ind_edges


class DRMs(object):
    def __init__(self, drm_dir):
        self.drm_dir = drm_dir
        self.drm_arr = get_drm_arr(drm_dir)

    def get_closest_ind(self, imx, imy):
        return np.argmin(np.hypot(imx - self.drm_arr["imx"], imy - self.drm_arr["imy"]))

    def get_drm(self, imx, imy, ret_pos=False):
        ind = self.get_closest_ind(imx, imy)

        fname = os.path.join(self.drm_dir, self.drm_arr["fname"][ind])

        # print "Opening DRM ", fname

        drm = fits.open(fname, memmap=False)

        if ret_pos:
            drm_imx = self.drm_arr["imx"][ind]
            drm_imy = self.drm_arr["imy"][ind]
            return drm, drm_imx, drm_imy

        return drm


class cnts_norm_intp(object):
    def __init__(self, cnt_ebins_norm_ind_mat, ind_ax):
        self.ind_ax = ind_ax
        self.cnt_ebins_norm_ind_mat = cnt_ebins_norm_ind_mat
        self.ind0 = np.min(ind_ax)
        self.ind1 = np.max(ind_ax)

    def __call__(self, ind):
        if (ind <= self.ind0) or (ind >= self.ind1):
            return np.nan * np.ones(np.shape(self.cnt_ebins_norm_ind_mat)[1])

        ind_ind0 = np.argmin(np.abs(ind - self.ind_ax))
        ind_ind1 = ind_ind0 + 1 if ind > self.ind_ax[ind_ind0] else ind_ind0 - 1

        A0 = np.abs(ind - self.ind_ax[ind_ind1]) / np.abs(
            self.ind_ax[ind_ind0] - self.ind_ax[ind_ind1]
        )
        A1 = 1.0 - A0

        cnts_norm = (
            A0 * self.cnt_ebins_norm_ind_mat[ind_ind0]
            + A1 * self.cnt_ebins_norm_ind_mat[ind_ind1]
        )

        return cnts_norm


class cnts_intp(object):
    def __init__(self, cnt_ebins_ind_mat, ind_ax):
        self.ind_ax = ind_ax
        self.cnt_ebins_ind_mat = cnt_ebins_ind_mat
        self.ind0 = np.min(ind_ax)
        self.ind1 = np.max(ind_ax)

    def __call__(self, ind):
        if (ind <= self.ind0) or (ind >= self.ind1):
            return np.nan * np.ones(np.shape(self.cnt_ebins_ind_mat)[1])

        ind_ind0 = np.argmin(np.abs(ind - self.ind_ax))
        ind_ind1 = ind_ind0 + 1 if ind > self.ind_ax[ind_ind0] else ind_ind0 - 1

        A0 = np.abs(ind - self.ind_ax[ind_ind1]) / np.abs(
            self.ind_ax[ind_ind0] - self.ind_ax[ind_ind1]
        )
        A1 = 1 - A0

        cnts = (
            A0 * self.cnt_ebins_ind_mat[ind_ind0]
            + A1 * self.cnt_ebins_ind_mat[ind_ind1]
        )

        return cnts


def get_cnts_intp_obj(ind_ax, drm, ebin_ind_edges, abs_cor, E0=50.0, normed=True):
    nebins = len(ebin_ind_edges)
    cnt_ebins_ind_mat = np.zeros((len(ind_ax), nebins))

    for i in range(len(ind_ax)):
        if normed:
            cnt_ebins_ind_mat[i] = get_cnt_ebins_normed(
                ind_ax[i], drm, ebin_ind_edges, abs_cor=abs_cor, E0=E0
            )
        else:
            cnt_ebins_ind_mat[i] = get_cnt_ebins(
                1.0, ind_ax[i], drm, ebin_ind_edges, abs_cor=abs_cor, E0=E0
            )

    if np.any(np.isnan(cnt_ebins_ind_mat)):
        print("Bad cnt_ebins_ind_mat")
        print((np.sum(np.isnan(cnt_ebins_ind_mat))))

    if normed:
        intp_obj = cnts_norm_intp(cnt_ebins_ind_mat, ind_ax)
    else:
        intp_obj = cnts_intp(cnt_ebins_ind_mat, ind_ax)

    return intp_obj
