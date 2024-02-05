import numpy as np
from astropy.io import fits
from scipy import stats
import os
from scipy import optimize
from ..lib.counting_and_quad_funcs import (
    get_quad_cnts_tbins,
    get_cnts_per_tbins,
    get_quad_cnts_tbins_fast,
)
from ..lib.dbread_funcs import get_rate_fits_tab


def lin_func(x, m, b):
    return m * x + b


def cubic_func(x, a0, a1, a2, a3):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3


def cov2err(cov_mat, t_ax):
    sigs2 = np.diag(cov_mat)
    cov = cov_mat[1, 0]
    err = np.sqrt((t_ax**2) * sigs2[0] + sigs2[1] + 2.0 * t_ax * cov)
    return err


def cubic_err(cov_mat, t_ax):
    sigs2 = np.sqrt(np.diag(cov_mat))
    err = np.sqrt(
        sigs2[0] ** 2
        + (sigs2[1] * t_ax) ** 2
        + (sigs2[2] * (t_ax**2)) ** 2
        + (sigs2[3] * (t_ax**3)) ** 2
    )
    # ignoring correlated errors
    return err


def get_cub_rate_quad_objs(
    quad_dicts,
    ev_data,
    trig_time,
    ebins0,
    ebins1,
    bin_size=0.512,
    tstep=0.512,
    quad_cnts_mat=None,
    post=True,
    trng=60,
    poly_trng=15,
):
    t_bins0 = np.arange(-trng * 1.024, trng * 1.024, tstep) + trig_time
    t_bins1 = t_bins0 + bin_size

    if quad_cnts_mat is None:
        quad_cnts_mat = get_quad_cnts_tbins(t_bins0, t_bins1, ebins0, ebins1, ev_data)

    rate_quad_dict = {}

    for direc, quad_dict in quad_dicts.items():
        cnts_per_tbin = np.sum(
            [quad_cnts_mat[:, :, q] for q in quad_dict["quads"]], axis=0
        )

        rate_quad_dict[direc] = Cubic_Rates(
            cnts_per_tbin,
            t_bins0,
            t_bins1,
            trig_time,
            bkg_post=post,
            poly_trng=poly_trng,
        )

        rate_quad_dict[direc].do_fits()

    return rate_quad_dict


def get_lin_rate_quad_objs(
    quad_dicts,
    ev_data,
    trig_time,
    ebins0,
    ebins1,
    bin_size=0.512,
    tstep=0.512,
    quad_cnts_mat=None,
    post=True,
    trng=45,
    poly_trng=15,
):
    t_bins0 = np.arange(-trng * 1.024, trng * 1.024, tstep) + trig_time
    t_bins1 = t_bins0 + bin_size

    if quad_cnts_mat is None:
        quad_cnts_mat = get_quad_cnts_tbins(t_bins0, t_bins1, ebins0, ebins1, ev_data)

    lin_rate_quad_dict = {}

    for direc, quad_dict in quad_dicts.items():
        cnts_per_tbin = np.sum(
            [quad_cnts_mat[:, :, q] for q in quad_dict["quads"]], axis=0
        )

        lin_rate_quad_dict[direc] = Linear_Rates(
            cnts_per_tbin,
            t_bins0,
            t_bins1,
            trig_time,
            bkg_post=post,
            poly_trng=poly_trng,
        )

        lin_rate_quad_dict[direc].do_fits()

    return lin_rate_quad_dict


def get_avg_rate_quad_objs(
    quad_dicts,
    ev_data,
    trig_time,
    ebins0,
    ebins1,
    bin_size=0.512,
    tstep=0.512,
    quad_cnts_mat=None,
    post=True,
    trng=45,
    poly_trng=15,
):
    t_bins0 = np.arange(-trng * 1.024, trng * 1.024, tstep) + trig_time
    t_bins1 = t_bins0 + bin_size

    if quad_cnts_mat is None:
        quad_cnts_mat = get_quad_cnts_tbins(t_bins0, t_bins1, ebins0, ebins1, ev_data)

    avg_rate_quad_dict = {}

    for direc, quad_dict in quad_dicts.items():
        cnts_per_tbin = np.sum(
            [quad_cnts_mat[:, :, q] for q in quad_dict["quads"]], axis=0
        )

        avg_rate_quad_dict[direc] = Average_Rates(
            cnts_per_tbin,
            t_bins0,
            t_bins1,
            trig_time,
            bkg_post=post,
            poly_trng=poly_trng,
        )

        avg_rate_quad_dict[direc].do_fits()

    return avg_rate_quad_dict


def get_avg_lin_cub_rate_quad_obs(
    quad_dicts,
    ev_data,
    trig_time,
    ebins0,
    ebins1,
    bin_size=0.512,
    tstep=0.512,
    trng=60,
    post=True,
    poly_trng=15,
):
    t_bins0 = np.arange(-trng * 1.024, trng * 1.024, tstep) + trig_time
    t_bins1 = t_bins0 + bin_size

    quad_cnts_mat = get_quad_cnts_tbins_fast(t_bins0, t_bins1, ebins0, ebins1, ev_data)

    avg_rate_quad_dict = get_avg_rate_quad_objs(
        quad_dicts,
        ev_data,
        trig_time,
        ebins0,
        ebins1,
        bin_size=bin_size,
        tstep=tstep,
        quad_cnts_mat=quad_cnts_mat,
        poly_trng=poly_trng,
        trng=trng,
    )

    lin_rate_quad_dict = get_lin_rate_quad_objs(
        quad_dicts,
        ev_data,
        trig_time,
        ebins0,
        ebins1,
        bin_size=bin_size,
        tstep=tstep,
        quad_cnts_mat=quad_cnts_mat,
        poly_trng=poly_trng,
        trng=trng,
    )

    cub_rate_quad_dict = get_cub_rate_quad_objs(
        quad_dicts,
        ev_data,
        trig_time,
        ebins0,
        ebins1,
        bin_size=bin_size,
        tstep=tstep,
        quad_cnts_mat=quad_cnts_mat,
        poly_trng=poly_trng,
        trng=trng,
    )

    return avg_rate_quad_dict, lin_rate_quad_dict, cub_rate_quad_dict


def get_lin_rate_obj(
    ev_data,
    trig_time,
    ebins0,
    ebins1,
    bin_size=0.512,
    tstep=0.512,
    trng=45,
    sig_clip=None,
):
    t_bins0 = np.arange(-trng * 1.024, trng * 1.024, tstep) + trig_time
    t_bins1 = t_bins0 + bin_size

    cnts_per_tbin = get_cnts_per_tbins(t_bins0, t_bins1, ebins0, ebins1, ev_data, None)

    lin_rate_obj = Linear_Rates(
        cnts_per_tbin, t_bins0, t_bins1, trig_time, sig_clip=sig_clip
    )

    lin_rate_obj.do_fits()

    return lin_rate_obj


def get_chi2(cnts, predics):
    chi2 = np.sum(np.square(cnts - predics) / cnts)

    return chi2


class Cubic_Rates(object):
    def __init__(
        self,
        cnts_per_tbin,
        t_bins0,
        t_bins1,
        trig_time,
        t_poly_step=1.024,
        bkg_post=True,
        bkg_rng=45,
        poly_trng=15,
    ):
        self.cnts_per_tbin = cnts_per_tbin
        self.t_bins0 = t_bins0
        self.t_bins1 = t_bins1
        self.tstep = t_bins0[1] - t_bins0[0]
        self.bin_size = t_bins1[0] - t_bins0[0]
        self.sig_window = (-5.0 * 1.024, 10.0 * 1.024)
        self.sig_exp = self.sig_window[1] - self.sig_window[0]
        self.post = bkg_post
        self.deg = 3
        if bkg_post:
            self.bkg_window = (-bkg_rng * 1.024, bkg_rng * 1.024)
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0] - self.sig_exp
        else:
            self.bkg_window = (-bkg_rng * 1.024, self.sig_window[0])
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0]
        self.trig_time = trig_time
        self.nebins = cnts_per_tbin.shape[1]

        self.t_poly_step = t_poly_step

        self.t0 = trig_time - poly_trng * self.t_poly_step
        self.t1 = trig_time + poly_trng * self.t_poly_step

        self.t_poly_ax = np.arange(self.t0, self.t1, self.t_poly_step)
        self.n_lin_pnts = len(self.t_poly_ax)

        self.A0s = np.zeros((self.n_lin_pnts, self.nebins))
        self.A1s = np.zeros((self.n_lin_pnts, self.nebins))
        self.A2s = np.zeros((self.n_lin_pnts, self.nebins))
        self.A3s = np.zeros((self.n_lin_pnts, self.nebins))
        self.errs = np.zeros((self.n_lin_pnts, self.nebins))
        self.chi2s = np.zeros_like(self.errs)
        self.dof = np.zeros((self.n_lin_pnts, self.nebins), dtype=np.int64)

        self.npars = self.deg + 1
        # self.dof = int(self.bkg_exp/self.bin_size) - self.npars

    def do_fits(self):
        for i in range(self.n_lin_pnts):
            t_mid = self.t_poly_ax[i]

            t_0 = t_mid + self.bkg_window[0]
            t_1 = t_mid + self.bkg_window[1]

            t_sig0 = t_mid + self.sig_window[0]
            t_sig1 = t_mid + self.sig_window[1]

            ind0 = np.argmin(np.abs(self.t_bins0 - t_0))
            ind1 = np.argmin(np.abs(self.t_bins1 - t_1))

            ind0_sig = np.argmin(np.abs(self.t_bins1 - t_sig0))
            ind1_sig = np.argmin(np.abs(self.t_bins0 - t_sig1))

            _t_ax0 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind0:ind0_sig]
            _t_ax1 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind1_sig:ind1]
            _t_ax = np.append(_t_ax0, _t_ax1) - self.trig_time

            _cnts = np.append(
                self.cnts_per_tbin[ind0:ind0_sig],
                self.cnts_per_tbin[ind1_sig:ind1],
                axis=0,
            )

            for j in range(self.nebins):
                res_ = optimize.curve_fit(
                    cubic_func,
                    _t_ax,
                    _cnts[:, j],
                    sigma=np.sqrt(_cnts[:, j]),
                    absolute_sigma=False,
                )

                tot_cnts = np.sum(_cnts[:, j])
                cnt_err = np.sqrt(tot_cnts) / (len(_cnts[:, j]))

                fit_err = cubic_err(np.array(res_[1]), t_mid - self.trig_time)

                err = np.hypot(cnt_err, fit_err)

                self.A0s[i, j] = res_[0][0]
                self.A1s[i, j] = res_[0][1]
                self.A2s[i, j] = res_[0][2]
                self.A3s[i, j] = res_[0][3]
                self.errs[i, j] = err

                preds = cubic_func(
                    _t_ax,
                    self.A0s[i, j],
                    self.A1s[i, j],
                    self.A2s[i, j],
                    self.A3s[i, j],
                )

                self.chi2s[i, j] = get_chi2(_cnts[:, j], preds)
                self.dof[i, j] = len(_cnts[:, j]) - self.npars

    def get_rate(self, t, chi2=False):
        ind = np.argmin(np.abs(t - self.t_poly_ax))

        rate = (
            cubic_func(
                t - self.trig_time,
                self.A0s[ind],
                self.A1s[ind],
                self.A2s[ind],
                self.A3s[ind],
            )
            / self.bin_size
        )

        error = self.errs[ind] / self.bin_size
        if chi2:
            chi2 = self.chi2s[ind]
            dof = self.dof[ind]
            return rate, error, chi2 / dof

        return rate, error


class Linear_Rates(object):
    def __init__(
        self,
        cnts_per_tbin,
        t_bins0,
        t_bins1,
        trig_time,
        t_poly_step=1.024,
        bkg_post=True,
        poly_trng=15,
        sig_clip=None,
    ):
        self.cnts_per_tbin = cnts_per_tbin
        self.t_bins0 = t_bins0
        self.t_bins1 = t_bins1
        self.tstep = t_bins0[1] - t_bins0[0]
        self.bin_size = t_bins1[0] - t_bins0[0]
        self.sig_window = (-5.0 * 1.024, 10.0 * 1.024)
        self.sig_exp = self.sig_window[1] - self.sig_window[0]
        self.post = bkg_post
        self.deg = 1
        if bkg_post:
            self.bkg_window = (-30.0 * 1.024, 30.0 * 1.024)
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0] - self.sig_exp
        else:
            self.bkg_window = (-30.0 * 1.024, self.sig_window[0])
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0]
        self.trig_time = trig_time
        self.nebins = cnts_per_tbin.shape[1]

        self.t_poly_step = t_poly_step

        self.t0 = trig_time - poly_trng * self.t_poly_step
        self.t1 = trig_time + poly_trng * self.t_poly_step

        self.t_poly_ax = np.arange(self.t0, self.t1, self.t_poly_step)
        self.n_lin_pnts = len(self.t_poly_ax)

        self.slopes = np.zeros((self.n_lin_pnts, self.nebins))
        self.ints = np.zeros_like(self.slopes)
        self.errs = np.zeros_like(self.slopes)
        self.chi2s = np.zeros_like(self.errs)
        self.dof = np.zeros((self.n_lin_pnts, self.nebins), dtype=np.int64)
        self.sig_clip = sig_clip

        self.npars = self.deg + 1
        # self.dof = int(self.bkg_exp/self.bin_size) - self.npars

    def do_fits(self):
        for i in range(self.n_lin_pnts):
            t_mid = self.t_poly_ax[i]

            t_0 = t_mid + self.bkg_window[0]
            t_1 = t_mid + self.bkg_window[1]

            t_sig0 = t_mid + self.sig_window[0]
            t_sig1 = t_mid + self.sig_window[1]

            ind0 = np.argmin(np.abs(self.t_bins0 - t_0))
            ind1 = np.argmin(np.abs(self.t_bins1 - t_1))

            ind0_sig = np.argmin(np.abs(self.t_bins1 - t_sig0))
            ind1_sig = np.argmin(np.abs(self.t_bins0 - t_sig1))

            _t_ax0 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind0:ind0_sig]
            _t_ax1 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind1_sig:ind1]
            _t_ax = np.append(_t_ax0, _t_ax1) - self.trig_time

            _cnts = np.append(
                self.cnts_per_tbin[ind0:ind0_sig],
                self.cnts_per_tbin[ind1_sig:ind1],
                axis=0,
            )

            for j in range(self.nebins):
                try:
                    bl = np.ones(len(_cnts[:, j]), dtype=bool)
                    if self.sig_clip is not None:
                        avg = np.mean(_cnts[:, j])
                        std = np.std(_cnts[:, j])
                        std_res = np.abs(_cnts[:, j] - avg) / std
                        while np.any(std_res[bl] > self.sig_clip):
                            bl[np.argmax(std_res)] = False
                            avg = np.mean(_cnts[:, j][bl])
                            std = np.std(_cnts[:, j][bl])
                            std_res = np.zeros_like(_cnts[:, j])
                            std_res[bl] = np.abs(_cnts[:, j][bl] - avg) / std
                            if (np.sum(bl) / float(len(bl)) < 0.7) or (np.sum(bl) < 10):
                                break

                    res_lin = optimize.curve_fit(
                        lin_func,
                        _t_ax[bl],
                        _cnts[:, j][bl],
                        sigma=np.sqrt(_cnts[:, j][bl]),
                        absolute_sigma=False,
                    )
                except Exception as E:
                    print(E)
                    print(("_cnts[:,j].shape: ", _cnts[:, j].shape))
                    print(("_t_ax.shape: ", _t_ax.shape))
                    raise E

                tot_cnts = np.sum(_cnts[:, j][bl])
                cnt_err = np.sqrt(tot_cnts) / (len(_cnts[:, j][bl]))

                fit_err = cov2err(np.array(res_lin[1]), t_mid - self.trig_time)

                err = np.hypot(cnt_err, fit_err)

                self.slopes[i, j] = res_lin[0][0]
                self.ints[i, j] = res_lin[0][1]
                self.errs[i, j] = err

                preds = lin_func(_t_ax[bl], self.slopes[i, j], self.ints[i, j])
                self.chi2s[i, j] = get_chi2(_cnts[:, j][bl], preds)
                self.dof[i, j] = len(_cnts[:, j][bl]) - self.npars

    def get_rate(self, t, chi2=False):
        ind = np.argmin(np.abs(t - self.t_poly_ax))

        rate = (
            lin_func(t - self.trig_time, self.slopes[ind], self.ints[ind])
            / self.bin_size
        )

        error = self.errs[ind] / self.bin_size
        if chi2:
            chi2 = self.chi2s[ind]
            dof = self.dof[ind]
            return rate, error, chi2 / dof

        return rate, error


class Average_Rates(object):
    def __init__(
        self,
        cnts_per_tbin,
        t_bins0,
        t_bins1,
        trig_time,
        t_poly_step=1.024,
        bkg_post=True,
        poly_trng=15,
    ):
        self.cnts_per_tbin = cnts_per_tbin
        self.t_bins0 = t_bins0
        self.t_bins1 = t_bins1
        self.tstep = t_bins0[1] - t_bins0[0]
        self.bin_size = t_bins1[0] - t_bins0[0]
        self.sig_window = (-5.0 * 1.024, 10.0 * 1.024)
        self.sig_exp = self.sig_window[1] - self.sig_window[0]
        self.post = bkg_post
        self.deg = 0
        if bkg_post:
            self.bkg_window = (-30.0 * 1.024, 30.0 * 1.024)
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0] - self.sig_exp
        else:
            self.bkg_window = (-30.0 * 1.024, self.sig_window[0])
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0]
        self.trig_time = trig_time
        self.nebins = cnts_per_tbin.shape[1]

        self.t_poly_step = t_poly_step

        self.t0 = trig_time - poly_trng * self.t_poly_step
        self.t1 = trig_time + poly_trng * self.t_poly_step

        self.t_poly_ax = np.arange(self.t0, self.t1, self.t_poly_step)
        self.n_lin_pnts = len(self.t_poly_ax)

        self.means = np.zeros((self.n_lin_pnts, self.nebins))
        self.errs = np.zeros_like(self.means)
        self.chi2s = np.zeros_like(self.errs)
        self.dof = np.zeros((self.n_lin_pnts, self.nebins), dtype=np.int64)

        self.npars = self.deg + 1
        # self.dof = int(self.bkg_exp/self.bin_size) - self.npars

    def do_fits(self):
        for i in range(self.n_lin_pnts):
            t_mid = self.t_poly_ax[i]

            t_0 = t_mid + self.bkg_window[0]
            t_1 = t_mid + self.bkg_window[1]

            t_sig0 = t_mid + self.sig_window[0]
            t_sig1 = t_mid + self.sig_window[1]

            ind0 = np.argmin(np.abs(self.t_bins0 - t_0))
            ind1 = np.argmin(np.abs(self.t_bins1 - t_1))

            ind0_sig = np.argmin(np.abs(self.t_bins1 - t_sig0))
            ind1_sig = np.argmin(np.abs(self.t_bins0 - t_sig1))

            _t_ax0 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind0:ind0_sig]
            _t_ax1 = ((self.t_bins0 + self.t_bins1) / 2.0)[ind1_sig:ind1]
            _t_ax = np.append(_t_ax0, _t_ax1) - self.trig_time

            _cnts = np.append(
                self.cnts_per_tbin[ind0:ind0_sig],
                self.cnts_per_tbin[ind1_sig:ind1],
                axis=0,
            )

            for j in range(self.nebins):
                mean = np.mean(_cnts[:, j])

                tot_cnts = np.sum(_cnts[:, j])
                cnt_err = np.sqrt(tot_cnts) / (len(_cnts[:, j]))

                fit_err = np.std(_cnts[:, j])
                err = np.hypot(cnt_err, fit_err)

                self.means[i, j] = mean
                self.errs[i, j] = err
                self.chi2s[i, j] = get_chi2(_cnts[:, j], mean)
                self.dof[i, j] = len(_cnts[:, j]) - self.npars

    def get_rate(self, t, chi2=False):
        ind = np.argmin(np.abs(t - self.t_poly_ax))
        rate = self.means[ind] / self.bin_size
        error = self.errs[ind] / self.bin_size
        if chi2:
            chi2 = self.chi2s[ind]
            dof = self.dof[ind]
            return rate, error, chi2 / dof

        return rate, error


def get_linear_bkg_rates(quad_cnts_mat, t_bins0, t_bins1, trig_time, quad_dicts):
    tstep = 0.512
    bin_size = 0.512
    tstep = t_bins0[1] - t_bins0[0]
    bin_size = t_bins1[0] - t_bins0[0]
    # t_bins0 = np.arange(-15.008, 15.008, tstep) + trig_time
    # t_bins0 = np.arange(-150.*1.024, 300.*1.024, tstep) + trig_time
    # t_bins1 = t_bins0 + bin_size
    ntbins = len(t_bins0)
    print(ntbins)
    nebins = quad_cnts_mat.shape[1]

    sig_window = (-5.0 * 1.024, 10.24)
    bkg_window = (-30.0 * 1.024, 30.0 * 1.024)

    # fit for signal windows at 1.024 intervals

    t_poly_step = 1.024

    t0 = trig_time - 15.0 * t_poly_step
    t1 = trig_time + 15.0 * t_poly_step

    t_poly_ax = np.arange(t0, t1, t_poly_step)
    nptbins = len(t_poly_ax)
    print(nptbins)

    lin_params = np.zeros((nptbins, nebins, 3))
    # lin_cov = np.zeros((nptbins, nebins, 2, 2))
    quads_lin_resdict = {}
    # quads_lin_covdict = {}
    for k in list(quad_dicts.keys()):
        quads_lin_resdict[k] = np.copy(lin_params)
        # quads_lin_covdict[k] = np.copy(lin_cov)

    for i in range(len(t_poly_ax)):
        t_mid = t_poly_ax[i]

        t_0 = t_mid + bkg_window[0]
        t_1 = t_mid + bkg_window[1]

        t_sig0 = t_mid + sig_window[0]
        t_sig1 = t_mid + sig_window[1]

        ind0 = np.argmin(np.abs(t_bins0 - t_0))
        ind1 = np.argmin(np.abs(t_bins1 - t_1))

        ind0_sig = np.argmin(np.abs(t_bins1 - t_sig0))
        ind1_sig = np.argmin(np.abs(t_bins0 - t_sig1))

        _t_ax0 = ((t_bins0 + t_bins1) / 2.0)[ind0:ind0_sig]
        _t_ax1 = ((t_bins0 + t_bins1) / 2.0)[ind1_sig:ind1]
        _t_ax = np.append(_t_ax0, _t_ax1) - trig_time

        quad_cnts = np.append(
            quad_cnts_mat[ind0:ind0_sig], quad_cnts_mat[ind1_sig:ind1], axis=0
        )

        for direc, quad_dict in quad_dicts.items():
            cnts_per_tbin = np.sum(
                [quad_cnts[:, :, q] for q in quad_dict["quads"]], axis=0
            )

            for ii in range(nebins):
                res_lin = optimize.curve_fit(
                    lin_func,
                    _t_ax,
                    cnts_per_tbin[:, ii],
                    sigma=np.sqrt(cnts_per_tbin[:, ii]),
                )

                quads_lin_covdict[direc][i, ii] = np.array(res_lin[1])

                tot_cnts = np.sum(cnts_per_tbin[:, ii])
                cnt_err = np.sqrt(tot_cnts) / (len(cnts_per_tbin[:, ii]))

                fit_err = cov2err(np.array(res_lin[1]), t_mid - trig_time)

                err = np.hypot(cnt_err, fit_err)

                quads_lin_resdict[direc][i, ii] = np.array(np.append(res_lin[0], [err]))

    return quads_lin_resdict, t_poly_ax


class rate_obj_from_sqltab(object):
    def __init__(self, rate_fits_df, quadID, deg):
        self.deg = deg
        self.quadID = quadID
        bl = (rate_fits_df.quadID == quadID) & (rate_fits_df.deg == deg)
        self.df = rate_fits_df[bl]
        self.groups = self.df.groupby("ebin")
        self.nebins = len(self.groups)
        self.bkg_exp = 1.0
        self.t0 = np.min(self.df["time"])
        self.t1 = np.max(self.df["time"])

    def get_rate(self, t, chi2=False):
        rates = np.zeros(self.nebins)
        errors = np.zeros(self.nebins)
        chi2s = np.zeros(self.nebins)
        for i, ebin_grp in enumerate(self.groups):
            ind = np.argmin(np.abs(t - ebin_grp[1]["time"]))
            rates[i] = ebin_grp[1]["rate"][ind]
            errors[i] = ebin_grp[1]["error"][ind]
            chi2s[i] = ebin_grp[1]["chi2"][ind]
        if chi2:
            return rates, errors, chi2s
        return rates, errors


def get_quad_rate_objs_from_db(conn, quad_dicts):
    rate_fits_tab = get_rate_fits_tab(conn)

    lin_rate_quad_obj = {}
    avg_rate_quad_obj = {}

    for direc, quad_dict in quad_dicts.items():
        avg_rate_quad_obj[direc] = rate_obj_from_sqltab(
            rate_fits_tab, quad_dict["id"], 0
        )
        lin_rate_quad_obj[direc] = rate_obj_from_sqltab(
            rate_fits_tab, quad_dict["id"], 1
        )

    return avg_rate_quad_obj, lin_rate_quad_obj
