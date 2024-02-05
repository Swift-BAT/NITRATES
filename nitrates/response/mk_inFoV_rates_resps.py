import numpy as np
from scipy import optimize, stats, interpolate
from astropy.io import fits
from astropy.table import Table
import os
import argparse
import logging, traceback


from ..config import rt_dir, rates_resp_dir
from ..response.ray_trace_funcs import RayTraces
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..models.models import Source_Model_InFoV, Source_Model_InOutFoV
from ..response.response import (
    Swift_Mask_Interactions,
    bldmask2batxys,
    get_fixture_struct,
    dpi_shape,
)
from ..lib.coord_conv_funcs import imxy2theta_phi, theta_phi2imxy
from ..models.flux_models import Cutoff_Plaw_Flux, Plaw_Flux
from ..response.Polygons import Polygon2D


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Njobs", type=int, help="Number of jobs being run", default=None
    )
    parser.add_argument("--job_id", type=int, help="Job ID number", default=-1)
    args = parser.parse_args()
    return args


def get_bldmask_alldets():
    detxs_by_sand0 = np.arange(0, 286 - 15, 18)
    detxs_by_sand1 = detxs_by_sand0 + 15

    detys_by_sand0 = np.arange(0, 173 - 7, 11)
    detys_by_sand1 = detys_by_sand0 + 7

    all_good_detxs = np.ravel(
        [
            np.arange(detxs_by_sand0[i], detxs_by_sand1[i] + 1, 1, dtype=np.int64)
            for i in range(16)
        ]
    )
    all_good_detys = np.ravel(
        [
            np.arange(detys_by_sand0[i], detys_by_sand1[i] + 1, 1, dtype=np.int64)
            for i in range(16)
        ]
    )

    detxax = np.arange(286, dtype=np.int64)
    detyax = np.arange(173, dtype=np.int64)
    detx_dpi, dety_dpi = np.meshgrid(detxax, detyax)
    bl_alldets = np.isin(detx_dpi, all_good_detxs) & np.isin(dety_dpi, all_good_detys)
    return bl_alldets


def get_in_out_rates4EpeakGamma(sig_mod, Epeak, gamma):
    flux_params = {"A": 1.0, "Epeak": Epeak, "gamma": gamma}
    sig_mod.set_flux_params(flux_params)
    #     in_fov_bl = sig_mod.resp_obj.mask_obj.does_int_mask
    in_fov_bl = (sig_mod.resp_obj.mask_obj.does_int_mask) & (
        sig_mod.resp_obj.mask_obj.fix_trans[:, 10] > 0.99
    )
    out_fov_bl = ~in_fov_bl
    rate_dpis = sig_mod.normed_rate_dpis
    in_fov_rates = np.sum(rate_dpis[:, in_fov_bl], axis=1) / np.sum(in_fov_bl)
    out_fov_rates = np.sum(rate_dpis[:, out_fov_bl], axis=1) / np.sum(out_fov_bl)
    return in_fov_rates, out_fov_rates


def get_in_out_rates(sig_mod):
    Epeaks = np.logspace(1, 3.2, 11 * 2 + 1)
    print(Epeaks)
    gammas = np.linspace(-0.2, 2.3, 4 * 5 + 1)
    print(gammas)
    Gs = np.meshgrid(Epeaks, gammas)
    Epeaks = Gs[0].ravel()
    gammas = Gs[1].ravel()
    print(len(Epeaks))
    Npnts = len(Epeaks)

    res_dicts = []

    for j in range(Npnts):
        res_dict = {"Epeak": Epeaks[j], "gamma": gammas[j]}
        in_fov_rates, out_fov_rates = get_in_out_rates4EpeakGamma(
            sig_mod, Epeaks[j], gammas[j]
        )
        res_dict["RatesIn"] = in_fov_rates
        res_dict["RatesOut"] = out_fov_rates
        res_dicts.append(res_dict)

    return res_dicts


def mk_in_out_rates_tab_masks(sig_mod, theta, phi):
    dpi_shape = (173, 286)

    sig_mod.set_theta_phi(theta, phi)
    #     in_fov_bl = sig_mod.resp_obj.mask_obj.does_int_mask
    in_fov_bl = (sig_mod.resp_obj.mask_obj.does_int_mask) & (
        sig_mod.resp_obj.mask_obj.fix_trans[:, 10] > 0.99
    )
    out_fov_bl = ~in_fov_bl
    in_ndets = np.sum(in_fov_bl)
    if in_ndets < 100:
        print("Only %d dets in FoV" % (in_ndets))
        return None, None, None

    mask_in = np.zeros(dpi_shape, dtype=bool)
    mask_out = np.zeros(dpi_shape, dtype=bool)
    mask_in[sig_mod.bl_dmask] = in_fov_bl
    mask_out[sig_mod.bl_dmask] = out_fov_bl

    res_dicts = get_in_out_rates(sig_mod)

    tab = Table(data=res_dicts)

    return tab, mask_in, mask_out


def mk_npz_file_in_out_rates(sig_mod, theta, phi):
    dname = rates_resp_dir
    tab, mask_in, mask_out = mk_in_out_rates_tab_masks(sig_mod, theta, phi)
    if tab is None:
        return
    imx, imy = theta_phi2imxy(theta, phi)
    fname = "resp_imx_%.3f_imy_%.3f_" % (
        np.round(imx, decimals=3),
        np.round(imy, decimals=3),
    )
    Epeak = tab["Epeak"]
    gamma = tab["gamma"]
    RatesIn = tab["RatesIn"]
    RatesOut = tab["RatesOut"]
    save_fname = os.path.join(dname, fname)
    print(save_fname)
    np.savez(
        save_fname,
        RatesIn=RatesIn,
        RatesOut=RatesOut,
        Epeak=Epeak,
        gamma=gamma,
        mask_in=mask_in,
        mask_out=mask_out,
    )


def main(args):
    fname = "mk_inFoV_resp" + "_" + str(args.job_id)

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    xbins = np.linspace(-1.8, 1.8, 30 + 1)
    ybins = np.linspace(-1.0, 1.0, 25 + 1)

    Nx = len(xbins) - 1
    Ny = len(ybins) - 1
    xs = np.empty(0)
    ys = np.empty(0)

    for i in range(Nx):
        xmid = (xbins[i] + xbins[i + 1]) / 2.0 - 0.12 / 4
        for j in range(Ny):
            ymid = (ybins[j] + ybins[j + 1]) / 2.0
            if j % 2 == 0:
                xmid += 0.12 / 2
            else:
                xmid -= 0.12 / 2
            nsteps = 1
            yax = np.linspace(-0.1 / 2.0, 0.1 / 2.0, nsteps + 1)
            yax = (yax[1:] + yax[:-1]) / 2.0
            xax = np.linspace(-0.16 / 2, 0.16 / 2, nsteps + 1)
            xax = (xax[1:] + xax[:-1]) / 2.0
            for jj in range(nsteps):
                x0_ = 0.1 / (nsteps + 1)
                if jj % 2 == 1:
                    dx = -x0_
                else:
                    dx = x0_
                y = yax[jj] + ymid
                for ii in range(nsteps):
                    x = xax[ii] + dx + xmid
                    xs = np.append(xs, [x])
                    ys = np.append(ys, [y])
    print(len(xs))
    print(len(ys))

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)

    flux_mod = Cutoff_Plaw_Flux(E0=100.0)
    flux_params = {"A": 1.0, "Epeak": 1000.0, "gamma": 1.5}

    rt_obj = RayTraces(rt_dir, max_nbytes=1e9)

    bl_alldets = get_bldmask_alldets()
    # sig_mod = Source_Model_InFoV(flux_mod, [ebins0,ebins1], bl_alldets, rt_obj)

    Ntot_pnts = len(xs)
    Npnts2do = 1 + int(Ntot_pnts / args.Njobs)
    i0 = Npnts2do * args.job_id
    i1 = i0 + Npnts2do

    for i in range(i0, i1):
        theta, phi = imxy2theta_phi(xs[i], ys[i])
        logging.info("Starting %d of %d points" % (1 + i - i0, Npnts2do))
        logging.info("imx, imy: %.3f, %.3f" % (xs[i], ys[i]))
        logging.info("theta, phi: %.3f, %.3f" % (theta, phi))

        sig_mod = Source_Model_InOutFoV(flux_mod, [ebins0, ebins1], bl_alldets, rt_obj)

        mk_npz_file_in_out_rates(sig_mod, theta, phi)


if __name__ == "__main__":
    args = cli()

    main(args)
