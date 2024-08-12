import pytest
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
from scipy import optimize, stats, interpolate
from scipy.integrate import quad
import argparse
import time
import multiprocessing as mp
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import sys
import pandas as pd

pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
import healpy as hp
from copy import copy, deepcopy
import logging, traceback
import sys

import nitrates
from nitrates.config import rt_dir, solid_angle_dpi_fname
from nitrates.lib import (
    get_conn,
    det2dpi,
    mask_detxy,
    get_info_tab,
    get_twinds_tab,
    ang_sep,
    theta_phi2imxy,
    imxy2theta_phi,
    convert_imxy2radec,
    convert_radec2thetaphi,
    convert_radec2imxy,
    convert_theta_phi2radec,
)
from nitrates.response import RayTraces
from nitrates.models import (
    Cutoff_Plaw_Flux,
    Plaw_Flux,
    get_eflux_from_model,
    Source_Model_InOutFoV,
    Bkg_Model_wFlatA,
    CompoundModel,
    Point_Source_Model_Binned_Rates,
    im_dist,
    Sig_Bkg_Model,
)
from nitrates.llh_analysis import parse_bkg_csv, LLH_webins, NLLH_ScipyMinimize_Wjacob
from nitrates.llh_analysis.LLH import LLH_webins2


#################################################################################
# Function for calculating the likelihood ratio test statistic in the same manner
# as in the Example_LLH_setup_fixed_dirs.ipynb notebook. Here, we specify the
# theta and phi values of the source signal as arguments.
def old_calculate_LLH_ratio_test_statistic(theta, phi):
    script_dir = os.path.dirname(__file__)
    work_dir = os.path.join(script_dir, "nitrates_resp_dir")
    rt_dir = os.path.join(work_dir, "ray_traces_detapp_npy")
    nitrates.config.COMP_FLOR_RESP_DNAME = os.path.join(work_dir, "comp_flor_resps")
    nitrates.config.HP_FLOR_RESP_DNAME = os.path.join(work_dir, "hp_flor_resps")
    solid_angle_dpi_fname = os.path.join(work_dir, "solid_angle_dpi.npy")

    nitrates.config.RESP_TAB_DNAME = os.path.join(work_dir, "resp_tabs_ebins")
    nitrates.config.bright_source_table_fname = os.path.join(
        work_dir, "bright_src_cat.fits"
    )
    nitrates.config.ELEMENT_CROSS_SECTION_DNAME = os.path.join(
        work_dir, "element_cross_sections"
    )

    conn = get_conn(os.path.join(work_dir, "results.db"))
    info_tab = get_info_tab(conn)

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])

    trigger_time = info_tab["trigtimeMET"][0]
    t_end = trigger_time + 1e3
    t_start = trigger_time - 1e3

    evfname = os.path.join(work_dir, "filter_evdata.fits")
    ev_data = fits.open(evfname)[1].data

    GTI_PNT = Table.read(evfname, hdu="GTI_POINTING")
    GTI_SLEW = Table.read(evfname, hdu="GTI_SLEW")

    attfile = fits.open(os.path.join(work_dir, "attitude.fits"))[1].data
    att_ind = np.argmin(np.abs(attfile["TIME"] - trigger_time))
    att_quat = attfile["QPARAM"][att_ind]

    dmask = fits.open(os.path.join(work_dir, "detmask.fits"))[0].data

    mask_vals = mask_detxy(dmask, ev_data)
    bl_dmask = dmask == 0.0

    bl_ev = (
        (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] <= 500.0)
        & (ev_data["ENERGY"] >= 14.0)
        & (mask_vals == 0.0)
        & (ev_data["TIME"] <= t_end)
        & (ev_data["TIME"] >= t_start)
    )

    ev_data0 = ev_data[bl_ev]

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}
    flux_mod = Cutoff_Plaw_Flux(E0=100.0)

    rt_obj = RayTraces(rt_dir)

    sig_mod = Source_Model_InOutFoV(
        flux_mod,
        [ebins0, ebins1],
        bl_dmask,
        rt_obj,
        use_deriv=True,
        resp_tab_dname=nitrates.config.RESP_TAB_DNAME,
        comp_flor_resp_dname=nitrates.config.COMP_FLOR_RESP_DNAME,
        hp_flor_resp_dname=nitrates.config.HP_FLOR_RESP_DNAME,
    )

    sig_mod.get_batxys()
    sig_mod.set_theta_phi(theta, phi)

    bkg_fname = os.path.join(work_dir, "bkg_estimation.csv")
    solid_ang_dpi = np.load(solid_angle_dpi_fname)

    bkg_df, bkg_name, PSnames, bkg_mod, ps_mods = parse_bkg_csv(
        bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir
    )

    bkg_mod.has_deriv = False
    bkg_mod_list = [bkg_mod]

    Nsrcs = len(ps_mods)
    if Nsrcs > 0:
        bkg_mod_list += ps_mods
        for ps_mod in ps_mods:
            ps_mod.has_deriv = False
        bkg_mod = CompoundModel(bkg_mod_list)

    tmid = trigger_time
    bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
    bkg_params = {pname: bkg_row[pname] for pname in bkg_mod.param_names}
    bkg_name = bkg_mod.name

    pars_ = {}
    pars_["Signal_theta"] = theta
    pars_["Signal_phi"] = phi
    for pname, val in list(bkg_params.items()):
        pars_[bkg_name + "_" + pname] = val
    for pname, val in list(flux_params.items()):
        pars_["Signal_" + pname] = val

    sig_mod.set_flux_params(flux_params)
    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_llh_obj = LLH_webins(ev_data0, ebins0, ebins1, bl_dmask, has_err=True)
    sig_llh_obj.set_model(comp_mod)

    sig_miner = NLLH_ScipyMinimize_Wjacob("")
    sig_miner.set_llh(sig_llh_obj)

    fixed_pnames = list(pars_.keys())
    fixed_vals = list(pars_.values())
    trans = [None for i in range(len(fixed_pnames))]
    sig_miner.set_trans(fixed_pnames, trans)
    sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    sig_miner.set_fixed_params(["Signal_A"], fixed=False)

    flux_params["gamma"] = 0.8
    flux_params["Epeak"] = 350.0
    sig_mod.set_flux_params(flux_params)

    t0 = trigger_time - 0.512
    t1 = t0 + 2.048
    sig_llh_obj.set_time(t0, t1)

    _, nllh, _ = sig_miner.minimize()
    pars_["Signal_A"] = 1e-10
    bkg_nllh = -sig_llh_obj.get_logprob(pars_)

    sqrtTS = np.sqrt(2.0 * (bkg_nllh - nllh[0]))
    return sqrtTS


# Modifying previous test function to handle new LLH_webins2 class
def calculate_LLH_ratio_test_statistic(theta, phi):
    script_dir = os.path.dirname(__file__)
    work_dir = os.path.join(script_dir, "nitrates_resp_dir")
    rt_dir = os.path.join(work_dir, "ray_traces_detapp_npy")
    nitrates.config.COMP_FLOR_RESP_DNAME = os.path.join(work_dir, "comp_flor_resps")
    nitrates.config.HP_FLOR_RESP_DNAME = os.path.join(work_dir, "hp_flor_resps")
    solid_angle_dpi_fname = os.path.join(work_dir, "solid_angle_dpi.npy")

    nitrates.config.RESP_TAB_DNAME = os.path.join(work_dir, "resp_tabs_ebins")
    nitrates.config.bright_source_table_fname = os.path.join(
        work_dir, "bright_src_cat.fits"
    )
    nitrates.config.ELEMENT_CROSS_SECTION_DNAME = os.path.join(
        work_dir, "element_cross_sections"
    )

    evfname = os.path.join(work_dir, "filter_evdata.fits")
    conn = get_conn(os.path.join(work_dir, "results.db"))
    solid_ang_dpi = np.load(solid_angle_dpi_fname)
    info_tab = get_info_tab(conn)

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])

    trigger_time = info_tab["trigtimeMET"][0]
    t_end = trigger_time + 1e3
    t_start = trigger_time - 1e3

    ev_data = fits.open(evfname)[1].data

    GTI_PNT = Table.read(evfname, hdu="GTI_POINTING")
    GTI_SLEW = Table.read(evfname, hdu="GTI_SLEW")

    dmask = fits.open(os.path.join(work_dir, "detmask.fits"))[0].data

    mask_vals = mask_detxy(dmask, ev_data)
    bl_dmask = dmask == 0.0

    bl_ev = (
        (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] <= 500.0)
        & (ev_data["ENERGY"] >= 14.0)
        & (mask_vals == 0.0)
        & (ev_data["TIME"] <= t_end)
        & (ev_data["TIME"] >= t_start)
    )

    ev_data0 = ev_data[bl_ev]

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}
    flux_mod = Cutoff_Plaw_Flux(E0=100.0)

    rt_obj = RayTraces(rt_dir)

    sig_mod = Source_Model_InOutFoV(
        flux_mod,
        [ebins0, ebins1],
        bl_dmask,
        rt_obj,
        use_deriv=True,
        resp_tab_dname=nitrates.config.RESP_TAB_DNAME,
        comp_flor_resp_dname=nitrates.config.COMP_FLOR_RESP_DNAME,
        hp_flor_resp_dname=nitrates.config.HP_FLOR_RESP_DNAME,
    )

    sig_mod.get_batxys()
    sig_mod.set_theta_phi(theta, phi)

    bkg_fname = os.path.join(work_dir, "bkg_estimation.csv")
    bkg_df, bkg_name, _, bkg_mod, ps_mods = parse_bkg_csv(
        bkg_fname, solid_ang_dpi, ebins0, ebins1, bl_dmask, rt_dir
    )

    bkg_mod.has_deriv = False
    bkg_mod_list = [bkg_mod]

    Nsrcs = len(ps_mods)
    if Nsrcs > 0:
        bkg_mod_list += ps_mods
        for ps_mod in ps_mods:
            ps_mod.has_deriv = False
        bkg_mod = CompoundModel(bkg_mod_list)

    tmid = trigger_time
    bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df["time"]))]
    bkg_params = {pname: bkg_row[pname] for pname in bkg_mod.param_names}
    bkg_name = bkg_mod.name

    pars_ = {}
    pars_["Signal_theta"] = theta
    pars_["Signal_phi"] = phi
    for pname, val in list(bkg_params.items()):
        pars_[bkg_name + "_" + pname] = val
    for pname, val in list(flux_params.items()):
        pars_["Signal_" + pname] = val

    sig_llh_obj = LLH_webins2(ev_data0, ebins0, ebins1, bl_dmask, has_err=True)
    sig_mod.set_flux_params(flux_params)

    sig_bkg_mod = Sig_Bkg_Model(bl_dmask, sig_mod, bkg_mod, use_deriv=True)
    sig_pars = copy(flux_params)
    sig_pars["A"] = 1.0
    sig_pars["theta"] = theta
    sig_pars["phi"] = phi

    sig_bkg_mod.set_bkg_params(bkg_params)
    sig_bkg_mod.set_sig_params(sig_pars)
    sig_llh_obj.set_model(sig_bkg_mod)

    sig_miner = NLLH_ScipyMinimize_Wjacob("")
    sig_miner.set_llh(sig_llh_obj)
    sig_miner.set_fixed_params(["A"], fixed=False)

    flux_params["gamma"] = 0.8
    flux_params["Epeak"] = 350.0
    sig_mod.set_flux_params(flux_params)
    sig_bkg_mod.set_bkg_params(bkg_params)
    sig_bkg_mod.set_sig_params(sig_pars)

    t0 = trigger_time - 0.512
    t1 = t0 + 2.048
    sig_llh_obj.set_time(t0, t1)

    _, nllh, _ = sig_miner.minimize()
    pars_["A"] = 1e-10
    bkg_nllh = -sig_llh_obj.get_logprob(pars_)

    sqrtTS = np.sqrt(2.0 * (bkg_nllh - nllh[0]))

    return sqrtTS


#################################################################################
# Testing the calculation of the LLH ratio test statistic for an IFOV source
# signal at theta, phi = 38.541, 137.652, (or given as ra, dec = 233.117, -26.213)
def test_LLH_ratio_test_statistic_IFOV():
    sqrtTS = calculate_LLH_ratio_test_statistic(38.541, 137.652)
    assert math.isclose(sqrtTS, 17.0090525819679, rel_tol=1e-04) == True


#################################################################################
# Testing the calculation of the LLH ratio test statistic for an OFOV source
# signal at theta, phi = 125.0, 25.0.
def test_LLH_ratio_test_statistic_OFOV():
    sqrtTS = calculate_LLH_ratio_test_statistic(125.0, 25.0)
    assert math.isclose(sqrtTS, 15.516484046540924, rel_tol=1e-04) == True


#################################################################################
# Older versions of the above tests using LLH_webins and using CompoundModel
# in place of LLH_webins2 and Sig_Bkg_Model respectively.
def old_test_LLH_ratio_test_statistic_IFOV():
    sqrtTS = old_calculate_LLH_ratio_test_statistic(38.541, 137.652)
    print("Old sqrtTS IFOV: ", sqrtTS)
    assert math.isclose(sqrtTS, 17.0090525819679, rel_tol=1e-04) == True


def old_test_LLH_ratio_test_statistic_OFOV():
    sqrtTS = old_calculate_LLH_ratio_test_statistic(125.0, 25.0)
    print("Old sqrtTS OFOV: ", sqrtTS)
    assert math.isclose(sqrtTS, 15.516484046540924, rel_tol=1e-04) == True


if __name__ == "__main__":
    sqrtTS_1 = calculate_LLH_ratio_test_statistic(38.541, 137.652)
    sqrtTS_2 = calculate_LLH_ratio_test_statistic(125.0, 25.0)
    print("sqrtTS_1 for theta, phi = 38.541, 137.652: ", sqrtTS_1)
    print("sqrtTS_2 for theta, phi = 125.0, 25.0: ", sqrtTS_2)
    print("\n")

    sqrtTS_1 = old_calculate_LLH_ratio_test_statistic(38.541, 137.652)
    sqrtTS_2 = old_calculate_LLH_ratio_test_statistic(125.0, 25.0)
    print("old sqrtTS_1 for theta, phi = 38.541, 137.652: ", sqrtTS_1)
    print("old sqrtTS_2 for theta, phi = 125.0, 25.0: ", sqrtTS_2)
    print("\n")
