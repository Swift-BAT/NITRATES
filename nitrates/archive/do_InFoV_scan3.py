import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from numba import jit, njit, prange
from scipy import interpolate
from math import erf
import healpy as hp
import pandas as pd
import argparse
import logging, traceback
from copy import copy, deepcopy

# import ..config

from ..response.StructFunc import get_full_struct_manager
from ..models.flux_models import Plaw_Flux, Cutoff_Plaw_Flux, Band_Flux
from ..config import (
    rt_dir,
    fp_dir,
    solid_angle_dpi_fname,
    drm_dir,
    bright_source_table_fname,
)
from ..lib.logllh_ebins_funcs import log_pois_prob, get_eflux, get_gammaln
from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..models.models import Model
from ..llh_analysis.minimizers import (
    NLLH_DualAnnealingMin,
    NLLH_ScipyMinimize,
    NLLH_ScipyMinimize_Wjacob,
)
from ..lib.coord_conv_funcs import (
    convert_radec2imxy,
    convert_imxy2radec,
    convert_radec2batxyz,
    convert_radec2thetaphi,
)
from ..response.ray_trace_funcs import RayTraces, FootPrints
from ..lib.hp_funcs import ang_sep
from ..response.StructClasses import Swift_Structure, Swift_Structure_Manager
from ..response.Materials import PB, TI, Korex, CarbonFibre, Mylar
from ..response.Polygons import Polygon2D, Box_Polygon
from ..archive.do_bkg_estimation_wPSs_mp import get_srcs_infov
from ..lib.gti_funcs import add_bti2gti, bti2gti


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument("--attfname", type=str, help="attitude fname", default=None)
    parser.add_argument(
        "--job_id", type=int, help="ID to tell it what seeds to do", default=-1
    )
    parser.add_argument(
        "--Njobs", type=int, help="Total number of jobs submitted", default=64
    )
    parser.add_argument("--work_dir", type=str, help="work directory", default=None)
    parser.add_argument(
        "--log_fname", type=str, help="log file name", default="in_fov_scan"
    )
    parser.add_argument("--Nside", type=int, help="Healpix Nside", default=2**4)
    parser.add_argument("--trig_time", type=float, help="Trigger time", default=None)
    parser.add_argument(
        "--Ntdbls", type=int, help="Number of times to double duration size", default=3
    )
    parser.add_argument("--min_dur", type=float, help="Trigger time", default=0.256)
    parser.add_argument(
        "--min_dt",
        type=float,
        help="Min time offset from trigger time to start at",
        default=1.25,
    )
    parser.add_argument(
        "--max_dt",
        type=float,
        help="Min time offset from trigger time to start at",
        default=3.75,
    )
    parser.add_argument(
        "--bkg_dt0",
        type=float,
        help="Time offset from trigger time to start bkg at",
        default=6.0,
    )
    parser.add_argument(
        "--bkg_dur", type=float, help="Duration to use for bkg", default=4.0
    )
    args = parser.parse_args()
    return args


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


@njit(cache=True)
def shift_pha_bins(spec, pha_bins0, pha_bins1, new_pha_bins0, new_pha_bins1):
    new_spec = np.zeros_like(new_pha_bins0)
    for i in range(len(new_spec)):
        e0 = new_pha_bins0[i]
        e1 = new_pha_bins1[i]
        bl = (pha_bins0 >= e0) & (pha_bins1 <= e1)
        new_spec[i] += np.sum(spec[bl])
        bl = (pha_bins0 < e0) & (pha_bins1 > e0)
        if np.sum(bl) > 0:
            ind = np.where(bl)[0][0]
            dE = pha_bins1[ind] - pha_bins0[ind]
            frac_in_bin = (pha_bins1[ind] - e0) / dE
            new_spec[i] += frac_in_bin * spec[ind]
        bl = (pha_bins0 < e1) & (pha_bins1 > e1)
        if np.sum(bl) > 0:
            ind = np.where(bl)[0][0]
            dE = pha_bins1[ind] - pha_bins0[ind]
            frac_in_bin = (e1 - pha_bins0[ind]) / dE
            new_spec[i] += frac_in_bin * spec[ind]

    return new_spec


@njit(cache=True)
def shift_flor_dpi_pha_bins(
    flor_dpi, pha_bins0, pha_bins1, new_pha_bins0, new_pha_bins1
):
    Nphabins_new = new_pha_bins0.size
    Ndets = flor_dpi.shape[0]
    NphotonEs = flor_dpi.shape[1]
    new_shp = (Ndets, NphotonEs, Nphabins_new)
    new_flor_dpi = np.zeros(new_shp)

    for i in range(Ndets):
        for j in range(NphotonEs):
            new_flor_dpi[i, j] += shift_pha_bins(
                flor_dpi[i, j], pha_bins0, pha_bins1, new_pha_bins0, new_pha_bins1
            )
    return new_flor_dpi


def shift_resp_tab_pha_bins(
    resp_tab, pha_bins0, pha_bins1, new_pha_bins0, new_pha_bins1
):
    new_tab = Table()
    new_tab["ENERG_LO"] = np.copy(resp_tab["ENERG_LO"])
    new_tab["ENERG_HI"] = np.copy(resp_tab["ENERG_HI"])
    NphotonEs = len(resp_tab["ENERG_LO"])

    for cname in resp_tab.colnames:
        if "ENERG" in cname:
            continue
        new_resp = np.zeros((NphotonEs, len(new_pha_bins0)))
        for i in range(NphotonEs):
            new_resp[i] += shift_pha_bins(
                resp_tab[cname][i].astype(np.float64),
                pha_bins0.astype(np.float64),
                pha_bins1.astype(np.float64),
                new_pha_bins0.astype(np.float64),
                new_pha_bins1.astype(np.float64),
            )
        new_tab[cname] = new_resp

    return new_tab


def get_dist2(x0, y0, z0, x1, y1, z1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2


def get_dist(x0, y0, z0, x1, y1, z1):
    return np.sqrt(get_dist2(x0, y0, z0, x1, y1, z1))


def get_dist_wts(x0, y0, z0, x1, y1, z1):
    wts = 1.0 / get_dist2(x0, y0, z0, x1, y1, z1)
    wts /= np.sum(wts)
    return wts


def get_sa_divA(x0, y0, z0, x1, y1, z1):
    dist3 = get_dist2(x0, y0, z0, x1, y1, z1) ** 1.5
    return np.abs(z1 - z0) / dist3


def get_sa_wts(x0, y0, z0, x1, y1, z1):
    wts = get_sa_divA(x0, y0, z0, x1, y1, z1)
    wts /= np.sum(wts)
    return wts


class Comp_Resp_Obj(object):
    def __init__(self, batxs, batys, batzs, struct4comp):
        self.ndets = len(batxs)
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs

        self.Ne = struct4comp.Ne

        self.struct_obj = struct4comp
        self.ncomp_pnts = len(self.struct_obj.batxs)
        self.comp_batxs = self.struct_obj.batxs
        self.comp_batys = self.struct_obj.batys
        self.comp_batzs = self.struct_obj.batzs

        self.calc_inds_wts4comp_dets()

    def calc_inds_wts4comp_dets(self, dmax=16):
        self.wts_list = []
        self.inds_list = []

        for i in range(self.ndets):
            dists = get_dist(
                self.comp_batxs,
                self.comp_batys,
                self.comp_batzs,
                self.batxs[i],
                self.batys[i],
                self.batzs[i],
            )
            bl = dists <= dmax
            wts = get_sa_wts(
                self.comp_batxs[bl],
                self.comp_batys[bl],
                self.comp_batzs[bl],
                self.batxs[i],
                self.batys[i],
                self.batzs[i],
            )
            inds = np.where(bl)[0]
            self.wts_list.append(wts)
            self.inds_list.append(inds)

    def set_theta_phi(self, theta, phi):
        self.struct_obj.set_theta_phi(theta, phi)
        self.struct_obj.calc_tot_rhomu_dist()
        self.calc_trans()

    def calc_trans(self):
        self.trans = np.zeros((self.ndets, self.Ne))

        self.comp_trans = np.zeros((self.ncomp_pnts, self.Ne))

        self.comp_trans[: self.ncomp_pnts] += self.struct_obj.get_trans()

        print(
            np.shape(self.trans[0]),
            np.shape(self.wts_list[0]),
            np.shape(self.comp_trans[self.inds_list[0], :]),
        )
        print(
            np.shape(
                np.sum(
                    self.comp_trans[self.inds_list[0], :]
                    * self.wts_list[0][:, np.newaxis],
                    axis=0,
                )
            )
        )
        for i in range(self.ndets):
            self.trans[i] += np.sum(
                self.comp_trans[self.inds_list[i], :] * self.wts_list[i][:, np.newaxis],
                axis=0,
            )

    def get_trans(self):
        return self.trans


def get_dual_struct_obj(Ephotons):
    dual_xs = []
    dual_ys = []
    for bi in range(8):
        x_b = -52.92 + bi * 15.12
        y_b = 23.555
        for i in range(2):
            x = x_b - 3.78 + i * 7.56
            for j in range(4):
                y = y_b - 18.935 + j * 9.24
                dual_xs.append(x)
                dual_ys.append(y)

    for bi in range(8):
        x_b = -52.92 + bi * 15.12
        y_b = -23.555
        for i in range(2):
            x = x_b - (-3.78 + i * 7.56)
            for j in range(4):
                y = y_b - (-18.935 + j * 9.24)
                dual_xs.append(x)
                dual_ys.append(y)

    dual_xs = np.array(dual_xs)
    dual_ys = np.array(dual_ys)
    print(len(dual_xs), len(dual_ys))

    BATZ_offset = 35.799
    dual_elec_x_halfwidth = 3.55
    dual_elec_y_halfwidth = 4.41

    dual_elec_z0 = -3.725 - 32.612 + BATZ_offset + 1.15 - 1.06 - 1.865
    dual_elec_z1 = -3.725 - 32.612 + BATZ_offset + 1.15 - 1.06 + 1.865
    dual_elec_zmid = -3.725 - 32.612 + BATZ_offset + 1.15 - 1.06
    dual_elec_z_halfwidth = 1.865

    # for each dual lets do 8 pnts (+/- x_hw/2, +/- y_hw/2, +/- z_hw/2)

    batxs4duals = []
    batys4duals = []
    batzs4duals = []

    Nduals = len(dual_xs)

    for ii in range(Nduals):
        dualx = dual_xs[ii]
        dualy = dual_ys[ii]
        for i in range(2):
            x = dualx - dual_elec_x_halfwidth / 2.0 + i * dual_elec_x_halfwidth
            for j in range(2):
                y = dualy - dual_elec_y_halfwidth / 2.0 + j * dual_elec_y_halfwidth
                for k in range(2):
                    z = (
                        dual_elec_zmid
                        - dual_elec_z_halfwidth / 2.0
                        + k * dual_elec_z_halfwidth
                    )
                    batxs4duals.append(x)
                    batys4duals.append(y)
                    batzs4duals.append(z)
    batxs4duals = np.array(batxs4duals)
    batys4duals = np.array(batys4duals)
    batzs4duals = np.array(batzs4duals)
    print(len(batxs4duals))

    dual_struct_obj = get_full_struct_manager(Es=Ephotons)
    dual_struct_obj.set_batxyzs(batxs4duals, batys4duals, batzs4duals)

    return dual_struct_obj


detxs_by_sand0 = np.arange(0, 286 - 15, 18)
detxs_by_sand1 = detxs_by_sand0 + 15
print(len(detxs_by_sand0))

detys_by_sand0 = np.arange(0, 173 - 7, 11)
detys_by_sand1 = detys_by_sand0 + 7
print(len(detys_by_sand0))

detxs_in_cols_not_edges = [
    np.arange(detxs_by_sand0[i] + 1, detxs_by_sand1[i], 1, dtype=np.int64)
    for i in range(16)
]
detys_in_rows_not_edges = [
    np.arange(detys_by_sand0[i] + 1, detys_by_sand1[i], 1, dtype=np.int64)
    for i in range(16)
]
print(detxs_in_cols_not_edges)

dpi_shape = (173, 286)
detxax = np.arange(286, dtype=np.int64)
detyax = np.arange(173, dtype=np.int64)
detx_dpi, dety_dpi = np.meshgrid(detxax, detyax)
print(np.shape(detx_dpi), np.shape(dety_dpi))
print(np.max(detx_dpi), np.max(dety_dpi))


def get_detxys_from_colrows(col0, col1, row0, row1, orientation="NonEdges"):
    if orientation == "NonEdges":
        good_detxs = np.array(detxs_in_cols_not_edges[col0:col1])
        good_detys = np.array(detys_in_rows_not_edges[row0:row1])
    elif orientation == "left":
        good_detxs = np.array(detxs_by_sand0[col0:col1])
        good_detys = np.array(detys_in_rows_not_edges[row0:row1])
        good_detys = np.append(good_detys, np.array(detys_by_sand1[row0:row1]))
    elif orientation == "top":
        good_detxs = np.array(detxs_in_cols_not_edges[col0:col1])
        good_detys = np.array(detys_by_sand1[row0:row1])
    elif orientation == "bot":
        good_detxs = np.array(detxs_in_cols_not_edges[col0:col1])
        good_detxs = np.append(good_detxs, np.array(detxs_by_sand0[col0:col1]))
        good_detys = np.array(detys_by_sand0[row0:row1])
    elif orientation == "right":
        good_detxs = np.array(detxs_by_sand1[col0:col1])
        good_detys = np.array(detys_in_rows_not_edges[row0:row1])
        good_detys = np.append(good_detys, np.array(detys_by_sand1[row0:row1]))
        good_detys = np.append(good_detys, np.array(detys_by_sand0[row0:row1]))
    else:
        print("bad orientation")
    blx = np.isin(detx_dpi, good_detxs)
    bly = np.isin(dety_dpi, good_detys)
    bl = blx & bly
    inds = np.where(bl)
    return inds


def rot_col_row_orientation(col0, col1, row0, row1, orientation, phi_rot):
    if phi_rot < 0:
        phi_rot = phi_rot + 2 * np.pi

    if (phi_rot >= np.pi / 4) and (phi_rot < np.pi / 2):
        # bot is strong
        # right is weak

        new_row0 = 16 - col1
        new_row1 = 16 - col0

        new_col0 = 16 - row1
        new_col1 = 16 - row0

        if orientation == "right":
            new_orientation = "bot"
        elif orientation == "bot":
            new_orientation = "right"
        else:
            new_orientation = orientation

    elif (phi_rot >= np.pi / 2) and (phi_rot < 3 * np.pi / 4):
        # bot is strong
        # left is weak

        new_row0 = 16 - col1
        new_row1 = 16 - col0

        new_col0 = row0
        new_col1 = row1

        if orientation == "right":
            new_orientation = "bot"
        elif orientation == "bot":
            new_orientation = "left"
        elif orientation == "left":
            new_orientation = "right"
        else:
            new_orientation = orientation

    elif (phi_rot >= 3 * np.pi / 4) and (phi_rot < np.pi):
        # left is strong
        # bot is weak

        new_row0 = row0
        new_row1 = row1

        new_col0 = 16 - col1
        new_col1 = 16 - col0

        if orientation == "right":
            new_orientation = "left"
        elif orientation == "left":
            new_orientation = "right"
        else:
            new_orientation = orientation

    elif (phi_rot >= np.pi) and (phi_rot < 5 * np.pi / 4):
        # left is strong
        # top is weak

        new_row0 = 16 - row1
        new_row1 = 16 - row0

        new_col0 = 16 - col1
        new_col1 = 16 - col0

        if orientation == "right":
            new_orientation = "left"
        elif orientation == "bot":
            new_orientation = "top"
        elif orientation == "left":
            new_orientation = "right"
        elif orientation == "top":
            new_orientation = "bot"
        else:
            new_orientation = orientation

    elif (phi_rot >= 5 * np.pi / 4) and (phi_rot < 6 * np.pi / 4):
        # top is strong
        # left is weak

        new_row0 = col0
        new_row1 = col1

        new_col0 = row0
        new_col1 = row1

        if orientation == "right":
            new_orientation = "top"
        elif orientation == "bot":
            new_orientation = "left"
        elif orientation == "left":
            new_orientation = "right"
        elif orientation == "top":
            new_orientation = "bot"
        else:
            new_orientation = orientation

    elif (phi_rot >= 6 * np.pi / 4) and (phi_rot < 7 * np.pi / 4):
        # top is strong
        # right is weak

        new_row0 = col0
        new_row1 = col1

        new_col0 = 16 - row1
        new_col1 = 16 - row0

        if orientation == "right":
            new_orientation = "top"
        elif orientation == "bot":
            new_orientation = "right"
        elif orientation == "top":
            new_orientation = "bot"
        else:
            new_orientation = orientation

    elif (phi_rot >= 7 * np.pi / 4) and (phi_rot < 8 * np.pi / 4):
        # right is strong
        # top is weak

        new_row0 = 16 - row1
        new_row1 = 16 - row0

        new_col0 = col0
        new_col1 = col1

        if orientation == "bot":
            new_orientation = "top"
        elif orientation == "top":
            new_orientation = "bot"
        else:
            new_orientation = orientation

    else:
        new_orientation = orientation
        new_row0 = row0
        new_row1 = row1
        new_col0 = col0
        new_col1 = col1

    return new_col0, new_col1, new_row0, new_row1, new_orientation


def resp_tab2resp_dpis(resp_tab, phi_rot=0.0):
    line_cnames = [
        cname
        for cname in resp_tab.colnames
        if (not "ENERG" in cname) and (not "comp" in cname)
    ]
    comp_cnames = [
        cname
        for cname in resp_tab.colnames
        if (not "ENERG" in cname) and ("comp" in cname)
    ]

    NphotonEs, Nphabins = resp_tab[line_cnames[0]].shape

    lines_resp_dpi = np.zeros((173, 286, NphotonEs, Nphabins))

    for cname in line_cnames:
        cname_list = cname.split("_")
        col0 = int(cname_list[-5])
        col1 = int(cname_list[-4])
        row0 = int(cname_list[-2])
        row1 = int(cname_list[-1])
        orientation = cname_list[0]

        (
            new_col0,
            new_col1,
            new_row0,
            new_row1,
            new_orientation,
        ) = rot_col_row_orientation(col0, col1, row0, row1, orientation, phi_rot)

        det_inds = get_detxys_from_colrows(
            new_col0, new_col1, new_row0, new_row1, orientation=new_orientation
        )

        lines_resp_dpi[det_inds[0], det_inds[1], :, :] = resp_tab[cname].data.copy()

    comp_resp_dpi = np.zeros((173, 286, NphotonEs, Nphabins))

    for cname in comp_cnames:
        cname_list = cname.split("_")
        col0 = int(cname_list[-6])
        col1 = int(cname_list[-5])
        row0 = int(cname_list[-3])
        row1 = int(cname_list[-2])
        orientation = cname_list[0]

        (
            new_col0,
            new_col1,
            new_row0,
            new_row1,
            new_orientation,
        ) = rot_col_row_orientation(col0, col1, row0, row1, orientation, phi_rot)

        det_inds = get_detxys_from_colrows(
            new_col0, new_col1, new_row0, new_row1, orientation=new_orientation
        )

        comp_resp_dpi[det_inds[0], det_inds[1], :, :] = resp_tab[cname].data.copy()

    return lines_resp_dpi, comp_resp_dpi


def get_resp_arr(drm_dir):
    fnames = np.array([fn for fn in os.listdir(drm_dir) if "drm_" in fn])
    thetas = np.array([float(fn.split("_")[2]) for fn in fnames])
    phis = np.array([float(fn.split("_")[4]) for fn in fnames])

    dtp = [("theta", np.float64), ("phi", np.float64), ("fname", fnames.dtype)]
    drm_arr = np.empty(len(thetas), dtype=dtp)
    drm_arr["theta"] = thetas
    drm_arr["phi"] = phis
    drm_arr["fname"] = fnames
    return drm_arr


class ResponseDPI(object):
    def __init__(self, resp_fname, pha_emins, pha_emaxs, phi0, bl_dmask):
        print("initing ResponseDPI, with fname")
        print(resp_fname)
        self.orig_resp_tab = Table.read(resp_fname)
        self.pha_tab = Table.read(resp_fname, hdu="EBOUNDS")

        self.orig_pha_emins = self.pha_tab["E_MIN"]
        self.orig_pha_emaxs = self.pha_tab["E_MAX"]

        self.photonEmins = self.orig_resp_tab["ENERG_LO"]
        self.photonEmaxs = self.orig_resp_tab["ENERG_HI"]
        self.photonEs = (self.photonEmins + self.photonEmaxs) / 2.0
        self.NphotonEs = len(self.photonEs)
        self.phi0 = phi0  # should be in radians

        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask

        self.set_pha_bins(pha_emins, pha_emaxs)
        self.mk_resp_dpis()

    def set_pha_bins(self, pha_emins, pha_emaxs):
        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(self.pha_emins)

        self.resp_tab = shift_resp_tab_pha_bins(
            self.orig_resp_tab,
            self.orig_pha_emins,
            self.orig_pha_emaxs,
            self.pha_emins,
            self.pha_emaxs,
        )

    def set_phi0(self, phi0):
        if np.abs(phi0 - self.phi0) > 1e-2:
            self.phi0 = phi0
            self.mk_resp_dpis()

    def mk_resp_dpis(self):
        lines_resp_dpis, comp_resp_dpis = resp_tab2resp_dpis(
            self.resp_tab, phi_rot=self.phi0
        )
        self.lines_resp_dpis = lines_resp_dpis[self.bl_dmask]
        self.comp_resp_dpis = comp_resp_dpis[self.bl_dmask]

    def get_lines_resp_dpis(self):
        return self.lines_resp_dpis

    def get_comp_resp_dpis(self):
        return self.comp_resp_dpis


def get_flor_intp_inds_wts(batxs, batys):
    detxax = np.arange(-1, 286 + 2, 8, dtype=np.int64)
    detyax = np.arange(-2, 173 + 2, 8, dtype=np.int64)
    batxax, batyax = detxy2batxy(detxax, detyax)
    flor_detx_dpi, flor_dety_dpi = np.meshgrid(detxax, detyax)
    shp = flor_detx_dpi.shape
    flor_batxs, flor_batys = detxy2batxy(flor_detx_dpi.ravel(), flor_dety_dpi.ravel())

    x0inds = np.digitize(batxs, batxax) - 1
    x1inds = x0inds + 1
    y0inds = np.digitize(batys, batyax) - 1
    y1inds = y0inds + 1

    x0s = batxax[x0inds]
    x1s = batxax[x1inds]
    dxs = x1s - x0s
    x0wts = (x1s - batxs) / dxs
    x1wts = (batxs - x0s) / dxs

    y0s = batyax[y0inds]
    y1s = batyax[y1inds]
    dys = y1s - y0s
    y0wts = (y1s - batys) / dys
    y1wts = (batys - y0s) / dys

    inds00 = np.ravel_multi_index((y0inds, x0inds), shp)
    inds01 = np.ravel_multi_index((y0inds, x1inds), shp)
    inds10 = np.ravel_multi_index((y1inds, x0inds), shp)
    inds11 = np.ravel_multi_index((y1inds, x1inds), shp)

    inds = [inds00, inds01, inds10, inds11]
    wts = [y0wts * x0wts, y0wts * x1wts, y1wts * x0wts, y1wts * x1wts]

    return inds, wts


@njit(cache=True)
def flor_resp2dpis(flor_resp, flor_inds, flor_wts):
    ndets = len(flor_inds[0])
    NphotonEs = flor_resp.shape[1]
    Nphabins = flor_resp.shape[2]

    flor_dpis = np.zeros((ndets, NphotonEs, Nphabins))

    for i in range(4):
        for j in range(ndets):
            flor_dpis[j] += flor_resp[flor_inds[i][j]] * (flor_wts[i][j])

    return flor_dpis


class FlorResponseDPI(object):
    def __init__(
        self,
        resp_dname,
        pha_tab,
        pha_emins,
        pha_emaxs,
        bl_dmask,
        Nside=2**3,
        NphotonEs=187,
    ):
        self.resp_dname = resp_dname
        self.pha_tab = pha_tab
        self.orig_pha_emins = self.pha_tab["E_MIN"].astype(np.float64)
        self.orig_pha_emaxs = self.pha_tab["E_MAX"].astype(np.float64)
        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(pha_emins)
        self.NphotonEs = NphotonEs

        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask
        self.batxs, self.batys = bldmask2batxys(self.bl_dmask)

        self.flor_inds, self.flor_wts = get_flor_intp_inds_wts(self.batxs, self.batys)
        self.orig_ndets = 851

        self.Nside = Nside

        self.resp_dict = {}  # hp inds will be the keys

    def set_theta_phi(self, theta, phi):
        self.phi = phi
        self.theta = theta
        self.lat = 90.0 - self.theta

        self.hp_inds2use, self.hp_wts = hp.get_interp_weights(
            self.Nside, self.phi, self.lat, lonlat=True
        )

        self.calc_resp_dpi()

    def open_new_file(self, hp_ind):
        fname = "hp_order_3_ind_%d_.npy" % (hp_ind)
        resp_arr = np.load(os.path.join(self.resp_dname, fname))
        self.resp_dict[hp_ind] = shift_flor_dpi_pha_bins(
            resp_arr,
            self.orig_pha_emins,
            self.orig_pha_emaxs,
            self.pha_emins,
            self.pha_emaxs,
        )

    def calc_resp_dpi(self):
        resp_dpi0 = np.zeros((self.orig_ndets, self.NphotonEs, self.Nphabins))

        for hp_ind, wt in zip(self.hp_inds2use, self.hp_wts):
            if not hp_ind in list(self.resp_dict.keys()):
                self.open_new_file(hp_ind)
            resp_dpi0 += wt * self.resp_dict[hp_ind]

        self.resp_dpi = flor_resp2dpis(resp_dpi0, self.flor_inds, self.flor_wts)

    def get_resp_dpi(self):
        return self.resp_dpi


class FlorResponseDPI(object):
    def __init__(
        self,
        resp_dname,
        pha_tab,
        pha_emins,
        pha_emaxs,
        bl_dmask,
        Nside=2**3,
        NphotonEs=187,
    ):
        self.resp_dname = resp_dname
        self.pha_tab = pha_tab
        self.orig_pha_emins = self.pha_tab["E_MIN"].astype(np.float64)
        self.orig_pha_emaxs = self.pha_tab["E_MAX"].astype(np.float64)
        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(pha_emins)
        self.NphotonEs = NphotonEs

        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask
        self.batxs, self.batys = bldmask2batxys(self.bl_dmask)

        self.flor_inds, self.flor_wts = get_flor_intp_inds_wts(self.batxs, self.batys)
        self.orig_ndets = 851

        fname = "/storage/work/jjd330/local/bat_data/OutFoVbursts/GRB131014A/flor_Aeff_adjust.npz"
        ratio_file = np.load(fname)
        self.sn_ratios = ratio_file["sn_ratios"]
        self.ta_ratios = ratio_file["ta_ratios"]
        self.pb_ratios = ratio_file["pb_ratios"]

        self.Nside = Nside

        self.resp_dict = {}  # hp inds will be the keys

    def set_theta_phi(self, theta, phi):
        self.phi = phi
        self.theta = theta
        self.lat = 90.0 - self.theta

        self.hp_inds2use, self.hp_wts = hp.get_interp_weights(
            self.Nside, self.phi, self.lat, lonlat=True
        )

        self.calc_resp_dpi()

    def open_new_file(self, hp_ind):
        fname = "hp_order_3_ind_%d_.npy" % (hp_ind)
        resp_arr = np.load(os.path.join(self.resp_dname, fname))
        sn_inds = np.arange(1, 13, dtype=np.int64)
        ta_inds = np.arange(14, 29, dtype=np.int64)
        pb_inds = np.arange(29, 39, dtype=np.int64)
        for sn_ind in sn_inds:
            resp_arr[:, :, sn_ind] *= self.sn_ratios
        for ta_ind in ta_inds:
            resp_arr[:, :, ta_ind] *= self.ta_ratios
        for pb_ind in pb_inds:
            resp_arr[:, :, pb_ind] *= self.pb_ratios

        self.resp_dict[hp_ind] = shift_flor_dpi_pha_bins(
            resp_arr,
            self.orig_pha_emins,
            self.orig_pha_emaxs,
            self.pha_emins,
            self.pha_emaxs,
        )

    def calc_resp_dpi(self):
        resp_dpi0 = np.zeros((self.orig_ndets, self.NphotonEs, self.Nphabins))

        for hp_ind, wt in zip(self.hp_inds2use, self.hp_wts):
            if not hp_ind in list(self.resp_dict.keys()):
                self.open_new_file(hp_ind)
            resp_dpi0 += wt * self.resp_dict[hp_ind]

        self.resp_dpi = flor_resp2dpis(resp_dpi0, self.flor_inds, self.flor_wts)

    #         for sn_ind in sn_inds:
    #             self.resp_dpi[:,:,sn_ind] *= self.sn_ratios
    #         for ta_ind in ta_inds:
    #             self.resp_dpi[:,:,ta_ind] *= self.ta_ratios
    #         for pb_ind in pb_inds:
    #             self.resp_dpi[:,:,pb_ind] *= self.pb_ratios

    def get_resp_dpi(self):
        return self.resp_dpi


class ResponseOutFoV(object):
    def __init__(self, resp_dname, pha_emins, pha_emaxs, bl_dmask):
        self.resp_dname = resp_dname
        self.resp_arr = get_resp_arr(self.resp_dname)
        self.thetas = np.unique(self.resp_arr["theta"])

        tab = Table.read(os.path.join(self.resp_dname, self.resp_arr["fname"][0]))
        pha_tab = Table.read(
            os.path.join(self.resp_dname, self.resp_arr["fname"][0]), hdu=2
        )
        self.PhotonEmins = tab["ENERG_LO"]
        self.PhotonEmaxs = tab["ENERG_HI"]
        self.PhotonEs = ((self.PhotonEmins + self.PhotonEmaxs) / 2.0).astype(np.float64)
        self.NphotonEs = len(self.PhotonEs)

        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(pha_emins)
        #         self.NphotonEs = NphotonEs
        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask
        self.batxs, self.batys = bldmask2batxys(self.bl_dmask)
        self.batzs = 3.087 + np.zeros(self.ndets)

        #         self.resp_dpi_shape = (173, 286, self.NphotonEs, self.Nphabins)
        self.resp_dpi_shape = (self.ndets, self.NphotonEs, self.Nphabins)

        self.resp_files = {}

        self.full_struct = get_full_struct_manager(Es=self.PhotonEs)
        self.full_struct.set_batxyzs(self.batxs, self.batys, self.batzs)

        dual_struct = get_dual_struct_obj(self.PhotonEs)
        self.comp_obj = Comp_Resp_Obj(self.batxs, self.batys, self.batzs, dual_struct)

        self.flor_resp_obj = FlorResponseDPI(
            "/gpfs/scratch/jjd330/bat_data/flor_resps/",
            pha_tab,
            self.pha_emins,
            self.pha_emaxs,
            self.bl_dmask,
            NphotonEs=self.NphotonEs,
        )

    def set_theta_phi(self, theta, phi):
        # use radians or degs ?

        self.theta = theta
        self.phi = phi

        self.thetas2use, self.phis2use, self.wts = self.get_intp_theta_phi_wts(
            self.theta, self.phi
        )
        self.inds4intp = []
        for i in range(len(self.wts)):
            ind = np.where(
                np.isclose(self.thetas2use[i], self.resp_arr["theta"])
                & np.isclose(self.phis2use[i], self.resp_arr["phi"])
            )[0][0]
            self.inds4intp.append(ind)

        self.full_struct.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        self.lines_trans_dpis = self.full_struct.get_trans()

        #         self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        #         self.comp_trans_dpis = self.comp_obj.get_trans()

        if theta > 90.0:
            self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.flor_resp_obj.set_theta_phi(self.theta, self.phi)

        self.calc_resp_dpis()
        self.calc_tot_resp_dpis()

    def update_trans(self, theta, phi):
        self.full_struct.set_theta_phi(np.radians(theta), np.radians(phi))
        self.lines_trans_dpis = self.full_struct.get_trans()

        #         self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        #         self.comp_trans_dpis = self.comp_obj.get_trans()

        if theta > 90.0:
            self.comp_obj.set_theta_phi(np.radians(theta), np.radians(phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.calc_tot_resp_dpis()

    def open_resp_file_obj(self, fname):
        resp_file_obj = ResponseDPI(
            os.path.join(self.resp_dname, fname),
            self.pha_emins,
            self.pha_emaxs,
            np.radians(self.phi),
            self.bl_dmask,
        )
        self.resp_files[fname] = resp_file_obj

    def calc_resp_dpis(self):
        self.lines_resp_dpis = np.zeros(self.resp_dpi_shape)
        self.comp_resp_dpis = np.zeros(self.resp_dpi_shape)

        for i in range(len(self.wts)):
            k = self.resp_arr["fname"][self.inds4intp[i]]
            if not k in list(self.resp_files.keys()):
                self.open_resp_file_obj(k)
            self.lines_resp_dpis += (
                self.wts[i] * self.resp_files[k].get_lines_resp_dpis()
            )
            self.comp_resp_dpis += self.wts[i] * self.resp_files[k].get_comp_resp_dpis()

    def calc_tot_resp_dpis(self):
        lines_dpi = self.lines_resp_dpis * (self.lines_trans_dpis[:, :, np.newaxis])
        comp_dpi = self.comp_resp_dpis * (self.comp_trans_dpis[:, :, np.newaxis])
        self.comp_resp_dpi = comp_dpi
        self.lines_resp_dpi = lines_dpi
        self.non_flor_resp_dpi = lines_dpi + comp_dpi
        self.flor_resp_dpi = self.flor_resp_obj.get_resp_dpi()
        self.tot_resp_dpis = self.non_flor_resp_dpi + self.flor_resp_dpi

    def get_lines_resp_dpis(self):
        return self.lines_resp_dpis

    def get_comp_resp_dpis(self):
        return self.comp_resp_dpis

    def get_flor_resp_dpis(self):
        return self.flor_resp_obj.get_resp_dpi()

    def get_tot_resp_dpis(self):
        return self.tot_resp_dpis

    def get_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = np.zeros((self.ndets, self.Nphabins))
        for j in range(self.Nphabins):
            rate_dpis[:, j] += np.sum(
                photon_fluxes * self.tot_resp_dpis[:, :, j], axis=1
            )
        return rate_dpis

    def get_flor_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = np.zeros((self.ndets, self.Nphabins))
        for j in range(self.Nphabins):
            rate_dpis[:, j] += np.sum(
                photon_fluxes * self.flor_resp_dpi[:, :, j], axis=1
            )
        return rate_dpis

    def get_comp_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = np.zeros((self.ndets, self.Nphabins))
        for j in range(self.Nphabins):
            rate_dpis[:, j] += np.sum(
                photon_fluxes * self.comp_resp_dpi[:, :, j], axis=1
            )
        return rate_dpis

    def get_photoe_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = np.zeros((self.ndets, self.Nphabins))
        for j in range(self.Nphabins):
            rate_dpis[:, j] += np.sum(
                photon_fluxes * self.lines_resp_dpi[:, :, j], axis=1
            )
        return rate_dpis

    def get_non_flor_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = np.zeros((self.ndets, self.Nphabins))
        for j in range(self.Nphabins):
            rate_dpis[:, j] += np.sum(
                photon_fluxes * self.non_flor_resp_dpi[:, :, j], axis=1
            )
        return rate_dpis

    def get_intp_theta_phi_wts(self, theta, phi, eps=0.1):
        thetas = np.sort(np.unique(self.resp_arr["theta"]))
        phis = np.sort(np.unique(self.resp_arr["phi"]))

        th0 = np.digitize(theta, thetas) - 1
        if theta == 180.0:
            th0 -= 1
        theta0 = thetas[th0]
        theta1 = thetas[th0 + 1]
        print(theta0, theta1)
        if np.abs(theta0 - theta) < eps:
            ths = [theta0]
            th_wts = [1.0]
        elif np.abs(theta1 - theta) < eps:
            ths = [theta1]
            th_wts = [1.0]
        else:
            ths = [theta0, theta1]
            dth = theta1 - theta0
            th_wts = [(theta1 - theta) / dth, (theta - theta0) / dth]

        phi_ = phi - (int(phi) / 45) * 45.0
        print(phi_)
        if (int(phi) / 45) % 2 == 1:
            phi_ = 45.0 - phi_
        print(phi_)
        ph0 = np.digitize(phi_, phis) - 1
        if phi_ == 45.0:
            ph0 -= 1

        phi0 = phis[ph0]
        phi1 = phis[ph0 + 1]

        if np.abs(phi0 - phi_) < eps:
            phs = [phi0]
            ph_wts = [1.0]
        elif np.abs(phi1 - phi_) < eps:
            phs = [phi1]
            ph_wts = [1.0]
        else:
            phs = [phi0, phi1]
            dph = phi1 - phi0
            ph_wts = [(phi1 - phi_) / dph, (phi_ - phi0) / dph]

        ths_ = []
        phs_ = []
        wts = []
        for i in range(len(ths)):
            if ths[i] == 0.0 or ths[i] == 180.0:
                ths_.append(ths[i])
                phs_.append(0.0)
                wts.append(th_wts[i])
                continue
            for j in range(len(phs)):
                ths_.append(ths[i])
                phs_.append(phs[j])
                wts.append(th_wts[i] * ph_wts[j])

        return ths_, phs_, wts


@njit(cache=True, fastmath=True)
def pois_norm_conv_n0(mu, sig):
    sig2 = sig**2
    return np.exp(((sig2 - mu) ** 2 - mu**2) / (2.0 * sig2))


@njit(cache=True, fastmath=True)
def pois_norm_conv_n1(mu, sig):
    sig2 = sig**2
    return ((mu - sig2)) * np.exp((sig2 / 2.0) - mu)


@njit(cache=True, fastmath=True)
def pois_norm_conv_n2(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    return eterm * (-mu * sig2 + 0.5 * (mu**2 + sig2**2 + sig2))


@njit(cache=True, fastmath=True)
def pois_norm_conv_n3(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    return eterm * 0.5 * (((mu - sig2) ** 3) / 3.0 + sig2 * (mu - sig2))


@njit(cache=True, fastmath=True)
def pois_norm_conv_n4(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    mu_sig2 = mu - sig2
    return (eterm / 24.0) * (
        (mu_sig2) ** 4 + 6 * (sig2 * mu_sig2**2) + 3 * (sig2**2)
    )


@njit(cache=True, fastmath=True)
def pois_norm_conv_n5(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    mu_sig2 = mu - sig2
    return (eterm / (5 * 24.0)) * (
        (mu_sig2) ** 5 + 5 * 2 * (sig2 * mu_sig2**3) + 5 * 3 * (sig2**2) * mu_sig2
    )


@njit(cache=True, fastmath=True)
def pois_norm_conv_n6(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    mu_sig2 = mu - sig2
    return (eterm / (6 * 5 * 24.0)) * (
        (mu_sig2) ** 6
        + 5 * 3 * (sig2 * mu_sig2**4)
        + 5 * 3 * 3 * (sig2**2) * mu_sig2**2
        + 5 * 3 * (sig2**3)
    )


@njit(cache=True, fastmath=True)
def pois_norm_conv_n7(mu, sig):
    sig2 = sig**2
    eterm = np.exp((sig2 / 2.0) - mu)
    mu_sig2 = mu - sig2
    return (eterm / (7 * 6 * 5 * 24.0)) * (
        (mu_sig2) ** 7
        + 7 * 3 * (sig2 * mu_sig2**5)
        + 7 * 5 * 3 * (sig2**2) * mu_sig2**3
        + 7 * 5 * 3 * (sig2**3) * mu_sig2
    )


@njit(cache=True, fastmath=True)
def num_factorial(N):
    res = 1.0
    for i in range(1, N + 1):
        res *= i
    return res


@njit(cache=True, fastmath=True)
def pois_norm_num_conv(mu, sig, N):
    res = 0.0
    Nmu = 256
    dmu = 8.0 * sig / Nmu
    norm_A = (1.0 / (2.0 * np.pi * sig**2)) ** 0.5
    fact = num_factorial(N)
    mu0 = mu - dmu * Nmu / 2
    if mu0 < 0:
        mu0 = 0.0
    for i in range(Nmu):
        mu_ = mu0 + i * dmu
        norm_prob = norm_A * np.exp(-((mu_ - mu) ** 2) / (2 * sig**2))
        pois_prob = ((mu_**N) / fact) * np.exp(-mu_)
        res += norm_prob * pois_prob * dmu
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


def det2dpis(tab, ebins0, ebins1, bl_dmask=None):
    xbins = np.arange(286 + 1) - 0.5
    ybins = np.arange(173 + 1) - 0.5
    ebins = np.append(ebins0, [ebins1[-1]])

    dpis = np.histogramdd(
        [tab["ENERGY"], tab["DETY"], tab["DETX"]], bins=[ebins, ybins, xbins]
    )[0]

    if bl_dmask is None:
        return dpis

    return dpis[:, bl_dmask]


class LLH_webins(object):
    def __init__(
        self,
        event_data,
        ebins0,
        ebins1,
        bl_dmask,
        t0=None,
        t1=None,
        model=None,
        has_err=False,
    ):
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

        self.data_dpis = det2dpis(
            self.data, self.ebins0, self.ebins1, bl_dmask=self.bl_dmask
        )
        self.data_dpis_flat = np.ravel(self.data_dpis)
        self.gamma_vals = get_gammaln(self.data_dpis)

        self.data_size = self.data_dpis.size

    def set_model(self, model):
        self.model = model
        self.nparams = self.model.nparams

    def set_ebin(self, j):
        if "all" in str(j):
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
                mod_rate, mod_rate_err = self.model.get_rate_dpis_err(
                    params, ret_rate_dpis=True
                )
                if not np.all(mod_rate > 0):
                    return -np.inf

                llh = logl_pois_norm_conv(
                    np.ravel(mod_rate * self.dt),
                    np.ravel(mod_rate_err * self.dt),
                    self.data_dpis_flat,
                    self.data_size,
                )
            else:
                mod_rate, mod_rate_err = self.model.get_rate_dpi_err(
                    params, self.ebin, ret_rate_dpis=True
                )
                if np.any(mod_rate <= 0):
                    return -np.inf
                llh = logl_pois_norm_conv(
                    mod_rate * self.dt,
                    mod_rate_err * self.dt,
                    self.data_dpis[self.ebin],
                    len(self.data_dpis[self.ebin]),
                )

        else:
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
                j = None
            else:
                j = self.ebin
            lp = self.model.get_log_prior(params, j=j)
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

    def get_logprob_hess(self, params):
        if self.ebin < 0:
            print("Not supported for multiple ebins yet")
            return 0

        else:
            mod_cnts = self.model.get_rate_dpi(params, self.ebin) * self.dt
            if np.any(np.isclose(mod_cnts, 0)):
                mod_cnts = 1e-6 * np.ones_like(mod_cnts)

            fact = (self.data_dpis[self.ebin]) / np.square(mod_cnts)

            dR_dparams = self.model.get_dr_dp(params, self.ebin)
            Ndim = len(dR_dparams)

            dNLProb_hess = np.zeros((Ndim, Ndim))

            for i in range(Ndim):
                dNLProb_hess[i, i] = np.sum(np.square(dR_dparams[i] * self.dt) * fact)
                for j in range(i + 1, Ndim):
                    dNLProb_hess[i, j] = np.sum(
                        (dR_dparams[i] * self.dt) * (dR_dparams[j] * self.dt) * fact
                    )
                    dNLProb_hess[j, i] += dNLProb_hess[i, j]

            if self.model.has_prior:
                dNLProb_hess += self.model.get_hess_nlogprior(params, self.ebin)

        return dNLProb_hess


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


def min_by_ebin(miner, params_):
    nebins = miner.llh_obj.nebins
    params = copy(params_)
    bf_params = copy(params_)
    NLLH = 0.0

    for e0 in range(nebins):
        miner.set_fixed_params(list(params.keys()), values=list(params.values()))
        e0_pnames = []
        for pname in miner.param_names:
            try:
                if int(pname[-1]) == e0:
                    e0_pnames.append(pname)
            except:
                pass
        miner.set_fixed_params(e0_pnames, fixed=False)
        miner.llh_obj.set_ebin(e0)

        bf_vals, nllh, res = miner.minimize()
        NLLH += nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]

    return NLLH, bf_params


def bkg_withPS_fit(
    PS_tab, model, llh_obj, t0s, t1s, dimxy=2e-3, im_steps=5, test_null=False, Nprocs=1
):
    Nps = len(PS_tab)
    imax = np.linspace(-dimxy, dimxy, im_steps)
    if im_steps == 3:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
    elif im_steps == 2:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
        imax0 = [-dimxy, 0.0, dimxy]
        imax1 = [dimxy, 0.0, -dimxy]
    elif im_steps == 1:
        imax = np.array([0.0])

    imlist = []
    for i in range(Nps):
        if im_steps == 2:
            imlist += [imax0, imax1]
        else:
            imlist += [imax, imax]
    imgs = np.meshgrid(*imlist)
    Npnts = imgs[0].size
    if im_steps == 2:
        ind_grids = np.meshgrid(*(np.arange(3, dtype=np.int64) for i in range(Nps)))
        Npnts = ind_grids[0].size
    logging.info("Npnts: %d" % (Npnts))

    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_miner.set_llh(llh_obj)
    llh_obj.set_time(t0s, t1s)

    bf_params_list = []
    bkg_nllhs = np.zeros(Npnts)

    nebins = model.nebins
    param_list = []

    for i in range(Npnts):
        bf_params = {}
        im_names = []
        params_ = {
            pname: val["val"] for pname, val in bkg_miner.param_info_dict.items()
        }

        for j in range(Nps):
            row = PS_tab[j]
            psname = row["Name"]
            if im_steps == 2:
                params_[psname + "_imx"] = (
                    imlist[2 * j][ind_grids[j].ravel()[i]] + row["imx"]
                )
                params_[psname + "_imy"] = (
                    imlist[2 * j + 1][ind_grids[j].ravel()[i]] + row["imy"]
                )
            else:
                params_[psname + "_imx"] = imgs[2 * j].ravel()[i] + row["imx"]
                params_[psname + "_imy"] = imgs[2 * j + 1].ravel()[i] + row["imy"]
            im_names = [psname + "_imx", psname + "_imy"]
        #         im_vals = [bf_params[nm] for nm in im_names]
        if Nprocs > 1:
            param_list.append(params_)
            continue
        im_vals = [params_[nm] for nm in im_names]

        bkg_nllhs[i], bf_params = min_by_ebin(bkg_miner, params_)

        bf_params_list.append(bf_params)

    if Nprocs > 1:
        res_q = mp.Queue()
        workers = []
        Nper_worker = 1 + int(len(param_list) / (1.0 * Nprocs))
        logging.info("Nper_worker: %d" % (Nper_worker))
        for i in range(Nprocs):
            i0 = i * Nper_worker
            i1 = i0 + Nper_worker
            w = Worker(res_q, deepcopy(bkg_miner), param_list[i0:i1])
            workers.append(w)

        for w in workers:
            w.start()
        res_dicts = []
        Ndone = 0
        while True:
            res = res_q.get()
            if res is None:
                Ndone += 1
                logging.info("%d of %d done" % (Ndone, Nprocs))
                if Ndone >= Nprocs:
                    break
            else:
                res_dicts.append(res)
        for w in workers:
            w.join()

        df = pd.DataFrame(res_dicts)
        min_ind = np.argmin(df["nllh"])
        bf_df_row = df.iloc[min_ind]
        bf_nllh = bf_df_row["nllh"]
        bf_params = {
            name: bf_df_row[name] for name in bf_df_row.index if not "nllh" in name
        }

    else:
        bf_ind = np.argmin(bkg_nllhs)
        bf_params = bf_params_list[bf_ind]
        bf_nllh = bkg_nllhs[bf_ind]

    if test_null:
        TS_nulls = {}
        for i in range(Nps):
            params_ = copy(bf_params)
            row = PS_tab[i]
            psname = row["Name"]
            for j in range(nebins):
                params_[psname + "_rate_" + str(j)] = 1e-8
            llh_obj.set_ebin(-1)
            nllh_null = -llh_obj.get_logprob(params_)
            TS_nulls[psname] = np.sqrt(2.0 * (nllh_null - bf_nllh))
            logging.info("nllh_null: %.3f", (nllh_null))
            logging.info("bf_nllh: %.3f" % (bf_nllh))

            if np.isnan(TS_nulls[psname]):
                TS_nulls[psname] = 0.0
        return bf_nllh, bf_params, TS_nulls

    return bf_nllh, bf_params


def do_init_bkg_wPSs(
    bkg_mod,
    llh_obj,
    src_tab,
    rt_obj,
    GTI,
    sig_twind,
    TSmin=7.0,
    Nprocs=1,
    tmin=None,
    tmax=None,
):
    if not tmin is None:
        bti = (-np.inf, tmin)
        GTI = add_bti2gti(bti, GTI)
    if not tmax is None:
        bti = (tmax, np.inf)
        GTI = add_bti2gti(bti, GTI)
    gti_bkg = add_bti2gti(sig_twind, GTI)
    bkg_t0s = gti_bkg["START"]
    bkg_t1s = gti_bkg["STOP"]
    exp = 0.0
    for i in range(len(bkg_t0s)):
        exp += bkg_t1s[i] - bkg_t0s[i]
    logging.info("exp: %.3f" % (exp))

    logging.info("bkg_t0s: ")
    logging.info(bkg_t0s)
    logging.info("bkg_t1s: ")
    logging.info(bkg_t1s)

    Nsrcs = len(src_tab)
    nebins = bkg_mod.nebins

    for ii in range(Nsrcs):
        mod_list = [bkg_mod]
        im_steps = 1  # 5
        TSmin_ = TSmin
        # if Nsrcs >= 3:
        #     im_steps = 3
        #     TSmin_ = TSmin - 1.0
        # if Nsrcs >= 5:
        #     im_steps = 2
        #     TSmin_ = TSmin - 1.5
        # if Nsrcs >= 9:
        #     im_steps = 1
        #     TSmin_ = TSmin - 2.5

        ps_mods = []
        for i in range(Nsrcs):
            row = src_tab[i]
            mod = Point_Source_Model_Binned_Rates(
                row["imx"],
                row["imy"],
                0.1,
                [llh_obj.ebins0, llh_obj.ebins1],
                rt_obj,
                llh_obj.bl_dmask,
                use_deriv=True,
                name=row["Name"],
            )
            ps_mods.append(mod)

        mod_list += ps_mods
        comp_mod = CompoundModel(mod_list)
        print(comp_mod.name)

        llh_obj.set_model(comp_mod)

        bf_nllh, bf_params, TS_nulls = bkg_withPS_fit(
            src_tab,
            comp_mod,
            llh_obj,
            bkg_t0s,
            bkg_t1s,
            test_null=True,
            im_steps=im_steps,
            Nprocs=Nprocs,
        )

        logging.info("TS_nulls: ")
        logging.info(TS_nulls)

        bkg_rates = np.array(
            [bf_params["Background" + "_bkg_rate_" + str(j)] for j in range(nebins)]
        )
        min_rate = 1e-1 * bkg_rates
        logging.debug("min_rate: ")
        logging.debug(min_rate)
        PSs2keep = []
        for name, TS in TS_nulls.items():
            ps_rates = np.array(
                [bf_params[name + "_rate_" + str(j)] for j in range(nebins)]
            )
            logging.info(name + " rates: ")
            logging.info(ps_rates)
            if TS < TSmin_:
                # ps_rates = np.array([bf_params[name+'_rate_'+str(j)] for j in range(nebins)])
                # print ps_rates
                if np.all(ps_rates < min_rate):
                    continue
            if np.all(ps_rates < (min_rate / 20.0)):
                continue
            PSs2keep.append(name)

        if len(PSs2keep) == len(src_tab):
            break
        if len(PSs2keep) == 0:
            Nsrcs = 0
            src_tab = src_tab[np.zeros(len(src_tab), dtype=bool)]
            break
        bl = np.array([src_tab["Name"][i] in PSs2keep for i in range(Nsrcs)])
        src_tab = src_tab[bl]
        Nsrcs = len(src_tab)
        logging.debug("src_tab: ")
        logging.debug(src_tab)

    return bf_params, src_tab  # , errs_dict, corrs_dict


class Source_Model_OutFoV(Model):
    def __init__(
        self,
        flux_model,
        ebins,
        bl_dmask,
        name="Signal",
        use_deriv=False,
        use_prior=False,
    ):
        self.fmodel = flux_model

        self.ebins = ebins
        self.ebins0 = ebins[0]
        self.ebins1 = ebins[1]
        nebins = len(self.ebins0)

        self.flor_resp_dname = "/storage/work/jjd330/local/bat_data/resp_tabs/"

        param_names = ["theta", "phi"]
        param_names += self.fmodel.param_names

        param_dict = {}

        for pname in param_names:
            pdict = {}
            if pname == "theta":
                pdict["bounds"] = (0.0, 180.0)
                pdict["val"] = 90.0
                pdict["nuis"] = False
            elif pname == "phi":
                pdict["bounds"] = (0.0, 360.0)
                pdict["val"] = 180.0
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

        super(Source_Model_OutFoV, self).__init__(
            name, bl_dmask, param_names, param_dict, nebins, has_prior=use_prior
        )

        if use_deriv:
            self.has_deriv = True

        self.get_batxys()
        self.flor_err = 0.2
        self.non_flor_err = 0.05

        self.ones = np.ones(self.ndets)

    def get_batxys(self):
        yinds, xinds = np.where(self.bl_dmask)
        self.batxs, self.batys = detxy2batxy(xinds, yinds)

    def set_theta_phi(self, theta, phi):
        self.resp_obj = ResponseOutFoV(
            self.flor_resp_dname, self.ebins0, self.ebins1, self.bl_dmask
        )
        self._theta = theta
        self._phi = phi

        self.resp_obj.set_theta_phi(theta, phi)

    def set_flux_params(self, flux_params):
        self.flux_params = flux_params
        resp_ebins = np.append(
            self.resp_obj.PhotonEmins, [self.resp_obj.PhotonEmaxs[-1]]
        )
        self.flux_params["A"] = 1.0
        self.normed_photon_fluxes = self.fmodel.get_photon_fluxes(
            resp_ebins, self.flux_params
        )

        self.normed_rate_dpis = np.swapaxes(
            self.resp_obj.get_rate_dpis_from_photon_fluxes(self.normed_photon_fluxes),
            0,
            1,
        )
        self.normed_err_rate_dpis = np.swapaxes(
            np.sqrt(
                (
                    self.flor_err
                    * self.resp_obj.get_flor_rate_dpis_from_photon_fluxes(
                        self.normed_photon_fluxes
                    )
                )
                ** 2
                + (
                    self.non_flor_err
                    * self.resp_obj.get_non_flor_rate_dpis_from_photon_fluxes(
                        self.normed_photon_fluxes
                    )
                )
                ** 2
            ),
            0,
            1,
        )

    def get_rate_dpis(self, params):
        theta = params["theta"]
        phi = params["phi"]
        A = params["A"]

        return A * self.normed_rate_dpis

    def get_rate_dpis_err(self, params, ret_rate_dpis=False):
        err_rate_dpis = params["A"] * self.normed_err_rate_dpis
        if ret_rate_dpis:
            rate_dpis = self.get_rate_dpis(params)
            return rate_dpis, err_rate_dpis
        return err_rate_dpis

    def get_rate_dpi(self, params, j):
        return A * self.normed_rate_dpis[:, j]

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


def theta_phi2imxy(theta, phi):
    imr = np.tan(np.radians(theta))
    imx = imr * np.cos(np.radians(phi))
    imy = imr * np.sin(np.radians(-phi))
    return imx, imy


def imxy2theta_phi(imx, imy):
    theta = np.rad2deg(np.arctan(np.sqrt(imx**2 + imy**2)))
    phi = np.rad2deg(np.arctan2(-imy, imx))
    if np.isscalar(phi):
        if phi < 0:
            phi += 360.0
    else:
        bl = phi < 0
        if np.sum(bl) > 0:
            phi[bl] += 360.0
    return theta, phi


mask_off_vec = np.array([-0.145, 0.114, 0.0])
mask_off_vec = np.zeros(3)


def get_fixture_struct():
    fixture_half_dims01 = [11.656, 6.944, 0.477]
    fixture_pos0 = (0.0, 53.578, 35.799 + 98.489 - 32.612)
    fixture_pos1 = (0.0, -53.578, 35.799 + 98.489 - 32.612)

    fixture_half_dims23 = [5.95, 5.95, 0.477]
    fixture_pos2 = (114.974, 53.822, 35.799 + 98.489 - 32.612)
    fixture_pos3 = (-114.974, 53.822, 35.799 + 98.489 - 32.612)

    fixture_half_dims45 = [6.198, 6.198, 0.477]
    fixture_pos4 = (-59.448, -53.518, 35.799 + 98.489 - 32.612)
    fixture_pos5 = (59.448, -53.518, 35.799 + 98.489 - 32.612)

    fixture_half_dims67 = [6.942, 6.2, 0.477]
    fixture_pos6 = (113.85, 1.984, 35.799 + 98.489 - 32.612)
    fixture_pos7 = (-113.85, 1.984, 35.799 + 98.489 - 32.612)

    fixture_box0 = Box_Polygon(
        fixture_half_dims01[0],
        fixture_half_dims01[1],
        fixture_half_dims01[2],
        np.array(fixture_pos0) + mask_off_vec,
    )
    fixture_box1 = Box_Polygon(
        fixture_half_dims01[0],
        fixture_half_dims01[1],
        fixture_half_dims01[2],
        np.array(fixture_pos1) + mask_off_vec,
    )
    fixture_box2 = Box_Polygon(
        fixture_half_dims23[0],
        fixture_half_dims23[1],
        fixture_half_dims23[2],
        np.array(fixture_pos2) + mask_off_vec,
    )
    fixture_box3 = Box_Polygon(
        fixture_half_dims23[0],
        fixture_half_dims23[1],
        fixture_half_dims23[2],
        np.array(fixture_pos3) + mask_off_vec,
    )
    fixture_box4 = Box_Polygon(
        fixture_half_dims45[0],
        fixture_half_dims45[1],
        fixture_half_dims45[2],
        np.array(fixture_pos4) + mask_off_vec,
    )
    fixture_box5 = Box_Polygon(
        fixture_half_dims45[0],
        fixture_half_dims45[1],
        fixture_half_dims45[2],
        np.array(fixture_pos5) + mask_off_vec,
    )
    fixture_box6 = Box_Polygon(
        fixture_half_dims67[0],
        fixture_half_dims67[1],
        fixture_half_dims67[2],
        np.array(fixture_pos6) + mask_off_vec,
    )
    fixture_box7 = Box_Polygon(
        fixture_half_dims67[0],
        fixture_half_dims67[1],
        fixture_half_dims67[2],
        np.array(fixture_pos7) + mask_off_vec,
    )
    Fixture0 = Swift_Structure(fixture_box0, TI, Name="Fix0")
    Fixture1 = Swift_Structure(fixture_box1, TI, Name="Fix1")
    Fixture2 = Swift_Structure(fixture_box2, TI, Name="Fix2")
    Fixture3 = Swift_Structure(fixture_box3, TI, Name="Fix3")
    Fixture4 = Swift_Structure(fixture_box4, TI, Name="Fix4")
    Fixture5 = Swift_Structure(fixture_box5, TI, Name="Fix5")
    Fixture6 = Swift_Structure(fixture_box6, TI, Name="Fix6")
    Fixture7 = Swift_Structure(fixture_box7, TI, Name="Fix7")
    Fixtures = [
        Fixture0,
        Fixture1,
        Fixture2,
        Fixture3,
        Fixture4,
        Fixture5,
        Fixture6,
        Fixture7,
    ]

    Fixture_Struct = Swift_Structure_Manager()
    for fix in Fixtures:
        Fixture_Struct.add_struct(fix)
    return Fixture_Struct


mask_off_vec = np.array([-0.145, 0.114, 0.0])


class Swift_Mask_Interactions(object):
    """
    Should say whether photon goes through the mask poly or not
    Also want it to get ray trace
    Should have also contain the lead tiles where the struts screw in (which aren't included in the ray traces)

    Should be able to give trans to each det,
    assume each photon that goes through lead tile, goes through 0.1cm/cos(theta)
    trans = (shadow_frac)*exp[-rhomu_pb * 0.1 / cos(theta)]
    """

    def __init__(self, rt_obj, bl_dmask):
        self.rt_obj = rt_obj
        self.bl_dmask = bl_dmask

        self.ds_base = 0.1

        self.material = PB
        self.Nmaterials = 1
        self.Name = "Mask"

        self.mask_materials = [Korex, CarbonFibre, Mylar]
        self.mask_ds_base = np.array([1.6 * 2, 0.032 * 4, 0.0413 * 2])

        self.norm_vec = np.array([0.0, 0.0, -1.0])

        self.verts = np.array(
            [
                (121.92, 60.95, 103.187),
                (121.92, -1.41, 103.187),
                (61.5, -60.95, 103.187),
                (-61.5, -60.95, 103.187),
                (-121.92, -1.41, 103.187),
                (-121.92, 60.95, 103.187),
            ]
        )
        shrink_size = 0.1
        self.verts = np.array(
            [
                (121.92 - shrink_size, 60.96 - shrink_size, 103.187),
                (121.92 - shrink_size, -0.634, 103.187),
                (61.594 - shrink_size, -60.96 + shrink_size, 103.187),
                (-61.594 + shrink_size, -60.96 + shrink_size, 103.187),
                (-121.92 + shrink_size, -0.634, 103.187),
                (-121.92 + shrink_size, 60.96 - shrink_size, 103.187),
            ]
        )
        shrink_size = 0.0
        self.verts = np.array(
            [
                (121.92 - shrink_size, 60.96 - shrink_size, 103.187),
                (121.92 - shrink_size, -0.034 + 0.2, 103.187),
                (60.794 - shrink_size, -60.96 + shrink_size, 103.187),
                (-60.794 + shrink_size, -60.96 + shrink_size, 103.187),
                (-121.92 + shrink_size, -0.034 + 0.2, 103.187),
                (-121.92 + shrink_size, 60.96 - shrink_size, 103.187),
            ]
        )
        trans_vec = np.zeros(3)
        #         trans_vec = mask_off_vec

        self.mask_poly = Polygon2D(self.verts, trans_vec)

        self.fix_struct = get_fixture_struct()

        self._rt_imx = 10.0
        self._rt_imy = 10.0
        self._rt_im_update = 1e-6

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus = self.material.get_tot_rhomu(self.energy)
        self.comp_rho_mus = self.material.get_comp_rhomu(self.energy)
        self.photoe_rho_mus = self.material.get_photoe_rhomu(self.energy)

        self.mask_tot_rho_mus_list = [
            mat.get_tot_rhomu(self.energy) for mat in self.mask_materials
        ]

        if hasattr(self, "dists"):
            self.calc_tot_rhomu_dist()

        self.fix_struct.set_energy_arr(self.energy)

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.imx, self.imy = theta_phi2imxy(theta, phi)
        self.d = self.ds_base / np.cos(np.radians(theta))
        self.mask_d = self.mask_ds_base / np.cos(np.radians(theta))

        self.calc_does_int_mask()
        self.calc_dists()
        self.calc_tot_rhomu_dist()
        self.does_int_fix()

    def calc_does_int_mask(self):
        self.does_int_mask = self.mask_poly.does_intersect(
            np.radians(self.theta),
            np.radians(self.phi),
            self.batxs,
            self.batys,
            self.batzs,
        )

    def calc_dists(self):
        self.dists = (self.does_int_mask.astype(np.float64)) * self.d

    def calc_tot_rhomu_dist(self):
        #         self.tot_rhomu_dists = np.zeros((self.ndets,self.Ne))
        self.tot_rhomu_dists = self.dists[:, np.newaxis] * self.tot_rho_mus
        self.mask_tot_rhomu_dists = np.zeros(self.Ne)
        for i in range(len(self.mask_d)):
            self.mask_tot_rhomu_dists += self.mask_d[i] * self.mask_tot_rho_mus_list[i]
        self.mask_trans = np.exp(-self.mask_tot_rhomu_dists)

    def does_int_fix(self):
        #         does_int_fix = np.zeros(self.ndets, dtype=bool)
        self.fix_trans = np.ones((self.ndets, self.Ne))
        self.fix_struct.set_batxyzs(
            self.batxs[self.does_int_mask],
            self.batys[self.does_int_mask],
            self.batzs[self.does_int_mask],
        )
        self.fix_struct.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        self.fix_trans[self.does_int_mask] = self.fix_struct.get_trans()

    def get_trans(self):
        self.trans = np.ones((self.ndets, self.Ne))

        rt = self.get_rt(self.imx, self.imy)[:, np.newaxis]

        self.trans[self.does_int_mask] = (
            self.mask_trans
            * self.fix_trans[self.does_int_mask]
            * (
                rt[self.does_int_mask]
                + (1.0 - rt[self.does_int_mask])
                * np.exp(-self.tot_rhomu_dists[self.does_int_mask])
            )
        )
        return self.trans

    def get_rt(self, imx, imy):
        if np.hypot(imx - self._rt_imx, imy - self._rt_imy) < self._rt_im_update:
            return self._rt
        else:
            rt = self.rt_obj.get_intp_rt(imx, imy, get_deriv=False)
            self._rt = np.copy(rt[self.bl_dmask])
            self.max_rt = np.max(self._rt)
            print(("max rt: %.4f" % (self.max_rt)))
            self._rt /= self.max_rt
            self._shadow = 1.0 - self._rt
            #             self._shadow = (self.max_rt - self._rt)
            #             fp = self.get_fp(imx, imy)
            #             self._shadow[self.uncoded] = 0.0
            # self._drt_dx = drt_dx[self.bl_dmask]
            # self._drt_dy = drt_dy[self.bl_dmask]
            self._rt_imx = imx
            self._rt_imy = imy

        return self._rt


@njit(cache=True)
def get_rate_dpis_from_photon_fluxes(resp_dpi, photon_fluxes):
    ndets, NphotonEs, Nphabins = resp_dpi.shape
    rate_dpis = np.zeros((ndets, Nphabins))

    for i in range(ndets):
        for j in range(Nphabins):
            for k in range(NphotonEs):
                rate_dpis[i, j] += photon_fluxes[k] * resp_dpi[i, k, j]
    return rate_dpis


@njit(cache=True)
def multiply_resp_trans_dpis(resp_dpis, trans_dpis):
    ndets, NphotonEs, Nphabins = resp_dpis.shape
    dpis = np.zeros((ndets, NphotonEs, Nphabins))

    for i in range(ndets):
        for j in range(Nphabins):
            for k in range(NphotonEs):
                dpis[i, k, j] += resp_dpis[i, k, j] * trans_dpis[i, k]

    return dpis


class ResponseInFoV(object):
    def __init__(self, resp_dname, pha_emins, pha_emaxs, bl_dmask, rt_obj):
        self.resp_dname = resp_dname
        self.resp_arr = get_resp_arr(self.resp_dname)
        self.thetas = np.unique(self.resp_arr["theta"])

        tab = Table.read(os.path.join(self.resp_dname, self.resp_arr["fname"][0]))
        pha_tab = Table.read(
            os.path.join(self.resp_dname, self.resp_arr["fname"][0]), hdu=2
        )
        self.PhotonEmins = tab["ENERG_LO"]
        self.PhotonEmaxs = tab["ENERG_HI"]
        self.PhotonEs = ((self.PhotonEmins + self.PhotonEmaxs) / 2.0).astype(np.float64)
        self.NphotonEs = len(self.PhotonEs)

        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(pha_emins)
        #         self.NphotonEs = NphotonEs
        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask
        self.batxs, self.batys = bldmask2batxys(self.bl_dmask)
        self.batzs = 3.187 + np.zeros(self.ndets)

        #         self.resp_dpi_shape = (173, 286, self.NphotonEs, self.Nphabins)
        self.resp_dpi_shape = (self.ndets, self.NphotonEs, self.Nphabins)

        self.resp_files = {}

        self.full_struct = get_full_struct_manager(Es=self.PhotonEs)
        self.full_struct.set_batxyzs(self.batxs, self.batys, self.batzs)

        dual_struct = get_dual_struct_obj(self.PhotonEs)
        self.comp_obj = Comp_Resp_Obj(self.batxs, self.batys, self.batzs, dual_struct)

        self.flor_resp_obj = FlorResponseDPI(
            "/gpfs/scratch/jjd330/bat_data/flor_resps/",
            pha_tab,
            self.pha_emins,
            self.pha_emaxs,
            self.bl_dmask,
            NphotonEs=self.NphotonEs,
        )

        self.mask_obj = Swift_Mask_Interactions(rt_obj, self.bl_dmask)
        self.mask_obj.set_energy_arr(self.PhotonEs)
        self.mask_obj.set_batxyzs(self.batxs, self.batys, self.batzs)

    def set_theta_phi(self, theta, phi):
        # use radians or degs ?

        self.theta = theta
        self.phi = phi

        self.thetas2use, self.phis2use, self.wts = self.get_intp_theta_phi_wts(
            self.theta, self.phi
        )
        self.inds4intp = []
        for i in range(len(self.wts)):
            ind = np.where(
                np.isclose(self.thetas2use[i], self.resp_arr["theta"])
                & np.isclose(self.phis2use[i], self.resp_arr["phi"])
            )[0][0]
            self.inds4intp.append(ind)

        self.mask_obj.set_theta_phi(theta, phi)
        self.mask_trans = self.mask_obj.get_trans()

        self.full_struct.set_theta_phi(np.radians(theta), np.radians(phi))
        self._lines_trans_dpis = self.full_struct.get_trans()
        self.lines_trans_dpis = self._lines_trans_dpis * self.mask_trans

        #         self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        #         self.comp_trans_dpis = self.comp_obj.get_trans()

        if theta > 90.0:
            self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.flor_resp_obj.set_theta_phi(self.theta, self.phi)

        self.calc_resp_dpis()
        self.calc_tot_resp_dpis()

    def update_trans(self, theta, phi):
        self.mask_obj.set_theta_phi(theta, phi)
        self.mask_trans = self.mask_obj.get_trans()

        self.full_struct.set_theta_phi(np.radians(theta), np.radians(phi))
        self._lines_trans_dpis = self.full_struct.get_trans()
        self.lines_trans_dpis = self._lines_trans_dpis * self.mask_trans

        #         self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        #         self.comp_trans_dpis = self.comp_obj.get_trans()

        if theta > 90.0:
            self.comp_obj.set_theta_phi(np.radians(theta), np.radians(phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.calc_tot_resp_dpis()

    def open_resp_file_obj(self, fname):
        resp_file_obj = ResponseDPI(
            os.path.join(self.resp_dname, fname),
            self.pha_emins,
            self.pha_emaxs,
            np.radians(self.phi),
            self.bl_dmask,
        )
        self.resp_files[fname] = resp_file_obj

    def calc_resp_dpis(self):
        self.lines_resp_dpis = np.zeros(self.resp_dpi_shape)
        self.comp_resp_dpis = np.zeros(self.resp_dpi_shape)

        for i in range(len(self.wts)):
            k = self.resp_arr["fname"][self.inds4intp[i]]
            if not k in list(self.resp_files.keys()):
                self.open_resp_file_obj(k)
            self.lines_resp_dpis += (
                self.wts[i] * self.resp_files[k].get_lines_resp_dpis()
            )
            self.comp_resp_dpis += self.wts[i] * self.resp_files[k].get_comp_resp_dpis()

    def calc_tot_resp_dpis(self):
        lines_dpi = self.lines_resp_dpis * (self.lines_trans_dpis[:, :, np.newaxis])
        comp_dpi = self.comp_resp_dpis * (self.comp_trans_dpis[:, :, np.newaxis])
        self.comp_resp_dpi = comp_dpi
        self.lines_resp_dpi = lines_dpi
        self.non_flor_resp_dpi = lines_dpi + comp_dpi
        self.flor_resp_dpi = self.flor_resp_obj.get_resp_dpi()
        self.tot_resp_dpis = self.non_flor_resp_dpi + self.flor_resp_dpi

    def get_lines_resp_dpis(self):
        return self.lines_resp_dpis

    def get_comp_resp_dpis(self):
        return self.comp_resp_dpis

    def get_flor_resp_dpis(self):
        return self.flor_resp_obj.get_resp_dpi()

    def get_tot_resp_dpis(self):
        return self.tot_resp_dpis

    def get_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(self.tot_resp_dpi, photon_fluxes)
        #         rate_dpis = np.zeros((self.ndets,self.Nphabins))
        #         for j in range(self.Nphabins):
        #             rate_dpis[:,j] += np.sum(photon_fluxes*self.tot_resp_dpis[:,:,j],axis=1)
        return rate_dpis

    def get_flor_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(self.flor_resp_dpi, photon_fluxes)
        #         rate_dpis = np.zeros((self.ndets,self.Nphabins))
        #         for j in range(self.Nphabins):
        #             rate_dpis[:,j] += np.sum(photon_fluxes*self.flor_resp_dpi[:,:,j],axis=1)
        return rate_dpis

    def get_comp_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(self.comp_resp_dpi, photon_fluxes)
        #         rate_dpis = np.zeros((self.ndets,self.Nphabins))
        #         for j in range(self.Nphabins):
        #             rate_dpis[:,j] += np.sum(photon_fluxes*self.comp_resp_dpi[:,:,j],axis=1)
        return rate_dpis

    def get_photoe_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(self.lines_resp_dpi, photon_fluxes)
        #         rate_dpis = np.zeros((self.ndets,self.Nphabins))
        #         for j in range(self.Nphabins):
        #             rate_dpis[:,j] += np.sum(photon_fluxes*self.lines_resp_dpi[:,:,j],axis=1)
        return rate_dpis

    def get_non_flor_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(
            self.non_flor_resp_dpi, photon_fluxes
        )
        #         rate_dpis = np.zeros((self.ndets,self.Nphabins))
        #         for j in range(self.Nphabins):
        #             rate_dpis[:,j] += np.sum(photon_fluxes*self.non_flor_resp_dpi[:,:,j],axis=1)
        return rate_dpis

    def get_intp_theta_phi_wts(self, theta, phi, eps=0.1):
        thetas = np.sort(np.unique(self.resp_arr["theta"]))
        phis = np.sort(np.unique(self.resp_arr["phi"]))

        th0 = np.digitize(theta, thetas) - 1
        if theta == 180.0:
            th0 -= 1
        theta0 = thetas[th0]
        theta1 = thetas[th0 + 1]
        print(theta0, theta1)
        if np.abs(theta0 - theta) < eps:
            ths = [theta0]
            th_wts = [1.0]
        elif np.abs(theta1 - theta) < eps:
            ths = [theta1]
            th_wts = [1.0]
        else:
            ths = [theta0, theta1]
            dth = theta1 - theta0
            th_wts = [(theta1 - theta) / dth, (theta - theta0) / dth]

        phi_ = phi - (int(phi) / 45) * 45.0
        print(phi_)
        if (int(phi) / 45) % 2 == 1:
            phi_ = 45.0 - phi_
        print(phi_)
        ph0 = np.digitize(phi_, phis) - 1
        if phi_ == 45.0:
            ph0 -= 1

        phi0 = phis[ph0]
        phi1 = phis[ph0 + 1]

        if np.abs(phi0 - phi_) < eps:
            phs = [phi0]
            ph_wts = [1.0]
        elif np.abs(phi1 - phi_) < eps:
            phs = [phi1]
            ph_wts = [1.0]
        else:
            phs = [phi0, phi1]
            dph = phi1 - phi0
            ph_wts = [(phi1 - phi_) / dph, (phi_ - phi0) / dph]

        ths_ = []
        phs_ = []
        wts = []
        for i in range(len(ths)):
            if ths[i] == 0.0 or ths[i] == 180.0:
                ths_.append(ths[i])
                phs_.append(0.0)
                wts.append(th_wts[i])
                continue
            for j in range(len(phs)):
                ths_.append(ths[i])
                phs_.append(phs[j])
                wts.append(th_wts[i] * ph_wts[j])

        return ths_, phs_, wts


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

        self.flor_resp_dname = "/storage/work/jjd330/local/bat_data/resp_tabs/"

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
        self._resp_phi = 0.0
        self._resp_theta = 180.0

        self._trans_update = 5e-3
        self._trans_phi = 0.0
        self._trans_theta = 180.0

        self.ones = np.ones(self.ndets)

    #     def get_fp(self, imx, imy):

    #         if np.hypot(imx-self._fp_imx, imy-self._fp_imy) <\
    #                 self._fp_im_update:
    #                 return self._fp
    #         else:
    #             fp = self.fp_obj.get_fp(imx, imy)
    #             self._fp = fp[self.bl_dmask].astype(np.int64)
    #             self._fp[(self._rt>1e-2)] = 1
    #             self._unfp = 1 - self._fp
    #             self.uncoded = (self._fp<.1)
    #             self.coded = ~self.uncoded
    #             # self._drt_dx = drt_dx[self.bl_dmask]
    #             # self._drt_dy = drt_dy[self.bl_dmask]
    #             self._fp_imx = imx
    #             self._fp_imy = imy

    #         return self._fp

    #     def get_rt(self, imx, imy):

    #         if np.hypot(imx-self._rt_imx, imy-self._rt_imy) <\
    #                 self._rt_im_update:
    #                 return self._rt
    #         else:
    #             rt = self.rt_obj.get_intp_rt(imx, imy, get_deriv=False)
    #             self._rt = rt[self.bl_dmask]
    #             self.max_rt = np.max(self._rt)
    #             print("max rt: %.4f"%(self.max_rt))
    #             self._rt /= self.max_rt
    #             self._shadow = (1. - self._rt)
    # #             self._shadow = (self.max_rt - self._rt)
    #             fp = self.get_fp(imx, imy)
    #             self._shadow[self.uncoded] = 0.0
    #             # self._drt_dx = drt_dx[self.bl_dmask]
    #             # self._drt_dy = drt_dy[self.bl_dmask]
    #             self._rt_imx = imx
    #             self._rt_imy = imy

    #         return self._rt

    def get_batxys(self):
        yinds, xinds = np.where(self.bl_dmask)
        self.batxs, self.batys = detxy2batxy(xinds, yinds)

    def set_theta_phi(self, theta, phi):
        if (
            ang_sep(phi, 90.0 - theta, self._resp_phi, 90.0 - self._resp_theta)
            > self._resp_update
        ):
            logging.info("Making new response object")
            self.resp_obj = ResponseInFoV(
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


def min_at_Epeaks_gammas(sig_miner, sig_mod, Epeaks, gammas):
    nllhs = []
    As = []
    flux_params = {"A": 1.0, "Epeak": 150.0, "gamma": -0.25}

    Npnts = len(gammas)

    for i in range(Npnts):
        flux_params["gamma"] = gammas[i]
        flux_params["Epeak"] = Epeaks[i]
        sig_mod.set_flux_params(flux_params)
        pars, nllh, res = sig_miner.minimize()
        nllhs.append(nllh[0])
        As.append(pars[0][0])
    return nllhs, As


def analysis_for_imxy_square(
    imx0,
    imx1,
    imy0,
    imy1,
    bkg_bf_params,
    bkg_mod,
    flux_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
):
    bl_dmask = bkg_mod.bl_dmask

    # dimxy = 0.0025
    dimxy = np.round(imx1 - imx0, decimals=4)
    imstep = 0.003
    imxstep = 0.004

    # imx_ax = np.arange(imx0, imx1+dimxy/2., dimxy)
    # imy_ax = np.arange(imy0, imy1+dimxy/2., dimxy)
    # imxg,imyg = np.meshgrid(imx_ax, imy_ax)

    # imx_ax = np.arange(imx0, imx1, imxstep)
    # imy_ax = np.arange(imy0, imy1, imstep)
    imx_ax = np.arange(0, dimxy, imxstep)
    imy_ax = np.arange(0, dimxy, imstep)
    imxg, imyg = np.meshgrid(imx_ax, imy_ax)
    bl = np.isclose((imyg * 1e4).astype(np.int64) % int(imstep * 2 * 1e4), 0)
    imxg[bl] += imxstep / 2.0
    imxs = np.ravel(imxg) + imx0
    imys = np.ravel(imyg) + imy0
    Npnts = len(imxs)

    print(Npnts)
    logging.info("%d imxy points to do" % (Npnts))

    thetas, phis = imxy2theta_phi(imxs, imys)

    gamma_ax = np.linspace(-0.4, 1.6, 8 + 1)
    gamma_ax = np.linspace(-0.4, 1.6, 4 + 1)[1:-1]
    # gamma_ax = np.array([0.4, 0.9])
    #     gamma_ax = np.linspace(-0.4, 1.6, 3+1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 10 + 1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 5 + 1)[2:-1]
    #     Epeak_ax = np.logspace(np.log10(45.0), 3, 5+1)[3:]
    logging.info("Epeak_ax: ")
    logging.info(Epeak_ax)
    logging.info("gammas_ax: ")
    logging.info(gamma_ax)
    #     Epeak_ax = np.logspace(np.log10(25.0), 3, 3+1)
    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    Nspec_pnts = len(Epeaks)
    ntbins = len(tbins0)

    rt_obj = RayTraces(rt_dir)
    # fp_obj = FootPrints(fp_dir)

    sig_mod = Source_Model_InFoV(
        flux_mod, [ebins0, ebins1], bl_dmask, rt_obj, use_deriv=True
    )
    sig_mod.set_theta_phi(np.mean(thetas), np.mean(phis))

    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_miner = NLLH_ScipyMinimize_Wjacob("")

    tmin = np.min(tbins0)
    tmax = np.max(tbins1)
    tbl = (ev_data["TIME"] >= (tmin - 1.0)) & (ev_data["TIME"] < (tmax + 1.0))
    sig_llh_obj = LLH_webins(ev_data[tbl], ebins0, ebins1, bl_dmask, has_err=True)

    sig_llh_obj.set_model(comp_mod)

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}

    bkg_name = bkg_mod.name

    pars_ = {}
    pars_["Signal_theta"] = np.mean(thetas)
    pars_["Signal_phi"] = np.mean(phis)
    for pname, val in bkg_bf_params.items():
        # pars_['Background_'+pname] = val
        pars_[bkg_name + "_" + pname] = val
    for pname, val in flux_params.items():
        pars_["Signal_" + pname] = val

    sig_miner.set_llh(sig_llh_obj)

    fixed_pnames = list(pars_.keys())
    fixed_vals = list(pars_.values())
    trans = [None for i in range(len(fixed_pnames))]
    sig_miner.set_trans(fixed_pnames, trans)
    sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    sig_miner.set_fixed_params(["Signal_A"], fixed=False)

    res_dfs_ = []

    for ii in range(Npnts):
        print(imxs[ii], imys[ii])
        print(thetas[ii], phis[ii])
        sig_miner.set_fixed_params(
            ["Signal_theta", "Signal_phi"], values=[thetas[ii], phis[ii]]
        )

        res_dfs = []

        for j in range(Nspec_pnts):
            flux_params["gamma"] = gammas[j]
            flux_params["Epeak"] = Epeaks[j]
            sig_mod.set_flux_params(flux_params)

            res_dict = {}

            res_dict["Epeak"] = Epeaks[j]
            res_dict["gamma"] = gammas[j]

            nllhs = np.zeros(ntbins)
            As = np.zeros(ntbins)

            for i in range(ntbins):
                t0 = tbins0[i]
                t1 = tbins1[i]
                dt = t1 - t0
                sig_llh_obj.set_time(tbins0[i], tbins1[i])

                pars, nllh, res = sig_miner.minimize()
                # print "res: "
                # print res
                As[i] = pars[0][0]
                nllhs[i] = nllh[0]

            res_dict["nllh"] = nllhs
            res_dict["A"] = As
            res_dict["time"] = np.array(tbins0)
            res_dict["dur"] = np.array(tbins1) - np.array(tbins0)

            res_dict["theta"] = thetas[ii]
            res_dict["phi"] = phis[ii]
            res_dict["imx"] = imxs[ii]
            res_dict["imy"] = imys[ii]

            res_dfs.append(pd.DataFrame(res_dict))

            # logging.info("Done with spec %d of %d" %(j+1,Nspec_pnts))

        res_df = pd.concat(res_dfs, ignore_index=True)
        bkg_nllhs = np.zeros(len(res_df))

        for i in range(ntbins):
            t0 = tbins0[i]
            t1 = tbins1[i]
            dt = t1 - t0
            sig_llh_obj.set_time(tbins0[i], tbins1[i])
            pars_["Signal_theta"] = thetas[ii]
            pars_["Signal_phi"] = phis[ii]
            pars_["Signal_A"] = 1e-10
            bkg_nllh = -sig_llh_obj.get_logprob(pars_)
            bl = np.isclose(res_df["time"] - t0, t0 - t0) & np.isclose(
                res_df["dur"], dt
            )
            bkg_nllhs[bl] = bkg_nllh

        # pars_['Signal_A'] = 1e-10
        # bkg_nllh = -sig_llh_obj.get_logprob(pars_)

        res_df["bkg_nllh"] = bkg_nllhs
        res_df["TS"] = np.sqrt(2.0 * (bkg_nllhs - res_df["nllh"]))

        res_dfs_.append(res_df)

        logging.info("Done with imxy %d of %d" % (ii + 1, Npnts))

    return pd.concat(res_dfs_, ignore_index=True)


def analysis_at_theta_phi(
    theta,
    phi,
    bkg_bf_params,
    bkg_mod,
    flux_mod,
    ev_data,
    ebins0,
    ebins1,
    tbins0,
    tbins1,
):
    bl_dmask = bkg_mod.bl_dmask

    sig_mod = Source_Model_OutFoV(flux_mod, [ebins0, ebins1], bl_dmask, use_deriv=True)
    sig_mod.set_theta_phi(theta, phi)
    print("theta, phi set")

    comp_mod = CompoundModel([bkg_mod, sig_mod])
    sig_miner = NLLH_ScipyMinimize_Wjacob("")
    sig_llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask, has_err=True)

    sig_llh_obj.set_model(comp_mod)

    flux_params = {"A": 1.0, "gamma": 0.5, "Epeak": 1e2}

    pars_ = {}
    pars_["Signal_theta"] = theta
    pars_["Signal_phi"] = phi
    for pname, val in bkg_bf_params.items():
        pars_["Background_" + pname] = val
    for pname, val in flux_params.items():
        pars_["Signal_" + pname] = val

    sig_miner.set_llh(sig_llh_obj)

    fixed_pnames = list(pars_.keys())
    fixed_vals = list(pars_.values())
    trans = [None for i in range(len(fixed_pnames))]
    sig_miner.set_trans(fixed_pnames, trans)
    sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
    sig_miner.set_fixed_params(["Signal_A"], fixed=False)

    gamma_ax = np.linspace(-0.4, 1.6, 8 + 1)
    gamma_ax = np.linspace(-0.4, 1.6, 4 + 1)
    #     gamma_ax = np.linspace(-0.4, 1.6, 3+1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 10 + 1)
    Epeak_ax = np.logspace(np.log10(45.0), 3, 5 + 1)
    #     Epeak_ax = np.logspace(np.log10(25.0), 3, 3+1)
    gammas, Epeaks = np.meshgrid(gamma_ax, Epeak_ax)
    gammas = gammas.ravel()
    Epeaks = Epeaks.ravel()

    res_dfs = []

    ntbins = len(tbins0)

    for i in range(ntbins):
        t0 = tbins0[i]
        t1 = tbins1[i]
        dt = t1 - t0
        sig_llh_obj.set_time(tbins0[i], tbins1[i])

        res_dict = {"theta": theta, "phi": phi, "time": t0, "dur": dt}

        res_dict["Epeak"] = Epeaks
        res_dict["gamma"] = gammas

        nllhs, As = min_at_Epeaks_gammas(sig_miner, sig_mod, Epeaks, gammas)

        pars_["Signal_A"] = 1e-10
        bkg_nllh = -sig_llh_obj.get_logprob(pars_)

        res_dict["nllh"] = np.array(nllhs)
        res_dict["A"] = np.array(As)
        res_dict["TS"] = np.sqrt(2 * (bkg_nllh - res_dict["nllh"]))
        res_dict["bkg_nllh"] = bkg_nllh

        res_dfs.append(pd.DataFrame(res_dict))
        print("done with %d of %d tbins" % (i + 1, ntbins))
    return pd.concat(res_dfs, ignore_index=True)


def main(args):
    fname = os.path.join(args.work_dir, args.log_fname + "_" + str(args.job_id))

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    resp_fname = (
        "/storage/work/jjd330/local/bat_data/resp_tabs/drm_theta_126.0_phi_30.0_.fits"
    )

    resp_file = fits.open(resp_fname)
    pha_emins, pha_emaxs = resp_file[2].data["E_MIN"].astype(np.float64), resp_file[
        2
    ].data["E_MAX"].astype(np.float64)

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)

    ev_data = fits.open(args.evfname)[1].data
    if args.trig_time is None:
        trigger_time = np.min(ev_data["TIME"])
    else:
        trigger_time = args.trig_time

    if "bdecb" in args.dmask:
        enb_tab = Table.read(args.dmask)
        enb_ind = np.argmin(np.abs(enb_tab["TIME"] - (trigger_time + args.min_dt)))
        dmask = enb_tab[enb_ind]["FLAG"]
    else:
        dmask = fits.open(args.dmask)[0].data
    mask_vals = mask_detxy(dmask, ev_data)
    bl_dmask = dmask == 0.0

    bl_ev = (
        (ev_data["EVENT_FLAGS"] < 1)
        & (ev_data["ENERGY"] < 1e3)
        & (ev_data["ENERGY"] >= 10.0)
        & (mask_vals == 0.0)
    )
    ev_data0 = ev_data[bl_ev]

    attfile = Table.read(args.attfname)
    att_ind = np.argmin(np.abs(attfile["TIME"] - (trigger_time + args.min_dt)))
    att_q = attfile["QPARAM"][att_ind]
    pnt_ra, pnt_dec = attfile["POINTING"][att_ind, :2]

    solid_angle_dpi = np.load(solid_angle_dpi_fname)
    bkg_mod = Bkg_Model_wFlatA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)
    llh_obj = LLH_webins(ev_data0, ebins0, ebins1, bl_dmask, has_err=True)

    # bkg_miner = NLLH_ScipyMinimize('')
    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_t0 = trigger_time + args.bkg_dt0  # 6.0
    bkg_dt = args.bkg_dur  # 4.0
    bkg_t1 = bkg_t0 + bkg_dt

    brt_src_tab = get_srcs_infov(attfile, bkg_t0 + bkg_dt / 2.0)

    Nsrcs = len(brt_src_tab)
    if Nsrcs < 1:
        llh_obj.set_time(bkg_t0, bkg_t1)
        llh_obj.set_model(bkg_mod)
        bkg_miner.set_llh(llh_obj)
        pars, bkg_nllh, res = bkg_miner.minimize()
        i = 0
        bkg_bf_params = {}
        for cname in bkg_mod.param_names:
            if cname in bkg_miner.fixed_params:
                continue
            bkg_bf_params[cname] = pars[0][i]
            i += 1
    else:
        rt_obj = RayTraces(rt_dir)
        GTI = Table.read(args.evfname, hdu="GTI")
        sig_twind = (
            args.min_dt + trigger_time - 5.0,
            args.max_dt + trigger_time + 10.0,
        )
        tmin = bkg_t0
        tmax = bkg_t1
        bkg_bf_params, src_tab = do_init_bkg_wPSs(
            bkg_mod, llh_obj, brt_src_tab, rt_obj, GTI, sig_twind, tmin=tmin, tmax=tmax
        )

        Nsrcs = len(src_tab)

        if Nsrcs < 1:
            llh_obj.set_time(bkg_t0, bkg_t1)
            llh_obj.set_model(bkg_mod)
            bkg_miner.set_llh(llh_obj)
            pars, bkg_nllh, res = bkg_miner.minimize()
            i = 0
            bkg_bf_params = {}
            for cname in bkg_mod.param_names:
                if cname in bkg_miner.fixed_params:
                    continue
                bkg_bf_params[cname] = pars[0][i]
                i += 1

        else:
            mod_list = [bkg_mod]
            ps_mods = []
            for i in range(Nsrcs):
                row = src_tab[i]
                mod = Point_Source_Model_Binned_Rates(
                    row["imx"],
                    row["imy"],
                    0.1,
                    [llh_obj.ebins0, llh_obj.ebins1],
                    rt_obj,
                    llh_obj.bl_dmask,
                    use_deriv=True,
                    name=row["Name"],
                )
                ps_mods.append(mod)

            mod_list += ps_mods
            bkg_mod = CompoundModel(mod_list)

    # llh_obj.set_time(bkg_t0, bkg_t1)
    # llh_obj.set_model(bkg_mod)
    #
    # bkg_miner.set_llh(llh_obj)
    #
    # pars, bkg_nllh, res = bkg_miner.minimize()
    #
    # bkg_bf_params = {bkg_mod.param_names[i]:pars[0][i] for i in range(len(bkg_mod.param_names))}

    flux_mod = Cutoff_Plaw_Flux(E0=100.0)

    dur = args.min_dur
    tbins0 = np.arange(args.min_dt, args.max_dt, dur / 2.0) + trigger_time
    tbins1 = tbins0 + dur
    for i in range(args.Ntdbls):
        dur *= 2
        tbins0_ = np.arange(args.min_dt, args.max_dt, dur / 2.0) + trigger_time
        tbins1_ = tbins0_ + dur
        tbins0 = np.append(tbins0, tbins0_)
        tbins1 = np.append(tbins1, tbins1_)

    ntbins = len(tbins0)
    logging.info("ntbins: %d" % (ntbins))

    Njobs = args.Njobs
    job_id = args.job_id

    dimxy = 0.048
    imxax = np.arange(-1.8, 1.8, dimxy)  # [12:-12]
    imyax = np.arange(-1.0, 1.0, dimxy)  # [5:-5]
    imxg, imyg = np.meshgrid(imxax, imyax)
    imx0s = np.ravel(imxg)
    imy0s = np.ravel(imyg)

    Nsquares = len(imx0s)

    Npix2do = 1 + Nsquares / Njobs
    logging.info("Npix2do: %d" % (Npix2do))

    ind0 = job_id * Npix2do
    ind1 = min(ind0 + Npix2do, Nsquares)

    # logging.info("hp_ind0: %d" %(hp_ind0))
    # logging.info("hp_ind1: %d" %(hp_ind1))

    for ind in range(ind0, ind1):
        imx0 = imx0s[ind]
        imx1 = imx0 + dimxy
        imy0 = imy0s[ind]
        imy1 = imy0 + dimxy

        logging.info("Starting ind %d" % (ind))
        logging.info("imx0, imx1: %.3f, %.3f" % (imx0, imx1))
        logging.info("imy0, imy1: %.3f, %.3f" % (imy0, imy1))

        res_df = analysis_for_imxy_square(
            imx0,
            imx1,
            imy0,
            imy1,
            bkg_bf_params,
            bkg_mod,
            flux_mod,
            ev_data0,
            ebins0,
            ebins1,
            tbins0,
            tbins1,
        )

        res_df["dt"] = res_df["time"] - trigger_time
        res_df["square_ind"] = ind
        save_fname = os.path.join(args.work_dir, "square_ind_%d_.csv" % (ind))
        res_df.to_csv(save_fname)
        logging.info("wrote results to, ")
        logging.info(save_fname)


if __name__ == "__main__":
    args = cli()
    main(args)
