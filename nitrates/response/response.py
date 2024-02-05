import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from numba import jit, njit, prange
from scipy import interpolate
import healpy as hp
import logging, traceback
from copy import copy, deepcopy

from ..lib.drm_funcs import get_ebin_ind_edges
from ..lib.trans_func import get_pb_absortion
from ..response.StructFunc import get_full_struct_manager
from ..response.StructClasses import Swift_Structure, Swift_Structure_Manager
from ..response.Materials import PB, TI, Korex, CarbonFibre, Mylar
from ..response.Polygons import Polygon2D, Box_Polygon
from ..response.ray_trace_funcs import RayTraces
from ..lib.coord_conv_funcs import theta_phi2imxy, imxy2theta_phi


class Response(object):
    def __init__(self, drm, ebins=None, cor_type="op"):
        self.drm = drm
        self.drm_mat = drm[1].data["MATRIX"]
        self.drm_e0s = (drm[1].data["ENERG_LO"]).astype(np.float64)
        self.drm_e1s = (drm[1].data["ENERG_HI"]).astype(np.float64)
        self.drm_emids = (
            (drm[1].data["ENERG_LO"] + drm[1].data["ENERG_HI"]) / 2.0
        ).astype(np.float64)
        self.drm_nebins = len(self.drm_e0s)
        if ebins is not None:
            self.set_ebins(ebins[0], ebins[1])
        self.imx, self.imy = self.get_imxy()
        self.absorbs = self.get_abs()
        self.set_abs_cor(cor_type)

    def set_ebins(self, ebins0, ebins1):
        self.ebins0 = ebins0
        self.ebins1 = ebins1
        self.nebins = len(ebins0)
        #         print self.nebins, "ebins"
        self.ebin_ind_edges = get_ebin_ind_edges(self.drm, ebins0, ebins1)
        # might want to make a drm_mat copy that has ebins that I want
        self.drm_mat_ebins = [
            np.sum(
                self.drm_mat[
                    :, self.ebin_ind_edges[i][0] : self.ebin_ind_edges[i][1] + 1
                ],
                axis=1,
            )
            for i in range(self.nebins)
        ]
        self.drm_mat_ebins = np.array(self.drm_mat_ebins).T

    def get_imxy(self):
        x = self.drm[1].header["BAT_XOBJ"]
        y = self.drm[1].header["BAT_YOBJ"]
        z = self.drm[1].header["BAT_ZOBJ"]
        imx = x / z
        imy = y / z
        # print "DRM made at imx, imy: ", imx, imy
        return imx, imy

    def get_abs(self):
        absorbs = get_pb_absortion(self.drm_emids, self.imx, self.imy)

        return absorbs

    def set_abs_cor(self, cor_type="op"):
        cor_types_allowed = ["op", "trans", "none"]
        if cor_type == "op":
            self.abs_cor = 1.0 / self.absorbs
        elif cor_type == "trans":
            self.abs_cor = (2.0 - self.absorbs) / (self.absorbs)
        elif cor_type == "none" or cor_type is None:
            self.abs_cor = np.ones_like(self.absorbs)
        else:
            print("cor_type should be one of: ")
            print(cor_types_allowed)
            self.abs_cor = np.ones_like(self.absorbs)
        self.abs_cor = self.abs_cor.reshape(len(self.abs_cor), 1)

    def get_rates_from_flux(self, photon_fluxes):
        phot_fluxes = photon_fluxes.reshape(len(photon_fluxes), 1)
        rates = self.abs_cor * self.drm_mat_ebins * phot_fluxes

        return rates

    def get_rate_from_flux(self, photon_fluxes, ebin):
        #         phot_fluxes = photon_fluxes.reshape(len(photon_fluxes),1)
        rates = self.abs_cor[:, 0] * (self.drm_mat_ebins[:, ebin]) * photon_fluxes

        return rates


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
            (
                np.shape(self.trans[0]),
                np.shape(self.wts_list[0]),
                np.shape(self.comp_trans[self.inds_list[0], :]),
            )
        )
        print(
            (
                np.shape(
                    np.sum(
                        self.comp_trans[self.inds_list[0], :]
                        * self.wts_list[0][:, np.newaxis],
                        axis=0,
                    )
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

    dual_struct_obj = get_full_struct_manager(Es=Ephotons)
    dual_struct_obj.set_batxyzs(batxs4duals, batys4duals, batzs4duals)

    return dual_struct_obj


detxs_by_sand0 = np.arange(0, 286 - 15, 18)
detxs_by_sand1 = detxs_by_sand0 + 15

detys_by_sand0 = np.arange(0, 173 - 7, 11)
detys_by_sand1 = detys_by_sand0 + 7

detxs_in_cols_not_edges = [
    np.arange(detxs_by_sand0[i] + 1, detxs_by_sand1[i], 1, dtype=np.int64)
    for i in range(16)
]
detys_in_rows_not_edges = [
    np.arange(detys_by_sand0[i] + 1, detys_by_sand1[i], 1, dtype=np.int64)
    for i in range(16)
]

dpi_shape = (173, 286)
detxax = np.arange(286, dtype=np.int64)
detyax = np.arange(173, dtype=np.int64)
detx_dpi, dety_dpi = np.meshgrid(detxax, detyax)


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


orientations = ["NonEdges", "left", "top", "bot", "right"]


def resp_tab2resp_dpis(resp_tab, phi_rot=0.0, correct_comp_theta_gt110=False):
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

    if correct_comp_theta_gt110:
        col0 = 1
        col1 = 15
        row0 = 1
        row1 = 15
        for orientation in orientations:
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
            cname = [cn for cn in comp_cnames if orientation in cn][0]

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
    def __init__(
        self, resp_fname, pha_emins, pha_emaxs, phi0, bl_dmask, correct_comp=False
    ):
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

        self.correct_comp = correct_comp

        self.set_pha_bins(pha_emins, pha_emaxs)
        self.mk_resp_dpis()

    def set_pha_bins(self, pha_emins, pha_emaxs):
        self.pha_emins = pha_emins
        self.pha_emaxs = pha_emaxs
        self.Nphabins = len(self.pha_emins)
        self.ShiftEbins = True
        if self.Nphabins == len(self.orig_pha_emins):
            if np.all(np.isclose(self.pha_emins, self.orig_pha_emins)) and np.all(
                np.isclose(self.pha_emaxs, self.orig_pha_emaxs)
            ):
                self.ShiftEbins = False

        if self.ShiftEbins:
            self.resp_tab = shift_resp_tab_pha_bins(
                self.orig_resp_tab,
                self.orig_pha_emins,
                self.orig_pha_emaxs,
                self.pha_emins,
                self.pha_emaxs,
            )
        else:
            self.resp_tab = self.orig_resp_tab

    def set_phi0(self, phi0):
        if np.abs(phi0 - self.phi0) > 1e-2:
            self.phi0 = phi0
            self.mk_resp_dpis()

    def mk_resp_dpis(self):
        lines_resp_dpis, comp_resp_dpis = resp_tab2resp_dpis(
            self.resp_tab, phi_rot=self.phi0, correct_comp_theta_gt110=self.correct_comp
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
        self.ShiftEbins = True
        if self.Nphabins == len(self.orig_pha_emins):
            if np.all(np.isclose(self.pha_emins, self.orig_pha_emins)) and np.all(
                np.isclose(self.pha_emaxs, self.orig_pha_emaxs)
            ):
                self.ShiftEbins = False

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
        if self.ShiftEbins:
            sn_inds = np.arange(1, 13, dtype=np.int64)
            ta_inds = np.arange(14, 29, dtype=np.int64)
            pb_inds = np.arange(29, 39, dtype=np.int64)
            for sn_ind in sn_inds:
                resp_arr[:, :, sn_ind] *= self.sn_ratios
            for ta_ind in ta_inds:
                resp_arr[:, :, ta_ind] *= self.ta_ratios
            for pb_ind in pb_inds:
                resp_arr[:, :, pb_ind] *= self.pb_ratios

        if self.ShiftEbins:
            self.resp_dict[hp_ind] = shift_flor_dpi_pha_bins(
                resp_arr,
                self.orig_pha_emins,
                self.orig_pha_emaxs,
                self.pha_emins,
                self.pha_emaxs,
            )
        else:
            self.resp_dict[hp_ind] = resp_arr

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


psv = [
    798.38422,
    -2.5190028,
    25.51,
    313.545,
    -2.36265,
    0.0851972,
    -0.1008352,
    0.882394,
    80.72,
    30609.4,
    -22.9841,
    0.577041,
    -0.477249,
    0.334311,
]


def passive_trans(E, cos_theta):
    if E < psv[2]:
        #          /* a power law
        #           *
        #           * psv[0]*energy^psv[1]
        #           */

        passive_ind = psv[0] * np.power(E, psv[1])

    #          headas_chat(5,"using case 2 (low energies): \n");

    elif (E >= psv[2]) and (E < psv[8]):
        #          /* a smoothly broken power law
        #           *
        #           * [(psv[3]*energy^psv[4])^(1/psv[7])+
        #           *  (psv[5]*energy^psv[6])^(1/psv[7])]^psv[7]
        #           */

        passive_ind = np.power(
            np.power(psv[3] * np.power(E, psv[4]), 1.0 / psv[7])
            + np.power(psv[5] * np.power(E, psv[6]), 1.0 / psv[7]),
            psv[7],
        )

    #          headas_chat(5,"using case 2 (mid energies): \n");

    else:
        #          /* a smoothly broken power law
        #           *
        #           * [(psv[9]*energy^psv[10])^(1/psv[13])+
        #           *  (psv[11]*energy^psv[12])^(1/psv[13])]^psv[13]
        #           */

        passive_ind = np.power(
            np.power(psv[9] * np.power(E, psv[10]), 1.0 / psv[13])
            + np.power(psv[11] * np.power(E, psv[12]), 1.0 / psv[13]),
            psv[13],
        )

    #          headas_chat(5,"using case 2 (high energies): \n");

    t_passive = np.exp(-1.0 * passive_ind / cos_theta)

    return t_passive


def get_passive_trans(Es, theta):
    pass_trans = np.array([passive_trans(E, np.cos(np.radians(theta))) for E in Es])
    return pass_trans


class Swift_Mask_Interactions(object):
    """
    Should say whether photon goes through the mask poly or not
    Also want it to get ray trace
    Should have also contain the lead tiles where the struts screw in (which aren't included in the ray traces)

    Should be able to give trans to each det,
    assume each photon that goes through lead tile, goes through 0.1cm/cos(theta)
    trans = (shadow_frac)*exp[-rhomu_pb * 0.1 / cos(theta)]
    """

    def __init__(self, rt_obj, bl_dmask, pass_trans=False):
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

        self.pass_trans = pass_trans

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
        self.Ndets_int_mask = np.sum(self.does_int_mask)

    def calc_dists(self):
        self.dists = (self.does_int_mask.astype(np.float64)) * self.d

    def calc_tot_rhomu_dist(self):
        #         self.tot_rhomu_dists = np.zeros((self.ndets,self.Ne))
        self.tot_rhomu_dists = self.dists[:, np.newaxis] * self.tot_rho_mus
        self.mask_tot_rhomu_dists = np.zeros(self.Ne)
        for i in range(len(self.mask_d)):
            self.mask_tot_rhomu_dists += self.mask_d[i] * self.mask_tot_rho_mus_list[i]
        self.mask_trans = np.exp(-self.mask_tot_rhomu_dists)
        if self.pass_trans:
            self.mask_trans *= get_passive_trans(self.energy, self.theta)

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
        if self.Ndets_int_mask < 1:
            return self.trans

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
    def __init__(
        self, resp_dname, flor_resp_dname, pha_emins, pha_emaxs, bl_dmask, rt_obj
    ):
        self.flor_resp_dname = flor_resp_dname
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
            self.flor_resp_dname,
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

        if theta > 78.0:
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
        fname_theta = float(fname.split("_")[2])
        correct_comp = False
        if fname_theta > 110.0:
            correct_comp = True
        resp_file_obj = ResponseDPI(
            os.path.join(self.resp_dname, fname),
            self.pha_emins,
            self.pha_emaxs,
            np.radians(self.phi),
            self.bl_dmask,
            correct_comp=correct_comp,
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
        # py3
        # stop
        thetas = np.sort(np.unique(self.resp_arr["theta"]))
        phis = np.sort(np.unique(self.resp_arr["phi"]))

        th0 = np.digitize(theta, thetas) - 1
        if theta == 180.0:
            th0 -= 1
        theta0 = thetas[th0]
        theta1 = thetas[th0 + 1]
        print((theta0, theta1))
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

        phi_ = phi - (int(phi) // 45) * 45.0
        print(phi_)
        if (int(phi) // 45) % 2 == 1:
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


def sand_img2dpi(sand_img):
    dpi = np.zeros(dpi_shape)
    for i in range(16):
        x0 = detxs_by_sand0[i]
        x1 = detxs_by_sand1[i] + 1
        for j in range(16):
            y0 = detys_by_sand0[j]
            y1 = detys_by_sand1[j] + 1
            dpi[y0:y1, x0:x1] = sand_img[j, i]
    return dpi


def sand_imgs2dpis(sand_imgs):
    dpis = np.zeros(
        (dpi_shape[0], dpi_shape[1], sand_imgs.shape[-2], sand_imgs.shape[-1])
    )

    for i in range(16):
        x0 = detxs_by_sand0[i]
        x1 = detxs_by_sand1[i] + 1
        for j in range(16):
            y0 = detys_by_sand0[j]
            y1 = detys_by_sand1[j] + 1
            dpis[y0:y1, x0:x1] = sand_imgs[j, i]
    return dpis


@njit(cache=True)
def shift_sand_img_pha_bins(
    sand_imgs, pha_bins0, pha_bins1, new_pha_bins0, new_pha_bins1
):
    Nphabins_new = new_pha_bins0.size
    NphotonEs = sand_imgs.shape[0]
    new_shp = (NphotonEs, 16, 16, Nphabins_new)
    new_sand_imgs = np.zeros(new_shp)

    for k in range(NphotonEs):
        for i in range(16):
            for j in range(16):
                new_sand_imgs[k, i, j] += shift_pha_bins(
                    sand_imgs[k, i, j],
                    pha_bins0,
                    pha_bins1,
                    new_pha_bins0,
                    new_pha_bins1,
                )
    return new_sand_imgs


class CompFlorResponseDPI(object):
    def __init__(self, resp_dname, bl_dmask, Nside=2**2):
        self.resp_dname = resp_dname

        try:
            fits_fname0 = os.path.join(self.resp_dname, "resp_by_sand_hpind_191.fits")
            fits_file0 = fits.open(fits_fname0)
        except:
            fits_fname0 = os.path.join(self.resp_dname, "resp_by_sand_hpind_51.fits")
            fits_file0 = fits.open(fits_fname0)
        self.orig_photonEs = fits_file0[1].data["Ephoton"]
        self.Norig_photonEs = len(self.orig_photonEs)
        self.orig_pha_emins = fits_file0[2].data["E_MIN"].astype(np.float64)
        self.orig_pha_emaxs = fits_file0[2].data["E_MAX"].astype(np.float64)
        fits_file0.close()

        self.ndets = np.sum(bl_dmask)
        self.bl_dmask = bl_dmask
        #         self.batxs, self.batys = bldmask2batxys(self.bl_dmask)

        self.Nside = Nside
        all_hp_phis, all_hp_lats = hp.pix2ang(
            self.Nside, np.arange(hp.nside2npix(self.Nside), dtype=np.int64), lonlat=True
        )
        all_hp_thetas = 90.0 - all_hp_lats
        self.max_hp_theta = np.max(all_hp_thetas)
        self.min_hp_theta = np.min(all_hp_thetas)
        self.min_hp_theta = (
            60.0  # adjust this when the next theta set of flor resps are made
        )
        self.hp_phis_at_max_theta = np.sort(
            all_hp_phis[(np.isclose(all_hp_thetas, self.max_hp_theta))]
        )
        self.hp_phis_at_min_theta = np.sort(
            all_hp_phis[(np.isclose(all_hp_thetas, self.min_hp_theta))]
        )

        self.resp_dict = {}  # hp inds will be the keys

    def set_ebins(self, ebins0, ebins1):
        self.pha_emins = ebins0
        self.pha_emaxs = ebins1
        self.Nphabins = len(ebins0)
        self.ShiftEbins = True
        if self.Nphabins == len(self.orig_pha_emins):
            if np.all(np.isclose(self.pha_emins, self.orig_pha_emins)) and np.all(
                np.isclose(self.pha_emaxs, self.orig_pha_emaxs)
            ):
                self.ShiftEbins = False

    def set_photonEs(self, photonEs):
        self.NphotonEs = len(photonEs)
        self.photonEs = photonEs

        self.Einds0 = np.digitize(self.photonEs, self.orig_photonEs) - 1
        self.Einds1 = self.Einds0 + 1
        self.Einds1 = np.clip(self.Einds1, 1, len(self.orig_photonEs) - 1)

        self.E_A0s = (self.orig_photonEs[self.Einds1] - self.photonEs) / (
            self.orig_photonEs[self.Einds1] - self.orig_photonEs[self.Einds0]
        )
        self.E_A0s[np.isnan(self.E_A0s)] = 0.5
        self.E_A1s = 1.0 - self.E_A0s
        self.E_A0s = self.E_A0s[:, None, None, None]
        self.E_A1s = self.E_A1s[:, None, None, None]

    def do_Eintp(self, resp_sand_imgs):
        new_resp_sand_imgs = (
            resp_sand_imgs[self.Einds0] * self.E_A0s
            + resp_sand_imgs[self.Einds0] * self.E_A1s
        )
        return new_resp_sand_imgs

    def set_theta_phi(self, theta, phi):
        self.phi = phi
        self.theta = theta
        self.lat = 90.0 - self.theta

        if self.theta > self.max_hp_theta:
            dphi = self.hp_phis_at_max_theta[1] - self.hp_phis_at_max_theta[0]
            if self.phi <= np.min(self.hp_phis_at_max_theta):
                phi0 = self.hp_phis_at_max_theta[0]
                phi1 = self.hp_phis_at_max_theta[-1]
                phi_wt1 = (phi0 - self.phi) / dphi
                phi_wt0 = 1.0 - phi_wt1

            elif self.phi >= np.max(self.hp_phis_at_max_theta):
                phi0 = self.hp_phis_at_max_theta[0]
                phi1 = self.hp_phis_at_max_theta[-1]
                phi_wt0 = (self.phi - phi1) / dphi
                phi_wt1 = 1.0 - phi_wt0

            else:
                ind0 = np.digitize(self.phi, self.hp_phis_at_max_theta) - 1
                ind1 = ind0 + 1
                phi0 = self.hp_phis_at_max_theta[ind0]
                phi1 = self.hp_phis_at_max_theta[ind1]
                phi_wt0 = (self.phi - phi1) / dphi
                phi_wt1 = 1.0 - phi_wt0

            self.hp_inds2use = [
                hp.ang2pix(self.Nside, phi0, 90.0 - self.max_hp_theta, lonlat=True),
                hp.ang2pix(self.Nside, phi1, 90.0 - self.max_hp_theta, lonlat=True),
                hp.nside2npix(self.Nside),
            ]
            dtheta = 180.0 - self.max_hp_theta
            theta_wt0 = (180.0 - self.theta) / dtheta
            theta_wt1 = 1.0 - theta_wt0
            self.hp_wts = [phi_wt0 * theta_wt0, phi_wt1 * theta_wt0, theta_wt1]

        elif self.theta < self.min_hp_theta:
            dphi = self.hp_phis_at_min_theta[1] - self.hp_phis_at_min_theta[0]
            # then do the rest of the last if thing but with mins instead
            if self.phi <= np.min(self.hp_phis_at_min_theta):
                phi0 = self.hp_phis_at_min_theta[0]
                phi1 = self.hp_phis_at_min_theta[-1]
                phi_wt1 = (phi0 - self.phi) / dphi
                phi_wt0 = 1.0 - phi_wt1

            elif self.phi >= np.max(self.hp_phis_at_min_theta):
                phi0 = self.hp_phis_at_min_theta[0]
                phi1 = self.hp_phis_at_min_theta[-1]
                phi_wt0 = (self.phi - phi1) / dphi
                phi_wt1 = 1.0 - phi_wt0

            else:
                ind0 = np.digitize(self.phi, self.hp_phis_at_min_theta) - 1
                ind1 = ind0 + 1
                phi0 = self.hp_phis_at_min_theta[ind0]
                phi1 = self.hp_phis_at_min_theta[ind1]
                phi_wt0 = (self.phi - phi1) / dphi
                phi_wt1 = 1.0 - phi_wt0

            self.hp_inds2use = [
                hp.ang2pix(self.Nside, phi0, 90.0 - self.min_hp_theta, lonlat=True),
                hp.ang2pix(self.Nside, phi1, 90.0 - self.min_hp_theta, lonlat=True),
                -1,
            ]
            dtheta = self.min_hp_theta
            theta_wt0 = (self.theta) / dtheta
            theta_wt1 = 1.0 - theta_wt0
            self.hp_wts = [phi_wt0 * theta_wt0, phi_wt1 * theta_wt0, theta_wt1]

        else:
            self.hp_inds2use, self.hp_wts = hp.get_interp_weights(
                self.Nside, self.phi, self.lat, lonlat=True
            )

        self.calc_resp_dpi()

    def open_new_file(self, hp_ind, fname=None):
        if fname is None:
            fname = "resp_by_sand_hpind_%d.fits" % (hp_ind)
        logging.debug("Openning file: %s" % (fname))
        fits_file = fits.open(os.path.join(self.resp_dname, fname))
        resp_sand_imgs0 = fits_file[1].data["RESPONSE"].astype(np.float64)
        if self.ShiftEbins:
            resp_sand_imgs0 = shift_sand_img_pha_bins(
                resp_sand_imgs0,
                self.orig_pha_emins,
                self.orig_pha_emaxs,
                self.pha_emins,
                self.pha_emaxs,
            )

        self.resp_dict[hp_ind] = np.swapaxes(
            np.swapaxes(self.do_Eintp(resp_sand_imgs0), 0, 2), 0, 1
        )

    def calc_resp_dpi(self):
        resp_sand_img0 = np.zeros((self.NphotonEs, 16, 16, self.Nphabins))
        resp_sand_img0 = np.zeros((16, 16, self.NphotonEs, self.Nphabins))

        for hp_ind, wt in zip(self.hp_inds2use, self.hp_wts):
            if not hp_ind in list(self.resp_dict.keys()):
                if hp_ind >= hp.nside2npix(self.Nside):
                    self.open_new_file(hp_ind, fname="resp_by_sand_theta_180.fits")
                elif hp_ind < 0:
                    self.open_new_file(hp_ind, fname="resp_by_sand_theta_0.fits")
                else:
                    self.open_new_file(hp_ind)
            resp_sand_img0 += wt * self.resp_dict[hp_ind]

        self.resp_dpi = sand_imgs2dpis(resp_sand_img0)[self.bl_dmask]

    #         for sn_ind in sn_inds:
    #             self.resp_dpi[:,:,sn_ind] *= self.sn_ratios
    #         for ta_ind in ta_inds:
    #             self.resp_dpi[:,:,ta_ind] *= self.ta_ratios
    #         for pb_ind in pb_inds:
    #             self.resp_dpi[:,:,pb_ind] *= self.pb_ratios

    def get_resp_dpi(self):
        return self.resp_dpi


class ResponseInFoV2(object):
    def __init__(
        self,
        resp_dname,
        flor_resp_dname,
        comp_flor_resp_dname,
        pha_emins,
        pha_emaxs,
        bl_dmask,
        rt_obj,
    ):
        self.flor_resp_dname = flor_resp_dname
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

        # self.flor_resp_obj = FlorResponseDPI(self.flor_resp_dname,\
        #                                      pha_tab, self.pha_emins, self.pha_emaxs,\
        #                                      self.bl_dmask, NphotonEs=self.NphotonEs)

        # self.flor_resp_dname2 = '/gpfs/scratch/jjd330/bat_data/hp_flor_resps/'
        self.flor_resp_dname2 = flor_resp_dname

        self.comp_flor_resp_dname = "/gpfs/scratch/jjd330/bat_data/comp_flor_resps/"
        self.comp_flor_resp_dname = comp_flor_resp_dname
        self.comp_flor_resp_obj = CompFlorResponseDPI(
            self.comp_flor_resp_dname, self.bl_dmask
        )
        self.comp_flor_resp_obj.set_photonEs(self.PhotonEs)
        self.comp_flor_resp_obj.set_ebins(self.pha_emins, self.pha_emaxs)

        self.use_comp_flor = True
        self.use_comp_flor4flor_only = False

        self.mask_obj = Swift_Mask_Interactions(rt_obj, self.bl_dmask)
        self.mask_obj.set_energy_arr(self.PhotonEs)
        self.mask_obj.set_batxyzs(self.batxs, self.batys, self.batzs)

    def setup_comp_flor_obj(self):
        if self.use_comp_flor4flor_only:
            self.comp_flor_resp_obj = CompFlorResponseDPI(
                self.flor_resp_dname2, self.bl_dmask
            )
        else:
            self.comp_flor_resp_obj = CompFlorResponseDPI(
                self.comp_flor_resp_dname, self.bl_dmask
            )
        self.comp_flor_resp_obj.set_photonEs(self.PhotonEs)
        self.comp_flor_resp_obj.set_ebins(self.pha_emins, self.pha_emaxs)

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

        if self.theta < 90.0 - hp.pix2ang(2**2, 56, lonlat=True)[1]:
            if not self.use_comp_flor4flor_only:
                self.use_comp_flor4flor_only = True
                self.setup_comp_flor_obj()
            self.use_comp_flor4flor_only = True

        #         self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
        #         self.comp_trans_dpis = self.comp_obj.get_trans()

        if theta > 78.0:
            self.comp_obj.set_theta_phi(np.radians(self.theta), np.radians(self.phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.comp_flor_resp_obj.set_theta_phi(self.theta, self.phi)

        # if self.use_comp_flor:
        #     try:
        #         self.comp_flor_resp_obj.set_theta_phi(self.theta, self.phi)
        #     except Exception as E:
        #         logging.warning("Can't use comp flor object")
        #         logging.error(E)
        #         self.use_comp_flor = False
        #         self.flor_resp_obj.set_theta_phi(self.theta, self.phi)
        #
        # else:
        #     self.flor_resp_obj.set_theta_phi(self.theta, self.phi)

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

        if theta > 78.0:
            self.comp_obj.set_theta_phi(np.radians(theta), np.radians(phi))
            self.comp_trans_dpis = self.comp_obj.get_trans()
        else:
            self.comp_trans_dpis = self.lines_trans_dpis

        self.calc_tot_resp_dpis()

    def open_resp_file_obj(self, fname):
        fname_theta = float(fname.split("_")[2])
        correct_comp = False
        if fname_theta > 110.0:
            correct_comp = True
        resp_file_obj = ResponseDPI(
            os.path.join(self.resp_dname, fname),
            self.pha_emins,
            self.pha_emaxs,
            np.radians(self.phi),
            self.bl_dmask,
            correct_comp=correct_comp,
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
        # self.non_flor_resp_dpi = lines_dpi + comp_dpi
        # if not self.use_comp_flor:
        #     self.flor_resp_dpi = self.flor_resp_obj.get_resp_dpi()

        #         self.tot_resp_dpis = self.non_flor_resp_dpi + self.flor_resp_dpi
        if self.use_comp_flor:
            if self.use_comp_flor4flor_only:
                self.flor_resp_dpi = self.comp_flor_resp_obj.get_resp_dpi()
                self.comp_flor_resp_dpis = self.comp_resp_dpi + self.flor_resp_dpi
            else:
                self.comp_flor_resp_dpis = self.comp_flor_resp_obj.get_resp_dpi()
        else:
            self.comp_flor_resp_dpis = self.comp_resp_dpi + self.flor_resp_dpi

        # self.tot_resp_dpis = self.lines_resp_dpi + self.comp_flor_resp_dpis

    def get_lines_resp_dpis(self):
        return self.lines_resp_dpis

    def get_comp_resp_dpis(self):
        return self.comp_resp_dpis

    # def get_flor_resp_dpis(self):
    #     return self.flor_resp_obj.get_resp_dpi()

    def get_comp_flor_resp_dpis(self):
        return self.comp_flor_resp_dpis

    def get_tot_resp_dpis(self):
        return self.tot_resp_dpis

    def get_rate_dpis_from_photon_flux(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(self.tot_resp_dpis, photon_fluxes)
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

    def get_comp_flor_rate_dpis_from_photon_fluxes(self, photon_fluxes):
        rate_dpis = get_rate_dpis_from_photon_fluxes(
            self.comp_flor_resp_dpis, photon_fluxes
        )
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
        # py3
        # stop
        thetas = np.sort(np.unique(self.resp_arr["theta"]))
        phis = np.sort(np.unique(self.resp_arr["phi"]))

        th0 = np.digitize(theta, thetas) - 1
        if theta == 180.0:
            th0 -= 1
        theta0 = thetas[th0]
        theta1 = thetas[th0 + 1]
        print((theta0, theta1))
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

        phi_ = phi - (int(phi) // 45) * 45.0
        print(phi_)
        if (int(phi) // 45) % 2 == 1:
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

    
def get_pc(bl_dmask, theta, phi):
    '''
    Returns fraction of detectors that have los go through mask
    doesn't exactly match partial coding from bat tools but close
    
    bl_dmask: boolean array of which dets are active
    theta, phi: instrument coordinates, can be scalars or arrays
    
    returns partial coding fraction (array if theta, phi are arrays)
    '''
    
    mask_obj = Swift_Mask_Interactions(None, bl_dmask)
    batxs, batys = bldmask2batxys(bl_dmask)
    batzs = 3.187 + np.zeros(np.sum(bl_dmask))
    mask_obj.set_batxyzs(batxs, batys, batzs)
    mask_obj.set_energy_arr(np.array([15.0]))
    
    if np.isscalar(theta):
        mask_obj.set_theta_phi(theta, phi)
        pc = float(mask_obj.Ndets_int_mask) / mask_obj.ndets
    else:            
        pc = np.zeros_like(theta)

        for i in range(len(pc)):
            mask_obj.set_theta_phi(theta[i], phi[i])
            pc[i] = float(mask_obj.Ndets_int_mask) / mask_obj.ndets
        
    return pc
        
    
