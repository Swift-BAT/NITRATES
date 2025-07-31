import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
from numba import jit, njit, prange
import argparse
import healpy as hp

from ..response.StructFunc import get_full_struct_manager
from ..response.Materials import PB, TA, SN, CU, CZT
from ..response.StructClasses import (
    Swift_Structure_Shield,
    Swift_Structure,
    Swift_Structure_Mask,
)
from ..response.Polygons import Polygon2D, Box_Polygon
from ..response.shield_structure import Shield_Interactions, Shield_Structure


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind", type=int, help="healpy index to do", default=None)
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


def get_resp4line(resp_fname, Eline):
    resp_tab = Table.read(resp_fname)
    photonEs = (resp_tab["ENERG_HI"] + resp_tab["ENERG_LO"]) / 2.0

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

    for cname in line_cnames:
        cname_list = cname.split("_")
        col0 = int(cname_list[-5])
        col1 = int(cname_list[-4])
        row0 = int(cname_list[-2])
        row1 = int(cname_list[-1])
        orientation = cname_list[0]
        if (orientation == "NonEdges") and (col0 == 0) and (row0 == 0):
            cname2use = cname
            break

    for cname in comp_cnames:
        cname_list = cname.split("_")
        col0 = int(cname_list[-6])
        col1 = int(cname_list[-5])
        row0 = int(cname_list[-3])
        row1 = int(cname_list[-2])
        orientation = cname_list[0]
        if (orientation == "NonEdges") and (col0 == 0) and (row0 == 0):
            comp_cname2use = cname
            break

    E_ind0 = np.digitize(Eline, photonEs) - 1
    E_ind1 = E_ind0 + 1

    Elo = photonEs[E_ind0]
    Ehi = photonEs[E_ind1]
    dE = Ehi - Elo

    a0 = (Ehi - Eline) / dE
    a1 = 1.0 - a0

    resp = a0 * resp_tab[cname2use][E_ind0] + a1 * resp_tab[cname2use][E_ind1]
    comp_resp = (
        a0 * resp_tab[comp_cname2use][E_ind0] + a1 * resp_tab[comp_cname2use][E_ind1]
    )
    return resp + comp_resp


def get_theta_resps4line(resp_dir, Eline):
    resp_arr = get_resp_arr(resp_dir)
    bl = np.isclose(resp_arr["phi"], 0.0)
    resp_arr = resp_arr[bl]
    resp_arr.sort(order="theta")

    Nfiles = len(resp_arr)

    resps = []

    for i in range(Nfiles):
        resps.append(get_resp4line(os.path.join(resp_dir, resp_arr["fname"][i]), Eline))
    return resps, resp_arr["theta"]


@njit(cache=True)
def calc_resp4shield_pnt(
    th,
    phi,
    th_inds0,
    r2,
    shield_dA,
    shield_vec,
    rho_mud,
    trans2shields,
    resp_thetas,
    resp_list,
    rhomu_photoEs,
    rhomu_tot0s,
    rhomu_tot1,
    cos_theta0,
    h,
    trans_inshields,
):
    det_pnts = len(th)
    NphotonEs = len(trans2shields)
    Nphabins = len(resp_list[0])
    specs = np.zeros((det_pnts, NphotonEs, Nphabins))

    from_out = cos_theta0 > 0
    cos_theta0 = abs(cos_theta0)

    th_deg = np.rad2deg(th)
    th_inds1 = th_inds0 + 1

    for j in range(det_pnts):
        gam_vec = -np.array(
            [
                np.sin(th[j]) * np.cos(-phi[j]),
                np.sin(th[j]) * np.sin(-phi[j]),
                np.cos(th[j]),
            ]
        )
        #             angs[j] = np.arccos(np.dot(gam_vec,shield_vec))
        cos_ang = np.abs(np.dot(gam_vec, shield_vec))

        #             photoE_abs = get_photoE_abs(rhomu_photoEs, rhomu_tot0s, rhomu_tot1,\
        #                                         cos_theta0, cos_ang, h, from_out)

        mu_A = (rhomu_tot1 / cos_ang) + (rhomu_tot0s / cos_theta0)
        mu_ratio = rhomu_photoEs / mu_A
        #         exp_term = np.exp(-rhomu_tot0s*h/cos_theta0) - np.exp(-rhomu_tot1*h/cos_ang)
        exp_term = 1.0 - np.exp(-h * mu_A)

        if from_out:
            mu_diff = (rhomu_tot1 / cos_ang) - (rhomu_tot0s / cos_theta0)
            mu_ratio = rhomu_photoEs / mu_diff
            exp_term = np.exp(-rhomu_tot0s * h / cos_theta0) - np.exp(
                -rhomu_tot1 * h / cos_ang
            )
        #             else:
        #                 mu_A = (rhomu_tot1/cos_ang) + (rhomu_tot0s/cos_theta0)
        #                 mu_ratio = rhomu_photoEs / mu_A
        #                 exp_term = np.exp(-rhomu_tot0s*h/cos_theta0) - np.exp(-rhomu_tot1*h/cos_ang)
        photoE_abs = mu_ratio * exp_term

        th0 = resp_thetas[th_inds0[j]]
        th1 = resp_thetas[th_inds1[j]]
        dTH = th1 - th0
        a0 = (th1 - th_deg[j]) / dTH
        a1 = 1.0 - a0

        resp = a0 * resp_list[th_inds0[j]] + a1 * resp_list[th_inds1[j]]
        solid_angs = resp / r2[j]

        #             ds_fact = np.abs(1./np.cos(angs[j]))
        ds_fact = 1.0 / cos_ang
        trans2dets = np.exp(-rho_mud * ds_fact)

        for k in range(NphotonEs):
            if rhomu_tot0s[k] <= 0.0:
                continue
            int_flux = trans_inshields[k] * photoE_abs[k] * cos_theta0
            specs[j, k] += (
                int_flux
                * trans2shields[k]
                * shield_dA
                * trans2dets
                * (solid_angs / (4 * np.pi))
            )

    return specs


@njit(cache=True)
def calc_shield_resp4line(
    shield_vec,
    batxs,
    batys,
    shield_xs,
    shield_ys,
    shield_zs,
    shield_dA,
    rho_mud,
    trans2shields,
    resp_thetas,
    resp_list,
    rhomu_photoEs,
    rhomu_tot0s,
    rhomu_tot1,
    cos_theta0,
    h,
    trans_inshields,
):
    det_pnts = len(batxs)
    NphotonEs = len(trans2shields[0])
    Nphabins = len(resp_list[0])
    specs = np.zeros((det_pnts, NphotonEs, Nphabins))

    Nshield_pnts = len(shield_xs)
    #     from_out = (cos_theta0 > 0)
    #     cos_theta0 = abs(cos_theta0)

    for i in range(Nshield_pnts):
        shield_x = shield_xs[i]
        shield_y = shield_ys[i]
        shield_z = shield_zs[i]
        r2 = (
            np.square(shield_x - batxs)
            + np.square(shield_y - batys)
            + np.square(shield_z - 3.187)
        )
        r = np.sqrt(r2)
        rho2 = np.square(shield_x - batxs) + np.square(shield_y - batys)
        rho = np.sqrt(rho2)
        th = np.pi / 2.0 - np.arctan2((shield_z - 3.187), rho)
        th_deg = np.rad2deg(th)
        th_inds0 = np.digitize(th_deg, resp_thetas) - 1
        th_inds1 = th_inds0 + 1

        phi = np.arctan2(-(shield_y - batys), (shield_x - batxs))

        specs += calc_resp4shield_pnt(
            th,
            phi,
            th_inds0,
            r2,
            shield_dA,
            shield_vec,
            rho_mud,
            trans2shields[i],
            resp_thetas,
            resp_list,
            rhomu_photoEs,
            rhomu_tot0s,
            rhomu_tot1,
            cos_theta0,
            h,
            trans_inshields,
        )

    return specs


def calc_flor_resp(
    th,
    phi,
    Elines,
    line_wts,
    E_edge,
    material,
    mat_ind,
    in_layers,
    out_layers,
    photonEs,
    Nphabins,
    dx_big=2.0,
    dx_small=0.5,
):
    resp_dir = "/storage/work/jjd330/local/bat_data/resp_tabs/"
    Nlines = len(Elines)

    dpi_shape = (173, 286)
    detxax = np.arange(-1, 286 + 2, 8, dtype=np.int64)
    detyax = np.arange(-2, 173 + 2, 8, dtype=np.int64)
    detx_dpi, dety_dpi = np.meshgrid(detxax, detyax)
    batxs, batys = detxy2batxy(detx_dpi.ravel(), dety_dpi.ravel())

    gam_vec = -np.array(
        [np.sin(th) * np.cos(-phi), np.sin(th) * np.sin(-phi), np.cos(th)]
    )

    mask_obj = Swift_Structure_Mask()
    shield_int_obj = Shield_Interactions()

    det_arr_half_dims = [59.85, 36.2, 0.1]
    det_arr_pos = (0.0, 0.0, 3.087)
    det_arr_box = Box_Polygon(
        det_arr_half_dims[0],
        det_arr_half_dims[1],
        det_arr_half_dims[2],
        np.array(det_arr_pos),
    )
    DetArr = Swift_Structure(det_arr_box, CZT, Name="DetArr")

    ds_base = [
        0.00254 * np.array([3, 3, 2, 1]),
        0.00254 * np.array([8, 7, 6, 2]),
        0.00254 * np.array([5, 5, 4, 1]),
        0.00254 * np.array([3, 3, 2, 1]),
    ]
    materials = [PB, TA, SN, CU]

    line_resps = np.zeros((len(batxs), len(photonEs), Nphabins))

    for jj in range(Nlines):
        print("Line %d of %d" % (jj + 1, Nlines))

        Eline = Elines[jj]

        resp_list, resp_thetas = get_theta_resps4line(resp_dir, Eline)

        line_resp_list = []
        names = []

        rhomu_photoEs = material.get_photoe_rhomu(photonEs)
        rhomu_photoEs[(photonEs < E_edge)] = 0.0
        rhomu_tot0s = material.get_tot_rhomu(photonEs)
        rhomu_tot0s[(photonEs < E_edge)] = 0.0
        rhomu_tot1 = material.get_tot_rhomu(Eline)

        for i in range(shield_int_obj.Npolys):
            poly = shield_int_obj.get_poly(i)
            name = shield_int_obj.shield_struct.shield_names[i]
            print(i)
            print(name)

            struct_obj = get_full_struct_manager(Es=photonEs, structs2ignore=["Shield"])
            shield_obj = Swift_Structure_Shield()
            shield_obj.add_polyID2ignore(i)

            struct_obj.add_struct(shield_obj)
            struct_obj.add_struct(mask_obj)
            struct_obj.add_struct(DetArr)

            if "Bb" in name:
                dx = dx_small
            else:
                dx = dx_big
            dA = dx**2
            shield_xs, shield_ys, shield_zs = poly.get_grid_pnts(dx=dx)

            if (np.all(shield_zs < 0)) and (th <= np.pi):
                continue
            shield_vec = np.copy(poly.norm_vec)

            layer = shield_int_obj.get_shield_layer(i)
            dss = ds_base[layer]

            struct_obj.set_batxyzs(shield_xs, shield_ys, shield_zs)
            struct_obj.set_theta_phi(th, phi)

            trans2shield = struct_obj.get_trans()  # [:,0]
            print(np.min(trans2shield), np.max(trans2shield), np.mean(trans2shield))

            cos_theta = np.dot(shield_vec, gam_vec)
            print("cos(theta): ", cos_theta)
            if np.dot(shield_vec, gam_vec) < 0:
                #                 rho_mud = (SN.get_tot_rhomu(photonEs)*dss[2] + CU.get_tot_rhomu(photonEs)*dss[3])
                rho_mud = np.zeros_like(photonEs)
                for lay in in_layers:
                    rho_mud += materials[lay].get_tot_rhomu(photonEs) * dss[lay]
                trans = np.exp(-rho_mud / np.abs(cos_theta))
            else:
                #                 d = ds_base[layer][0]/cos_theta
                #                 trans = np.exp(-PB.get_tot_rhomu(photonEs)*d)
                rho_mud = np.zeros_like(photonEs)
                for lay in out_layers:
                    rho_mud += materials[lay].get_tot_rhomu(photonEs) * dss[lay]
                trans = np.exp(-rho_mud / np.abs(cos_theta))
            print(np.min(trans), np.max(trans), np.mean(trans))
            #             d = ds_base[layer][1]/np.abs(cos_theta)
            #             photoE_abs = 1. - np.exp(-TA.get_photoe_rhomu(photonEs)*d)
            #             print np.min(photoE_abs), np.max(photoE_abs), np.mean(photoE_abs)
            #             int_fluxs = photoE_abs*trans*cos_theta
            #             int_fluxs[(photonEs<E_edge)] = 0.0

            print(len(shield_xs), len(shield_xs) * dA)
            print("min, max shield ys: ", np.min(shield_ys), np.max(shield_ys))
            #             rho_mud = (SN.get_tot_rhomu(ta_lines2use[0])*dss[2] + CU.get_tot_rhomu(ta_lines2use[0])*dss[3])
            rho_mud = 0.0
            for lay in in_layers:
                rho_mud += materials[lay].get_tot_rhomu(Eline) * dss[lay]

            print(np.exp(-rho_mud))

            line_resp = calc_shield_resp4line(
                shield_vec,
                batxs,
                batys,
                shield_xs,
                shield_ys,
                shield_zs,
                dA,
                rho_mud,
                trans2shield,
                resp_thetas,
                resp_list,
                rhomu_photoEs,
                rhomu_tot0s,
                rhomu_tot1,
                cos_theta,
                ds_base[layer][mat_ind],
                trans,
            )
            line_resps += line_wts[jj] * line_resp
            line_resp_list.append(line_resp)
            names.append(shield_int_obj.shield_struct.shield_names[i])
            print(
                np.sum(line_resp),
                np.sum(line_resp) / len(batxs),
                32768 * np.sum(line_resp) / len(batxs),
            )
            print()

    return line_resps


def main(args):
    pb_lines2use = np.array([73.03, 75.25, 84.75, 85.23])
    pb_line_wts = np.array([0.2956, 0.4926, 0.0591, 0.1133])
    pb_line_wts /= pb_line_wts.sum()
    ta_lines2use = np.array([56.41, 57.69, 65.11, 65.39, 67.17])
    ta_line_wts = np.array([0.28934, 0.5076, 0.05584, 0.11675, 0.03553])
    sn_lines2use = np.array([25.03, 25.25, 28.47])
    sn_line_wts = np.array([0.288, 0.5435, 0.0489 + 0.09239])
    sn_line_wts /= sn_line_wts.sum()

    hp_ind = args.ind
    Nside = 2**3

    phi, theta = hp.pix2ang(Nside, hp_ind, lonlat=True)
    th = np.pi / 2.0 - np.radians(theta)
    phi = np.radians(phi)

    resp_dir = "/storage/work/jjd330/local/bat_data/resp_tabs/"
    resp_arr = get_resp_arr(resp_dir)
    tab = Table.read(os.path.join(resp_dir, resp_arr["fname"][0]))
    photonEs = ((tab["ENERG_LO"] + tab["ENERG_HI"]) / 2.0).astype(np.float64)
    Nphabins = len(tab[2][2])

    in_layers = [1, 2, 3]
    out_layers = []
    mat_ind = 0
    E_edge = 88.0
    material = PB

    line_resps = calc_flor_resp(
        th,
        phi,
        pb_lines2use,
        pb_line_wts,
        E_edge,
        material,
        mat_ind,
        in_layers,
        out_layers,
        photonEs,
        Nphabins,
    )

    in_layers = [2, 3]
    out_layers = [0]
    mat_ind = 1
    E_edge = 67.4
    material = TA

    line_resps += calc_flor_resp(
        th,
        phi,
        ta_lines2use,
        ta_line_wts,
        E_edge,
        material,
        mat_ind,
        in_layers,
        out_layers,
        photonEs,
        Nphabins,
    )

    in_layers = [3]
    out_layers = [0, 1]
    mat_ind = 2
    E_edge = 29.2
    material = SN

    line_resps += calc_flor_resp(
        th,
        phi,
        sn_lines2use,
        sn_line_wts,
        E_edge,
        material,
        mat_ind,
        in_layers,
        out_layers,
        photonEs,
        Nphabins,
    )

    save_fname = "/gpfs/scratch/jjd330/bat_data/flor_resps/hp_order_3_ind_%d_" % (
        hp_ind
    )
    np.save(save_fname, line_resps)


if __name__ == "__main__":
    args = cli()

    main(args)
