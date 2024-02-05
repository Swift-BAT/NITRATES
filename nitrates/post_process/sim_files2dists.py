import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
import argparse

try:
    import ROOT
except ModuleNotFoundError as err:
    # Error handling
    print(err)
    print(
        "Please install the Python ROOT package to be able to run the full forward modeling calculations."
    )


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dname", type=str, help="directory for the sim run files", default=None
    )
    args = parser.parse_args()
    return args


def comp_Enew(E, theta):
    """
    E' = E/(1 + (E/m_e*c2)*(1 - cos(theta)))
    """
    denom = 1.0 + (E / 511) * (1.0 - np.cos(theta))
    return E / denom


def detxy2batxy(detx, dety):
    batx = 0.42 * detx - (285 * 0.42) / 2
    baty = 0.42 * dety - (172 * 0.42) / 2
    return batx, baty


def batxy2detxy(batx, baty):
    detx = (batx + (285 * 0.42) / 2) / 0.42
    dety = (baty + (172 * 0.42) / 2) / 0.42
    return detx, dety


def mk_comp_ebins(PrimaryE):
    comp_E90 = comp_Enew(PrimaryE, np.pi / 2.0)
    comp_E180 = comp_Enew(PrimaryE, np.pi)
    print((comp_E90, comp_E180))

    P_E180 = PrimaryE - comp_E180
    P_E90 = PrimaryE - comp_E90
    print((P_E90, P_E180))

    if PrimaryE - comp_E180 < 1.0:
        N_Esteps = int((PrimaryE - 8.0) / 4.0) + 2
        ebins = np.linspace(8.0, PrimaryE - 0.1, N_Esteps)
        ebins0 = ebins[:-1]
        ebins1 = ebins[1:]
        return ebins0, ebins1
    #     if PrimaryE - comp_E180 <= 2.5:
    if P_E180 <= 10.0:
        N_Esteps = int((comp_E180 - 8.0) / 4.0) + 2
        ebins = np.linspace(8.0, comp_E180, N_Esteps)
        ebins = np.append(ebins, [comp_E90, PrimaryE - 0.1])
        return ebins[:-1], ebins[1:]
    if P_E180 > 10.0 and (comp_E180 - P_E180) > 2.0:
        if P_E90 - 8.0 > 2.0:
            steps = int((P_E90 - 8.0) / 4.0) + 2
            steps = min(steps, 8)
            ebins = np.linspace(8.0, P_E90, steps)
            steps = int((P_E180 - P_E90) / 4.0) + 2
            steps = min(steps, 8)
            ebins = np.append(ebins[:-1], np.linspace(P_E90, P_E180, steps))
        else:
            steps = int((P_E180 - 8.0) / 4.0) + 2
            ebins = np.linspace(8.0, P_E180, steps)
        steps = int((comp_E180 - P_E180) / 5.0) + 2
        steps = min(steps, 10)
        ebins = np.append(ebins[:-1], np.linspace(P_E180, comp_E180, steps)[:-1])
        steps = int((comp_E90 - comp_E180) / 2.5) + 2
        ebins = np.append(ebins, np.linspace(comp_E180, comp_E90, steps)[:-1])
        steps = int((PrimaryE - comp_E90) / 5.0) + 2
        steps = min(steps, 10)
        ebins = np.append(ebins, np.linspace(comp_E90, PrimaryE - 0.1, steps))
        return ebins[:-1], ebins[1:]
    else:
        ebins = np.linspace(8.0, PrimaryE - 0.1, 25)
        if (ebins[1] - ebins[0]) < 2.0:
            steps = int((PrimaryE - 8.0) / 2.5) + 2
            ebins = np.linspace(8.0, PrimaryE - 0.1, steps)
        return ebins[:-1], ebins[1:]


def get_depth_dE4comp(
    ebins0, ebins1, Eline_bin0s, Eline_bin1s, edeps, zs, Ndets, gamma_percm2, Nzbins=20
):
    depth_dists = []
    Nebins = len(ebins0)
    Nlines = len(Eline_bin0s)
    zbins = np.linspace(0, 0.2, Nzbins + 1)
    zax = (zbins[1:] + zbins[:-1]) / 2.0
    dz = zbins[1] - zbins[0]
    zbins0_ = np.linspace(0.0, 0.02, Nzbins / 5 + 1)[:-1]
    zbins1_ = np.linspace(0.18, 0.2, Nzbins / 5 + 1)[1:]
    zbins = np.append(zbins0_, np.linspace(0.02, 0.18, Nzbins - 2 * len(zbins0_) + 1))
    zbins = np.append(zbins, zbins1_)
    zbins0 = zbins[:-1]
    zbins1 = zbins[1:]
    for j in range(Nebins):
        e0 = ebins0[j]
        e1 = ebins1[j]
        dE = e1 - e0
        ebl = (edeps >= e0) & (edeps < e1)
        for i in range(Nlines):
            if Eline_bin0s[i] > e1 or Eline_bin1s[i] < e0:
                continue
            if Eline_bin0s[i] <= e0:
                dE_ = Eline_bin1s[i] - e0
            elif Eline_bin1s[i] >= e1:
                dE_ = e1 - Eline_bin0s[i]
            else:
                dE_ = Eline_bin1s[i] - Eline_bin0s[i]
            ebl_ = (edeps < Eline_bin0s[i]) | (edeps >= Eline_bin1s[i])
            ebl = ebl & ebl_
            dE -= dE_

        N = np.sum(ebl)
        #         print N, dE
        #         wts_ = np.sum(ebl)*det_wts[ebl]/np.sum(det_wts[ebl])
        h = np.histogram(zs[ebl], bins=zbins)[0]
        dz = np.diff(zbins)
        depth_dists.append(h / (dz * Ndets * gamma_percm2 * dE))

    return depth_dists, zbins


def get_depth4lines(
    Eline_bin0s, Eline_bin1s, edeps, zs, Ndets, gamma_percm2, Nzbins=20
):
    depth_dists = []
    Nebins = len(Eline_bin0s)
    zbins0_ = np.linspace(0.0, 0.02, Nzbins / 5 + 1)[:-1]
    zbins1_ = np.linspace(0.18, 0.2, Nzbins / 5 + 1)[1:]
    zbins = np.append(zbins0_, np.linspace(0.02, 0.18, Nzbins - 2 * len(zbins0_) + 1))
    zbins = np.append(zbins, zbins1_)
    zbins0 = zbins[:-1]
    zbins1 = zbins[1:]
    dz = np.diff(zbins)
    for j in range(Nebins):
        ebl = (edeps >= Eline_bin0s[j]) & (edeps < Eline_bin1s[j])
        N = np.sum(ebl)
        #         wts_ = np.sum(ebl)*det_wts[ebl]/np.sum(det_wts[ebl])
        #         print N
        #         zbins = np.linspace(0, 0.2, Nzbins+1)
        #         zax = (zbins[1:] + zbins[:-1])/2.
        #         dz = zbins[1] - zbins[0]
        h = np.histogram(zs[ebl], bins=zbins)[0]
        depth_dists.append(h / (dz * Ndets * gamma_percm2))
    return depth_dists


def get_Es_Zs_PixIDs_from_root_file(fname):
    Edeps = []
    wtd_zs = []
    pix_ids = []
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    runID = int(fname.split("_")[-2])
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    PrimaryE = float(fname.split("_")[-6])
    Ngammas = int(fname.split("_")[-4])
    print(PrimaryE)
    print(Ngammas)
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        edep = getattr(tree, "sum_Edep")
        if edep > 0.0:
            Edeps.append(edep)
            wtd_zs.append(getattr(tree, "Ewtd_Z"))
            pix_ids.append(getattr(tree, "PixID"))
    File.Close()

    Edeps = np.array(Edeps)
    wtd_zs = np.array(wtd_zs)
    pix_ids = np.array(pix_ids)

    return Edeps * 1e3, (wtd_zs - 29.87) / 10.0, pix_ids


def get_Es_Zs_PixIDs_detxys_from_root_file(fname):
    Edeps = []
    wtd_zs = []
    pix_ids = []
    pos_xs = []
    pos_ys = []
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    runID = int(fname.split("_")[-2])
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    PrimaryE = float(fname.split("_")[-6])
    Ngammas = int(fname.split("_")[-4])
    print(PrimaryE)
    print(Ngammas)
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        edep = getattr(tree, "sum_Edep")
        if edep > 0.0:
            Edeps.append(edep)
            wtd_zs.append(getattr(tree, "Ewtd_Z"))
            pix_ids.append(getattr(tree, "PixID"))
            pos_xs.append(getattr(tree, "Position_X"))
            pos_ys.append(getattr(tree, "Position_Y"))
    File.Close()

    Edeps = np.array(Edeps)
    wtd_zs = np.array(wtd_zs)
    pix_ids = np.array(pix_ids)
    pos_xs = np.array(pos_xs)
    pos_ys = np.array(pos_ys)
    detxs, detys = batxy2detxy(pos_xs / 10.0, pos_ys / 10.0)
    detxs = np.round(detxs).astype(np.int64)
    detys = np.round(detys).astype(np.int64)

    return Edeps * 1e3, (wtd_zs - 29.87) / 10.0, pix_ids, detxs, detys


def get_stuff_from_Erun(dname):
    fnames = [
        os.path.join(dname, fname) for fname in os.listdir(dname) if ".root" in fname
    ]
    print(len(fnames))
    print(fnames)
    edeps = np.empty(0)
    zs = np.empty(0)
    pix_ids = np.empty(0, dtype=np.int64)
    detxs = np.empty(0, dtype=np.int64)
    detys = np.empty(0, dtype=np.int64)
    Ngammas = 0
    for i, fname in enumerate(fnames):
        print(i)
        Ngamma = int(fname.split("_")[-4])
        PrimaryE = float(fname.split("_")[-6])
        try:
            (
                edeps_,
                zs_,
                pix_ids_,
                detxs_,
                detys_,
            ) = get_Es_Zs_PixIDs_detxys_from_root_file(fname)
        except Exception as E:
            print(E)
            print("problem with,")
            print(fname)
            continue
        Ngammas += Ngamma
        edeps = np.append(edeps, edeps_)
        zs = np.append(zs, zs_)
        pix_ids = np.append(pix_ids, pix_ids_)
        detxs = np.append(detxs, detxs_)
        detys = np.append(detys, detys_)

    return PrimaryE, Ngammas, edeps, zs, pix_ids, detxs, detys


left_pixids = np.arange(8, dtype=np.int64)
right_pixids = np.arange(120, 128, dtype=np.int64)
bot_pixids = np.append(
    np.arange(0, 57, 8, dtype=np.int64), np.arange(71, 128, 8, dtype=np.int64)
)
top_pixids = np.append(
    np.arange(7, 64, 8, dtype=np.int64), np.arange(64, 121, 8, dtype=np.int64)
)
Ntot_dets = 32768
Nleft_dets = len(left_pixids) * 256
Nright_dets = len(right_pixids) * 256
Ntop_dets = len(top_pixids) * 256
Nbot_dets = len(bot_pixids) * 256
Nedge_dets = Nleft_dets + Nright_dets + Ntop_dets + Nbot_dets - 4 * 256
Nnot_edge_dets = Ntot_dets - Nedge_dets
Nrightmost_dets = len(right_pixids) * 16
Nbotmost_dets = len(bot_pixids) * 16

Ndets_per_sand = 128
Ndets_per_sand_no_edges = 84
Ndets_by_col = 16 * Ndets_per_sand
Ndets_by_row = 16 * Ndets_per_sand
Ndets_by_col_no_edges = 16 * Ndets_per_sand_no_edges
Ndets_by_row_no_edges = 16 * Ndets_per_sand_no_edges

detxs_by_sand0 = np.arange(0, 286 - 15, 18)
detxs_by_sand1 = detxs_by_sand0 + 15
detys_by_sand0 = np.arange(0, 173 - 7, 11)
detys_by_sand1 = detys_by_sand0 + 7

max_detx = 285
max_dety = 172

EK1_CD, EK1_TE = 23.172, 27.471
EK2_CD, EK2_TE = 26.084, 30.980


def get_dist_dicts(dname):
    NPrimary_Es = 50
    Primary_Es = []

    line_dicts = []
    comp_dicts = []

    for i in range(NPrimary_Es):
        direc = os.path.join(dname, "run_" + str(i))
        fnames = [fname for fname in os.listdir(direc) if "root" in fname]
        Edeps = np.empty(0)
        wtd_zs = np.empty(0)
        pix_ids = np.empty(0, dtype=np.int64)
        Ngammas_tot = 0

        for fname in fnames:
            PrimaryE = float(fname.split("_")[-6])
            Ngammas = int(fname.split("_")[-4])

            fname_ = os.path.join(direc, fname)
            try:
                Es, zs, pixids = get_Es_Zs_PixIDs_from_root_file(fname_)
            except Exception as E:
                print(E)
                print(("bad file: ", fname_))
                continue
            Edeps = np.append(Edeps, Es)
            wtd_zs = np.append(wtd_zs, zs)
            pix_ids = np.append(pix_ids, pixids)
            Ngammas_tot += Ngammas

        # if PrimaryE > 150.0:
        #     Ngammas_tot = 2147483647

        Primary_Es.append(PrimaryE)
        flux_sim_area = 750.0 * 750.0
        N_per_cm2 = Ngammas_tot / flux_sim_area

        bl_right = np.isin(pix_ids, right_pixids)
        bl_left = np.isin(pix_ids, left_pixids)
        bl_top = np.isin(pix_ids, top_pixids)
        bl_bot = np.isin(pix_ids, bot_pixids)
        edge_dets = bl_bot | bl_top | bl_right | bl_left

        print(np.shape(Edeps), np.shape(wtd_zs), np.shape(pix_ids))
        print(np.shape(edge_dets))

        bls = [~edge_dets, bl_right, bl_left, bl_top, bl_bot]
        Ndets_list = [Nnot_edge_dets, Nright_dets, Nleft_dets, Ntop_dets, Nbot_dets]
        orientation_names = ["NonEdges", "right", "left", "top", "bot"]

        line_names = ["TE", "CD", "PEAK"]
        line_ebins0 = np.array(
            [PrimaryE - EK1_TE - 0.1, PrimaryE - EK1_CD - 0.1, PrimaryE - 0.1]
        )
        line_ebins1 = np.array(
            [PrimaryE - EK1_TE + 0.1, PrimaryE - EK1_CD + 0.1, PrimaryE + 0.1]
        )

        comp_ebins0, comp_ebins1 = mk_comp_ebins(PrimaryE)

        line_dict = {}
        line_dict["Energy"] = PrimaryE
        comp_dict = {}
        comp_dict["Elow"] = comp_ebins0
        comp_dict["Ehi"] = comp_ebins1
        for ii in range(len(orientation_names)):
            bl = bls[ii]
            print(orientation_names[ii])
            print(np.sum(bl), np.shape(bl))
            print()
            ndets = Ndets_list[ii]
            zs = wtd_zs[bl]
            Es = Edeps[bl]
            k = orientation_names[ii] + "_comp_Depth_dE"
            comp_dict[k], zbins = get_depth_dE4comp(
                comp_ebins0,
                comp_ebins1,
                line_ebins0,
                line_ebins1,
                Es,
                zs,
                ndets,
                N_per_cm2,
                Nzbins=20,
            )
            #     k = orientation_names[i] + '_line_depths'
            line_depths = get_depth4lines(
                line_ebins0, line_ebins1, Es, zs, ndets, N_per_cm2, Nzbins=20
            )
            for j, depth in enumerate(line_depths):
                name = line_names[j] + "_" + orientation_names[ii]
                line_dict[name] = depth

        line_dicts.append(line_dict)
        comp_dicts.append(comp_dict)
    return line_dicts, comp_dicts, zbins, Primary_Es


def get_dist_dicts2(dname, theta, phi):
    NPrimary_Es = 50
    Primary_Es = []

    line_dicts = []
    comp_dicts = []

    for i in range(NPrimary_Es):
        direc = os.path.join(dname, "run_" + str(i))
        print(direc)
        PrimaryE, Ngammas, edeps, zs, pix_ids, detxs, detys = get_stuff_from_Erun(direc)

        Primary_Es.append(PrimaryE)
        flux_sim_area = 750.0 * 750.0
        N_per_cm2 = Ngammas / flux_sim_area

        bl_right = np.isin(pix_ids, right_pixids)
        bl_left = np.isin(pix_ids, left_pixids)
        bl_top = np.isin(pix_ids, top_pixids)
        bl_bot = np.isin(pix_ids, bot_pixids)
        edge_dets = bl_bot | bl_top | bl_right | bl_left
        bl_right_most = detxs == max_detx
        bl_bot_most = detys == 0

        orientation_names = ["NonEdges", "right", "left", "top", "bot"]
        bls = [~edge_dets, bl_right, bl_left, bl_top, bl_bot]
        Ndets_per_sand_ors = [Ndets_per_sand_no_edges, 8, 8, 16, 16]

        line_names = ["TE", "CD", "PEAK"]
        line_ebins0 = np.array(
            [PrimaryE - EK1_TE - 0.1, PrimaryE - EK1_CD - 0.1, PrimaryE - 0.1]
        )
        line_ebins1 = np.array(
            [PrimaryE - EK1_TE + 0.1, PrimaryE - EK1_CD + 0.1, PrimaryE + 0.1]
        )

        comp_ebins0, comp_ebins1 = mk_comp_ebins(PrimaryE)

        line_dict = {}
        line_dict["Energy"] = PrimaryE
        comp_dict = {}
        comp_dict["Elow"] = comp_ebins0
        comp_dict["Ehi"] = comp_ebins1

        if (theta > 70.0 and theta < 80.0) or (theta > 100.0 and theta < 110.0):
            if phi > 5.0:
                cols = [(0, 14), (14, 15), (15, 16), (0, 14), (0, 14)]
                rows = [(0, 14), (0, 16), (0, 16), (14, 15), (15, 16)]
                rows = [(2, 16), (0, 16), (0, 16), (0, 1), (1, 2)]
            else:
                cols = [(0, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16)]

        elif (theta > 80.0 and theta < 85.0) or (theta > 95.0 and theta < 100.0):
            if phi > 25.0:
                cols = [
                    (0, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 13),
                    (0, 13),
                    (0, 13),
                ]
                rows = [
                    (0, 13),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [(3, 16), (0, 16), (0, 16), (0, 16), (0, 1), (1, 2), (2, 3)]
            elif phi > 5.0:
                cols = [(0, 13), (13, 14), (14, 15), (15, 16), (0, 13), (0, 13)]
                rows = [(0, 14), (0, 16), (0, 16), (0, 16), (14, 15), (15, 16)]
                rows = [(2, 16), (0, 16), (0, 16), (0, 16), (0, 1), (1, 2)]
            else:
                cols = [(0, 13), (13, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16), (0, 16)]

        elif (theta > 85.0 and theta < 89.0) or (theta > 91.0 and theta < 95.0):
            if phi > 20.0:
                cols = [
                    (0, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                ]
                rows = [
                    (0, 11),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (5, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                ]

            elif phi > 5.0:
                cols = [
                    (0, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                ]
                rows = [
                    (0, 13),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (3, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                ]

            else:
                cols = [(0, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16), (0, 16), (0, 16), (0, 16)]

        elif theta >= 89.0 and theta < 91.0:
            if phi > 20.0:
                cols = [
                    (0, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                ]
                rows = [
                    (0, 8),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (8, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                ]

            elif phi > 5.0:
                cols = [
                    (0, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                    (0, 8),
                ]
                rows = [
                    (0, 12),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (4, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                ]

            else:
                cols = [
                    (0, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                ]

        elif theta >= 110.0:
            # do just edge sands
            # cols = [(0,1), (15,16), (0,16), (0,16)]
            # rows = [(0,16), (0,16), (0,1), (15,16)]
            cols = [(0, 1), (15, 16), (0, 16), (0, 16), (1, 15)]
            rows = [(0, 16), (0, 16), (0, 1), (15, 16), (1, 15)]

        Ncolrows = len(cols)

        for ii in range(Ncolrows):
            col0, col1 = cols[ii]
            row0, row1 = rows[ii]
            Ncols = col1 - col0
            Nrows = row1 - row0
            Nsands = Ncols * Nrows
            print("col0, col1: ", col0, col1)
            print("row0, row1: ", row0, row1)
            print("Nsands: ", Nsands)
            print()
            #     ndets = Nsands*Ndets_per_sand_no_edges
            #     print ndets
            #     bl_cols = np.zeros(len(bl), dtype=bool)
            bl_cols = (detxs >= detxs_by_sand0[col0]) & (
                detxs <= detxs_by_sand1[col1 - 1]
            )
            bl_rows = (detys >= detys_by_sand0[row0]) & (
                detys <= detys_by_sand1[row1 - 1]
            )
            bl_cr = bl_cols & bl_rows

            for j in range(len(orientation_names)):
                bl = bls[j] & bl_cr
                ndets = Ndets_per_sand_ors[j] * Nsands
                print(orientation_names[j])
                print("ndets: ", ndets)
                print(np.sum(bl))

                Zs = zs[bl]
                Es = edeps[bl]
                k = orientation_names[j] + "_cols_%d_%d_rows_%d_%d_comp_Depth_dE" % (
                    col0,
                    col1,
                    row0,
                    row1,
                )
                print(k)
                comp_dict[k], zbins = get_depth_dE4comp(
                    comp_ebins0,
                    comp_ebins1,
                    line_ebins0,
                    line_ebins1,
                    Es,
                    Zs,
                    ndets,
                    N_per_cm2,
                    Nzbins=20,
                )
                print()

        if theta > 70.0 and theta < 91.0:
            #             bls = [~edge_dets, bl_right&(~bl_right_most), bl_left,
            #                    bl_top, bl_bot&(~bl_bot_most), bl_right_most, bl_bot_most]
            #             Ndets_list = [Nnot_edge_dets, Nright_dets - Nrightmost_dets, Nleft_dets,
            #                           Ntop_dets, Nbot_dets - Nbotmost_dets, Nrightmost_dets, Nbotmost_dets]
            #             orientation_names = ['NonEdges', 'right', 'left', 'top', 'bot', 'rightmost', 'botmost']
            if phi > 5.0:
                cols = [(0, 15), (15, 16), (0, 15)]
                rows = [(0, 15), (0, 16), (15, 16)]
                rows = [(15, 16), (0, 16), (0, 1)]
            else:
                cols = [(0, 15), (15, 16)]
                rows = [(0, 16), (0, 16)]

        elif theta > 91.0 and theta < 95.0:
            if phi > 20.0:
                cols = [
                    (0, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                    (0, 11),
                ]
                rows = [
                    (0, 11),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (5, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                ]

            elif phi > 5.0:
                cols = [
                    (0, 11),
                    (11, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 11),
                    (0, 11),
                ]
                rows = [
                    (0, 14),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (2, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                ]

            else:
                cols = [(0, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16), (0, 16), (0, 16), (0, 16)]

        elif theta > 95.0 and theta < 100.0:
            if phi > 20.0:
                cols = [
                    (0, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 12),
                    (0, 12),
                    (0, 12),
                    (0, 12),
                ]
                rows = [
                    (0, 12),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [
                    (4, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                ]

            elif phi > 5.0:
                cols = [
                    (0, 12),
                    (12, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 12),
                    (0, 12),
                ]
                rows = [(0, 14), (0, 16), (0, 16), (0, 16), (0, 16), (14, 15), (15, 16)]
                rows = [(2, 16), (0, 16), (0, 16), (0, 16), (0, 16), (0, 1), (1, 2)]

            else:
                cols = [(0, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16), (0, 16), (0, 16)]

        elif theta > 100.0 and theta < 120.0:
            if phi > 20.0:
                cols = [
                    (0, 13),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                    (0, 13),
                    (0, 13),
                    (0, 13),
                ]
                rows = [
                    (0, 13),
                    (0, 16),
                    (0, 16),
                    (0, 16),
                    (13, 14),
                    (14, 15),
                    (15, 16),
                ]
                rows = [(3, 16), (0, 16), (0, 16), (0, 16), (0, 1), (1, 2), (2, 3)]
            elif phi > 5.0:
                cols = [(0, 13), (13, 14), (14, 15), (15, 16), (0, 13)]
                rows = [(0, 15), (0, 16), (0, 16), (0, 16), (15, 16)]
                rows = [(1, 16), (0, 16), (0, 16), (0, 16), (0, 1)]
            else:
                cols = [(0, 13), (13, 14), (14, 15), (15, 16)]
                rows = [(0, 16), (0, 16), (0, 16), (0, 16)]

        elif theta > 120.0 and theta < 150.0:
            if phi > 5.0:
                cols = [(0, 15), (15, 16), (0, 15)]
                rows = [(0, 15), (0, 16), (15, 16)]
                rows = [(1, 16), (0, 16), (0, 1)]
            else:
                cols = [(0, 15), (15, 16)]
                rows = [(0, 16), (0, 16)]

        elif theta >= 150.0:
            cols = [(0, 16)]
            rows = [(0, 16)]

        Ncolrows = len(cols)
        for ii in range(Ncolrows):
            col0, col1 = cols[ii]
            row0, row1 = rows[ii]
            Ncols = col1 - col0
            Nrows = row1 - row0
            Nsands = Ncols * Nrows
            print("col0, col1: ", col0, col1)
            print("row0, row1: ", row0, row1)
            print("Nsands: ", Nsands)
            print()
            #     ndets = Nsands*Ndets_per_sand_no_edges
            #     print ndets
            #     bl_cols = np.zeros(len(bl), dtype=bool)
            bl_cols = (detxs >= detxs_by_sand0[col0]) & (
                detxs <= detxs_by_sand1[col1 - 1]
            )
            bl_rows = (detys >= detys_by_sand0[row0]) & (
                detys <= detys_by_sand1[row1 - 1]
            )
            bl_cr = bl_cols & bl_rows

            for j in range(len(orientation_names)):
                bl = bls[j] & bl_cr
                ndets = Ndets_per_sand_ors[j] * Nsands
                print(orientation_names[j])
                print("ndets: ", ndets)
                print(np.sum(bl))

                Zs = zs[bl]
                Es = edeps[bl]
                k = orientation_names[j] + "_cols_%d_%d_rows_%d_%d" % (
                    col0,
                    col1,
                    row0,
                    row1,
                )
                print(k)

                line_depths = get_depth4lines(
                    line_ebins0, line_ebins1, Es, Zs, ndets, N_per_cm2, Nzbins=20
                )
                for j, depth in enumerate(line_depths):
                    name = line_names[j] + "_" + k
                    line_dict[name] = depth
        line_dicts.append(line_dict)
        comp_dicts.append(comp_dict)

    return line_dicts, comp_dicts, zbins, Primary_Es


def main(args):
    dname = args.dname

    theta = float(dname.split("_")[-3])
    phi = float(dname.split("_")[-1][:-1])

    if theta < 70.0:
        line_dicts, comp_dicts, zbins, Primary_Es = get_dist_dicts(dname)
    else:
        line_dicts, comp_dicts, zbins, Primary_Es = get_dist_dicts2(dname, theta, phi)

    lines_tab = Table(data=line_dicts)
    primary_hdu = fits.PrimaryHDU()
    lines_hdu = fits.table_to_hdu(lines_tab)
    hdu_list = [primary_hdu, lines_hdu]

    for i, comp_dict in enumerate(comp_dicts):
        comp_tab = Table(data=comp_dict)
        name = "PrimaryE_%.2f" % (Primary_Es[i])
        comp_hdu = fits.table_to_hdu(comp_tab)
        comp_hdu.name = name
        hdu_list.append(comp_hdu)

    zbins_tab = Table(data=[zbins[:-1], zbins[1:]], names=["Zlow", "Zhi"])
    zbin_hdu = fits.table_to_hdu(zbins_tab)
    zbin_hdu.name = "Zbins"
    hdu_list.append(zbin_hdu)

    save_fname = os.path.join(dname, "depth_dists.fits")
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(save_fname, overwrite=True)


if __name__ == "__main__":
    args = cli()

    main(args)
