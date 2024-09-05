import numpy as np
from numba import jit, njit, prange
from scipy import interpolate
import os
from astropy.io import fits
from astropy.table import Table, vstack
from math import erf
import multiprocessing as mp
import healpy as hp
import argparse

try:
    import ROOT
except ModuleNotFoundError as err:
    # Error handling
    print(err)
    print(
        "Please install the Python ROOT package to be able to run the full forward modeling calculations."
    )


from ..lib.event2dpi_funcs import det2dpis


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp_ind", type=int, help="hp_ind to use", default=0)
    parser.add_argument(
        "--theta",
        type=float,
        help="theta to use, instead of hp_ind, keep as None to use hp_ind",
        default=None,
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


detxs_by_sand0 = np.arange(0, 286 - 15, 18)
detxs_by_sand1 = detxs_by_sand0 + 15

detys_by_sand0 = np.arange(0, 173 - 7, 11)
detys_by_sand1 = detys_by_sand0 + 7


def dpi2sand_img(dpi):
    sand_img = np.zeros((16, 16))
    for i in range(16):
        x0 = detxs_by_sand0[i]
        x1 = detxs_by_sand1[i] + 1
        for j in range(16):
            y0 = detys_by_sand0[j]
            y1 = detys_by_sand1[j] + 1
            sand_img[j, i] += np.sum(dpi[y0:y1, x0:x1])
    return sand_img


def mk_sand_bl(detxs, detys, col_num, row_num):
    bl = (
        (detxs >= detxs_by_sand0[col_num])
        & (detxs <= detxs_by_sand1[col_num])
        & (detys >= detys_by_sand0[row_num])
        & (detys <= detys_by_sand1[row_num])
    )
    return bl


dpi_shape = (173, 286)


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


# cal_resp_fname = '/storage/work/jjd330/caldb_files/data/swift/bat/cpf/swbresponse20030101v007.rsp'
# resp_ebins_tab = Table.read(cal_resp_fname, hdu='EBOUNDS')
# print(resp_ebins_tab.colnames)

# params_fname = '/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbparams20030101v009.fits'
# mt_tab = Table.read(params_fname, hdu=2)
# params_tab = Table.read(params_fname)
# print(mt_tab.colnames)
# print(params_tab.colnames)

# params_header = fits.open(params_fname)[1].header

# psv = []
# for i in range(14):
#    psv.append(float(params_header['PSV_'+str(i)]))
# print(psv)

# depth_fname = '/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbdepthdis20030101v003.fits'
# dtab = Table.read(depth_fname)
# print(dtab.colnames)


# define MTFUNC_PARMS 3
# define N_MT   36
# define DET_THICKNESS  0.2
# define N_COEFFS 20
# define N_PEAKS 3
# define N_DEPTHS 1000
# define CD_EDGE 26.72
# define TE_EDGE 31.82
# define EK1_CD 23.172
# define EK1_TE 27.471
# define SUB_BINS 10

SUB_BINS = 10

CD_EDGE = 26.72
TE_EDGE = 31.82
EK1_CD = 23.172
EK1_TE = 27.471
N_DEPTHS = 1000
DET_THICKNESS = 0.2
dX = DET_THICKNESS / N_DEPTHS

EXP_REF = 31.0
EXP_CUTOFF = 0.01
EXP_LAMB = 0.276815564
# EXP_CFF = 0.0
EXP_CFF = 0.2197
EXP_IND = 1.956559051
NORM_ADJ = 1.069053483

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

GAIN_ = [400.0, -4.66162e-7, 2.19376e-4, 0.998148, -1.53554e-4, 1.07273]

SIGMA_ = [76.76, 0.0, 0.0, 1.34514, 0.0, 0.00917815, 0.640623]


def add_exp_tail(
    E, EXP_CUTOFF, exp_coeff, exp_index, emin_pre, emax_pre, dist_eff_area, norm, result
):
    exp_ratio = exp_coeff * np.power(max(E, EXP_REF) / EXP_REF, -exp_index)

    if exp_ratio > EXP_CUTOFF:
        peak_channel = 0
        while emax_pre[peak_channel] < E:
            peak_channel += 1

        norm_etail = (norm) * exp_ratio * dist_eff_area

        result *= 1.0 - exp_ratio

        temp_sum = 0
        for i in range(peak_channel):
            etail = np.exp((min(emax_pre[i], E) - E) / (E * EXP_LAMB)) - np.exp(
                (emin_pre[i] - E) / (E * EXP_LAMB)
            )
            temp_sum += norm_etail * etail
            result[i] += norm_etail * etail

    return result


def get_dist(cheb_coefs):
    cpoly = np.polynomial.chebyshev.Chebyshev(cheb_coefs, domain=(1e-4, 0.2 - 1e-4))
    x = np.linspace(0, 0.2, 1000 + 1)
    xax = (x[1:] + x[:-1]) / 2.0
    dx = x[1] - x[0]
    #     dist = np.exp(cpoly(x))
    dist = np.exp(cpoly(xax))
    return dist


def adjust_gain(E):
    if E < GAIN_[0]:
        gain_adjust = GAIN_[3] + GAIN_[2] * E + GAIN_[1] * E * E

    else:
        gain_adjust = GAIN_[5] + GAIN_[4] * E

    return gain_adjust


def get_sigma(E):
    if E < SIGMA_[0]:
        sigma = SIGMA_[3]

    else:
        sigma = SIGMA_[6] + SIGMA_[5] * E

    return sigma


@njit(cache=True, fastmath=True)
def hecht(lambda_e, lambda_h, depth):
    """
    Inputs:
        lambda_e: mean distance electrons travel in the detector (cm)
        lambda_h: mean distance holes travel in the detector (cm)
        depth: distance below the top surface of the detector (cm)

    Output:
        charge induction efficiency at that depth (dimensionless)
    """

    return (
        lambda_e * (1.0 - np.exp(-(DET_THICKNESS - depth) / lambda_e))
        + lambda_h * (1.0 - np.exp(-depth / lambda_h))
    ) / DET_THICKNESS


@njit(cache=True, fastmath=True, parallel=True)
def mutau_model(
    mutaue, mutauh, voltage, gain_adjust, zbins0, zbins1, n_bins, E, norm, emax, dist
):
    #     print(mutaue)
    #     print(voltage)
    #     print(DET_THICKNESS)
    #     print(mutaue*voltage/DET_THICKNESS)
    lambda_e = mutaue * voltage / DET_THICKNESS
    lambda_h = mutauh * voltage / DET_THICKNESS
    #     dx = DET_THICKNESS/n_depths
    dzs = zbins1 - zbins0
    zax = (zbins0 + zbins1) / 2.0
    #     print "dx: ", dx
    max_hecht_depth = lambda_h * DET_THICKNESS / (lambda_e + lambda_h)
    n_depths = len(zbins0)
    #     print "max_hecht_depth: ", max_hecht_depth

    result = np.zeros(n_bins)

    for i in prange(n_depths):
        #         depth = (i+0.5)*dx
        depth = DET_THICKNESS - zax[i]

        slice_eff_area = dist[i] * dzs[i]

        eff_energy = (
            E
            * gain_adjust
            * hecht(lambda_e, lambda_h, depth)
            / hecht(lambda_e, lambda_h, max_hecht_depth)
        )

        if eff_energy <= emax[n_bins - 1]:
            # find the bin (j) that corresponds to eff_energy
            j = 0
            while emax[j] < eff_energy:
                j += 1

            # add norm*slice_eff_area to the contents of that ph bin

            result[j] += norm * slice_eff_area

    return result


def pha_bins2pre_pha_bins(emins, emaxs, sub_bins=10):
    nphabins = len(emins)
    nphabins_pre = nphabins * sub_bins

    emins_pre = np.zeros(nphabins_pre)
    emaxs_pre = np.zeros(nphabins_pre)

    for i in range(nphabins):
        emin = emins[i]
        emax = emaxs[i]
        de = (emax - emin) / sub_bins
        for j in range(sub_bins):
            ind = i * sub_bins + j
            emins_pre[ind] = emin + j * de
            emaxs_pre[ind] = emins_pre[ind] + de

    return emins_pre, emaxs_pre


sqrt_half = 1.0 / np.sqrt(2.0)


@njit(cache=True)
def norm_cdf(x, sig):
    x_ = sqrt_half * x / sig
    cdf = 0.5 * (1.0 + erf(x_))
    return cdf


@njit(cache=True)
def gauss_conv(res_pre_gauss, emins, emaxs, emins_pre, emaxs_pre, sigma):
    Nphas_bins = len(emins)
    #     emins_pre, emaxs_pre = pha_bins2pre_pha_bins(emins, emaxs)
    ecents_pre = (emins_pre + emaxs_pre) / 2.0
    Npha_bins_pre = len(emins_pre)

    result = np.zeros(Nphas_bins)
    #     gauss = stats.norm(loc=0.0, scale=sigma)

    for i in range(Npha_bins_pre):
        ecent = ecents_pre[i]
        #         gauss = stats.norm(loc=ecent, scale=sigma)
        pre_res = res_pre_gauss[i]
        for j in range(Nphas_bins):
            gauss_prob = norm_cdf(emaxs[j] - ecent, sigma) - norm_cdf(
                emins[j] - ecent, sigma
            )
            #         gauss_probs = gauss.cdf(emaxs) - gauss.cdf(emins)
            result[j] += gauss_prob * pre_res

    return result


def multi_mutau_func(
    Es, nphabins, mt_tab, voltage, dist, zbins0, zbins1, pha_emins, pha_emaxs
):
    sigma = get_sigma(Es[-1])
    nphabins_pre = nphabins * SUB_BINS
    Ne = len(Es)

    result_pre_gauss = np.zeros(nphabins_pre)

    emin_pre, emax_pre = pha_bins2pre_pha_bins(pha_emins, pha_emaxs, sub_bins=SUB_BINS)

    #     dx = DET_THICKNESS/n_depths
    #     dist = dists[0]
    dzs = zbins1 - zbins0
    dist_eff_area = 0.0

    dist_tot = np.sum(dist * dzs)
    print(dist_tot)

    #     dE_max = E - comp_Enew(E, np.pi)
    #     comp_dE = 1.0
    #     dEs = np.arange(dE_max, 10.0, -comp_dE)[::-1]
    #     Ncomp_Es = len(dEs)

    for row in mt_tab:
        frac = row["fraction"]
        norm_this_mt = frac * NORM_ADJ
        #         print row
        #         print norm_this_mt
        for j, E in enumerate(Es):
            gain_adjust = adjust_gain(E)
            res_pre_gauss = mutau_model(
                row["mutau_e"],
                row["mutau_h"],
                voltage,
                gain_adjust,
                zbins0,
                zbins1,
                nphabins_pre,
                E,
                norm_this_mt,
                emax_pre,
                dist[j],
            )
            #         print np.sum(res_pre_gauss)
            result_pre_gauss += res_pre_gauss
    #         print np.sum(result_pre_gauss)

    #         if E > CD_EDGE:
    #             res_pre_gauss = mutau_model(row['mutau_e'], row['mutau_h'],\
    #                                             voltage, gain_adjust, n_depths,\
    #                                             nphabins_pre, E-EK1_CD, norm_this_mt,\
    #                                             emax_pre, dists[1])
    #             result_pre_gauss += res_pre_gauss

    #         if E > TE_EDGE:
    #             res_pre_gauss = mutau_model(row['mutau_e'], row['mutau_h'],\
    #                                             voltage, gain_adjust, n_depths,\
    #                                             nphabins_pre, E-EK1_TE, norm_this_mt,\
    #                                             emax_pre, dists[2])
    #             result_pre_gauss += res_pre_gauss

    #         for i in range(Ncomp_Es):

    #             res_comp = mutau_model(row['mutau_e'], row['mutau_h'],\
    #                                             voltage, gain_adjust, n_depths,\
    #                                             nphabins_pre, dEs[i], norm_this_mt,\
    #                                             emax_pre, dists[2])

    res_pre_gauss = result_pre_gauss

    result = gauss_conv(
        res_pre_gauss,
        pha_emins.astype(np.float64),
        pha_emaxs.astype(np.float64),
        emin_pre.astype(np.float64),
        emax_pre.astype(np.float64),
        sigma,
    )

    return result


def get_Es_Zs_from_root_file(fname):
    Edeps = []
    wtd_zs = []
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    runID = int(fname.split("_")[-2])
    File = ROOT.TFile.Open(fname, "READ")
    tree = File.Get("Crystal")
    PrimaryE = float(fname.split("_")[-6])
    Ngammas = int(fname.split("_")[-4])
    #     print(PrimaryE)
    #     print(Ngammas)
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        edep = getattr(tree, "sum_Edep")
        if edep > 0.0:
            Edeps.append(edep)
            wtd_zs.append(getattr(tree, "Ewtd_Z"))
    File.Close()

    Edeps = np.array(Edeps)
    wtd_zs = np.array(wtd_zs)

    return Edeps * 1e3, (wtd_zs - 29.87) / 10.0


def get_Es_Zs_Ngammas_PrimaryE_from_direc_root_files(dname):
    Edeps = np.empty(0)
    wtd_zs = np.empty(0)
    Ngammas = 0
    fnames = [
        os.path.join(dname, fname) for fname in os.listdir(dname) if ".root" in fname
    ]
    for fname in fnames:
        Ngs = int(fname.split("_")[-4])
        PrimaryE = float(fname.split("_")[-6])
        try:
            es, zs = get_Es_Zs_from_root_file(fname)
        except Exception as E:
            print(E)
            print(("messed up with file, ", fname))
            continue
        Edeps = np.append(Edeps, es)
        wtd_zs = np.append(wtd_zs, zs)
        Ngammas += Ngs
    return Edeps, wtd_zs, Ngammas, PrimaryE


def get_Es_Zs_PixIDs_detxys_from_root_file(fname):
    Edeps = []
    wtd_zs = []
    pix_ids = []
    pos_xs = []
    pos_ys = []
    try:
        File = ROOT.TFile.Open(fname, "READ")
        tree = File.Get("Crystal")
    except:
        return
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


def get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files(dname):
    Edeps = np.empty(0)
    wtd_zs = np.empty(0)
    detxs = np.empty(0, dtype=np.int64)
    detys = np.empty(0, dtype=np.int64)
    Ngammas = 0
    fnames = [
        os.path.join(dname, fname) for fname in os.listdir(dname) if ".root" in fname
    ]
    for fname in fnames:
        Ngs = int(fname.split("_")[-4])
        PrimaryE = float(fname.split("_")[-6])
        try:
            es, zs, pix_ids, detxs_, detys_ = get_Es_Zs_PixIDs_detxys_from_root_file(
                fname
            )
        except Exception as E:
            print(E)
            print(("messed up with file, ", fname))
            continue
        Edeps = np.append(Edeps, es)
        wtd_zs = np.append(wtd_zs, zs)
        detxs = np.append(detxs, detxs_)
        detys = np.append(detys, detys_)
        Ngammas += Ngs
    return Edeps, wtd_zs, detxs, detys, Ngammas, PrimaryE


def get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files_mp(dname, Nprocs=4):
    Edeps = np.empty(0)
    wtd_zs = np.empty(0)
    detxs = np.empty(0, dtype=np.int64)
    detys = np.empty(0, dtype=np.int64)
    Ngammas = 0
    fnames = [
        os.path.join(dname, fname) for fname in os.listdir(dname) if ".root" in fname
    ]
    p = mp.Pool(Nprocs)
    stuff_list = p.map(get_Es_Zs_PixIDs_detxys_from_root_file, fnames)
    p.close()
    p.join()
    print("pool closed")
    for i in range(len(fnames)):
        stuff = stuff_list[i]
        fname = fnames[i]
        if stuff is None:
            continue

        Ngs = int(fname.split("_")[-4])
        PrimaryE = float(fname.split("_")[-6])
        es, zs, pix_ids, detxs_, detys_ = stuff
        Edeps = np.append(Edeps, es)
        wtd_zs = np.append(wtd_zs, zs)
        detxs = np.append(detxs, detxs_)
        detys = np.append(detys, detys_)
        Ngammas += Ngs
    return Edeps, wtd_zs, detxs, detys, Ngammas, PrimaryE


def mk_zbins(Nzbins):
    zbins = np.linspace(0, 0.2, Nzbins + 1)
    zax = (zbins[1:] + zbins[:-1]) / 2.0
    dz = zbins[1] - zbins[0]
    zbins0_ = np.linspace(0.0, 0.02, Nzbins / 5 + 1)[:-1]
    zbins1_ = np.linspace(0.18, 0.2, Nzbins / 5 + 1)[1:]
    zbins = np.append(zbins0_, np.linspace(0.02, 0.18, Nzbins - 2 * len(zbins0_) + 1))
    zbins = np.append(zbins, zbins1_)
    zbins0 = zbins[:-1]
    zbins1 = zbins[1:]
    return zbins0, zbins1


def calc_resp_from_sim_by_sand(dname, pha_emins, pha_emaxs, Nzbins=20):
    #     Es, Zs, detxs, detys, Ngammas, PrimaryE = get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files(dname)
    (
        Es,
        Zs,
        detxs,
        detys,
        Ngammas,
        PrimaryE,
    ) = get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files_mp(dname)
    print("PrimaryE: ", PrimaryE)
    print("Tot Nevents: ", len(Es))

    bls = []
    Elines = [PrimaryE, PrimaryE - EK1_CD, PrimaryE - EK1_TE]
    for line in Elines:
        bls.append(((Es) >= (line - 0.05)) & ((Es) < (line + 0.05)))
    bl = bls[0]
    for bl_ in bls[1:]:
        bl = bl | bl_
    bl0 = ~bl
    print("Nevents without PhotoE: ", np.sum(bl0))

    params_fname = (
        "/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbparams20030101v009.fits"
    )
    mt_tab = Table.read(params_fname, hdu=2)
    print(mt_tab.colnames)

    Npha_bins = len(pha_emins)
    Nebins = int(PrimaryE - 9.0) + 2
    ebins = np.linspace(9, PrimaryE, Nebins)
    ebins += (ebins[1] - ebins[0]) / 2.0
    if PrimaryE >= 1e3:
        Nebins = int(PrimaryE / 8) + 1
        ebins = np.logspace(0.95, np.log10(PrimaryE), Nebins)
    elif PrimaryE > 450.0:
        Nebins = int(PrimaryE / 5) + 1
        ebins = np.logspace(0.95, np.log10(PrimaryE), Nebins)
    elif PrimaryE > 300.0:
        Nebins = int(PrimaryE / 4) + 1
        ebins = np.logspace(0.95, np.log10(PrimaryE), Nebins)
    elif PrimaryE > 200.0:
        Nebins = int(PrimaryE / 2) + 1
        ebins = np.logspace(0.95, np.log10(PrimaryE), Nebins)
    Eax = (ebins[1:] + ebins[:-1]) / 2.0
    print("Nebins: ", Nebins)
    print("dE: ", ebins[1] - ebins[0])

    flux_sim_area = 750.0 * 750.0
    N_per_cm2 = Ngammas / flux_sim_area

    Ndets_per_sand = 32768 / 16.0 / 16.0

    resps_by_sand = np.zeros((16, 16, Npha_bins))

    if np.sum(bl0) < (5e3):
        # just make resp for all sands together
        print("Only making one averaged resp")
        Ndets = 32768

        zbins0, zbins1 = mk_zbins(Nzbins)
        zbins = np.append(zbins0, zbins1[-1])

        h = np.histogram2d(Es[bl0], Zs[bl0], bins=[ebins, zbins])[0]

        dzs = zbins1 - zbins0
        dists = h / dzs / N_per_cm2 / Ndets

        res = multi_mutau_func(
            Eax, Npha_bins, mt_tab, 200.0, dists, zbins0, zbins1, pha_emins, pha_emaxs
        )

        resps_by_sand += res

        return resps_by_sand, PrimaryE

    for row_num in range(16):
        for col_num in range(16):
            sand_bl = mk_sand_bl(detxs, detys, col_num, row_num)
            bl = sand_bl & bl0
            print("col_num, row_num: ", col_num, row_num)
            print("Nevents: ", np.sum(bl))
            Ndets = 32768 / 16.0 / 16.0

            if np.sum(bl) < 400:
                row_num0 = max(0, row_num - 1)
                row_num1 = min(15, row_num + 1)
                col_num0 = max(0, col_num - 1)
                col_num1 = min(15, col_num + 1)
                bls = []
                for cnum in range(col_num0, col_num1 + 1):
                    for rnum in range(row_num0, row_num1 + 1):
                        bls.append(mk_sand_bl(detxs, detys, cnum, rnum))
                sand_bl = bls[0]
                for bl_ in bls[1:]:
                    sand_bl = sand_bl | bl_
                Nsands = (1 + col_num1 - col_num0) * (1 + row_num1 - row_num0)
                Ndets = Ndets_per_sand * Nsands
                bl = sand_bl & bl0
                print("Nsands: ", Nsands)
                print("Nevents: ", np.sum(bl))
            elif np.sum(bl) < 1e3:
                row_num0 = max(0, row_num - 1)
                row_num1 = min(15, row_num + 1)
                bls = []
                for rnum in range(row_num0, row_num1 + 1):
                    bls.append(mk_sand_bl(detxs, detys, col_num, rnum))
                sand_bl = bls[0]
                for bl_ in bls[1:]:
                    sand_bl = sand_bl | bl_
                Nsands = 1 + row_num1 - row_num0
                Ndets = Ndets_per_sand * Nsands
                bl = sand_bl & bl0
                print("Nsands: ", Nsands)
                print("Nevents: ", np.sum(bl))

            zbins0, zbins1 = mk_zbins(Nzbins)
            zbins = np.append(zbins0, zbins1[-1])

            h = np.histogram2d(Es[bl], Zs[bl], bins=[ebins, zbins])[0]

            dzs = zbins1 - zbins0
            dists = h / dzs / N_per_cm2 / Ndets

            res = multi_mutau_func(
                Eax,
                Npha_bins,
                mt_tab,
                200.0,
                dists,
                zbins0,
                zbins1,
                pha_emins,
                pha_emaxs,
            )
            resps_by_sand[row_num, col_num] = res
    #             resps_by_sand.append(res)

    return resps_by_sand, PrimaryE


pb_lines2use = np.array([73.03, 75.25, 84.75, 85.23])
ta_lines2use = np.array([56.41, 57.69, 65.11, 65.39, 67.17])
sn_lines2use = np.array([25.03, 25.25, 28.47])
pb_edge = 88.0
ta_edge = 67.4
sn_edge = 29.2


def calc_flor_resp_from_sim_by_sand(dname, pha_emins, pha_emaxs, Nzbins=20):
    #     Es, Zs, detxs, detys, Ngammas, PrimaryE = get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files(dname)
    (
        Es,
        Zs,
        detxs,
        detys,
        Ngammas,
        PrimaryE,
    ) = get_Es_Zs_detxys_Ngammas_PrimaryE_from_direc_root_files_mp(dname)
    print("PrimaryE: ", PrimaryE)
    print("Tot Nevents: ", len(Es))

    Elines2use = np.empty(0)
    if PrimaryE >= sn_edge:
        Elines2use = np.append(Elines2use, sn_lines2use)
    if PrimaryE >= ta_edge:
        Elines2use = np.append(Elines2use, ta_lines2use)
    if PrimaryE >= pb_edge:
        Elines2use = np.append(Elines2use, pb_lines2use)

    flux_sim_area = 750.0 * 750.0
    N_per_cm2 = Ngammas / flux_sim_area

    Ndets_per_sand = 32768 / 16.0 / 16.0

    resps_by_sand = np.zeros((16, 16, Npha_bins))

    NElines = len(Elines2use)

    if NElines < 1:
        return resps_by_sand, PrimaryE

    bls = []
    #     Elines = [PrimaryE, PrimaryE - EK1_CD, PrimaryE - EK1_TE]
    for line in Elines2use:
        bls.append(((Es) >= (line - 0.05)) & ((Es) < (line + 0.05)))
    bl = bls[0]
    for bl_ in bls[1:]:
        bl = bl | bl_
    bl0 = bl
    print("Nevents in Flor lines: ", np.sum(bl0))

    params_fname = (
        "/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbparams20030101v009.fits"
    )
    mt_tab = Table.read(params_fname, hdu=2)
    print(mt_tab.colnames)

    if np.sum(bl0) < (1e3):
        # just make resp for all sands together
        print("Only making one averaged resp")
        Ndets = 32768

        zbins0, zbins1 = mk_zbins(Nzbins)
        zbins = np.append(zbins0, zbins1[-1])
        Nzbins = len(zbins0)

        h = np.zeros((NElines, Nzbins))

        for j in range(NElines):
            ble = ((Es) >= (Elines2use[j] - 0.05)) & ((Es) < (Elines2use[j] + 0.05))
            h[j] += np.histogram(Zs[bl0 & ble], bins=zbins)[0]

        #         h=np.histogram2d(Es[bl0], Zs[bl0], bins=[ebins,zbins])[0]

        dzs = zbins1 - zbins0
        dists = h / dzs / N_per_cm2 / Ndets

        res = multi_mutau_func(
            Elines2use,
            Npha_bins,
            mt_tab,
            200.0,
            dists,
            zbins0,
            zbins1,
            pha_emins,
            pha_emaxs,
        )

        resps_by_sand += res

        return resps_by_sand, PrimaryE

    for row_num in range(16):
        for col_num in range(16):
            sand_bl = mk_sand_bl(detxs, detys, col_num, row_num)
            bl = sand_bl & bl0
            print("col_num, row_num: ", col_num, row_num)
            print("Nevents: ", np.sum(bl))
            Ndets = 32768 / 16.0 / 16.0

            if np.sum(bl) < 400:
                row_num0 = max(0, row_num - 1)
                row_num1 = min(15, row_num + 1)
                col_num0 = max(0, col_num - 1)
                col_num1 = min(15, col_num + 1)
                bls = []
                for cnum in range(col_num0, col_num1 + 1):
                    for rnum in range(row_num0, row_num1 + 1):
                        bls.append(mk_sand_bl(detxs, detys, cnum, rnum))
                sand_bl = bls[0]
                for bl_ in bls[1:]:
                    sand_bl = sand_bl | bl_
                Nsands = (1 + col_num1 - col_num0) * (1 + row_num1 - row_num0)
                Ndets = Ndets_per_sand * Nsands
                bl = sand_bl & bl0
                print("Nsands: ", Nsands)
                print("Nevents: ", np.sum(bl))
            elif np.sum(bl) < 1e3:
                row_num0 = max(0, row_num - 1)
                row_num1 = min(15, row_num + 1)
                bls = []
                for rnum in range(row_num0, row_num1 + 1):
                    bls.append(mk_sand_bl(detxs, detys, col_num, rnum))
                sand_bl = bls[0]
                for bl_ in bls[1:]:
                    sand_bl = sand_bl | bl_
                Nsands = 1 + row_num1 - row_num0
                Ndets = Ndets_per_sand * Nsands
                bl = sand_bl & bl0
                print("Nsands: ", Nsands)
                print("Nevents: ", np.sum(bl))

            zbins0, zbins1 = mk_zbins(Nzbins)
            zbins = np.append(zbins0, zbins1[-1])
            Nzbins = len(zbins0)

            h = np.zeros((NElines, Nzbins))

            for j in range(NElines):
                ble = ((Es) >= (Elines2use[j] - 0.05)) & ((Es) < (Elines2use[j] + 0.05))
                h[j] += np.histogram(Zs[bl & ble], bins=zbins)[0]

            #             h=np.histogram2d(Es[bl], Zs[bl], bins=[ebins,zbins])[0]

            dzs = zbins1 - zbins0
            dists = h / dzs / N_per_cm2 / Ndets

            res = multi_mutau_func(
                Elines2use,
                Npha_bins,
                mt_tab,
                200.0,
                dists,
                zbins0,
                zbins1,
                pha_emins,
                pha_emaxs,
            )
            resps_by_sand[row_num, col_num] = res
    #             resps_by_sand.append(res)

    return resps_by_sand, PrimaryE


def main(args):
    if args.theta is None:
        hp_ind = args.hp_ind
        dname = "/gpfs/scratch/jjd330/g4_runs/hp_ind_order_2_allEs/hp_ind_%d/" % (
            hp_ind
        )
    elif args.theta == 0:
        dname = "/gpfs/scratch/jjd330/g4_runs/hp_ind_order_2_allEs/theta_0/"
    elif args.theta == 180:
        dname = "/gpfs/scratch/jjd330/g4_runs/hp_ind_order_2_allEs/theta_180/"

    drm_fname = "/storage/work/jjd330/caldb_files/data/swift/bat/cpf/swbresponse20030101v007.rsp"
    ebins_tab = Table.read(drm_fname, hdu=2)
    print(ebins_tab.colnames)
    ebins_tab[-1]

    pha_emins = ebins_tab["E_MIN"]
    pha_emaxs = ebins_tab["E_MAX"]
    pha_emins = np.round(pha_emins.astype(np.float64)[:-1], decimals=1)
    pha_emaxs = np.round(pha_emaxs.astype(np.float64)[:-1], decimals=1)
    pha_extras = np.round(
        np.logspace(np.log10(194.9), np.log10(500.0), 24 + 1), decimals=1
    )
    pha_extras = np.append(pha_extras, [1e5])
    pha_emins = np.append(pha_emins, pha_extras[:-1])
    pha_emaxs = np.append(pha_emaxs, pha_extras[1:])
    Npha_bins = len(pha_emins)

    Primary_Es = []
    print(dname)
    Nprimary_Es = 60
    resps = []
    for i in range(Nprimary_Es):
        direc = os.path.join(dname, "run_%d" % (i))
        print()
        print("***************************************************")
        print(i)
        print(direc)
        res, PrimaryE = calc_flor_resp_from_sim_by_sand(direc, pha_emins, pha_emaxs)
        print(np.sum(res))
        resps.append(res)
        Primary_Es.append(PrimaryE)
        print()
        print("***************************************************")

    drm_tab = Table(data={"Ephoton": np.array(Primary_Es), "RESPONSE": np.array(resps)})

    ebounds_tab = Table(
        data=[np.arange(len(pha_emaxs), dtype=np.int64), pha_emins, pha_emaxs],
        names=["CHANNEL", "E_MIN", "E_MAX"],
        dtype=[np.int64, np.float64, np.float64],
    )

    primary_hdu = fits.PrimaryHDU()
    drm_hdu = fits.table_to_hdu(drm_tab)
    ebounds_hdu = fits.table_to_hdu(ebounds_tab)

    ebounds_hdu.name = "EBOUNDS"

    hdul = fits.HDUList([primary_hdu, drm_hdu, ebounds_hdu])

    if args.theta is None:
        hp_ind = args.hp_ind
        save_fname = (
            "/gpfs/scratch/jjd330/bat_data/hp_flor_resps/resp_by_sand_hpind_%d.fits"
            % (hp_ind)
        )
    elif args.theta == 0:
        save_fname = (
            "/gpfs/scratch/jjd330/bat_data/hp_flor_resps/resp_by_sand_theta_0.fits"
        )
    elif args.theta == 180:
        save_fname = (
            "/gpfs/scratch/jjd330/bat_data/comp_flor_resps/resp_by_sand_theta_180.fits"
        )

    print("save_fname: ")
    print(save_fname)
    hdul.writeto(save_fname)  # , overwrite=True)


if __name__ == "__main__":
    args = cli()

    main(args)
