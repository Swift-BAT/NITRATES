import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from numba import jit, njit, prange
from scipy import interpolate
from math import erf
import argparse


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", type=str, help="depth dist file name", default=None)
    args = parser.parse_args()
    return args


# cal_resp_fname = '/storage/work/jjd330/caldb_files/data/swift/bat/cpf/swbresponse20030101v007.rsp'
# resp_ebins_tab = Table.read(cal_resp_fname, hdu='EBOUNDS')
# print(resp_ebins_tab.colnames)

# params_fname = '/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbparams20030101v009.fits'
# mt_tab = Table.read(params_fname, hdu=2)
##params_tab = Table.read(params_fname)
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


# pha_emins = resp_ebins_tab['E_MIN']
# pha_emaxs = resp_ebins_tab['E_MAX']
# pha_emins = np.round(pha_emins.astype(np.float64)[:-1], decimals=1)
# pha_emaxs = np.round(pha_emaxs.astype(np.float64)[:-1], decimals=1)
# pha_extras = np.round(np.logspace(np.log10(194.9), np.log10(500.0), 24+1), decimals=1)
# pha_extras = np.append(pha_extras, [1e5])
# pha_emins = np.append(pha_emins, pha_extras[:-1])
# pha_emaxs = np.append(pha_emaxs, pha_extras[1:])
# Npha_bins = len(pha_emins)
# print(Npha_bins)
#
# Ephotons = np.linspace(10.0, 100.0, 90+1)[:-1] + 0.5
# Ephotons = np.append(Ephotons, np.linspace(100.5, 200.5, 50+1)[:-1])
# Ephotons = np.append(Ephotons, np.logspace(np.log10(200.5), 2.75, 40+1))
# Ephotons = np.append(Ephotons, [600.0, 700.0, 900.0, 1.5e3, 3e3, 6e3])


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


"""
Funcs outside the DRMgen stuff
"""


def get_comp_depths(comp_depth_tab, PrimaryE, comp_dEax, colname):
    #     comp_Eax = (comp_Ebins[:-1]+comp_Ebins[1:])/2.
    #     comp_dEax = PrimaryE - comp_Eax
    comp_Ebin_width = comp_dEax[0] - comp_dEax[1]
    Nebins = len(comp_dEax)
    Nz = len(comp_depth_tab[colname][0])
    #     print Nebins
    #     print Nz

    #     tab_Eax = (comp_depth_tab[colname])

    comp_emids = (comp_depth_tab["Ehi"] + comp_depth_tab["Elow"]) / 2.0
    comp_dEmids = PrimaryE - comp_emids

    depths = np.zeros((Nebins, Nz))

    for i in range(Nebins):
        ind0 = np.digitize(comp_dEax[i], comp_dEmids) - 1
        if ind0 < 0:
            depths[i] += comp_depth_tab[colname][0] * comp_Ebin_width
        elif ind0 >= (len(comp_dEmids) - 1):
            depths[i] += comp_depth_tab[colname][-1] * comp_Ebin_width
        else:
            ind1 = ind0 + 1
            dE = comp_dEmids[ind1] - comp_dEmids[ind0]
            wt0 = (comp_dEmids[ind1] - comp_dEax[i]) / dE
            wt1 = (comp_dEax[i] - comp_dEmids[ind0]) / dE
            depths[i] += wt0 * comp_depth_tab[colname][ind0] * comp_Ebin_width
            depths[i] += wt1 * comp_depth_tab[colname][ind1] * comp_Ebin_width

    return depths


def get_col_row_strs(cnames):
    col_row_strs = []
    for cname in cnames:
        if cname in ["Ehi", "Elow", "Energy"]:
            continue
        cname_list = cname.split("_")
        try:
            if "comp" in cname_list:
                col0 = int(cname_list[-8])
                col1 = int(cname_list[-7])
                row0 = int(cname_list[-5])
                row1 = int(cname_list[-4])
            else:
                col0 = int(cname_list[-5])
                col1 = int(cname_list[-4])
                row0 = int(cname_list[-2])
                row1 = int(cname_list[-1])
        except Exception as E:
            col0, col1, row0, row1 = 0, 16, 0, 16
        cr_str = "cols_%d_%d_rows_%d_%d" % (col0, col1, row0, row1)
        col_row_strs.append(cr_str)
    col_row_strs = set(col_row_strs)
    #     print len(col_row_strs)
    return col_row_strs


def get_resp_dicts(depth_file, Ephotons, pha_emins, pha_emaxs):
    depth_tab = depth_file[1].data
    PrimaryEs = depth_tab["Energy"]

    ztab = depth_file[-1].data
    z_lows = ztab["Zlow"]
    z_highs = ztab["Zhi"]
    Nzbins = len(z_lows)
    dzs = z_highs - z_lows

    Npha_bins = len(pha_emins)

    line_col_row_strs = get_col_row_strs(depth_tab.columns.names)
    comp_col_row_strs = get_col_row_strs(depth_file[2].data.columns.names)

    Elows = [10.0]
    Ehis = []
    for i in range(len(Ephotons)):
        Ehis.append(Ephotons[i] + (Ephotons[i] - Elows[i]))
        if i < len(Ephotons) - 1:
            Elows.append(Ehis[i])

    params_fname = (
        "/storage/work/jjd330/caldb_files/data/swift/bat/bcf/swbparams20030101v009.fits"
    )
    mt_tab = Table.read(params_fname, hdu=2)
    print(mt_tab.colnames)

    res_dicts = []
    orientation_names = ["NonEdges", "right", "left", "top", "bot"]

    for ii, Ephoton in enumerate(Ephotons):
        res_dict = {}
        res_dict["ENERG_LO"] = Elows[ii]
        res_dict["ENERG_HI"] = Ehis[ii]

        Primary_ind0 = np.digitize(Ephoton, PrimaryEs) - 1
        Primary_ind1 = Primary_ind0 + 1
        print(PrimaryEs[Primary_ind0], PrimaryEs[Primary_ind1])
        dE = PrimaryEs[Primary_ind1] - PrimaryEs[Primary_ind0]
        wt0 = (PrimaryEs[Primary_ind1] - Ephoton) / dE
        wt1 = (Ephoton - PrimaryEs[Primary_ind0]) / dE
        print(wt0, wt1)

        comp_Ebins = np.linspace(10.0, Ephoton, int(Ephoton - 10.0) + 1)
        comp_Eax = (comp_Ebins[:-1] + comp_Ebins[1:]) / 2.0
        comp_dEax = Ephoton - comp_Eax

        for col_row in line_col_row_strs:
            cname_list = col_row.split("_")
            col0 = int(cname_list[-5])
            col1 = int(cname_list[-4])
            row0 = int(cname_list[-2])
            row1 = int(cname_list[-1])

            for oname in orientation_names:
                depth_list = []
                Es = []
                cname = oname + "_" + col_row
                peak_depth = np.zeros(Nzbins)
                cd_depth = np.zeros(Nzbins)
                te_depth = np.zeros(Nzbins)

                for ei, wt in zip([Primary_ind0, Primary_ind1], [wt0, wt1]):
                    try:
                        peak_depth += (
                            wt * depth_tab["PEAK_" + oname + "_" + col_row][ei]
                        )
                        cd_depth += wt * depth_tab["CD_" + oname + "_" + col_row][ei]
                        te_depth += wt * depth_tab["TE_" + oname + "_" + col_row][ei]
                    except:
                        peak_depth += wt * depth_tab["PEAK_" + oname][ei]
                        cd_depth += wt * depth_tab["CD_" + oname][ei]
                        te_depth += wt * depth_tab["TE_" + oname][ei]

                Es = [Ephoton, Ephoton - EK1_CD, Ephoton - EK1_TE]
                depth_list = [peak_depth, cd_depth, te_depth]

                lines_res = multi_mutau_func(
                    Es,
                    Npha_bins,
                    mt_tab,
                    200.0,
                    depth_list,
                    z_lows.astype(np.float64),
                    z_highs.astype(np.float64),
                    pha_emins,
                    pha_emaxs,
                )
                res_dict[cname] = lines_res

        for col_row in comp_col_row_strs:
            cname_list = col_row.split("_")
            col0 = int(cname_list[-5])
            col1 = int(cname_list[-4])
            row0 = int(cname_list[-2])
            row1 = int(cname_list[-1])

            for oname in orientation_names:
                cname = oname + "_" + col_row
                depths = np.zeros((len(comp_dEax), Nzbins))
                colname = cname + "_comp_Depth_dE"

                if len(comp_Eax) < 2:
                    res_dict[cname + "_comp"] = np.zeros(Npha_bins)
                    continue

                for ei, wt in zip([Primary_ind0, Primary_ind1], [wt0, wt1]):
                    PrimaryE = PrimaryEs[ei]
                    print(PrimaryE)
                    print(depth_file[ei + 2].name)
                    comp_tab = depth_file[ei + 2].data
                    if not colname in comp_tab.columns.names:
                        colname = oname + "_comp_Depth_dE"
                    comp_emids = (comp_tab["Ehi"] + comp_tab["Elow"]) / 2.0
                    comp_dEmids = PrimaryE - comp_emids
                    comp_ebin_widths = comp_tab["Ehi"] - comp_tab["Elow"]
                    #                 print len(comp_tab)
                    #                 print comp_tab.columns.names

                    try:
                        depths_ = get_comp_depths(
                            comp_tab, PrimaryE, comp_dEax, colname
                        )
                        depths += wt * depths_
                    except Exception as E:
                        print(E)
                        print("trouble with depth from")
                        print("ind: ", ei)
                        print(colname)

                comp_res = multi_mutau_func(
                    comp_Eax,
                    Npha_bins,
                    mt_tab,
                    200.0,
                    depths,
                    z_lows.astype(np.float64),
                    z_highs.astype(np.float64),
                    pha_emins,
                    pha_emaxs,
                )
                res_dict[cname + "_comp"] = comp_res

        res_dicts.append(res_dict)
        print("********************************")
        print("done with Energy %.3f" % (Ephoton))
        print("********************************")
        print()

    return res_dicts


def main(args):
    depth_fname = args.fname
    depth_file = fits.open(depth_fname)

    theta = float(depth_fname.split("_")[-4])
    phi = float(depth_fname.split("_")[-2])

    cal_resp_fname = "/storage/work/jjd330/caldb_files/data/swift/bat/cpf/swbresponse20030101v007.rsp"
    resp_ebins_tab = Table.read(cal_resp_fname, hdu="EBOUNDS")
    print(resp_ebins_tab.colnames)

    pha_emins = resp_ebins_tab["E_MIN"]
    pha_emaxs = resp_ebins_tab["E_MAX"]
    pha_emins = np.round(pha_emins.astype(np.float64)[:-1], decimals=1)
    pha_emaxs = np.round(pha_emaxs.astype(np.float64)[:-1], decimals=1)
    pha_extras = np.round(
        np.logspace(np.log10(194.9), np.log10(500.0), 24 + 1), decimals=1
    )
    pha_extras = np.append(pha_extras, [1e5])
    pha_emins = np.append(pha_emins, pha_extras[:-1])
    pha_emaxs = np.append(pha_emaxs, pha_extras[1:])
    Npha_bins = len(pha_emins)
    print(Npha_bins)

    Ephotons = np.linspace(10.0, 100.0, 90 + 1)[:-1] + 0.5
    Ephotons = np.append(Ephotons, np.linspace(100.5, 200.5, 50 + 1)[:-1])
    Ephotons = np.append(Ephotons, np.logspace(np.log10(200.5), 2.75, 40 + 1))
    Ephotons = np.append(Ephotons, [600.0, 700.0, 900.0, 1.5e3, 3e3, 6e3])

    res_dicts = get_resp_dicts(depth_file, Ephotons, pha_emins, pha_emaxs)

    drm_tab = Table(data=res_dicts)

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

    save_dname = "/storage/work/jjd330/local/bat_data/resp_tabs/"
    fname = "drm_theta_%.1f_phi_%.1f_.fits" % (theta, phi)
    save_fname = os.path.join(save_dname, fname)

    hdul.writeto(save_fname, overwrite=True)


if __name__ == "__main__":
    args = cli()

    main(args)
