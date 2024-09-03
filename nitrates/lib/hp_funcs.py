import numpy as np
import healpy as hp
from astropy.wcs import WCS
from astropy.io import fits
from ..lib.coord_conv_funcs import convert_imxy2radec, convert_radec2imxy
from ..lib.wcs_funcs import world2val
import logging, traceback
import pandas as pd
from scipy import interpolate


def ang_sep(ra0, dec0, ra1, dec1):
    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(
        np.cos(np.radians(90 - dec0)) * np.cos(np.radians(90 - dec1))
        + np.sin(np.radians(90 - dec0)) * np.sin(np.radians(90 - dec1)) * dcos
    )
    return np.rad2deg(angsep)


def probm2perc(pmap):
    bl = pmap > 0
    p_map = np.copy(pmap)
    #     p_map[~bl] = 1e5

    inds_sort = np.argsort(p_map)[::-1]

    perc_map = np.zeros_like(p_map)

    perc_map[inds_sort] = np.cumsum(p_map[inds_sort])  # *\
    #                hp.nside2pixarea(hp.npix2nside(len(p_map)))

    perc_map[~bl] = 1.0

    return perc_map


def pcfile2hpmap(pc_fname, att_row, Nside, nest=True):
    Npix = hp.nside2npix(Nside)
    pc_map = np.zeros(Npix)

    att_q = att_row["QPARAM"]
    pnt_ra, pnt_dec = att_row["POINTING"][:2]

    vec = hp.ang2vec(pnt_ra, pnt_dec, lonlat=True)
    hp_inds = hp.query_disc(Nside, vec, np.radians(70.0), nest=nest)

    hp_ras, hp_decs = hp.pix2ang(Nside, hp_inds, nest=nest, lonlat=True)

    hp_imxs, hp_imys = convert_radec2imxy(hp_ras, hp_decs, att_q)

    bl = (np.abs(hp_imys) < 1.01) & (np.abs(hp_imxs) < 2.0)

    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key="T")
    pc = PC.data

    pc_vals = world2val(w_t, pc, hp_imxs[bl], hp_imys[bl])

    pc_map[hp_inds[bl]] = pc_vals

    return pc_map


def pc_probmap2good_outFoVmap_inds(
    pc_fname,
    sk_fname,
    att_tab,
    trig_time,
    pc_max=0.05,
    gw_perc_max=0.995,
    Nside_out=2**4,
):
    att_ind = np.argmin(np.abs(att_tab["TIME"] - trig_time))
    att_row = att_tab[att_ind]

    try:
        pc_map = pcfile2hpmap(pc_fname, att_row, Nside_out)
    except Exception as E:
        logging.error(E)
        logging.warn("Couldn't make PC map")
        pc_map = np.zeros(hp.nside2npix(Nside_out))

    try:
        sky_map = hp.read_map(sk_fname, field=(0,), nest=True)
        Nside = hp.npix2nside(len(sky_map))
        perc_map = probm2perc(sky_map)

        perc_map2 = hp.ud_grade(
            perc_map, nside_out=Nside_out, order_in="NESTED", order_out="NESTED"
        )
    except Exception as E:
        logging.error(E)
        logging.warn("Couldn't use skymap")
        perc_map2 = np.zeros_like(pc_map)

    good_map = (perc_map2 <= gw_perc_max) & (pc_map <= pc_max)

    good_hp_inds = np.where(good_map)[0]

    return good_map, good_hp_inds


def pc_gwmap2good_pix(
    pc_fname, sk_fname, att_tab, trig_time, pc_min=0.1, gw_perc_max=0.99
):
    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key="T")
    pc = PC.data
    pcbl = pc >= pc_min
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    sky_map = hp.read_map(sk_fname, field=(0,), nest=True)
    Nside = hp.npix2nside(len(sky_map))
    perc_map = probm2perc(sky_map)

    pnt_ra, pnt_dec = att_tab["POINTING"][
        np.argmin(np.abs(trig_time - att_tab["TIME"])), :2
    ]
    att_q = att_tab["QPARAM"][np.argmin(np.abs(att_tab["TIME"] - trig_time))]

    pc_ras, pc_decs = convert_imxy2radec(pc_imxs, pc_imys, att_q)

    pc_hp_inds = hp.ang2pix(Nside, pc_ras, pc_decs, nest=True, lonlat=True)
    pc_gw_percs = perc_map[pc_hp_inds]
    pc_gw_bl = pc_gw_percs < gw_perc_max

    good_imxs = pc_imxs[pc_gw_bl]
    good_imys = pc_imys[pc_gw_bl]
    good_ras = pc_ras[pc_gw_bl]
    good_decs = pc_decs[pc_gw_bl]

    dtp = np.dtype(
        [("imx", np.float64), ("imy", np.float64), ("ra", np.float64), ("dec", np.float64)]
    )

    pix_arr = np.empty(np.sum(pc_gw_bl), dtype=dtp)
    pix_arr["imx"] = pc_imxs[pc_gw_bl]
    pix_arr["imy"] = pc_imys[pc_gw_bl]
    pix_arr["ra"] = pc_ras[pc_gw_bl]
    pix_arr["dec"] = pc_decs[pc_gw_bl]

    return pix_arr


def err_circle2prob_map(ra, dec, err_rad, Nside=None, sys_err=5.0):
    if Nside is None:
        if err_rad >= 1.0:
            Nside = 2**8
        # elif err_rad >= 0.1:
        # Nside = 2**9
        else:
            Nside = 2**9  # 2**10
    m = np.zeros(hp.nside2npix(Nside))
    vec = hp.ang2vec(ra, dec, lonlat=True)
    if err_rad < 1e-2:
        err_rad = 1e-2
    sig = np.sqrt(err_rad**2 + sys_err**2)
    r0s = np.radians(err_rad) * np.array([0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.0, 7.25, 7.5])
    for i, r0 in enumerate(r0s):
        if r0 <= np.radians(160.0):
            pix = hp.query_disc(Nside, vec, r0, nest=True)
            m[pix] += 1.0
        else:
            i -= 1
            break
    pix = hp.query_disc(Nside, vec, 1.05 * r0s[i], nest=True)
    m[pix] += 0.1

    m /= m.sum()
    return m


def get_dlogl_skymap(res_peak_tab, res_in_tab, res_out_tab, timeID, att_q, pc_map):
    

    bl = np.isclose(res_peak_tab['timeID'],timeID)
    bl0 = np.isclose(res_in_tab['timeID'],timeID)
    bl_out = np.isclose(res_out_tab['timeID'],timeID)

    res_all_in = pd.concat([res_peak_tab[bl], res_in_tab[bl0]])
    idx = res_all_in.groupby(['imx','imy'])['TS'].transform(max) == res_all_in['TS']
    tab_imxy = res_all_in[idx]
    
    tab_imxy['ra'], tab_imxy['dec'] = convert_imxy2radec(tab_imxy['imx'], tab_imxy['imy'], att_q)
    
    min_nllh = np.min(tab_imxy['nllh'])
    
    Nside = 2**4
    res_out_tab['ra'], res_out_tab['dec'] = hp.pix2ang(Nside, res_out_tab['hp_ind'], nest=True, lonlat=True)

    idx = res_out_tab[bl_out].groupby(['hp_ind'])['TS'].transform(max) == res_out_tab[bl_out]['TS']
    res_hpmax_tab = res_out_tab[bl_out][idx]
    
    diff_nllh_out = np.max(res_hpmax_tab['nllh'][np.isfinite(res_hpmax_tab['nllh'])]) - np.min(res_hpmax_tab['nllh'])
    print('diff nllh out: ', diff_nllh_out)
    
    min_nllh = min(min_nllh, np.min(res_hpmax_tab['nllh']))
    
    ras, decs = tab_imxy['ra'].values, tab_imxy['dec'].values
    nllhs = tab_imxy['nllh'].values
    ras = np.append(ras, res_hpmax_tab['ra'].values)
    decs = np.append(decs, res_hpmax_tab['dec'].values)
    nllhs = np.append(nllhs, res_hpmax_tab['nllh'].values)
    
    bl = (ras>310)
    ras = np.append(ras, ras[bl] - 360.0)
    decs = np.append(decs, decs[bl])
    nllhs = np.append(nllhs, nllhs[bl])

    bl = (ras<50)
    ras = np.append(ras, ras[bl] + 360.0)
    decs = np.append(decs, decs[bl])
    nllhs = np.append(nllhs, nllhs[bl])

    max_dec = np.max(decs)

    dec_add = 2*(90.0 - max_dec)

    bl = (decs > (max_dec - 0.1))

    ras = np.append(ras, ras[bl])
    decs = np.append(decs, decs[bl] + dec_add)
    nllhs = np.append(nllhs, nllhs[bl])

    bl = (decs < -(max_dec - 0.1))

    ras = np.append(ras, ras[bl])
    decs = np.append(decs, decs[bl] - dec_add)
    nllhs = np.append(nllhs, nllhs[bl])
    
    pnts = np.array([ras, decs]).T
    interp = interpolate.LinearNDInterpolator(pnts, nllhs)
    
    nside = 2**11
    ra_m, dec_m = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int64), lonlat=True, nest=True)
    nllhs0 = np.zeros_like(ra_m)
    nllhs0 = interp(ra_m, dec_m)
    logging.debug('max(nllhs0) = %.3f'%(np.nanmax(nllhs0)))
    logging.debug('min(nllhs0) = %.3f'%(np.nanmin(nllhs0)))
    logging.debug('len(nllhs0) = %d'%(len(nllhs0)))
    logging.debug('sum(np.isnan(nllhs0)) = %d'%(np.sum(np.isnan(nllhs0))))

    return nllhs0