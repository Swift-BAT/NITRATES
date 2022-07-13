import numpy as np
import healpy as hp
from astropy.wcs import WCS
from astropy.io import fits
from ..lib.coord_conv_funcs import convert_imxy2radec, convert_radec2imxy
from ..lib.wcs_funcs import world2val
import logging, traceback

def ang_sep(ra0, dec0, ra1, dec1):

    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(np.cos(np.radians(90-dec0))*np.cos(np.radians(90-dec1)) +\
                        np.sin(np.radians(90-dec0))*np.sin(np.radians(90-dec1))*dcos)
    return np.rad2deg(angsep)

def probm2perc(pmap):

    bl = (pmap>0)
    p_map = np.copy(pmap)
#     p_map[~bl] = 1e5

    inds_sort = np.argsort(p_map)[::-1]

    perc_map = np.zeros_like(p_map)

    perc_map[inds_sort] = np.cumsum(p_map[inds_sort])#*\
    #                hp.nside2pixarea(hp.npix2nside(len(p_map)))

    perc_map[~bl] = 1.

    return perc_map


def pcfile2hpmap(pc_fname, att_row, Nside, nest=True):

    Npix = hp.nside2npix(Nside)
    pc_map = np.zeros(Npix)

    att_q = att_row['QPARAM']
    pnt_ra, pnt_dec = att_row['POINTING'][:2]

    vec = hp.ang2vec(pnt_ra, pnt_dec, lonlat=True)
    hp_inds = hp.query_disc(Nside, vec, np.radians(70.0), nest=nest)

    hp_ras, hp_decs = hp.pix2ang(Nside, hp_inds, nest=nest, lonlat=True)

    hp_imxs, hp_imys = convert_radec2imxy(hp_ras, hp_decs, att_q)

    bl = (np.abs(hp_imys)<1.01)&(np.abs(hp_imxs)<2.0)

    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key='T')
    pc = PC.data

    pc_vals = world2val(w_t, pc, hp_imxs[bl], hp_imys[bl])

    pc_map[hp_inds[bl]] = pc_vals

    return pc_map


def pc_probmap2good_outFoVmap_inds(pc_fname, sk_fname, att_tab, trig_time,\
                              pc_max=0.05, gw_perc_max=.995, Nside_out=2**4):

    att_ind = np.argmin(np.abs(att_tab['TIME']-trig_time))
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

        perc_map2 = hp.ud_grade(perc_map, nside_out=Nside_out,\
                                order_in='NESTED', order_out='NESTED')
    except Exception as E:
        logging.error(E)
        logging.warn("Couldn't use skymap")
        perc_map2 = np.zeros_like(pc_map)

    good_map = (perc_map2<=gw_perc_max)&(pc_map<=pc_max)

    good_hp_inds = np.where(good_map)[0]

    return good_map, good_hp_inds


def pc_gwmap2good_pix(pc_fname, sk_fname, att_tab, trig_time,\
                      pc_min=0.1, gw_perc_max=.99):

    PC = fits.open(pc_fname)[0]
    w_t = WCS(PC.header, key='T')
    pc = PC.data
    pcbl = (pc>=pc_min)
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    sky_map = hp.read_map(sk_fname, field=(0,), nest=True)
    Nside = hp.npix2nside(len(sky_map))
    perc_map = probm2perc(sky_map)


    pnt_ra, pnt_dec = att_tab['POINTING'][np.argmin(np.abs(trig_time-\
                                                    att_tab['TIME'])),:2]
    att_q = att_tab['QPARAM'][np.argmin(np.abs(att_tab['TIME']-trig_time))]

    pc_ras, pc_decs = convert_imxy2radec(pc_imxs, pc_imys, att_q)

    pc_hp_inds = hp.ang2pix(Nside, pc_ras, pc_decs, nest=True, lonlat=True)
    pc_gw_percs = perc_map[pc_hp_inds]
    pc_gw_bl = (pc_gw_percs<gw_perc_max)

    good_imxs = pc_imxs[pc_gw_bl]
    good_imys = pc_imys[pc_gw_bl]
    good_ras = pc_ras[pc_gw_bl]
    good_decs = pc_decs[pc_gw_bl]

    dtp = np.dtype([('imx', np.float), ('imy', np.float),
                    ('ra', np.float), ('dec', np.float)])

    pix_arr = np.empty(np.sum(pc_gw_bl), dtype=dtp)
    pix_arr['imx'] = pc_imxs[pc_gw_bl]
    pix_arr['imy'] = pc_imys[pc_gw_bl]
    pix_arr['ra'] = pc_ras[pc_gw_bl]
    pix_arr['dec'] = pc_decs[pc_gw_bl]

    return pix_arr


def err_circle2prob_map(ra, dec, err_rad, Nside=None, sys_err=5.0):

    if Nside is None:
        if err_rad >= 1.0:
            Nside = 2**8
        # elif err_rad >= 0.1:
            # Nside = 2**9
        else:
            Nside = 2**9 #2**10
    m = np.zeros(hp.nside2npix(Nside))
    vec = hp.ang2vec(ra, dec, lonlat=True)
    if err_rad < 1e-2:
        err_rad = 1e-2
    sig = np.sqrt(err_rad**2 + sys_err**2)
    r0s = np.radians(err_rad)*np.array([0.5, 1., 2., 3., 4.5, 6.0, 7.0, 7.25, 7.5])
    for i, r0 in enumerate(r0s):
        if r0 <= np.radians(160.0):
            pix = hp.query_disc(Nside, vec, r0, nest=True)
            m[pix] += 1.0
        else:
            i -= 1
            break
    pix = hp.query_disc(Nside, vec, 1.05*r0s[i], nest=True)
    m[pix] += 0.1

    m /= m.sum()
    return m
