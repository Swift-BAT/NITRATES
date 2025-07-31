import numpy as np
import healpy as hp
from scipy.special import logsumexp
import mhealpy as mhp
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
import logging, traceback
from astropy.wcs import WCS

from ..lib.coord_conv_funcs import (
    convert_imxy2radec,
    convert_radec2thetaphi,
    convert_theta_phi2radec,
    imxy2theta_phi,
    convert_radec2imxy,
)
from .hp_funcs import pcfile2hpmap
from .wcs_funcs import world2val


def get_prob_map(nllhs0, att_q, pc_map, ifov_fact=1.0, ofov_fact=0.5, dllh_out=0.0, infov_smooth=None):
    
    nside = 2**11
    ra_m, dec_m = hp.pix2ang(nside, np.arange(hp.nside2npix(nside), dtype=np.int64), lonlat=True, nest=True)
    
    bl_in = (pc_map>1e-3)
    bl_out = ~bl_in
    
    bl_good = np.isfinite(nllhs0)
    diff_nllh_out = np.max(nllhs0[bl_out&bl_good]) - np.min(nllhs0[bl_out&bl_good])
    print('diff nllh out: ', diff_nllh_out)
    
    logging.debug('len(ra_m) = %d'%(len(ra_m)))
    
    theta_m, phi_m = convert_radec2thetaphi(ra_m, dec_m, att_q)

    logging.debug('converted ra, dec to theta, phi')
    
    bl = (theta_m>=40)&(theta_m<71)&(phi_m>240)&(phi_m<=270)&(pc_map<=1e-3)
    nllhs0[bl] -= 0.25*diff_nllh_out*((71 - theta_m[bl])/(71-40))*(np.abs(phi_m[bl] - 240)/(270-240))
    bl = (theta_m>=40)&(theta_m<71)&(phi_m>270)&(phi_m<=300)&(pc_map<=1e-3)
    nllhs0[bl] -= 0.25*diff_nllh_out*((71 - theta_m[bl])/(71-40))*(np.abs(300 - phi_m[bl])/(270-240))
    
    bl = (theta_m>=93)
    nllhs0[bl] -= 0.05*diff_nllh_out*(np.abs(theta_m[bl] - 93) / (180-93))

    logging.debug('adjusted ofov llh')

    dlogls = (nllhs0 - np.nanmin(nllhs0))
    
    min_in_nllh = np.nanmin(dlogls[bl_in])
    min_out_nllh = np.nanmin(dlogls[bl_out])
    corr_fact = min_in_nllh*(1-ifov_fact) - min_out_nllh*(1-ofov_fact)
    print('corr_fact: ', corr_fact)
    
    m = dlogls*ofov_fact
    m[pc_map>1e-3] *= (ifov_fact/ofov_fact)
    m[pc_map<=1e-3] -= corr_fact

    
    
    mliks = np.exp(-m[np.isfinite(m)])
    prob_map = mliks / np.exp(logsumexp(-m[np.isfinite(m)]))
    prob_map0 = np.zeros_like(m)
    prob_map0[np.isfinite(m)] = prob_map
    prob_map0[(prob_map0<0)] = 0.0
    prob_map0 /= np.sum(prob_map0)

    
    nside2 = 2**10
    
    pc_map2 = hp.ud_grade(pc_map, nside2, order_in='NEST', order_out='NEST')

    if infov_smooth is None:
        smooth_prob_a3 = prob_map0
    else:
        smooth_prob_a3 = hp.sphtfunc.smoothing(prob_map0, sigma=np.radians(infov_smooth), nest=True)
    prob_map_a32 = hp.ud_grade(smooth_prob_a3 / hp.nside2pixarea(nside), nside2, order_in='NEST',\
                                order_out='NEST')*hp.nside2pixarea(nside2)

    smooth_prob_d5 = hp.sphtfunc.smoothing(prob_map_a32, sigma=np.radians((6.0)), nest=True)
    smooth_prob_d3 = hp.sphtfunc.smoothing(prob_map_a32, sigma=np.radians((3.0)), nest=True)
    
    
    ra_m, dec_m = hp.pix2ang(nside2, np.arange(hp.nside2npix(nside2), dtype=np.int64), lonlat=True, nest=True)
    theta_m, phi_m = convert_radec2thetaphi(ra_m, dec_m, att_q)


    prob_map = np.copy(prob_map_a32)
    prob_map[pc_map2<=0.001] = smooth_prob_d3[pc_map2<=0.001]
    bl = (theta_m>=40)&(theta_m<71)&(phi_m>240)&(phi_m<=270)&(pc_map2<=1e-3)
    prob_map[bl] = smooth_prob_d5[bl]
    
    
    bl = (theta_m>=40)&(theta_m<71)&(phi_m>240)&(phi_m<=270)&(pc_map2<=1e-3)
    A5 = ((71 - theta_m)/(71-40))*(np.abs(phi_m - 240)/(270-240))
    A3 = 1 - A5
    prob_map[bl] = smooth_prob_d5[bl]*A5[bl] + smooth_prob_d3[bl]*A3[bl]
    print(np.min(A5[bl]), np.max(A5[bl]))
    print(np.min(A3[bl]), np.max(A3[bl]))
    
    
    bl = (theta_m>=40)&(theta_m<71)&(phi_m>270)&(phi_m<=300)&(pc_map2<=1e-3)
    A5 = ((71 - theta_m)/(71-40))*(np.abs(300 - phi_m)/(270-240))
    A3 = 1 - A5
    prob_map[bl] = smooth_prob_d5[bl]*A5[bl] + smooth_prob_d3[bl]*A3[bl]
    print(np.min(A5[bl]), np.max(A5[bl]))
    print(np.min(A3[bl]), np.max(A3[bl]))
    
    bl = (theta_m>=93)
    A5 = (theta_m - 93)/(180-93)
    A3 = 1 - A5    
    prob_map[bl] = smooth_prob_d5[bl]*A5[bl] + smooth_prob_d3[bl]*A3[bl]
    print(np.min(A5[bl]), np.max(A5[bl]))
    print(np.min(A3[bl]), np.max(A3[bl]))
    
    print('sum(prob_map) = ', np.sum(prob_map))
    
    prob_map[(prob_map<0)] = 0.0
    print('sum(prob_map) = ', np.sum(prob_map))
    prob_map /= np.sum(prob_map)
    
    return prob_map


def get_earth_sat_pos(sao_row):
    
    altitude = sao_row['SAT_ALT']
    ra, dec = sao_row['EARTH_RA'], sao_row['EARTH_DEC']

    EARTH_RADIUS = 6378.140
    occultation_radius = np.rad2deg(np.arcsin(EARTH_RADIUS / altitude))

    return ra, dec, occultation_radius


def rm_earth_prob_map(prob_map, sao_tab, trigger_time):

    nside = hp.npix2nside(len(prob_map))

    sao_ind = np.argmin(np.abs(sao_tab['TIME']-trigger_time))
    sao_row = sao_tab[sao_ind]    

    earth_ra, earth_dec, earth_rad = get_earth_sat_pos(sao_row)
    logging.debug('Earth ra, dec = %.2f, %.2f, Radius = %.2f'%(earth_ra, earth_dec, earth_rad))
    logging.debug('sao dt = %.3f'%(sao_row['TIME'] - trigger_time))

    earth_vec = hp.ang2vec(earth_ra, earth_dec, lonlat=True)
    earth_inds = hp.query_disc(nside, earth_vec, np.radians(earth_rad), nest=True)
    prob_map[earth_inds] = 0.0
    prob_map /= np.sum(prob_map)

    return prob_map


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

def pmap2moc_map(prob_map, pcfname, att_row, max_bytes=2.9e6):

    nside = hp.npix2nside(len(prob_map))

    logging.debug('nside = %d'%(nside))

    prob_map2 = np.copy(prob_map)
    perc_map2 = probm2perc(prob_map)
    pc_map2 = pcfile2hpmap(pcfname, att_row, nside)

    max_perc = 0.999

    
    def split_func(start, stop):
    
        max_order = 10
        ofov_order = 7

        Npix = stop - start

        Npix_tot = hp.nside2npix(nside)/Npix
        order = hp.npix2order(Npix_tot)

    #     order = 10 - Npix/4
    #     print(order)
    #     return False

        if order >= 10:
            return False

        if order < ofov_order:
            return True


        for ind in range(start, stop):

            if pc_map2[ind] > 1e-3 and perc_map2[ind] < max_perc:
                return True


        if order < ofov_order and pc_map2[ind] <= 1e-3:
            return True
        else:
            return False
        
        
    mmap = mhp.HealpixMap.adaptive_moc_mesh(nside, split_func, density=True, unit=1/u.steradian)
    if mmap.uniq.nbytes >= max_bytes:
        max_perc = 0.99
        print(max_perc, mmap.uniq.nbytes)
        mmap = mhp.HealpixMap.adaptive_moc_mesh(nside, split_func, density=True, unit=1/u.steradian)
        if mmap.uniq.nbytes >= max_bytes:
            max_perc = 0.98
            print(max_perc, mmap.uniq.nbytes)
            mmap = mhp.HealpixMap.adaptive_moc_mesh(nside, split_func, density=True, unit=1/u.steradian)
            if mmap.uniq.nbytes >= max_bytes:
                max_perc = 0.97
                print(max_perc, mmap.uniq.nbytes)
                mmap = mhp.HealpixMap.adaptive_moc_mesh(nside, split_func, density=True, unit=1/u.steradian)
                if mmap.uniq.nbytes >= max_bytes:
                    nside2 = 2**9
                    print('nside to', nside2)
                    prob_map2 = hp.ud_grade(prob_map/hp.nside2pixarea(nside), nside2, order_in='nest',\
                            order_out='nest')*hp.nside2pixarea(nside2)
                    perc_map2 = probm2perc(prob_map2)
                    pc_map2 = pcfile2hpmap(pcfname, att_row, nside2)    
                    max_perc = 0.999
                    mmap = mhp.HealpixMap.adaptive_moc_mesh(nside2, split_func, density=True, unit=1/u.steradian)
                    if mmap.uniq.nbytes >= max_bytes:
                        max_perc = 0.99
                        print(max_perc, mmap.uniq.nbytes)
                        mmap = mhp.HealpixMap.adaptive_moc_mesh(nside2, split_func, density=True, unit=1/u.steradian)

    for pix in range(len(mmap.data)):
        # for each pix in moc, get integrated prob divided by solid angle of pix
        ind0, ind1 = mmap.pix2range(nside, pix)
        mmap[pix] = np.sum(prob_map[ind0:ind1]) / ((ind1 - ind0)*hp.nside2pixarea(nside)) / u.steradian

            
    return mmap


def write_moc_skymap(skymap, filename, name=0):
    
    tab = Table()
    tab['UNIQ'] = skymap.uniq.astype(np.int32)
#     tab['PROB'] = (skymap.data*skymap.pixarea()).astype(np.float32) / u.steradian
    prob_dens = (skymap.data).astype(np.float32)
    prob_dens[(prob_dens<=0)] = 0.0
    tab['PROBDENSITY'] = prob_dens / u.steradian
    
    tab.write(filename, overwrite=True)
    
    file = fits.open(filename)
    header = file[1].header
    header['PIXTYPE'] = ('HEALPIX', 'HEALPIX pixelisation')
    header['ORDERING'] = ('NUNIQ', 'Pixel ordering scheme: RING, NESTED, or NUNIQ')
    header['COORDSYS'] = ('C', 'Ecliptic, Galactic or Celestial (equatorial)')
    header['MOCORDER'] = (skymap.order, 'MOC resolution (best order)')
    header['INDXSCHM'] = ('EXPLICIT', 'Indexing: IMPLICIT or EXPLICIT')
    header['OBJECT'] = (name, 'Unique identifier for this event')
    header['INSTRUME'] = ('Swift BAT', 'Instrument')
    
    file.writeto(filename, overwrite=True)


def moc_prob2percs(moc_skymap):

    p_map = np.copy(moc_skymap.data)

    inds_sort = np.argsort(p_map)[::-1]

    perc_map = np.zeros_like(p_map)

    perc_map[inds_sort] = np.cumsum((p_map*moc_skymap.pixarea())[inds_sort])

    perc_map = mhp.HealpixMap(data=perc_map, uniq=moc_skymap.uniq)

    return perc_map

def pcfile2mocmap(moc_map, pcfname, att_row):

    print('enter pcfile2mocmap')
    att_q = att_row["QPARAM"]
    pnt_ra, pnt_dec = att_row["POINTING"][:2]
    print(att_row)

    Npix = len(moc_map.uniq)
    pc_map = np.zeros(Npix)
    print(Npix)

    vec = hp.ang2vec(pnt_ra, pnt_dec, lonlat=True)

    inds = moc_map.query_disc(vec, np.radians(70.0))

    moc_ras, moc_decs = moc_map.pix2ang(inds, lonlat=True)

    moc_imxs, moc_imys = convert_radec2imxy(moc_ras, moc_decs, att_q)

    bl = (np.abs(moc_imys) < 1.01) & (np.abs(moc_imxs) < 2.0)
    print("np.sum(bl), ", np.sum(bl))

    PC = fits.open(pcfname)[0]
    w_t = WCS(PC.header, key="T")
    pc = PC.data

    print('opened PC')

    pc_vals = world2val(w_t, pc, moc_imxs[bl], moc_imys[bl])

    pc_map[inds[bl]] = pc_vals

    pc_moc_map = mhp.HealpixMap(data=pc_map, uniq=moc_map.uniq)

    return pc_moc_map

def mk_probmap_plots(mmap, pc_mmap, sao_row):
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib import cm
    from astropy.visualization.wcsaxes import SphericalCircle
    from matplotlib.patches import Rectangle
    import ligo.skymap.plot

    print('imported stuff')

    moc_perc = moc_prob2percs(mmap)

    print('got moc_perc')
    
    vmax = np.percentile(mmap.data, 99.0)
    vmin = np.percentile(mmap.data[mmap.data>0], 10.0)
    vmin = max(vmax/1e3, vmin)
    logging.debug('vmin, vmax: %f, %f)'%(vmin, vmax))
    
    im,ax = mmap.plot(norm=LogNorm(vmin=vmin, vmax=vmax), ax='astro degrees mollweide', cmap=cm.Oranges, rasterize=False, cbar=False)
    img = mmap.get_wcs_img(ax)
    img_perc = moc_perc.get_wcs_img(ax, rasterize=False)
    CS = ax.contour(img_perc, levels=[0.5, 0.9], colors=['cyan','green'], linestyles=['-', '--'])
    plt.clabel(CS, fmt='%.1f')
    ax.grid(True)
    lon = ax.coords[0]
    lat = ax.coords[1]

    area_50 = np.sum(moc_perc.pixarea()[moc_perc.data<0.5])
    area_90 = np.sum(moc_perc.pixarea()[moc_perc.data<0.9])
    print(area_50.to(u.deg*u.deg))
    print(area_90.to_string(unit=(u.deg*u.deg)))
    prob_infov = np.sum((mmap.data*mmap.pixarea()/u.steradian)[pc_mmap.data>1e-3])
    print(prob_infov)

    ax.text(550,260,'90%% Area = %.2f deg$^2$ \n50%% Area = %.2f deg$^2$ \nProb In FoV = %.2f'\
            %(area_90.to(u.deg**2).value,area_50.to(u.deg**2).value,prob_infov),\
            size=8, bbox=dict(facecolor='none', alpha=0.5))

    theta_max,phi_max = mmap.pix2ang(np.argmax(mmap))

    ra_max = np.rad2deg(phi_max)
    dec_max = 90-np.rad2deg(theta_max)
    print(ra_max, dec_max)

    vec = mhp.ang2vec(ra_max, dec_max, lonlat=True)
    half_wind = 5.0
    inds = mmap.query_disc(vec, np.radians(half_wind))
    min_max = np.min(mmap[inds]) / np.max(mmap[inds])
    print('min_max: ', min_max)
    if min_max > 0.1:
        half_wind = 10.0
    print('half_wind = ', half_wind)

    lonra = np.rad2deg(phi_max) + np.array([-half_wind,half_wind])/np.cos(np.pi/2 - theta_max)
    latra = (90-np.rad2deg(theta_max)) + np.array([-half_wind,half_wind])

    r = Rectangle((np.rad2deg(phi_max) - half_wind/ np.cos(np.pi/2. - theta_max), 90 - np.rad2deg(theta_max) - half_wind),\
                  2*half_wind / np.cos(np.pi/2. - theta_max), 2*half_wind, edgecolor='black', facecolor='none',
                  transform=ax.get_transform('world'))

    ax.add_patch(r)

    if sao_row is not None:
        sun_ra, sun_dec = sao_row['SUN_RA'], sao_row['SUN_DEC']
        ax.scatter(sun_ra, sun_dec, s=15, marker="$\u263b$", transform=ax.get_transform('world'), c='yellow', label='Sun', alpha=0.5)
        ax.scatter(sun_ra, sun_dec, s=30, marker="$\u263c$", transform=ax.get_transform('world'), c='yellow', label='Sun', alpha=0.5)

    moll_fname = 'mollview_plot.png'
    plt.savefig(moll_fname, bbox_inches='tight')
    
    plt.close()
    
    

    im,ax = mmap.plot(ax = 'cartview',aspect='auto', norm=LogNorm(vmin=vmin, vmax=vmax), ax_kw = {'latra': latra, 'lonra': lonra},\
                      rasterize=False, cmap=cm.Oranges, cbar=False);

    img_perc = moc_perc.get_wcs_img(ax, rasterize=False)
    CS = ax.contour(img_perc, levels=[0.5, 0.9], colors=['cyan','green'], linestyles=['-', '--'])
    plt.clabel(CS, fmt='%.1f')

    ax.grid(True)

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_ticklabel_visible(True)
    lat.set_ticklabel_visible(True)

    lon.set_major_formatter('d')

    lon.set_axislabel('RA')
    lat.set_axislabel('Dec')
    
    if sao_row is not None:
        ax.scatter(sun_ra, sun_dec, s=30, marker="$\u263b$", transform=ax.get_transform('world'), c='yellow', label='Sun', alpha=0.5)
        ax.scatter(sun_ra, sun_dec, s=60, marker="$\u263c$", transform=ax.get_transform('world'), c='yellow', label='Sun', alpha=0.5)

    zoom_fname = 'zoom_plot.png'
    plt.savefig(zoom_fname, bbox_inches='tight')

    return moll_fname, zoom_fname

