import numpy as np
from astropy.io import fits 
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from copy import copy, deepcopy
import logging, traceback
import sys
import argparse
import time
import numpy
from datetime import datetime, timedelta
from astropy.time import Time
import json


from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_info_tab
from ..lib.coord_conv_funcs import  convert_theta_phi2radec

from ..analysis_seeds.do_full_rates import *

from ..lib.calc_BAT_ul import *
from ..config import resp_dname

from gbm.data import HealPix
from gbm.data import GbmHealPix

from matplotlib.colors import LogNorm
import ligo.skymap.plot
import ligo.skymap.io
from astropy.table import Table
import astropy_healpix as ah
import astropy.units as u
import healpy as hp
import sys
from ligo.gracedb.rest import GraceDb
#sys.path.append('/gpfs/group/jak51/default/UtilityBelt/UtilityBelt')
sys.path.append('/gpfs/group/jak51/default/UtilityBelt/UtilityBelt')
from skyplot import get_earth_sat_pos
import scipy.interpolate
from scipy.spatial import Delaunay
import pandas as pd
from gbm.finder import TriggerCatalog


import cProfile as profile
import pstats


# sys.path.append('/gpfs/group/jak51/default/TheEchoLocation')
# from app.models import *
# from BATtargeted import *
# app.app_context().push()


start_time = time.time()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, help="Results directory", default='.')
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument("--att_fname", type=str, help="Fname for that att file", default=None)
    parser.add_argument("--trig_time",type=str, help="Time of trigger, in either MET or a datetime string", default=None)
    parser.add_argument("--dbfname", type=str, help="Name to save the database to", default=None)
    parser.add_argument("--resp_dname",type=str,help="new responses directory",default=None)
    parser.add_argument("--api_token",type=str,help="EchoAPI key for interactions.",default=None)
    parser.add_argument("--coord_file",type=str,help="file with coordinates",default=None)
    parser.add_argument("--fermi_file",type=str,help="fermi fit file with localization",default=None)

    args = parser.parse_args()
    return args



def down(graceid,file_name,path):

    client = GraceDb()
    response = client.files(graceid, file_name)
    skymap = response.read()
    name = f'{file_name}'
    file_path = os.path.join(path, name)
    with open(file_path, 'wb') as binary_file:
        binary_file.write(skymap)



class LinearNDInterpolatorExt(object):

    def __init__(self, points,values):
        self.funcinterp=scipy.interpolate.LinearNDInterpolator(points,values)
        self.funcnearest=scipy.interpolate.NearestNDInterpolator(points,values)
    def __call__(self,*args):
        t=self.funcinterp(*args)
        if not numpy.isnan(t):
            return t.item(0)
        else:
            return self.funcnearest(*args)
        

def f_ul(ra,dec,ras,decs,z_int):
    x = ras  # Assuming x is the 5th column (index 4)
    y = decs  # Assuming y is the 6th column (index 5)

    tri = Delaunay(np.column_stack((x, y)))        

    interp = LinearNDInterpolatorExt(tri, z_int)

    return interp(ra,dec)
 


def ang_sep(ra0, dec0, ra1, dec1):
    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(
        np.cos(np.radians(90 - dec0)) * np.cos(np.radians(90 - dec1))
        + np.sin(np.radians(90 - dec0)) * np.sin(np.radians(90 - dec1)) * dcos
    )
    return np.rad2deg(angsep)
        


def plot_ul_map(ras,decs,z,skymap_file, ra_ea, red_ea, radius_ear,wdir,gwname):

    #print('check plot')

    def f_ul_int(x,y):
        return f_ul(x,y,ras,decs,z)


    #---- save the map as image


    nside = 32  # Resolution parameter, determines the number of pixels (higher resolution = more pixels)
    npix = hp.nside2npix(nside)  # Total number of pixels in the map

    map_values = np.zeros(npix)  # Initialize the array with zeros  
    minval=np.inf
    for n in range(npix):

        theta, phi = hp.pix2ang(nside, n)  # Retrieve theta and phi angles
        dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
        ra = np.degrees(phi)  # Convert phi to RA

        
        if ang_sep(ra_ea, red_ea, ra, dec)> radius_ear:

            map_values[n]=f_ul_int(ra,dec)
            minval=min(map_values[n],minval)
        
        else:
            map_values[n]='nan'


    hp.write_map(os.path.join(wdir,'ul.fits'), map_values, coord='C', overwrite=True)


    hpx, header = hp.read_map(os.path.join(wdir,'ul.fits'), h=True)

    fig = plt.figure(figsize=(15, 11), dpi=100)

    ax = plt.axes(projection='astro degrees mollweide')
    ax.grid()

    if skymap_file!='None':
        # Plot probability contours
        skymap, metadata = ligo.skymap.io.fits.read_sky_map(skymap_file, nest=True, distances=False)

        cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)

        ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='black', linewidths=1, levels=(50, 90), zorder=10, linestyles=['dashed', 'solid'])
        
    # Create a scalar mappable for the color map
    sm = plt.cm.ScalarMappable(cmap='viridis_r')


    image = ax.imshow_hpx((hpx, 'ICRS'), cmap='viridis_r', zorder=9, alpha=.6)

    cbar = plt.colorbar(sm)
    vmin=minval
    vmax=max(hpx)

    sm.set_norm(LogNorm(vmin=vmin, vmax=vmax))

    cbar.set_label('Flux upper limit (erg cm$^{-2}$ s$^{-1}$)', fontsize=14)

    if skymap_file!='None':
        plt.title('%s, skymap = %s, \n Temporal bin = 1.024 s, spectrum : normal ' %(gwname,skymap_file), fontsize=18)
    
    else:
        plt.title('Temporal bin = 1.024 s, spectrum : normal ' , fontsize=18)


    # plt.show()
    # Show the plot
    plt.savefig(os.path.join(wdir,'ul.pdf'))


def ul_gw(gwid,t0,ul_5sigma_new,head,ras,decs,wdir):

    client = GraceDb()

    gw=client.files('%s' %gwid).json().keys()

    #print('check gw ul')

    gw=list(gw)
    check=0

    for i in range(len(gw)):

        if 'Bilby.multiorder.fits'==gw[i]:
            check=1
            graceid=gwid
            map_name=gw[i]
            down(graceid,map_name,wdir)
            break

        if gw[i].endswith('multiorder.fits'):
            check=1
            graceid=gwid
            map_name=gw[i]
            down(graceid,map_name,wdir)  
            break

    if check==0:
        print('No skymaps for', gw[i])
    else:
        print(gwid, ' done')


    skymap_file=os.path.join(wdir,map_name)
    skymap = Table.read(os.path.join(wdir,map_name))

    # this part is needed to not consider part of sky behind the earth

    #print('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            break
        except:
            print('error in getting earth data')

    #print('check post earth')


    cumul=0
    tot=0
    for i in range(len(skymap['PROBDENSITY'])):

        # print(i)

        uniq = skymap[i]['UNIQ']
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)

        ra, dec = ah.healpix_to_lonlat(ipix, nside, order='nested')
        tot+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

        if ang_sep(ra_ea, red_ea, ra.deg, dec.deg)< radius_ear:

            cumul+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

            skymap[i]['PROBDENSITY']=0

    # this line prints the fraction of the gw posterior occulted by earth
    sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
    # Format the sentence with the number
    formatted_sentence = sentence.replace('%%%%', "%.2f" % (cumul/tot))

    # Save the sentence to a text file
    filename = os.path.join(wdir,'earth_occultation.txt')  # Specify the desired filename
    with open(filename, 'w') as file:
        file.write(formatted_sentence)



   
    df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

    #print('check pre ul')


    for n in range(len(ul_5sigma_new)):

       
        spec = head[n][0]
        time_bin = head[n][1]


        def f_ul_n(x,y):
            return f_ul(x,y,ras,decs,ul_5sigma_new[n])

        dp=0
        norm=0
        for i in range(len(skymap['PROBDENSITY'])):


            uniq = skymap[i]['UNIQ']
            level, ipix = ah.uniq_to_level_ipix(uniq)
            nside = ah.level_to_nside(level)

            ra, dec = ah.healpix_to_lonlat(ipix, nside, order='nested')
            ul=f_ul_n(ra.deg,dec.deg)



            if skymap[i]['PROBDENSITY']>0:

                dp+=ul*skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)
                norm+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)


        if spec=='normal' and time_bin=='1.024':
            plot_ul_map(ras,decs,ul_5sigma_new[n],skymap_file, ra_ea, red_ea, radius_ear, wdir, gwid)

        df.at[time_bin,spec] = "{:.2e}".format(dp/norm)

    #print('check post ul')

    df.to_csv(os.path.join(wdir,'ul_gw.txt'), sep='\t', index=True)

    


def ul_nogw(t0,ul_5sigma_new,head,ras,decs,wdir):

    region=None


    if args.coord_file != None:

        filename = os.path.join(wdir,args.coord_file)

        with open(filename, 'r') as file:
            # Read the first line to get the requested variable
            row = file.readline().strip()

            if row=='circle':
                region='circle'
                values = next(file).strip().split(',')
                ra_circ = float(values[0])
                dec_circ = float(values[1])
                radius_circ = float(values[2])
            
            elif row=='poly':
                region='poly'
                data = np.loadtxt(file, delimiter=',')
                ra_poly = data[:, 0]
                dec_poly = data[:, 1]

            elif row=='point':
                region='point'
                data = np.loadtxt(file, delimiter=',')
                ra_point = data[:, 0]
                dec_point = data[:, 1]

            else:
                print('no valid file')

        if region=='poly':
            ra_pts=ra_poly.tolist()
            dec_pts=dec_poly.tolist()
            verts_map = HealPix.from_vertices(ra_pts, dec_pts, nside=128)

            def weight(ra,dec):
                return verts_map.probability(ra, dec)
        
        if region=='circle':

            def weight(ra,dec):

                if ang_sep(ra_circ, dec_circ, ra, dec)< radius_circ:
                    return 1.0
                else:
                    return 0.0

    else:

        trigcat = TriggerCatalog()


        format = '%Y-%m-%d %H:%M:%S.%f'

        dt = datetime.strptime(t0, format)

        t1 = dt - timedelta(seconds=1)
        t2 = dt + timedelta(seconds=1)

        t1=t1.strftime(format)
        t2=t2.strftime(format)

        tri_name = trigcat.slice('trigger_time', lo=t1, hi= t2).get_table(columns=('trigger_name', 'trigger_time'))

        if len(tri_name)>0:
        
            tri_name= tri_name[0][0]

            os.system('curl -o %s/fermi_map.fit https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/20%s/%s/quicklook/glg_healpix_all_%s.fit' %(wdir,tri_name[2:4],tri_name,tri_name))

            if os.path.exists(os.path.join(wdir,'fermi_map.fit')):

                loc = GbmHealPix.open(os.path.join(wdir,'fermi_map.fit'))

                def weight(ra,dec):
                    return loc.probability(ra, dec)
                
            else:
                print('no Fermi skymap available')

                def weight(ra,dec):
                    return 1.0

        else:

            def weight(ra,dec):
                return 1.0



    #print('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            break
        except:
            print('error in getting earth data')

    
    if region=='point':

        df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

        for n in range(len(ul_5sigma_new)):

        
            spec = head[n][0]
            time_bin = head[n][1]


            def f_ul_n(x,y):
                return f_ul(x,y,ras,decs,ul_5sigma_new[n])

            ul=f_ul_n(ra,dec)

            df.at[time_bin,spec] = "{:.2e}".format(ul)


        df.to_csv(os.path.join(wdir,'ul.txt'), sep='\t', index=True)
        
    else:
        #print('check post earth')
    
        nside = 32  # Resolution parameter, determines the number of pixels (higher resolution = more pixels)
        npix = hp.nside2npix(nside)  # Total number of pixels in the map
    
        map_values = np.zeros(npix)  # Initialize the array with zeros
    
        tot=0
        cumul=0
    
        for n in range(npix):
    
            theta, phi = hp.pix2ang(nside, n)  # Retrieve theta and phi angles
            dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
            ra = np.degrees(phi)  # Convert phi to RA
    
    
            tot+=weight(ra,dec)
    
            if ang_sep(ra_ea, red_ea, ra, dec)< radius_ear:
                map_values[n]=0.0
                cumul+=weight(ra,dec)
    
            else:
                map_values[n]=1.0 * weight(ra,dec)
    
        # this line prints the fraction of the gw posterior occulted by earth
        sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
        # Format the sentence with the number
        formatted_sentence = sentence.replace('%%%%', "%.2f" %(cumul/tot))
    
        # Save the sentence to a text file
        filename = os.path.join(wdir,'earth_occultation.txt')  # Specify the desired filename
        with open(filename, 'w') as file:
            file.write(formatted_sentence)
    
        if max(map_values)==0.0:
            print('region occulted by earth, no upper limits available')
        
        else:
    
            #print('check pre ul')
    
            df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])
    
            for n in range(len(ul_5sigma_new)):
    
            
                spec = head[n][0]
                time_bin = head[n][1]
    
    
                def f_ul_n(x,y):
                    return f_ul(x,y,ras,decs,ul_5sigma_new[n])
    
                dp=0
                norm=0
    
                for i in range(npix):
    
                    theta, phi = hp.pix2ang(nside, i)  # Retrieve theta and phi angles
                    dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
                    ra = np.degrees(phi)   # Convert phi to RA
    
    
                    if map_values[i]>0:
    
                        ul=f_ul_n(ra,dec)
                        dp+=ul*map_values[i] 
                        norm+=map_values[i]
    
    
                if spec=='normal' and time_bin=='1.024':
                    plot_ul_map(ras,decs,ul_5sigma_new[n],'None', ra_ea, red_ea, radius_ear, wdir, None)
    
                df.at[time_bin,spec] = "{:.2e}".format(dp/norm)
    
            #print('check post ul')
    
            df.to_csv(os.path.join(wdir,'ul.txt'), sep='\t', index=True)



def main(args):
    logging.basicConfig(
        filename="upper_limits_analysis.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    work_dir=args.work_dir
    #resp_dname=args.resp_dname

    conn = get_conn(os.path.join(work_dir,'results.db'))
    logging.info("Connecting to DB") 

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    trigger_time = info_tab['trigtimeMET'][0]
    logging.debug("trigtime: %.3f" % (trigger_time))    
    # default file names made by do_data_setup.py
    evfname = os.path.join(work_dir,'filter_evdata.fits')
    ev_data = fits.open(evfname)[1].data
    # GTI extensions added to event fits file by do_data_setup.py
    GTI_PNT = Table.read(evfname, hdu='GTI_POINTING') # when the analysis can be run (good data time and pointing)
    GTI_SLEW = Table.read(evfname, hdu='GTI_SLEW') # slewing times
    # the final set of detectors to mask (disabled dets, hot/cold dets, and dets with glitches)  
    dmask = fits.open(os.path.join(work_dir,'detmask.fits'))[0].data
    attfile = fits.open(os.path.join(work_dir,'attitude.fits'))[1].data
    att_q = attfile["QPARAM"][np.argmin(np.abs(attfile["TIME"] - trigger_time))]


    # number of detectors being used
    ndets = np.sum(dmask==0)
    logging.debug("Ndets: ")
    logging.debug(np.sum(dmask==0))

    t_end = trigger_time + 1e3
    t_start = trigger_time - 1e3
    mask_vals = mask_detxy(dmask, ev_data)

    logging.debug(mask_vals)
    bl_dmask = (dmask==0.)

    # get rid of events:
    # far away from trigger time
    # from bad dets
    # with bad event flags
    bl_ev = (ev_data['EVENT_FLAGS']<1)&\
        (ev_data['ENERGY']<=350.)&(ev_data['ENERGY']>=15.)&\
        (mask_vals==0.)&(ev_data['TIME']<=t_end)&\
        (ev_data['TIME']>=t_start)

    #logging.info("Number of events passing cuts: ", np.sum(bl_ev))
    ev_data0 = ev_data[bl_ev]

   # logging.info(GTI_PNT)


        
    bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
    bkg_df = pd.read_csv(bkg_fname)


# create bkg object

    tmin = GTI_PNT["START"][0]
    tmax = GTI_PNT["STOP"][-1]
    twind = 20

    poly_trng = np.int(twind)

    bkg_obj = Linear_Rates(ev_data0, tmin, tmax, trigger_time, GTI_PNT, sig_clip=4.0, poly_trng=poly_trng)
    bkg_obj.do_fits()

# new using new responses ---- before it was in the loop below
    fnames = np.array([fname for fname in os.listdir(resp_dname) if '.rsp' in fname])
    theta_values = np.array([float(fname.split('_')[3]) for fname in fnames])
    phi_values = np.array([float(fname.split('_')[5]) for fname in fnames]) 
    thetas_string = np.array([fname.split('_')[3] for fname in fnames])
    phis_string = np.array([fname.split('_')[5] for fname in fnames])

    ras, decs = convert_theta_phi2radec(theta_values, phi_values, att_q)



# Getting rate and std deviations for all durations:

    dur_values=[0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 16.384]
    #dur_values=[1.024]
    ul_5sigma_new = []
    head=[]


    for k in range(len(dur_values)):

# calculate bkg rate and rate error
        twind = 20
        dtmin = -20
        dtmax = twind

        tstep = dur_values[k] / 4.0
        tbins0 = np.arange(dtmin, dtmax, tstep) + trigger_time
        tbins1 = tbins0 + dur_values[k]
        tcnts = get_cnts_tbins_fast(tbins0, tbins1, ev_data0)

        ntbins = len(tbins0)
        snrs = np.zeros(ntbins)
        sig2_bkg_values =[]
        
        for i in range(ntbins):
            dur = tbins1[i] - tbins0[i]
            tmid = (tbins1[i] + tbins0[i]) / 2.0
            bkg_rate, bkg_rate_err = bkg_obj.get_rate(tmid)
            sig2_bkg = (bkg_rate_err * dur) ** 2 + (bkg_rate * dur)
            sig2_bkg_values.append(sig2_bkg)
            snrs[i] = (tcnts[i] - bkg_rate * dur) / np.sqrt(sig2_bkg)
   
        max_sig2_bkg = np.max(sig2_bkg_values)
        min_sig2_bkg = np.min(sig2_bkg_values)
        logging.debug('max sig2_bkg:')
        logging.debug(max_sig2_bkg) 
        logging.debug('min sig2_bkg')
        logging.debug(min_sig2_bkg)
        logging.debug('bkg rate:')
        logging.debug(bkg_rate)
        logging.debug('bkg rate error:')
        logging.debug(bkg_rate_err)

# UL

        Ndets_tot = 32768.0 # total dets 
        Ndets_active = ndets   # number of active detectors

        Ndet_ratio = Ndets_active / Ndets_tot
    
        rate_std = (np.sqrt(max_sig2_bkg))/dur_values[k]
        rate_upper_limit = 5*rate_std

   # Using 4 spectral templates
        alpha_values = [-1.0,-1.9,-0.62,-1.5]
        beta_values = [-2.3,-3.7,0.0,0.0]
        Epeak_values = [230.0,70.0,185.0,1500.0]
        flux_elo=15.0
        flux_ehi=350.0
        for index,j in enumerate(alpha_values):
        #for index in range(0,1):
                      
            if index == 0:
                spec_temp = "Band_Normal_template_Ep230keV_alpha-1.0_beta-2.3"
                spe='normal'
            elif index == 1:
                spec_temp = "Band_Soft_template_Ep70keV_alpha-1.9_beta_-3.7"
                spe='soft'
            elif index == 2:
                spec_temp = "Cutoff_PL_GW170817-like_Ep185keV_alpha0.62"
                spe='170817'
            elif index == 3:
                spec_temp = "Cutoff_PL_hard_Ep1500_keV_alpha1.5"
                spe='hard'
            
            ul_5sigma_in=[]

            for i in range(len(theta_values)):
                drm_tab_new = get_resp4ul_tab(thetas_string[i],phis_string[i])

# using energy bin 15-350 and ignoring 350-500
                chan_low = 0
                chan_hi = 3
# response matrix using selected energy bins and corrected for number of active dets
                drm_matrix_new = drm_tab_new['MATRIX'][:,chan_low:(chan_hi+1)] * Ndet_ratio

# find the flux that gives an expected rate equal to the rate upper limit
                if index>=2:
                    flux_upper_limit_new = rate2comp_eflux(rate_upper_limit,drm_matrix_new,\
                                               drm_tab_new['ENERG_LO'], drm_tab_new['ENERG_HI'],\
                                               alpha_values[index],Epeak_values[index], flux_elo, flux_ehi)
                else:
                    flux_upper_limit_new = rate2band_eflux(rate_upper_limit, drm_matrix_new,\
                                                   drm_tab_new['ENERG_LO'], drm_tab_new['ENERG_HI'],\
                                                   alpha_values[index], beta_values[index], Epeak_values[index], flux_elo, flux_ehi)
                
                
                ul_5sigma_in.append(flux_upper_limit_new)

            head.append([spe,str(dur_values[k])])
            ul_5sigma_new.append(ul_5sigma_in)
            

            logging.debug('5 sigma UL values for:')
            logging.debug(spec_temp)
            logging.debug('and time bin')
            logging.debug(dur_values[k])
            logging.debug('-----')
            logging.debug(ul_5sigma_new)
            logging.debug('------')

    

    file_path = os.path.join(work_dir,'config.json')

    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    trig_time=json_data['trigtime']


    utc_time = Time(trig_time, scale='utc')
    t1=utc_time.gps-1
    t2=utc_time.gps+1

    client = GraceDb()

    ev=client.superevents(query='gpstime: %s .. %s' %(str(t1),str(t2)))

    ev_list=[]
    for event in ev:
        ev_list.append(event['superevent_id'])

    #print('ev list', ev_list)


    try:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

    if len(ev_list)>0:
        ul_gw(ev_list[0],trig_time,ul_5sigma_new,head,ras,decs,work_dir)
    else:
        ul_nogw(trig_time,ul_5sigma_new,head,ras,decs,work_dir)

    elapsed_time = time.time() - start_time

    logging.debug("Time taken in seconds:")
    logging.debug(elapsed_time)


if __name__ == "__main__":
    args = cli()
    prof = profile.Profile()
    prof.enable()

    main(args)

    prof.disable()

    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(20) # top 10 rows
