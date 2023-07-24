import numpy as np
from astropy.io import fits 
from astropy.table import Table
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
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

from .do_full_rates import *

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
sys.path.append('/gpfs/group/jak51/default/UtilityBelt/UtilityBelt')
from skyplot import get_earth_sat_pos
import scipy.interpolate
from scipy.spatial import Delaunay
import pandas as pd
from gbm.finder import TriggerCatalog


import cProfile as profile
import pstats



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
    x = ras  
    y = decs 

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
        

def extract_final_part(input_string):

    last_slash_index = input_string.rfind('/')

    final_part = input_string[last_slash_index + 1:]

    return final_part


def plot_ul_map(ras,decs,z,skymap_file, ra_ea, red_ea, radius_ear,name, path_results):

    #logging.info('check plot')

    def f_ul_int(x,y):
        return f_ul(x,y,ras,decs,z)


    #---- save the map as image


    nside = 32  
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


    hp.write_map(os.path.join(path_results,'ul.fits'), map_values, coord='C', overwrite=True)


    hpx, header = hp.read_map(os.path.join(path_results,'ul.fits'), h=True)

    fig = plt.figure(figsize=(15, 11), dpi=100)

    ax = plt.axes(projection='astro degrees mollweide')
    ax.grid()

    if skymap_file!='None':

        if skymap_file[0]=='point':

            ax.plot(skymap_file[1], skymap_file[2], transform=ax.get_transform('world'), marker='x' , c='red', zorder=1, ms='20', markeredgewidth=3)

        else:
            # Plot probability contours
            skymap, metadata = ligo.skymap.io.fits.read_sky_map(skymap_file, nest=True, distances=False)

            cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)

            ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='black', linewidths=1.5, levels=(50, 90), zorder=10, linestyles=['dashed', 'solid'])
    

    # Create a scalar mappable for the color map
    sm = plt.cm.ScalarMappable(cmap='viridis_r')


    image = ax.imshow_hpx((hpx, 'ICRS'), cmap='viridis_r', zorder=9, alpha=.6)

    cbar = plt.colorbar(sm)
    vmin=minval
    vmax=max(hpx)

    sm.set_norm(LogNorm(vmin=vmin, vmax=vmax))

    cbar.set_label('Flux upper limit (erg cm$^{-2}$ s$^{-1}$)', fontsize=14)

    if skymap_file!='None' and skymap_file[0]!='point':
        map_name=extract_final_part(skymap_file)
        plt.title('event = %s, skymap = %s, \n Temporal bin = 1.024 s, spectrum : normal ' %(name,map_name), fontsize=18)
    
    else:
        plt.title('Temporal bin = 1.024 s, spectrum : normal ' , fontsize=18)


    plt.savefig(os.path.join(path_results,'ul.pdf'))


def ul_gw(gwid,t0,head,ras,decs,wdir,path_results, ratio, ul_arr):

    ul_final={}

    client = GraceDb()

    gw=client.files('%s' %gwid).json().keys()

    logging.info('check gw ul')

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
        logging.info('No skymaps for %s' %gw[i])
    else:
        logging.info('%s done' %gwid)


    skymap_file=os.path.join(wdir,map_name)
    skymap = Table.read(skymap_file)

    # this part is needed to not consider part of sky behind the earth

    logging.info('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            break
        except:
            logging.info('error in getting earth data')

    logging.info('check post earth')


    # This part avoids looping 4 times in the conversion i-pixel to ra-dec
    ra_map=np.zeros(len(skymap['PROBDENSITY']))
    dec_map=np.zeros(len(skymap['PROBDENSITY']))
    nside_map=np.zeros(len(skymap['PROBDENSITY']))

    for i in range(len(skymap['PROBDENSITY'])):


        uniq = skymap[i]['UNIQ']
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)

        nside_map[i]=nside
        ra, dec = ah.healpix_to_lonlat(ipix, nside, order='nested')
        ra_map[i] = ra.deg
        dec_map[i] = dec.deg


    # print(nside_map)

    cumul=0
    tot=0
    for i in range(len(skymap['PROBDENSITY'])):

        # logging.info(i)

        ra, dec = ra_map[i], dec_map[i]
        nside = nside_map[i]

        tot+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

        if ang_sep(ra_ea, red_ea, ra, dec)< radius_ear:

            cumul+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

            skymap[i]['PROBDENSITY']=0

    # this line prints the fraction of the gw posterior occulted by earth
    sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
    # Format the sentence with the number
    formatted_sentence = sentence.replace('%%%%', "%.2f" % (cumul/tot*100))

    # Save the sentence to a text file
    filename = os.path.join(path_results,'earth_occultation.txt')  # Specify the desired filename
    with open(filename, 'w') as file:
        file.write(formatted_sentence)


   
    df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

    logging.info('check pre ul')


    for n in range(len(head)):
        logging.info('loop gw')

       
        spec = head[n][0]
        time_bin = head[n][1]

        if time_bin=='0.128':

            ul_5sigma_new = ul_arr['%s' %spec]


            def f_ul_n(x,y):
                return f_ul(x,y,ras,decs,ul_5sigma_new)

            dp=0
            norm=0
            for i in range(len(skymap['PROBDENSITY'])):


                ra, dec = ra_map[i], dec_map[i]
                nside = nside_map[i]    
                ul=f_ul_n(ra,dec)



                if skymap[i]['PROBDENSITY']>0:

                    dp+=ul*skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)
                    norm+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)


            df.at[time_bin,spec] = "{:.2e}".format(dp/norm)

            ul_final['%s' %spec]=dp/norm

        else:
             df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])
           

        if spec=='normal' and time_bin=='1.024':
            ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]
            plot_ul_map(ras,decs,ul_5sigma_new, skymap_file , ra_ea, red_ea, radius_ear, gwid, path_results)


    logging.info('check post ul')

    df.to_csv(os.path.join(path_results,'ul_gw.txt'), sep='\t', index=True)

    


def ul_nogw(t0,head,ras,decs,wdir, path_results, ratio, ul_arr):

    fermi=False

    ul_final={}


    logging.info('start nogw')

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
                values = next(file).strip().split(',')
                ra_point = float(values[0])
                dec_point = float(values[1])

            else:
                logging.info('no valid file')

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

            os.system('curl -o %s/glg_healpix_all_%s.fit https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/20%s/%s/quicklook/glg_healpix_all_%s.fit' %(wdir,tri_name,tri_name[2:4],tri_name,tri_name))

            name_fermi_map='glg_healpix_all_%s.fit' %tri_name

            if os.path.exists(os.path.join(wdir,name_fermi_map)):

                fermi=True

                skymap_file=os.path.join(wdir,name_fermi_map)

                loc = GbmHealPix.open(os.path.join(wdir,name_fermi_map))

                logging.info('fermi map received')
                def weight(ra,dec):
                    return loc.probability(ra, dec)
                
            else:
                logging.info('no Fermi skymap available')

                def weight(ra,dec):
                    return 1.0

        else:

            def weight(ra,dec):
                return 1.0



    #logging.info('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            logging.info('earth map received')
            break
        except:
            logging.info('error in getting earth data')

    
    if region=='point':

        df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

        for n in range(len(head)):

        
            spec = head[n][0]
            time_bin = head[n][1]

            if time_bin=='0.128':

                ul_5sigma_new = ul_arr['%s' %spec]


                def f_ul_n(x,y):
                    return f_ul(x,y,ras,decs,ul_5sigma_new)

                ul=f_ul_n(ra_point,dec_point)

                df.at[time_bin,spec] = "{:.2e}".format(ul)

                ul_final['%s' %spec]=ul
            
            else:
             
                df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])

            if spec=='normal' and time_bin=='1.024':
                ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]
                skymap_file=['point',ra_point,dec_point]
                plot_ul_map(ras,decs,ul_5sigma_new, skymap_file, ra_ea, red_ea, radius_ear, None, path_results)


        df.to_csv(os.path.join(path_results,'ul.txt'), sep='\t', index=True)
        
    else:
        #logging.info('check post earth')
    
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

        logging.info('earth fraction')

        # this line prints the fraction of the gw posterior occulted by earth
        sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
        # Format the sentence with the number
        formatted_sentence = sentence.replace('%%%%', "%.2f" %(cumul/tot*100))
    
        # Save the sentence to a text file
        filename = os.path.join(path_results,'earth_occultation.txt')  # Specify the desired filename
        with open(filename, 'w') as file:
            file.write(formatted_sentence)
    
        if max(map_values)==0.0:
            logging.info('region occulted by earth, no upper limits available')
        
        else:
    
            #logging.info('check pre ul')
    
            df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])
    
            for n in range(len(head)):

    
            
                spec = head[n][0]
                time_bin = head[n][1]
                
                logging.info('ul weighting %s %s' %(spec, time_bin))
  

                if time_bin=='0.128':

                    ul_5sigma_new = ul_arr['%s' %spec]


                    def f_ul_n(x,y):
                        return f_ul(x,y,ras,decs,ul_5sigma_new)
        
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


                    df.at[time_bin,spec] = "{:.2e}".format(dp/norm)

                    ul_final['%s' %spec]=dp/norm
                
                else:

                    df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])

    
                if spec=='normal' and time_bin=='1.024':

                    ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]

                    if fermi:
                        plot_ul_map(ras,decs,ul_5sigma_new, skymap_file , ra_ea, red_ea, radius_ear, tri_name, path_results)
                    else:
                        plot_ul_map(ras,decs,ul_5sigma_new,'None', ra_ea, red_ea, radius_ear, tri_name, path_results)

    
                
    
            #logging.info('check post ul')

            df.to_csv(os.path.join(path_results,'ul.txt'), sep='\t', index=True)

            logging.info('ul done')



def main(args):

    work_dir=args.work_dir

    file_path = os.path.join(work_dir,'config.json')

    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    trig_time=json_data['trigtime']

    trig_name = '%s_c%s' %(json_data['triggerID'], json_data['config']['id'])
    path_results='/gpfs/group/jak51/default/nitrates_all_results/UL/%s' %trig_name

    if os.path.exists(path_results)== False:
        'done'
        os.mkdir(path_results)

    logging.basicConfig(
        filename=os.path.join(path_results,'upper_limits_analysis.log'),
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )


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

    logging.info("b1")

    # number of detectors being used
    ndets = np.sum(dmask==0)
    logging.debug("Ndets: ")
    logging.debug(np.sum(dmask==0))

    t_end = trigger_time + 1e3
    t_start = trigger_time - 1e3
    mask_vals = mask_detxy(dmask, ev_data)

    # logging.debug(mask_vals)
    bl_dmask = (dmask==0.)

    logging.info("b2")

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
    logging.info("b3")


        
    bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
    bkg_df = pd.read_csv(bkg_fname)


# create bkg object

    tmin = GTI_PNT["START"][0]
    tmax = GTI_PNT["STOP"][-1]
    twind = 20

    poly_trng = np.int(twind)

    bkg_obj = Linear_Rates(ev_data0, tmin, tmax, trigger_time, GTI_PNT, sig_clip=4.0, poly_trng=poly_trng)
    bkg_obj.do_fits()

    logging.info("b4")


# new using new responses ---- before it was in the loop below
    fnames = np.array([fname for fname in os.listdir(resp_dname) if '.rsp' in fname])
    theta_values = np.array([float(fname.split('_')[3]) for fname in fnames])
    phi_values = np.array([float(fname.split('_')[5]) for fname in fnames]) 
    thetas_string = np.array([fname.split('_')[3] for fname in fnames])
    phis_string = np.array([fname.split('_')[5] for fname in fnames])

    ras, decs = convert_theta_phi2radec(theta_values, phi_values, att_q)

    logging.info("b5")


# Getting rate and std deviations for all durations:

    dur_values=[0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 16.384]
    #dur_values=[1.024]
    head=[]



    alpha_values = [-1.0,-1.9,-0.62,-1.5]
    # alpha_values = [-1.0,-1.9,-0.62,-1.215]

    beta_values = [-2.3,-3.7,0.0,0.0]
    Epeak_values = [230.0,70.0,185.0,1500.0]
    # Epeak_values = [230.0,70.0,185.0,150000.0]

    flux_elo=15.0
    flux_ehi=350.0

    ratio={'normal':{},'soft':{},'170817':{},'hard':{}}
    ul_arr={'normal':1.0,'soft': 1.0,'170817':1.0,'hard': 1.0}

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

        




        for k in range(len(dur_values)):

            logging.debug(spe)
            logging.debug(dur_values[k])

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

    # UL

            Ndets_tot = 32768.0 # total dets 
            Ndets_active = ndets   # number of active detectors

            Ndet_ratio = Ndets_active / Ndets_tot
        
            rate_std = (np.sqrt(max_sig2_bkg))/dur_values[k]
            rate_upper_limit = 5*rate_std

    # Using 4 spectral templates

            if k==0:
                ul_5sigma_zero=np.zeros(len(ras))

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
                    
                    if i==0:
                        ref=flux_upper_limit_new

                    ul_5sigma_zero[i]=flux_upper_limit_new


                ratio['%s' %spe]['%s' %str(dur_values[k])]=1.0


                ul_arr['%s' %spe]=ul_5sigma_zero
            else:

                for i in range(1):
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
                    
                    ref_i=flux_upper_limit_new

                ratio['%s' %spe]['%s' %str(dur_values[k])]=ref_i/ref

            head.append([spe,str(dur_values[k])])
            # ul_5sigma_new.append(ul_5sigma_in)


    # print(ul_arr)

    utc_time = Time(trig_time, scale='utc')
    t1=utc_time.gps-1
    t2=utc_time.gps+1

    client = GraceDb()

    ev=client.superevents(query='gpstime: %s .. %s' %(str(t1),str(t2)))

    ev_list=[]
    for event in ev:
        ev_list.append(event['superevent_id'])

    #logging.info('ev list', ev_list)


    try:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

    


    if len(ev_list)>0:
        ul_gw(ev_list[0],trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr)
    else:
        ul_nogw(trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr)

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
