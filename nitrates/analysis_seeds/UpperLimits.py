import numpy as np
from astropy.io import fits 
from astropy.table import Table
import os
import time

import pandas as pd
import logging
import sys
import argparse
import time
from datetime import datetime
from astropy.time import Time
import json


from ..lib.event2dpi_funcs import mask_detxy
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_info_tab
from ..lib.coord_conv_funcs import  convert_theta_phi2radec

from .do_full_rates import *

from ..lib.calc_BAT_ul import *
from ..config import resp_dname


from astropy.table import Table

from ligo.gracedb.rest import GraceDb

import cProfile as profile
import pstats

sys.path.append('/gpfs/group/jak51/default/UtilityBelt/UtilityBelt')
from ul_skyaverage import  ul_gw
from ul_skyaverage import  ul_nogw



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

    args = parser.parse_args()
    return args


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
        coord_file = args.coord_file
        ul_nogw(trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr, coord_file)

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
