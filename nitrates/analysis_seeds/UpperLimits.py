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

from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_info_tab
from ..lib.coord_conv_funcs import  convert_theta_phi2radec

from ..analysis_seeds.do_full_rates import *

from ..lib.calc_BAT_ul import *

start_time = time.time()

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, help="Results directory", default="/gpfs/group/jak51/default/F702399789/")
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--att_fname", type=str, help="Fname for that att file", default=None
    )
    parser.add_argument(
        "--trig_time",
        type=str,
        help="Time of trigger, in either MET or a datetime string",
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument(
        "--bkg_fname",
        type=str,
        help="Name of the file with the bkg fits",
        default="bkg_estimation.csv",
    )
    parser.add_argument(
        "--twind",
        type=float,
        help="Number of seconds to go +/- from the trigtime",
        default=20 * 1.024,
    )
    parser.add_argument(
        "--min_dt", type=float, help="Min time from trigger to do", default=None
    )
    parser.add_argument(
        "--drm_dir_old",
        type=str,
        help="drm_directory",
        default="/gpfs/group/jak51/default/responses/rsp_maskweight/",
    )
    parser.add_argument(
        "--drm_dir",
        type=str,
        help="drm_directory",
        default="/gpfs/group/jak51/default/responses/rsp_NITRATES/",
    )
    parser.add_argument(
        "--api_token",
        type=str,
        help="EchoAPI key for interactions.",
        default=None
    )

    args = parser.parse_args()
    return args







def main(args):
    logging.basicConfig(
        filename="upper_limits_analysis.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )


   # work_dir='/gpfs/group/jak51/default/F702399789'
    work_dir='/storage/home/gzr5209/Desktop/709523649_c0/'

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
    ra1, dec1 = convert_theta_phi2radec(100,100,att_q)
    print('test ra, dec=',ra1,dec1)

    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5+1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)
    logging.debug("Number of ebins: ")
    logging.debug(nebins)

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
        (ev_data['ENERGY']<=500.)&(ev_data['ENERGY']>=14.)&\
        (mask_vals==0.)&(ev_data['TIME']<=t_end)&\
        (ev_data['TIME']>=t_start)

    #logging.info("Number of events passing cuts: ", np.sum(bl_ev))
    ev_data0 = ev_data[bl_ev]

   # logging.info(GTI_PNT)

   # for row in GTI_PNT:
   #     print(row['START'] - trigger_time, row['STOP'] - trigger_time)

        
    bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
    bkg_df = pd.read_csv(bkg_fname)


# create bkg object

    try:
        GTI = Table.read(evfname, hdu="GTI_POINTING")
    except:
        GTI = Table.read(evfname, hdu="GTI")
    tmin = GTI_PNT["START"][0]
    tmax = GTI_PNT["STOP"][-1]
    twind = 20
# print(GTI,tmin,tmax,twind)

    poly_trng = np.int(twind)

    bkg_obj = Linear_Rates(ev_data, tmin, tmax, trigger_time, GTI_PNT, sig_clip=4.0, poly_trng=poly_trng)
#print(bkg_obj)
    bkg_obj.do_fits()



# Getting rate and std deviations for all durations:

   # dur_values=[0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 16.384]
    dur_values=[1.024]
    for k in range(len(dur_values)):
        #print('k=',k,'duration:',dur_values[k])
# calculate bkg rate and rate error
        twind = 20
        #dur = 1.024 #  1.024 s
# min_dt = -twind
        dtmin = -20
        dtmax = twind

        tstep = dur_values[k] / 4.0
        tbins0 = np.arange(dtmin, dtmax, tstep) + trigger_time
        tbins1 = tbins0 + dur_values[k]
        tcnts = get_cnts_tbins_fast(tbins0, tbins1, ev_data)

# snrs = calc_rate_snrs(tbins0, tbins1, tcnts, bkg_obj)
# print(snrs)
#
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
        #print('max of sig2_bkg=',max_sig2_bkg)
        #print('sig2_bkg=',sig2_bkg)
        logging.debug('max sig2_bkg:')
        logging.debug(max_sig2_bkg) 
        logging.debug('bkg rate:')
        logging.debug(bkg_rate)
        logging.debug('bkg rate error:')
        logging.debug(bkg_rate_err)


# UL


        Ndets_tot = 32768.0 # total dets 
        Ndets_active = ndets   # number of active detectors
        #print('No. of avtive detectors =',ndets)

        Ndet_ratio = Ndets_active / Ndets_tot
    
# new using new responses ----

        theta_values=["0.000","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","109.471","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","120.000","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","131.810","144.341","144.341","144.341","144.341","144.341","144.341","144.341","144.341","144.341","144.341","144.341","144.341","156.444","156.444","156.444","156.444","156.444","156.444","156.444","156.444","16.699","16.699","168.284","168.284","168.284","168.284","180.000","21.801","21.801","26.565","26.565","26.565","26.565","30.964","30.964","35.796","35.796","35.796","35.796","38.660","38.660","40.511","40.511","40.511","40.511","41.987","41.987","44.564","44.564","44.564","44.564","45.000","45.000","45.000","45.000","50.194","50.194","50.194","50.194","50.292","50.292","50.292","50.292","51.046","51.046","51.046","51.046","51.671","51.671","51.671","51.671","53.301","53.301","53.301","53.301","55.264","55.264","55.264","55.264","56.310","56.310","56.310","56.310","56.310","56.310","57.212","57.212","57.212","57.212","57.995","57.995","58.438","58.438","58.438","58.438","59.491","59.491","59.491","59.491","59.534","59.534","59.534","59.534","59.664","59.664","59.664","59.664","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","60.000","61.421","61.421","61.421","61.421","62.499","62.499","62.499","62.499","63.435","63.435","63.435","63.435","65.489","65.489","65.489","65.489","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","70.529","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","80.406","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","90.000","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594","99.594"]    

        phi_values=["-0.000","101.250","11.250","123.750","146.250","168.750","191.250","213.750","236.250","258.750","281.250","303.750","326.250","33.750","348.750","56.250","78.750","0.000","112.500","135.000","157.500","180.000","202.500","225.000","22.500","247.500","270.000","292.500","315.000","337.500","45.000","67.500","90.000","101.250","11.250","123.750","146.250","168.750","191.250","213.750","236.250","258.750","281.250","303.750","326.250","33.750","348.750","56.250","78.750","105.000","135.000","15.000","165.000","195.000","225.000","255.000","285.000","315.000","345.000","45.000","75.000","112.500","157.500","202.500","22.500","247.500","292.500","337.500","67.500","270.000","90.000","135.000","225.000","315.000","45.000","0.000","-0.000","180.000","143.130","216.870","323.130","36.870","270.000","90.000","123.690","236.310","303.690","56.310","-0.000","180.000","159.444","200.556","20.556","339.444","270.000","90.000","113.962","246.038","293.962","66.038","143.130","216.870","323.130","36.870","-0.000","180.000","270.000","90.000","131.634","228.366","311.634","48.366","14.036","165.964","194.036","345.964","108.435","251.565","288.435","71.565","153.435","206.565","26.565","333.435","123.690","236.310","303.690","56.310","143.130","216.870","270.000","323.130","36.870","90.000","104.931","255.069","284.931","75.069","-0.000","180.000","10.620","169.380","190.620","349.380","135.000","225.000","315.000","45.000","118.072","241.928","298.072","61.928","159.444","200.556","20.556","339.444","0.000","112.500","135.000","157.500","180.000","202.500","225.000","22.500","247.500","270.000","292.500","315.000","337.500","45.000","67.500","90.000","150.642","209.358","29.358","330.642","128.660","231.340","308.660","51.340","143.130","216.870","323.130","36.870","136.848","223.152","316.848","43.152","101.250","11.250","123.750","146.250","168.750","191.250","213.750","236.250","258.750","281.250","303.750","326.250","33.750","348.750","56.250","78.750","0.000","112.500","135.000","157.500","180.000","202.500","225.000","22.500","247.500","270.000","292.500","315.000","337.500","45.000","67.500","90.000","101.250","11.250","123.750","146.250","168.750","191.250","213.750","236.250","258.750","281.250","303.750","326.250","33.750","348.750","56.250","78.750","0.000","112.500","135.000","157.500","180.000","202.500","225.000","22.500","247.500","270.000","292.500","315.000","337.500","45.000","67.500","90.000"]

        float_theta = []
        float_phi = []

        for element in theta_values:
            float_theta_value = float(element)
            float_theta.append(float_theta_value)
        for element2 in phi_values:
            float_phi_value = float(element2)
            float_phi.append(float_phi_value)
       
        float_theta = np.array(float_theta)
        float_phi = np.array(float_phi) 
        #print('float_theta=',float_theta)
        #print('float_phi=',float_phi)
        ras, decs = convert_theta_phi2radec(float_theta, float_phi, att_q)
        #print('ras=',ras)
        #print('decs=',decs)            

        rate_std = np.sqrt(max_sig2_bkg)
        rate_upper_limit = 5*rate_std
       # ul_5sigma_new = []

   # Using 4 spectral templates
        alpha_values = [-1.0,-1.9,0.62,-1.5]
        beta_values = [-2.3,-3.7,0.0,0.0]
        Epeak_values = [230.0,70.0,185.0,1500.0]

        for index,j in enumerate(alpha_values):
        #for j in range(1):
                      
            if index == 0:
                spec_temp = "Band_Normal_template_Ep230keV_alpha-1.0_beta-2.3"
            elif index == 1:
                spec_temp = "Band_Soft_template_Ep70keV_alpha-1.9_beta_-3.7"
            elif index == 2:
                spec_temp = "Cutoff_PL_GW170817-like_Ep185keV_alpha0.62"
            elif index == 3:
                spec_temp = "Cutoff_PL_hard_Ep1500_keV_alpha-1.5"
            
            ul_5sigma_new = []

            for i in range(len(theta_values)):
           # for i in range(3):
                drm_tab_new = get_resp4ul_tab(theta_values[i],phi_values[i])

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
                
                ul_5sigma_new.append(flux_upper_limit_new)
            
            filename = f"ul_5sigma_{spec_temp}_{dur_values[k]}.txt"
            with open(filename, "w") as file:
                #for item in ul_5sigma_new:
                    #file.write(str(item) + "\n")
                for i in range(len(ul_5sigma_new)):
                    file.write(f"{ul_5sigma_new[i]}\t{theta_values[i]}\t{phi_values[i]}\t{ras[i]}\t{decs[i]}\n")    
            logging.debug('5 sigma UL values for:')
            logging.debug(spec_temp)
            logging.debug('and time bin')
            logging.debug(dur_values[k])
            logging.debug('-----')
            logging.debug(ul_5sigma_new)
            logging.debug('------')


    elapsed_time = time.time() - start_time

# Print the elapsed time
    print(f"Time taken: {elapsed_time} seconds")

# ------------------------------------
# Old using grid_ids for infov----

#    IDs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33]
#    rate_std = np.sqrt(sig2_bkg) # the std of rates from the LC
#    rate_upper_limit = 5*rate_std
##    # print("5sigma rate upper limit: ", rate_upper_limit)
#    
#    ul_5sigma = [] 
#    
#    for i in IDs:
#        grid_id = i 
##    # print("grid_id: ", grid_id)
##
##    # using energy bin 15-350 and ignoring 350-500
#        chan_low = 0 
#        chan_hi = 3
##
##    # getting the NITRATES DRM table
#        drm_tab = get_drm_tab(grid_id)
##
##    # response matrix using selected energy bins and corrected for number of active dets
#        drm_matrix = drm_tab['MATRIX'][:,chan_low:(chan_hi+1)] * Ndet_ratio 
##
##    # find the flux that gives an expected rate equal to the rate upper limit
#        flux_upper_limit = rate2band_eflux(rate_upper_limit, drm_matrix,\
#                                                   drm_tab['ENERG_LO'], drm_tab['ENERG_HI'],\
#                                                   alpha, beta, Epeak, flux_elo, flux_ehi)
##    #print("5-sigma flux upper limit [erg/cm2/s]: ", flux_upper_limit)
#        ul_5sigma.append(flux_upper_limit)
###     np.save(os.path.join('test-GW200112_155838', name + "_BAT_gridID_ul.npy"), ul_5sigma)
##
##
#    logging.debug('5 sigma UL=')
#    logging.debug(ul_5sigma)
#    #print("5 sigma UL old values=",ul_5sigma)
#

if __name__ == "__main__":
    args = cli()

    main(args)
