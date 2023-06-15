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

from ..lib.event2dpi_funcs import det2dpis, mask_detxy
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_info_tab


from ..analysis_seeds.do_full_rates import *

from ..lib.calc_BAT_ul import *

work_dir='/gpfs/group/jak51/default/F702399789'
#work_dir='/gpfs/group/jak51/default/nitrates_all_results/gw_subthresh/done/S200107i/'

conn = get_conn(os.path.join(work_dir,'results.db'))
info_tab = get_info_tab(conn)
info_tab.keys()

trigger_time = info_tab['trigtimeMET'][0]
print("trigger time: ", trigger_time)
# default file names made by do_data_setup.py
evfname = os.path.join(work_dir,'filter_evdata.fits')
ev_data = fits.open(evfname)[1].data
# GTI extensions added to event fits file by do_data_setup.py
GTI_PNT = Table.read(evfname, hdu='GTI_POINTING') # when the analysis can be run (good data time and pointing)
GTI_SLEW = Table.read(evfname, hdu='GTI_SLEW') # slewing times
# the final set of detectors to mask (disabled dets, hot/cold dets, and dets with glitches)  
dmask = fits.open(os.path.join(work_dir,'detmask.fits'))[0].data
attfile = fits.open(os.path.join(work_dir,'attitude.fits'))[1].data

ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5+1))[:-1]
ebins0 = np.round(ebins0, decimals=1)[:-1]
ebins1 = np.append(ebins0[1:], [350.0])
nebins = len(ebins0)
print("Number of ebins: ", nebins)

# number of detectors being used
ndets = np.sum(dmask==0)
print("Ndets: "), np.sum(dmask==0)

t_end = trigger_time + 1e3
t_start = trigger_time - 1e3
mask_vals = mask_detxy(dmask, ev_data)

print(mask_vals)
bl_dmask = (dmask==0.)

# get rid of events:
# far away from trigger time
# from bad dets
# with bad event flags
bl_ev = (ev_data['EVENT_FLAGS']<1)&\
        (ev_data['ENERGY']<=500.)&(ev_data['ENERGY']>=14.)&\
        (mask_vals==0.)&(ev_data['TIME']<=t_end)&\
        (ev_data['TIME']>=t_start)

print("Number of events passing cuts: ", np.sum(bl_ev))
ev_data0 = ev_data[bl_ev]

print(GTI_PNT)

for row in GTI_PNT:
    print(row['START'] - trigger_time, row['STOP'] - trigger_time)

        
bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
bkg_df = pd.read_csv(bkg_fname)


# create bkg object

try:
    GTI = Table.read(evfname, hdu="GTI_POINTING")
except:
    GTI = Table.read(evfname, hdu="GTI")
tmin = GTI["START"][0]
tmax = GTI["STOP"][-1]
twind = 20
# print(GTI,tmin,tmax,twind)

poly_trng = np.int(twind)

bkg_obj = Linear_Rates(ev_data, tmin, tmax, trigger_time, GTI_PNT, sig_clip=4.0, poly_trng=poly_trng)
print(bkg_obj)
bkg_obj.do_fits()

# calculate bkg rate and rate error
twind = 20
dur = 1.024 #  1.024 s
# min_dt = -twind
dtmin = -20
dtmax = twind

tstep = dur / 4.0
tbins0 = np.arange(dtmin, dtmax, tstep) + trigger_time
tbins1 = tbins0 + dur
tcnts = get_cnts_tbins_fast(tbins0, tbins1, ev_data)

# snrs = calc_rate_snrs(tbins0, tbins1, tcnts, bkg_obj)
# print(snrs)

ntbins = len(tbins0)
snrs = np.zeros(ntbins)
for i in range(ntbins):
    dur = tbins1[i] - tbins0[i]
    tmid = (tbins1[i] + tbins0[i]) / 2.0
    bkg_rate, bkg_rate_err = bkg_obj.get_rate(tmid)
    sig2_bkg = (bkg_rate_err * dur) ** 2 + (bkg_rate * dur)
    snrs[i] = (tcnts[i] - bkg_rate * dur) / np.sqrt(sig2_bkg)
    
print(bkg_rate,bkg_rate_err)


# UL


Ndets_tot = 32768.0 # total dets 
Ndets_active = ndets   # number of active detectors
print('No. of avtive detectors =',ndets)

Ndet_ratio = Ndets_active / Ndets_tot

IDs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33]
rate_std = np.sqrt(sig2_bkg) # the std of rates from the LC
rate_upper_limit = 5*rate_std
    # print("5sigma rate upper limit: ", rate_upper_limit)
    
ul_5sigma = [] 
    
for i in IDs:
    grid_id = i 
    # print("grid_id: ", grid_id)

    # using energy bin 15-350 and ignoring 350-500
    chan_low = 0 
    chan_hi = 3

    # getting the NITRATES DRM table
    drm_tab = get_drm_tab(grid_id)

    # response matrix using selected energy bins and corrected for number of active dets
    drm_matrix = drm_tab['MATRIX'][:,chan_low:(chan_hi+1)] * Ndet_ratio 

    # find the flux that gives an expected rate equal to the rate upper limit
    flux_upper_limit = rate2band_eflux(rate_upper_limit, drm_matrix,\
                                               drm_tab['ENERG_LO'], drm_tab['ENERG_HI'],\
                                               alpha, beta, Epeak, flux_elo, flux_ehi)
    #print("5-sigma flux upper limit [erg/cm2/s]: ", flux_upper_limit)
    ul_5sigma.append(flux_upper_limit)
#     np.save(os.path.join('test-GW200112_155838', name + "_BAT_gridID_ul.npy"), ul_5sigma)


print('5 sigma UL=',ul_5sigma)
