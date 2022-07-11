#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import pandas as pd
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
import healpy as hp
from copy import copy, deepcopy
import logging, traceback
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# In[31]:


# cd to code directory
get_ipython().run_line_magic('cd', '/Users/tparsota/Documents/BAT_SCRIPTS/NITRATES/NITRATES_GIT/NITRATES')
#/storage/work/j/jjd330/local/bat_data/BatML


# In[32]:


from event2dpi_funcs import det2dpis, mask_detxy
from flux_models import Cutoff_Plaw_Flux, Plaw_Flux, get_eflux_from_model
from sqlite_funcs import get_conn
from dbread_funcs import get_info_tab
from do_manage2 import im_dist, get_rate_res_fnames, get_peak_res_fnames, get_out_res_fnames,                    get_merged_csv_df, get_merged_csv_df_wpos
from hp_funcs import ang_sep
from coord_conv_funcs import theta_phi2imxy, imxy2theta_phi, convert_imxy2radec,                            convert_radec2thetaphi, convert_radec2imxy
from do_llh_inFoV4realtime2 import parse_bkg_csv
from LLH import LLH_webins
from minimizers import NLLH_ScipyMinimize_Wjacob


# In[33]:


ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5+1))[:-1]
ebins0 = np.round(ebins0, decimals=1)[:-1]
ebins1 = np.append(ebins0[1:], [350.0])
nebins = len(ebins0)
print("Number of ebins: ", nebins)


# directory with results
work_dir = '/Users/tparsota/Documents/BAT_SCRIPTS/NITRATES/F646018360/'
#'/storage/work/j/jjd330/local/bat_data/realtime_workdir/F646018360/'

conn = get_conn(os.path.join(work_dir,'results.db'))
info_tab = get_info_tab(conn)
# trigger time stored in results.db (sqlite DB)
# not much else stored there (results used to be stored there but not anymore)
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

# number of detectors being used
ndets = np.sum(dmask==0)
print ("Ndets: ", np.sum(dmask==0))

t_end = trigger_time + 1e3
t_start = trigger_time - 1e3
mask_vals = mask_detxy(dmask, ev_data)
bl_dmask = (dmask==0.)

# get rid of events:
# far away from trigger time
# from bad dets
# with bad event flags
bl_ev = (ev_data['EVENT_FLAGS']<1)&        (ev_data['ENERGY']<=500.)&(ev_data['ENERGY']>=14.)&        (mask_vals==0.)&(ev_data['TIME']<=t_end)&        (ev_data['TIME']>=t_start)

print("Number of events passing cuts: ", np.sum(bl_ev))
ev_data0 = ev_data[bl_ev]


# In[34]:


print (GTI_PNT)
print()
for row in GTI_PNT:
    print (row['START'] - trigger_time, row['STOP'] - trigger_time)


# In[35]:


bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
bkg_df = pd.read_csv(bkg_fname)


# In[36]:


bkg_df.head()


# In[37]:


# plotting the rate per detector in each energy bin for the diffuse models
nr = 3
nc = 3
nplt = 1
fig = plt.figure(dpi=80, figsize=(14,6))
for j in range(9):
    ax = fig.add_subplot(nr,nc,nplt)
    nplt+=1
    try:
        rate_name = 'bkg_rate_' + str(j)
        rate = bkg_df[rate_name]
    except:
        rate_name = 'Background_bkg_rate_' + str(j)
        rate = bkg_df[rate_name]
    err0 = rate - bkg_df['err_'+rate_name]
    err1 = rate + bkg_df['err_'+rate_name]
    plt.fill_between(bkg_df['dt'], err0, err1, alpha=.5)
    plt.plot(bkg_df['dt'], rate, 'o--')
    plt.grid(True)


# In[38]:


# plotting time hist of event data
tbins = np.arange(-40*1.024, 40*1.024, 0.064*4*4*1) # tbins with multiples of 64ms (64ms * 4 * 4 = 1.024s)
dt = tbins[1] - tbins[0]
tax = (tbins[1:] + tbins[:-1])/2.
ntbins = len(tax)
# choosing which tbins to estimate a flat bkg from for the plot
# bkg_bl = (np.abs(tax)>8.0)
bkg_bl = ((tax)<-10.0)&(tax>-40.0)

nc = 1
nr = 1
nplt = 1

fig = plt.figure(dpi=100, figsize=(8,3*nr))
ax = fig.add_subplot(nr,nc,nplt)
nplt += 1

h=plt.hist(ev_data0['TIME'] - trigger_time, bins=tbins,           histtype='step', label='data')[0]

bkg_mean = np.mean(h[bkg_bl])
bkg_std = np.std(h[bkg_bl])

print("time with min counts, max counts")
print (tbins[np.argmin(h)], tbins[np.argmax(h)])
print("counts at min time, max time")
print(np.min(h), np.max(h))
print("bkg mean, bkg std")
print(bkg_mean, bkg_std)

plt.axhline(bkg_mean)
plt.grid(True)
plt.xlim(-22, 26)
# plt.xlim(-4, 4.)
plt.ylim(np.around(.95*np.min(h[(h>0)]), decimals=-1),         np.round(1.01*np.max(h), decimals=-1))
plt.xlabel('t - trig_time (s)')
plt.ylabel('Counts 15-350 keV')
plt.legend(loc='lower left')


# In[39]:


# plotting time hist with same bins as last cell but for each Ebin
nc = 1
nr = nebins
nplt = 1
fig = plt.figure(dpi=100, figsize=(8,3.5*nr))

for ei in range(nebins):
    ax = fig.add_subplot(nr,nc,nplt)
    nplt += 1
    ebl = (ev_data0['ENERGY']>=ebins0[ei])&            (ev_data0['ENERGY']<ebins1[ei])
    h=plt.hist(ev_data0[ebl]['TIME'] - trigger_time, bins=tbins,               histtype='step')[0]
    bkg_mean = np.mean(h[bkg_bl])
    bkg_std = np.std(h[bkg_bl])
    ttl = '%.1f - %.1f keV' %(ebins0[ei],ebins1[ei])
    print(ttl)
    print("time with min counts, max counts")
    print (tbins[np.argmin(h)], tbins[np.argmax(h)])
    print("counts at min time, max time")
    print(np.min(h), np.max(h))
    print("bkg mean, bkg std")
    print(bkg_mean, bkg_std)
    print
    plt.axhline(bkg_mean)
    plt.title(ttl)
    plt.grid(True)
#     plt.xlim(-2, 2)
    plt.xlim(-22, 26)
    plt.ylim(np.around(.9*np.min(h[(h>0)]), decimals=-1),             np.round(1.1*np.max(h), decimals=-1))


# In[40]:


# getting attitude information at trigger time
att_ind = np.argmin(np.abs(attfile['TIME'] - trigger_time))
att_quat = attfile['QPARAM'][att_ind]
print("QUATERNION: ", att_quat)
pnt_ra, pnt_dec = attfile['POINTING'][att_ind,:2]
print("Pointing RA, Dec")
print(pnt_ra, pnt_dec)
plt.plot(attfile['TIME']-trigger_time, attfile['POINTING'], 'o')
plt.grid(True)
plt.xlim(-50,150)
plt.legend(['ra','dec','roll'])


# In[41]:


# if you have some ra, dec of interest (like a gbm localization or something)
# here's how to do conversions into detector coordinates (theta, phi) and (imx, imy)
ra_interest, dec_interest = 233.117, -26.213
print("RA of interest, Dec of interest")
print(ra_interest, dec_interest)
theta_interest, phi_interest = convert_radec2thetaphi(ra_interest, dec_interest, att_quat)
print("theta, phi")
print (theta_interest, phi_interest)
# imx, imy only valid at theta < 90 deg
imx_interest, imy_interest = convert_radec2imxy(ra_interest, dec_interest, att_quat)
print("imx, imy")
print (imx_interest, imy_interest)


# In[42]:


#%%time
# getting split rate analysis results

# get the file names
res_rate_fnames = get_rate_res_fnames(work_dir)
print("%d split rate result files"%(len(res_rate_fnames)))
# read files and merge into one Pandas Dataframe
res_rate_tab = get_merged_csv_df(res_rate_fnames, work_dir, ignore_index=True)
print("Merged split rate results table has %d rows"%(len(res_rate_tab)))
res_rate_tab['dt'] = res_rate_tab['time'] - trigger_time


# In[43]:


res_rate_tab.sort_values('TS', ascending=False).head(64)


# In[44]:


#%%time
# getting out of FoV analysis results

res_out_fnames = get_out_res_fnames(work_dir)
print("%d out of FoV result files"%(len(res_out_fnames)))

res_out_tab = get_merged_csv_df_wpos(res_out_fnames, attfile, direc=work_dir, ignore_index=True)
print("Merged out of FoV results table has %d rows"%(len(res_out_tab)))
res_out_tab['dt'] = res_out_tab['time'] - trigger_time


# In[45]:


#%%time
# getting in FoV analysis results

res_peak_fnames = get_peak_res_fnames(work_dir)
print("%d in FoV peaks result files"%(len(res_peak_fnames)))
# read files and merge in single dataframe, also convert detector coords into RA, Dec
res_peak_tab = get_merged_csv_df_wpos(res_peak_fnames, attfile, direc=work_dir, ignore_index=True)
print("Merged in FoV peaks results table has %d rows"%(len(res_peak_tab)))
res_peak_tab['dt'] = res_peak_tab['time'] - trigger_time


# In[46]:


# getting the max TS for each square/time seed combo
idx = res_peak_tab.groupby(['squareID','timeID'])['TS'].transform(max) == res_peak_tab['TS']
res_peak_maxSq_tab = res_peak_tab[idx]
print(len(res_peak_maxSq_tab))


# In[47]:


# initing flux model to calculate flux/fluences
flux_params = {'A':1.0, 'gamma':0.5, 'Epeak':1e2}
flux_mod = Cutoff_Plaw_Flux(E0=100.0)


# In[48]:


#%%time
# calculating fluence for each row based on the best fit spectral parameters 
fluncs = np.zeros(len(res_peak_maxSq_tab))
flux_pars = {'A':1.0, 'Epeak':1e2, 'gamma':0.5}
i = 0
for ind, row in res_peak_maxSq_tab.iterrows():
    flux_pars['A'] = row['A']
    flux_pars['gamma'] = row['gamma']
    flux_pars['Epeak'] = row['Epeak']
    fluncs[i] = get_eflux_from_model(flux_mod, flux_pars, 1e1, 1e3)*row['dur'] # fluence from 10keV to 1MeV
    i+=1
#     res_peak_maxSq_tab.loc[ind]['fluence'] = get_eflux_from_model(flux_mod, flux_pars, 1e1, 1e3)*row['dur']
res_peak_maxSq_tab['fluence'] = fluncs


# In[49]:


# max TS peak result
print (np.max(res_peak_maxSq_tab['TS']))
idx = res_peak_maxSq_tab['TS'].idxmax()
row = res_peak_maxSq_tab.loc[idx]
max_TS_timeID = row['timeID']
row


# In[50]:


res_peak_maxSq_tab.sort_values('TS', ascending=False).head(64)


# In[51]:


# max TS out result
print (np.max(res_out_tab['TS']))
idx = res_out_tab['TS'].idxmax()
row = res_out_tab.loc[idx]
max_TSout_timeID = row['timeID']
row


# In[52]:


res_out_tab.sort_values('TS', ascending=False).head(64)


# In[53]:


# get best out of FoV result for each healpix pixel for a certrain time bin
bl = np.isclose(res_out_tab['timeID'],max_TSout_timeID)
idx = res_out_tab[bl].groupby(['hp_ind'])['TS'].transform(max) == res_out_tab[bl]['TS']
res_hpmax_tab = res_out_tab[bl][idx]


# In[54]:


#%%time
# get fluence for each row of res_hpmax_tab
fluncs = np.zeros(len(res_hpmax_tab))
flux_pars = {'A':1.0, 'Epeak':1e2, 'gamma':0.5}
# for i in range(len(res_peak_maxSq_tab)):
i = 0
for ind, row in res_hpmax_tab.iterrows():
    flux_pars['A'] = row['A']
    flux_pars['gamma'] = row['gamma']
    flux_pars['Epeak'] = row['Epeak']
    fluncs[i] = get_eflux_from_model(flux_mod, flux_pars, 1e1, 1e3)*row['dur']
    i+=1
res_hpmax_tab['fluence'] = fluncs


# In[55]:


res_hpmax_tab.sort_values('TS', ascending=False).head(64)


# In[56]:


def get_dlogl_peak_out(res_peak_tab, res_out_tab):
    '''
    returns DeltaLLH_peak and DeltaLLH_out for the time bin with the max TS
    '''
    
    idx = res_peak_tab['TS'].idxmax()
    row = res_peak_tab.loc[idx]
    
    timeID = row['timeID']
    imdists = im_dist(row['imx'], row['imy'], res_peak_tab['imx'], res_peak_tab['imy'])
    bld = (imdists>0.012)&(res_peak_tab['timeID']==timeID)
    
    dlogl_peak = np.nanmin(res_peak_tab[bld]['nllh']) - row['nllh']
    
    blo = (res_out_tab['timeID']==timeID)
    
    dlogl_out = np.nanmin(res_out_tab[blo]['nllh']) - row['nllh']
    
    return dlogl_peak, dlogl_out

def get_dlogls_inout(res_tab, res_out_tab):
    '''
    returns DeltaLLH_peak, DeltaLLH_out, timeID for each time bin
    '''
    
    dlogls = []
    dlogls_in_out = []
    timeIDs = []
    for timeID, df in res_tab.groupby('timeID'):
        idx = df['TS'].idxmax()
        row = df.loc[idx]
        imdists = im_dist(row['imx'], row['imy'], df['imx'], df['imy'])
        bld = (imdists>0.012)
        try:
            dlogls.append(np.nanmin(df[bld]['nllh']) - row['nllh'])
        except Exception as E:
            print(E)
            dlogls.append(np.nan)
        blo = np.isclose(res_out_tab['timeID'],timeID,rtol=1e-9,atol=1e-3)
        dlogls_in_out.append(np.nanmin(res_out_tab[blo]['nllh']) - row['nllh'])
        timeIDs.append(timeID)
    return dlogls, dlogls_in_out, timeIDs


# In[57]:


dlogl_peak, dlogl_out = get_dlogl_peak_out(res_peak_tab, res_out_tab)
print("\Delta LLH_peak = %.3f"%(dlogl_peak))
print("\Delta LLH_out = %.3f"%(dlogl_out))


# In[58]:


# %%time
dlogls_peak, dlogls_out, list_timeIDs = get_dlogls_inout(res_peak_tab, res_out_tab)
for i in range(len(list_timeIDs)):
    print( "timeID = %d"%(list_timeIDs[i]))
    print( "\Delta LLH_peak = %.3f"%(dlogls_peak[i]))
    print ("\Delta LLH_out = %.3f"%(dlogls_out[i]))
    print()


# In[59]:




