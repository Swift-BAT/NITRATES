#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
from scipy import optimize, stats, interpolate
from scipy.integrate import quad
import argparse
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
import sys
import pandas as pd
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
import healpy as hp
from copy import copy, deepcopy
# sys.path.append('BatML/')
import logging, traceback
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# In[2]:


NITRATES_path = '/Users/tparsota/Documents/BAT_SCRIPTS/NITRATES/NITRATES_GIT/NITRATES'

#%cd {NITRATES_path}


# In[3]:


# if the response directories are all put in the same directory
# and given as an env variable (like in this cell), then config.py will 
# be able to find them correctly as long as they keep the original names 
# (Ex: ray trace directotry is still called 'ray_traces_detapp_npy')
# These paths can also just be hardcoded into config.py
# most of these paths can also be given directly as an argument to 
# some analysis objects, like Source_Model_InOutFoV()
os.environ['NITRATES_RESP_DIR'] = '/Users/tparsota/Documents/BAT_SCRIPTS/NITRATES_BAT_RSP_FILES/'


# In[4]:


from config import rt_dir
from ray_trace_funcs import RayTraces
from event2dpi_funcs import det2dpis, mask_detxy
from flux_models import Cutoff_Plaw_Flux, Plaw_Flux, get_eflux_from_model
from models import Source_Model_InOutFoV, Bkg_Model_wFlatA,                CompoundModel, Point_Source_Model_Binned_Rates, im_dist
from ray_trace_funcs import RayTraces
from sqlite_funcs import get_conn
from dbread_funcs import get_info_tab, get_twinds_tab
from hp_funcs import ang_sep
from coord_conv_funcs import theta_phi2imxy, imxy2theta_phi, convert_imxy2radec,                            convert_radec2thetaphi, convert_radec2imxy
from do_llh_inFoV4realtime2 import parse_bkg_csv
from LLH import LLH_webins
from minimizers import NLLH_ScipyMinimize_Wjacob


# In[5]:


#%matplotlib inline


# In[6]:


ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5+1))[:-1]
ebins0 = np.round(ebins0, decimals=1)[:-1]
ebins1 = np.append(ebins0[1:], [350.0])
nebins = len(ebins0)
print("Nebins: ", nebins)
print("ebins0: ", ebins0)
print("ebins1: ", ebins1)

# work_dir = '/storage/work/jjd330/local/bat_data/realtime_workdir/F646018360/'
work_dir = os.path.join(NITRATES_path, 'F646018360')
conn = get_conn(os.path.join(work_dir,'results.db'))
info_tab = get_info_tab(conn)
trigger_time = info_tab['trigtimeMET'][0]
print('trigger_time: ', trigger_time)

evfname = os.path.join(work_dir,'filter_evdata.fits')
ev_data = fits.open(evfname)[1].data
GTI_PNT = Table.read(evfname, hdu='GTI_POINTING')
GTI_SLEW = Table.read(evfname, hdu='GTI_SLEW')
dmask = fits.open(os.path.join(work_dir,'detmask.fits'))[0].data
attfile = fits.open(os.path.join(work_dir,'attitude.fits'))[1].data

att_ind = np.argmin(np.abs(attfile['TIME'] - trigger_time))
att_quat = attfile['QPARAM'][att_ind]
print('attitude quaternion: ', att_quat)

ndets = np.sum(dmask==0)
print("Ndets: ", np.sum(dmask==0))

t_end = trigger_time + 1e3
t_start = trigger_time - 1e3
mask_vals = mask_detxy(dmask, ev_data)
bl_dmask = (dmask==0.)

bl_ev = (ev_data['EVENT_FLAGS']<1)&        (ev_data['ENERGY']<=500.)&(ev_data['ENERGY']>=14.)&        (mask_vals==0.)&(ev_data['TIME']<=t_end)&        (ev_data['TIME']>=t_start)

print("Nevents: ",np.sum(bl_ev))
ev_data0 = ev_data[bl_ev]


# In[7]:


ra, dec = 233.117, -26.213 
theta, phi = convert_radec2thetaphi(ra, dec, att_quat)
print(theta, phi)
imx, imy = convert_radec2imxy(ra, dec, att_quat)
print(imx, imy)


# In[8]:


flux_params = {'A':1.0, 'gamma':0.5, 'Epeak':1e2}
flux_mod = Cutoff_Plaw_Flux(E0=100.0)


# In[9]:


# use rt_dir from config.py or
# rt_dir = '/path/to/ray_traces_detapp_npy/'
rt_obj = RayTraces(rt_dir)


# In[10]:


rt = rt_obj.get_intp_rt(imx, imy)
print(np.shape(rt))


# In[11]:


#%%time
# will by default use the resp directories from config.py
sig_mod = Source_Model_InOutFoV(flux_mod, [ebins0,ebins1], bl_dmask,                                rt_obj, use_deriv=True)
# or the paths can be given
# resp_tab_dname = '/path/to/resp_tabs_ebins/'
# hp_flor_resp_dname = '/path/to/hp_flor_resps/'
# comp_flor_resp_dname = '/path/to/comp_flor_resps/'
# sig_mod = Source_Model_InOutFoV(flux_mod, [ebins0,ebins1], bl_dmask,\
#                                 rt_obj, use_deriv=True,\
#                                 resp_tab_dname=resp_tab_dname,\
#                                 hp_flor_resp_dname=hp_flor_resp_dname,\
#                                 comp_flor_resp_dname=comp_flor_resp_dname)


# In[12]:


#%%time
sig_mod.set_theta_phi(theta, phi)


# In[13]:


from config import solid_angle_dpi_fname
# or give the path directly
# solid_angle_dpi_fname = '/path/to/solid_angle_dpi.npy'
bkg_fname = os.path.join(work_dir,'bkg_estimation.csv')
solid_ang_dpi = np.load(solid_angle_dpi_fname)
bkg_df, bkg_name, PSnames, bkg_mod, ps_mods = parse_bkg_csv(bkg_fname, solid_ang_dpi,                    ebins0, ebins1, bl_dmask, rt_dir)
bkg_mod.has_deriv = False
bkg_mod_list = [bkg_mod]
Nsrcs = len(ps_mods)
if Nsrcs > 0:
    bkg_mod_list += ps_mods
    for ps_mod in ps_mods:
        ps_mod.has_deriv = False
    bkg_mod = CompoundModel(bkg_mod_list)
tmid = trigger_time
bkg_row = bkg_df.iloc[np.argmin(np.abs(tmid - bkg_df['time']))]
bkg_params = {pname:bkg_row[pname] for pname in            bkg_mod.param_names}
bkg_name = bkg_mod.name
bkg_params


# In[14]:


pars_ = {}
pars_['Signal_theta'] = theta
pars_['Signal_phi'] = phi
for pname,val in list(bkg_params.items()):
    pars_[bkg_name+'_'+pname] = val
for pname,val in list(flux_params.items()):
    pars_['Signal_'+pname] = val
pars_


# In[15]:


comp_mod = CompoundModel([bkg_mod, sig_mod])

sig_miner = NLLH_ScipyMinimize_Wjacob('')

sig_llh_obj = LLH_webins(ev_data0, ebins0, ebins1, bl_dmask, has_err=True)

sig_llh_obj.set_model(comp_mod)

sig_miner.set_llh(sig_llh_obj)


# In[16]:


fixed_pnames = list(pars_.keys())
fixed_vals = list(pars_.values())
trans = [None for i in range(len(fixed_pnames))]
sig_miner.set_trans(fixed_pnames, trans)
sig_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
sig_miner.set_fixed_params(['Signal_A'], fixed=False)


# In[17]:


#%%time
flux_params['gamma'] = 0.8
flux_params['Epeak'] = 350.0
sig_mod.set_flux_params(flux_params)


# In[18]:


t0 = trigger_time - 0.512
t1 = t0 + 2.048
sig_llh_obj.set_time(t0, t1)


# In[19]:


#%%time
pars, nllh, res = sig_miner.minimize()


# In[20]:


print(res)
print(nllh)
print(pars)


# In[21]:


#%%time
pars_['Signal_A'] = 1e-10
bkg_nllh = -sig_llh_obj.get_logprob(pars_)
print (bkg_nllh)
sqrtTS = np.sqrt(2.*(bkg_nllh - nllh[0]))
print (sqrtTS)


# In[22]:


vars(sig_mod)


# In[ ]:


normed_comp_flor_rate_dpis, normed_photoe_rate_dpis,normed_rate_dpis, normed_err_rate_dpis


# In[24]:


vars(sig_mod.resp_obj)


# In[ ]:


resp_files, phis2use, wts

