import time
import numpy as np
import pandas as pd
import healpy as hp
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from scipy import interpolate
import os, socket, subprocess, shlex
import argparse
import logging, traceback
try:
    import paramiko
    # paramiko is only used when ssh commands are needed to submit jobs
except:
    pass

#from helper_funcs import send_email, send_error_email, send_email_attach, send_email_wHTML
#using the same helperfuncs as open grb even for vc because email settings are same.. July 4th 2022
from helper_funcs_open_grb_realtime import send_email, send_error_email, send_email_attach, send_email_wHTML

from sqlite_funcs import get_conn
from dbread_funcs import get_files_tab, get_info_tab, guess_dbfname
from coord_conv_funcs import convert_radec2imxy, convert_imxy2radec,\
                        convert_radec2thetaphi, convert_theta_phi2radec
from hp_funcs import pc_probmap2good_outFoVmap_inds


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evfname', type=str,\
            help="Event data file",
            default=None)
    parser.add_argument('--fp_dir', type=str,\
            help="Directory where the detector footprints are",
            default='/gpfs/group/jak51/default/gzr5209/bat-data/rtfp_dir_npy/')
    parser.add_argument('--Nrate_jobs', type=int,\
            help="Total number of jobs",
            default=16)
    parser.add_argument('--N_infov_jobs', type=int,\
            help="Number of infov jobs to submit",\
            default=96)
    parser.add_argument('--N_outfov_jobs', type=int,\
            help="Number of outfov jobs to submit",\
            default=24)
    parser.add_argument('--TSscan', type=float,\
            help="Min TS needed to do a full FoV scan",
            default=6.25)
    parser.add_argument('--pix_fname', type=str,\
            help="Name of the file with good imx/y coordinates",\
            default='good_pix2scan.npy')
    parser.add_argument('--bkg_fname', type=str,\
            help="Name of the file with the bkg fits",\
            default='bkg_estimation.csv')
    parser.add_argument('--dbfname', type=str,\
            help="Name to save the database to",\
            default=None)
    parser.add_argument('--GWname', type=str,\
            help="Name of the event to submit jobs as",\
            default='')
    parser.add_argument('--queue', type=str,\
            help="Name of the queue to submit jobs to",\
            default='jak51_b_g_vc_default')
    parser.add_argument('--qos', type=str,\
            help="Name of the qos to submit jobs to",\
            default=None)
    parser.add_argument('--q', type=str,\
            help="Name of the q to submit jobs to",\
	    default='hprc')
    parser.add_argument('--pcfname', type=str,\
            help="Name of the partial coding image",\
            default='pc_2.img')
    parser.add_argument('--BKGpyscript', type=str,\
            help="Name of python script for Bkg Estimation",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/do_bkg_estimation_wPSs_mp2.py')
    parser.add_argument('--RATEpyscript', type=str,\
            help="Name of python script for Rates analysis",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/do_rates_mle_InOutFoV2.py')
    parser.add_argument('--LLHINpyscript', type=str,\
            help="Name of python script for LLH analysis",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/do_llh_inFoV4realtime2.py')
    parser.add_argument('--LLHOUTpyscript', type=str,\
            help="Name of python script for LLH analysis",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/do_llh_outFoV4realtime2.py')
    # parser.add_argument('--SCANpyscript', type=str,\
    #         help="Name of python script for FoV scan",\
    #         default='do_llh_scan_uncoded.py')
    # parser.add_argument('--PEAKpyscript', type=str,\
    #         help="Name of python script for FoV scan",\
    #         default='do_intLLH_forPeaks.py')
    parser.add_argument('--do_bkg',\
            help="Submit the BKG estimation script",\
            action='store_true')
    parser.add_argument('--do_rates',\
            help="Submit the Rate jobs",\
            action='store_true')
    parser.add_argument('--do_llh',\
            help="Submit the llh jobs",\
            action='store_true')
    parser.add_argument('--do_scan',\
            help="Submit the scan jobs",\
            action='store_true')
    parser.add_argument('--skip_waiting',\
            help="Skip waiting for the stuff to finish and use what's there now",\
            action='store_true')
    parser.add_argument('--archive',\
            help="Run in archive mode, not realtime mode",\
            action='store_true')
    parser.add_argument('--rhel7',\
            help="Submit to a rhel7 node",\
            action='store_true')
    parser.add_argument('--pbs_fname', type=str,\
            help="Name of pbs script",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/submission_scripts/pyscript_template_g3.pbs')
    parser.add_argument('--pbs_rhel7_fname', type=str,\
            help="Name of pbs script",\
            default='/storage/home/gzr5209/work/BatML_code_work/NITRATES/submission_scripts/pyscript_template_rhel7_g3.pbs')
    parser.add_argument('--min_pc', type=float,\
            help="Min partical coding fraction to use",\
            default=0.1)
    parser.add_argument('--twind', type=float,\
            help="Number of seconds to go +/- from the trigtime",\
            default=20)
    parser.add_argument('--rateTScut', type=float,\
            help="Min split det TS for seeding",\
            default=4.5)
    args = parser.parse_args()
    return args

def im_dist(imx0, imy0, imx1, imy1):

    return np.hypot(imx0 - imx1, imy0 - imy1)


def get_rate_res_fnames(direc='.'):

    rate_fnames = [fname for fname in os.listdir(direc) if ('rates' in fname) and (fname[-4:]=='.csv')]
    return rate_fnames

def get_res_fnames(direc='.'):

    res_fnames = [fname for fname in os.listdir(direc) if (fname[-4:]=='.csv') and (fname[:4]=='res_')]
    return res_fnames

def get_scan_res_fnames(direc='.'):

    res_fnames = [fname for fname in os.listdir(direc) if (fname[-4:]=='.csv') and ('scan_res_' in fname)]
    return res_fnames

# def get_peak_res_fnames(direc='.'):
#
#     res_fnames = [fname for fname in os.listdir(direc) if (fname[-4:]=='.csv') and ('peak_scan_' in fname)]
#     return res_fnames


def get_in_res_fnames(direc='.'):

    rate_fnames = [fname for fname in os.listdir(direc) if (fname[:3]=='res') and\
                   (fname[-4:]=='.csv') and (not 'hpind' in fname)]
    return rate_fnames

def get_peak_res_fnames(direc='.'):

    rate_fnames = [fname for fname in os.listdir(direc) if (fname[:4]=='peak') and\
                   (fname[-4:]=='.csv') and (not 'hpind' in fname)]
    return rate_fnames


def get_out_res_fnames(direc='.'):

    rate_fnames = [fname for fname in os.listdir(direc) if (fname[:3]=='res') and\
                   (fname[-4:]=='.csv') and ('hpind' in fname)]
    return rate_fnames



def get_merged_csv_df(csv_fnames, direc=None, ignore_index=False):
    dfs = []
    for csv_fname in csv_fnames:
        try:
            if direc is None:
                dfs.append(pd.read_csv(csv_fname, dtype={'timeID':np.int}))
            else:
                dfs.append(pd.read_csv(os.path.join(direc,csv_fname), dtype={'timeID':np.int}))
        except Exception as E:
            logging.error(E)
            continue
    df = pd.concat(dfs, ignore_index=ignore_index)
    return df


def probm2perc(pmap):

    bl = (pmap>0)
    p_map = np.copy(pmap)
    inds_sort = np.argsort(p_map)[::-1]
    perc_map = np.zeros_like(p_map)
    perc_map[inds_sort] = np.cumsum(p_map[inds_sort])#*\
    perc_map[~bl] = 1.
    return perc_map

def get_merged_csv_df_wpos(csv_fnames, attfile, perc_map=None, direc=None, dur_name='dur', ignore_index=False):
    dfs = []
    for csv_fname in csv_fnames:
        try:
            if direc is None:
                tab = pd.read_csv(csv_fname, dtype={'timeID':np.int})
            else:
                tab = pd.read_csv(os.path.join(direc,csv_fname), dtype={'timeID':np.int})
            if len(tab) > 0:
                # att_ind = np.argmin(np.abs(attfile['TIME'] - trigger_time))
                # att_quat = attfile['QPARAM'][att_ind]
                # ras = np.zeros(len(tab))
                # decs = np.zeros(len(tab))
                # for i in xrange(len(ras)):
                # #     print np.shape(res_tab['time'][i]), np.shape(attfile['TIME'])
                #     att_ind0 = np.argmin(np.abs(tab['time'][i] + tab['duration'][i]/2. - attfile['TIME']))
                #     att_quat0 = attfile['QPARAM'][att_ind0]
                #     ras[i], decs[i] = convert_imxy2radec(tab['imx'][i],\
                #                                          tab['imy'][i],\
                #                                         att_quat0)
                t0_ = np.nanmean(tab['time'] + tab[dur_name]/2.)
                att_ind0 = np.argmin(np.abs(t0_ - attfile['TIME']))
                att_quat0 = attfile['QPARAM'][att_ind0]
                try:
                    ras, decs = convert_imxy2radec(tab['imx'], tab['imy'], att_quat0)
                except:
                    ras, decs = convert_theta_phi2radec(tab['theta'], tab['phi'], att_quat0)
                tab['ra'] = ras
                tab['dec'] = decs
                if not perc_map is None:
                    Nside = hp.npix2nside(len(perc_map))
                    hp_inds = hp.ang2pix(Nside, ras, decs, lonlat=True, nest=True)
                    cred_lvl = perc_map[hp_inds]
                    tab['cls'] = cred_lvl
                dfs.append(tab)
        except Exception as E:
            logging.warning(E)
            continue
    df = pd.concat(dfs, ignore_index=ignore_index)
    return df


def mk_seed_tab4scans(res_tab, pc_fname, rate_seed_tab, TS_min=6.5, im_steps=20, pc_min=0.1):

    PC = fits.open(pc_fname)[0]
    pc = PC.data
    w_t = WCS(PC.header, key='T')

    pcbl = (pc>=(pc_min*.99))
    pc_inds = np.where(pcbl)
    pc_imxs, pc_imys = w_t.all_pix2world(pc_inds[1], pc_inds[0], 0)

    imxax = np.linspace(-2,2,im_steps*4+1)
    imyax = np.linspace(-1,1,im_steps*2+1)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]

    h = np.histogram2d(pc_imxs, pc_imys, bins=bins)[0]
    inds = np.where(h>=10)
    squareIDs_all = np.ravel_multi_index(inds, h.shape)

    df_twinds = res_tab.groupby('timeID')
    seed_tabs = []
    for twind, dft in df_twinds:
        if np.max(dft['TS']) >= TS_min:
            seed_dict = {}
            seed_dict['timeID'] = twind
            seed_dict['dur'] = dft['duration'].values[0]
            seed_dict['time'] = dft['time'].values[0]

            bl_rate_seed = (rate_seed_tab['timeID']==twind)
            squareIDs_done = rate_seed_tab['squareID'][bl_rate_seed]
            squareIDs = squareIDs_all[~np.isin(squareIDs_all, squareIDs_done)]

            seed_dict['squareID'] = squareIDs
            seed_tabs.append(pd.DataFrame(seed_dict))

    seed_tab = pd.concat(seed_tabs)

    return seed_tab

def get_Nrank_val(vals, N):

    if N >= len(vals):
        return np.max(vals)
    val_sort = np.sort(vals)
    return val_sort[N-1]

def do_trng_overlap(tstart0, tend0, tstart1, tend1):

    if tstart0 >= tstart1 and tstart0 < tend1:
        return True
    elif tstart0 < tend1 and tend0 >= tstart1:
        return True
    else:
        return False

def mk_seed_tab(rates_res, TS_min=3.75, im_steps=20):

    imxax = np.linspace(-2,2,im_steps*4+1)
    imyax = np.linspace(-1,1,im_steps*2+1)
    # imyg, imxg = np.meshgrid((imyax[1:]+imyax[:-1])/2., (imxax[1:]+imxax[:-1])/2.)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]

    df_twinds = rates_res.groupby('timeID')
    seed_tabs = []
    for twind, dft in df_twinds:
        maxTS = np.nanmax(dft['TS'])
        if maxTS >= TS_min:

            TS_min2_ = min(maxTS-2.5, .8*maxTS)
            TS_min2 = max(TS_min2_, np.nanmedian(dft['TS']), TS_min-.5)
            seed_dict = {}
            seed_dict['timeID'] = twind
            seed_dict['dur'] = dft['dur'].values[0]
            seed_dict['time'] = dft['time'].values[0]

            pnts = np.vstack([dft['imx'],dft['imy']]).T
            TSintp = interpolate.LinearNDInterpolator(pnts, dft['TS'])
            imxax = np.linspace(-1.8, 1.8, 8*36+1)
            imyax = np.linspace(-1.0, 1.0, 8*20+1)
            xgrid, ygrid = np.meshgrid(imxax, imyax)
            pnts = np.vstack([xgrid.ravel(),ygrid.ravel()]).T
            TSgrid = TSintp(pnts)
            bl = (TSgrid>=(TS_min2))
            xs = xgrid.ravel()[bl]
            ys = ygrid.ravel()[bl]

            h = np.histogram2d(xs, ys, bins=bins)[0]
            inds = np.where(h>0)

            squareIDs = np.ravel_multi_index(inds, h.shape)
            seed_dict['squareID'] = squareIDs
            seed_tabs.append(pd.DataFrame(seed_dict))

    seed_tab = pd.concat(seed_tabs)

    return seed_tab

def get_hist2d_neib_inds(shape, indss, bins):

    sqids = [] #np.empty(0,dtype=np.int)
    imx0s = [] #np.empty(0)
    imx1s = [] #np.empty(0)
    imy0s = [] #np.empty(0)
    imy1s = [] #np.empty(0)

    for ii in range(len(indss[0])):
        inds = (indss[0][ii],indss[1][ii])
        for i in range(-1,2):
            indx0 = inds[0] + i
            indx1 = indx0 + 1
            if indx0 < 0:
                continue
            if indx1 >= len(bins[0]):
                continue
            for j in range(-1,2):
                indy0 = inds[1] + j
                indy1 = indy0 + 1
                if indy0 < 0:
                    continue
                if indy1 >= len(bins[1]):
                    continue
                sqids.append(np.ravel_multi_index([indx0,indy0], shape))
                imx0s.append(bins[0][indx0])
                imx1s.append(bins[0][indx1])
                imy0s.append(bins[1][indy0])
                imy1s.append(bins[1][indy1])
    return sqids, imx0s, imx1s, imy0s, imy1s

def mk_in_seed_tab(rates_res, TS_min=4.5, im_steps=25, max_Ntwinds=8, max_overlaps=4, min_dlogl_cut=12.0):

    imxax = np.linspace(-2,2,im_steps*4+1)
    imyax = np.linspace(-1,1,im_steps*2+1)
    # imyg, imxg = np.meshgrid((imyax[1:]+imyax[:-1])/2., (imxax[1:]+imxax[:-1])/2.)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]

    df_twinds = rates_res.groupby('timeID')

    dur_bins = [0.0, 0.2, 0.5, 1.0, 2.0, 1e4]
    ts_adj_facts = np.array([1.0, 1.05, 1.2, 1.3, 1.35])
    dur_inds = np.digitize(rates_res['dur'], dur_bins)-1

    ts_adj_fact = ts_adj_facts[dur_inds]
    rates_res['TS2'] = rates_res['TS']*ts_adj_fact
    idx = rates_res.groupby(['timeID'])['TS2'].transform(max) == rates_res['TS2']
    rates_res_max_tab = rates_res[idx]
    rates_max_sort = rates_res_max_tab.sort_values('TS2', ascending=False)


    timeIDs = []
    TS2s = []
    dts0 = []
    dts1 = []
    Ntime_seeds = 0

    for row_ind, row in rates_max_sort.iterrows():
        dt0 = row['dt']
        dt1 = dt0 + row['dur']
        TS2 = row['TS2']
        Noverlaps = 0

        if TS2 >= 0.7*np.max(rates_max_sort['TS2']):
            keep = True
        else:
            continue
        for i in range(len(dts0)):
            if do_trng_overlap(dts0[i], dts1[i], dt0, dt1):
                Noverlaps += 1
                if Noverlaps >= max_overlaps:
                    keep = False
                    break
                if TS2 < 0.8*TS2s[i]:
                    keep = False
                    break

        if keep:
            timeIDs.append(row['timeID'])
            dts0.append(dt0)
            dts1.append(dt1)
            TS2s.append(TS2)
            Ntime_seeds += 1
            if Ntime_seeds >= max_Ntwinds:
                break

    print(Ntime_seeds)
    print(timeIDs)
    print(TS2s)

    seed_tabs = []
    for twind, dft in df_twinds:
        if not np.any(np.isclose(twind,timeIDs)):
            continue
        maxTS = np.nanmax(dft['TS'])
        if maxTS >= TS_min:

            print(twind)
            logging.info("twind: %d"%(twind))
            logging.info("maxTS: %.3f"%(maxTS))

            bl = np.isfinite(dft['imx'])&(dft['TS']>1e-2)
            df = dft[bl]

            dlogls = np.max(np.square(df['TS'])) - np.square(df['TS'])

            max_dlogl = np.max(dlogls)
            med_dlogl = np.median(dlogls)
            rank48_dlogl = get_Nrank_val(dlogls.values, 48)
            rank196_dlogl = get_Nrank_val(dlogls.values, 196)

            print("max, med dlogl: ", max_dlogl, med_dlogl)
            print("rank 48, 196 dlogl: ", rank48_dlogl, rank196_dlogl)


            # dlogl_cut at most the median
            # at least dlogl of the 48th ranked pnt
            # dlogl_cut at least 12
            # TS at least TS_min-0.5

            dlogl_cut = max(rank48_dlogl, min_dlogl_cut)
            if len(dlogls) > 96:
                dlogl_cut = min(dlogl_cut, med_dlogl)
                dlogl_cut = min(dlogl_cut, rank196_dlogl)

            print(dlogl_cut)
            print(np.sum(dlogls<=dlogl_cut), len(dlogls))



#             TS_min2_ = min(maxTS-2.5, .8*maxTS)
#             TS_min2 = max(TS_min2_, np.nanmedian(dft['TS']), TS_min-.5)
            seed_dict = {}
            seed_dict['timeID'] = twind
            seed_dict['dur'] = dft['dur'].values[0]
            seed_dict['time'] = dft['time'].values[0]

            pnts = np.vstack([df['imx'],df['imy']]).T
#             TSintp = interpolate.LinearNDInterpolator(pnts, dft['TS'])
            try:
                DLintp = interpolate.LinearNDInterpolator(pnts, np.sqrt(dlogls))
            except Exception as E:
                logging.warn("trouble with creating interp object for timeID %s"%(str(twind)))
                logging.debug("shape(pnts): "+str(np.shape(pnts)))
                logging.error(E)
                logging.error(traceback.format_exc())
                try:
                    logging.info("trying nearest interp")
                    DLintp = interpolate.NearestNDInterpolator(pnts, np.sqrt(dlogls))
                except Exception as E:
                    logging.warn("trouble with creating interp object for timeID %s"%(str(twind)))
                    logging.warn("This failed too so skipping this timeID")
                    logging.error(E)
                    logging.error(traceback.format_exc())
                    continue
            imxax = np.linspace(-1.8, 1.8, 20*36+1)
            imyax = np.linspace(-1.0, 1.0, 20*20+1)
            xgrid, ygrid = np.meshgrid(imxax, imyax)
            pnts = np.vstack([xgrid.ravel(),ygrid.ravel()]).T
#             TSgrid = TSintp(pnts)
            try:
                DLgrid = DLintp(pnts)**2
            except Exception as E:
                logging.warn("trouble calling the interp object so skipping timeID "+str(twind))
                logging.error(E)
                logging.error(traceback.format_exc())
                continue
            TSgrid = np.sqrt( np.max(np.square(df['TS'])) - DLgrid )
            logging.debug("TSgrid min, max: %.3f, %.3f"\
                    %(np.nanmin(TSgrid),np.nanmax(TSgrid)))
#             bl = (TSgrid>=(TS_min2))
            bl = (DLgrid<=(dlogl_cut+0.5))&(TSgrid>=(TS_min-0.5))
            xs = xgrid.ravel()[bl]
            ys = ygrid.ravel()[bl]

            h = np.histogram2d(xs, ys, bins=bins)[0]
            inds = np.where(h>0)

            seed_dict['squareID'],seed_dict['imx0'],seed_dict['imx1'],seed_dict['imy0'],seed_dict['imy1'] =\
            get_hist2d_neib_inds(h.shape, inds, bins)

#             squareIDs = np.ravel_multi_index(inds, h.shape)
#             seed_dict['squareID'] = squareIDs
#             seed_dict['imx0'] = bins[0][inds[0]]
#             seed_dict['imx1'] = bins[0][inds[0]+1]
#             seed_dict['imy0'] = bins[1][inds[1]]
#             seed_dict['imy1'] = bins[1][inds[1]+1]
            seed_tabs.append(pd.DataFrame(seed_dict))

    if len(seed_tabs) < 1:
        return []
    seed_tab = pd.concat(seed_tabs)
    seed_tab.drop_duplicates(inplace=True)

    return seed_tab


def mk_in_seed_tab_archive(rates_res, twind_size, tbin_size=60.0, TS_min=4.5):

    tbins0 = np.arange(-twind_size - 1.0, twind_size + 1.0, tbin_size)
    tbins1 = tbins0 + tbin_size

    Ntbins = len(tbins0)

    seed_tabs = []

    for i in range(Ntbins):

        tbl = (rates_res['dt']>=tbins0[i])&(rates_res['dt']<tbins1[i])
        df = mk_in_seed_tab(rates_res[tbl], TS_min=TS_min)
        if len(df) > 0:
            seed_tabs.append(df)

    seed_tab = pd.concat(seed_tabs, ignore_index=True)
    return seed_tab

def assign_in_seeds2jobs(seed_tab, Nmax_jobs=96):

    SquareIDs, sqID_cnts = np.unique(seed_tab['squareID'], return_counts=True)
    logging.debug("min, max sqID_cnts: %d, %d"%(np.min(sqID_cnts), np.max(sqID_cnts)))
    Nsquares = len(SquareIDs)
    logging.info("Nsquares: %d"%(Nsquares))
    Ntbins = len(np.unique(seed_tab['timeID']))
    logging.info("Ntbins: %d" %(Ntbins))
    Ntot = len(seed_tab)
    logging.info("Ntot Seeds: %d" %(Ntot))
    Ntbins_perSq = float(Ntot) / Nsquares
    logging.info("Tbins per Square: %.2f" %(Ntbins_perSq))

    job_ids = np.zeros(Ntot, dtype=np.int) - 1

    if Nsquares <= Nmax_jobs and Ntbins_perSq < 4:
        Njobs = Nsquares
        for i in range(Nsquares):
            bl = (seed_tab['squareID']==SquareIDs[i])
            job_ids[bl] = i
        logging.debug("Njobs: %d"%(Njobs))

    elif Nsquares <= Nmax_jobs/2 and Ntbins_perSq >= 6:
        Njobs = 2*Nsquares
        job_iter = 0
        for i in range(Nsquares):
            bl = (seed_tab['squareID']==SquareIDs[i])
            Nt = np.sum(bl)
            if Nt == 1:
                job_ids[bl] = job_iter%Njobs
                job_iter += 1
                continue
            job_ids[bl][:(Nt/2)] = job_iter%Njobs
            job_iter += 1
            job_ids[bl][(Nt/2):] = job_iter%Njobs
            job_iter += 1
        logging.debug("Njobs, job_iter: %d, %d"%(Njobs, job_iter))

    elif Nsquares > Nmax_jobs:

        Njobs = Nmax_jobs
        job_iter = 0
        for i in range(Nsquares):
            bl = np.isclose(seed_tab['squareID'],SquareIDs[i])
            Nt = np.sum(bl)
            if Nt > (2*Ntbins_perSq):
                inds_ = np.where(bl)[0]
                job_ids[inds_[:(Nt/2)]] = job_iter%Njobs
                job_iter += 1
                job_ids[inds_[(Nt/2):]] = job_iter%Njobs
                job_iter += 1
                continue
            job_ids[bl] = job_iter%Njobs
            job_iter += 1
        logging.debug("Njobs, job_iter: %d, %d"%(Njobs, job_iter))

    else:

        Njobs = min(Ntot, Nmax_jobs)
        job_iter = 0
        for i in range(Nsquares):
            bl = (seed_tab['squareID']==SquareIDs[i])
            Nt = np.sum(bl)
            if Nt == 1:
                job_ids[bl] = job_iter%Njobs
                job_iter += 1
                continue
            job_ids[bl][:(Nt/2)] = job_iter%Njobs
            job_iter += 1
            job_ids[bl][(Nt/2):] = job_iter%Njobs
            job_iter += 1
        logging.debug("Njobs, job_iter: %d, %d"%(Njobs, job_iter))

    logging.debug("sum(job_ids<0): %d"%(np.sum(job_ids<0)))
    seed_tab['proc_group'] = job_ids

    return seed_tab


def mk_out_seed_tab(in_seed_tab, hp_inds, att_q, Nside=2**4, Nmax_jobs=24):

    tgrps = in_seed_tab.groupby('timeID')
    Ntbins = len(tgrps)

    Npix = len(hp_inds)

    print("Npix: ", Npix)
    Njobs = min(Npix, Nmax_jobs)
    print("Njobs: ", Njobs)

    timeIDs = []
    t0s = []
    durs = []
    for timeID, df in tgrps:
        timeIDs.append(timeID)
        t0s.append(np.nanmean(df['time']))
        durs.append(np.nanmean(df['dur']))

    Nseeds = Npix*Ntbins
    job_ids = np.zeros(Nseeds, dtype=np.int) - 1
    times = np.zeros(Nseeds)
    durss = np.zeros(Nseeds)
    hpinds = np.zeros(Nseeds, dtype=np.int)
    ras = np.zeros(Nseeds)
    decs = np.zeros(Nseeds)
    thetas = np.zeros(Nseeds)
    phis = np.zeros(Nseeds)
    timeIDss = np.empty(Nseeds, dtype=np.array(timeIDs).dtype)

    job_iter = 0
    for i in range(Npix):
        hpind = hp_inds[i]
        ra, dec = hp.pix2ang(Nside, hpind, nest=True, lonlat=True)
        theta, phi = convert_radec2thetaphi(ra, dec, att_q)
        jobid = job_iter%Njobs
        job_iter += 1
        for j in range(Ntbins):
            ind = i*Ntbins + j
            ras[ind] = ra
            decs[ind] = dec
            thetas[ind] = theta
            phis[ind] = phi
            hpinds[ind] = hpind
            job_ids[ind] = jobid
            times[ind] = t0s[j]
            durss[ind] = durs[j]
            timeIDss[ind] = timeIDs[j]

    seed_dict = {'ra':ras, 'dec':decs, 'hp_ind':hpinds,
                 'theta':thetas, 'phi':phis, 'proc_group':job_ids,
                 'time':times, 'dur':durss, 'timeID':timeIDss}
    out_seed_tab = pd.DataFrame(data=seed_dict)

    return out_seed_tab

# def mk_seed_tab(rates_res, TS_min=3.5, im_steps=20):
#
#     imxax = np.linspace(-2,2,im_steps*4+1)
#     imyax = np.linspace(-1,1,im_steps*2+1)
#     im_step = imxax[1] - imxax[0]
#     bins = [imxax, imyax]
#
#     df_twinds = rates_res.groupby('timeID')
#     seed_tabs = []
#     for twind, dft in df_twinds:
#         if np.max(dft['TS']) >= TS_min:
#             seed_dict = {}
#             seed_dict['timeID'] = twind
#             seed_dict['dur'] = dft['dur'].values[0]
#             seed_dict['time'] = dft['time'].values[0]
#
#             pnts = np.vstack([dft['imx'],dft['imy']]).T
#             TSintp = interpolate.LinearNDInterpolator(pnts, dft['TS'])
#             imxax = np.linspace(-1.5, 1.5, 8*30+1)
#             imyax = np.linspace(-.85, .85, 8*17+1)
#             xgrid, ygrid = np.meshgrid(imxax, imyax)
#             pnts = np.vstack([xgrid.ravel(),ygrid.ravel()]).T
#             TSgrid = TSintp(pnts)
#             bl = (TSgrid>=(TS_min-.1))
#             xs = xgrid.ravel()[bl]
#             ys = ygrid.ravel()[bl]
#
#             h = np.histogram2d(xs, ys, bins=bins)[0]
#             inds = np.where(h>0)
#             squareIDs = np.ravel_multi_index(inds, h.shape)
#             seed_dict['squareID'] = squareIDs
#             seed_tabs.append(pd.DataFrame(seed_dict))
#
#     seed_tab = pd.concat(seed_tabs)
#
#     return seed_tab

def mk_job_tab(seed_tab, Njobs, im_steps=20):

    imxax = np.linspace(-2,2,im_steps*4+1)
    imyax = np.linspace(-1,1,im_steps*2+1)
    im_step = imxax[1] - imxax[0]
    bins = [imxax, imyax]
    squareIDs = np.unique(seed_tab['squareID'])
    shp = (len(imxax)-1,len(imyax)-1)
    data_dicts = []
    for i, squareID in enumerate(squareIDs):
        data_dict = {}
        data_dict['proc_group'] = i%Njobs
        indx, indy = np.unravel_index(squareID, shp)
        data_dict['imx0'] = bins[0][indx]
        data_dict['imx1'] = bins[0][indx+1]
        data_dict['imy0'] = bins[1][indy]
        data_dict['imy1'] = bins[1][indy+1]
        data_dict['squareID'] = squareID
        data_dicts.append(data_dict)

    job_tab = pd.DataFrame(data_dicts)

    return job_tab

def execute_ssh_cmd(client, cmd, server, retries=5):

    tries = 0
    while tries < retries:
        try:
            stdin, stdout, stderr = client.exec_command(cmd)
            logging.info("stdout: ")
            sto = stdout.read()
            logging.info(sto)
            return sto
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.error("Messed up with ")
            logging.error(cmd)
            client.close()
            client = get_ssh_client(server)
            tries += 1
            logging.debug("retry %d of %d"%(tries,retries))
    return

def get_ssh_client(server, retries=5):

    tries = 0
    try:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
    except Exception as E:
        logging.error(E)
        logging.error(traceback.format_exc())
        return

    while tries < retries:
        try:
            client.connect(server)
            return client
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            tries += 1
    return







def sub_jobs(njobs, name, pyscript, pbs_fname, queue='jak51_b_g_vc_default',\
                workdir=None, qos=None, q='hprc', ssh=True, extra_args=None,\
                ppn=1, rhel7=False):

    hostname = socket.gethostname()

    if len(name) > 15:
        name = name[:15]

    if 'aci.ics' in hostname and 'amon' not in hostname:
        ssh=False

    if ssh:
        ssh_cmd = 'ssh aci-b.aci.ics.psu.edu "'
        server = 'aci-b.aci.ics.psu.edu'
        server = 'submit.aci.ics.psu.edu'
        #server = 'submit-001.aci.ics.psu.edu'
       # server = 'submit-010.aci.ics.psu.edu'
        # client = paramiko.SSHClient()
        # client.load_system_host_keys()
        # client.connect(server)
        client = get_ssh_client(server)
        # base_sub_cmd = 'qsub %s -A %s -N %s -v '\
        #             %(args.pbs_fname, args.queue, args.name)
        if qos is not None:
            if rhel7:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l qos=%s -l feature=rhel7 -v '\
                            %(pbs_fname, queue, q, name, ppn, qos)
            else:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l qos=%s -v '\
                            %(pbs_fname, queue, q, name, ppn, qos)
        else:
            if rhel7:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l feature=rhel7 -v '\
                            %(pbs_fname, queue, q, name, ppn)
            else:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -v '\
                            %(pbs_fname, queue, q, name, ppn)

    else:
        if qos is not None:
            if rhel7:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l qos=%s -l feature=rhel7 -v '\
                            %(pbs_fname, queue, q, name, ppn, qos)
            else:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l qos=%s -v '\
                            %(pbs_fname, queue, q, name, ppn, qos)
        else:
            if rhel7:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -l feature=rhel7 -v '\
                            %(pbs_fname, queue, q, name, ppn)
            else:
                base_sub_cmd = 'qsub %s -A %s -q %s -N %s -l nodes=1:ppn=%d -v '\
                            %(pbs_fname, queue, q, name, ppn)

    if workdir is None:
        workdir = os.getcwd()
    if extra_args is None:
        extra_args = ""

    cmd = ''
    jobids = []

    for i in xrange(njobs):

        # cmd_ = 'jobid=%d,workdir=%s,njobs=%d,pyscript=%s' %(i,workdir,njobs,pyscript)
        cmd_ = 'jobid=%d,workdir=%s,njobs=%d,pyscript=%s,extra_args="%s"' %(i,workdir,njobs,pyscript,extra_args)
        if ssh:
            cmd += base_sub_cmd + cmd_
            if i < (njobs-1):
                cmd += ' | '
            # cmd = base_sub_cmd + cmd_
            # jbid = execute_ssh_cmd(client, cmd, server)
            # jobids.append(jbid)
            # try:
            #     stdin, stdout, stderr = client.exec_command(cmd)
            #     logging.info("stdout: ")
            #     sto = stdout.read()
            #     logging.info(sto)
            #     jobids.append(sto)
            # except Exception as E:
            #     logging.error(E)
            #     logging.error(traceback.format_exc())
            #     logging.error("Messed up with ")
            #     logging.error(cmd)
        else:
            cmd = base_sub_cmd + cmd_
            logging.info("Trying to submit: ")
            logging.info(cmd)

            try:
                os.system(cmd)
                # subprocess.check_call(cmd, shell=True)
            except Exception as E:
                logging.error(E)
                logging.error("Messed up with ")
                logging.error(cmd)

            time.sleep(0.1)
    if ssh:

        # ssh_cmd = 'ssh aci-b.aci.ics.psu.edu "'
        # cmd = ssh_cmd + cmd + '"'
        logging.info("Full cmd to run:")
        logging.info(cmd)
        try:
            jobids = execute_ssh_cmd(client, cmd, server)
            # os.system(cmd)
            # subprocess.check_call(cmd, shell=True)
            # cmd_list = ['ssh', 'aci-b.aci.ics.psu.edu', '"'+cmd+'"']
            # cmd_list = shlex.split(cmd)
            # logging.info(cmd_list)
            # subprocess.check_call(cmd_list)
        except Exception as E:
            logging.error(E)
            logging.error("Messed up with ")
            logging.error(cmd)
    if ssh:
        client.close()
    return jobids



def find_peaks2scan(res_df, max_dv=10.0, min_sep=8e-3, max_Npeaks=48, min_Npeaks=2, minTS=6.0):

    tgrps = res_df.groupby('timeID')

    peak_dfs = []

    for timeID, df_ in tgrps:

        if np.nanmax(df_['TS']) < minTS:
            continue

        df = df_.sort_values('sig_nllh')
        vals = df['sig_nllh']
#         ind_sort = np.argsort(vals)
        min_val = np.nanmin(df['sig_nllh'])

        peak_dict = {'timeID': int(timeID), 'time':np.nanmean(df['time']),
                     'duration':np.nanmean(df['duration'])}

        imxs_ = np.empty(0)
        imys_ = np.empty_like(imxs_)
        As_ = np.empty_like(imxs_)
        Gs_ = np.empty_like(imxs_)



        for row_ind, row in df.iterrows():

            if row['sig_nllh'] > (min_val + max_dv) and len(imxs_) >= min_Npeaks:
                break
            if len(imxs_) >= max_Npeaks:
                break

            if len(imxs_) > 0:
                imdist = np.min(im_dist(row['imx'], row['imy'], imxs_, imys_))
                if imdist <= min_sep:
                    continue

            imxs_ = np.append(imxs_, [row['imx']])
            imys_ = np.append(imys_, [row['imy']])
            As_ = np.append(As_, [row['A']])
            Gs_ = np.append(Gs_, [row['ind']])

        peak_dict['imx'] = imxs_
        peak_dict['imy'] = imys_
        peak_dict['Signal_A'] = As_
        peak_dict['Signal_gamma'] = Gs_
        peak_dfs.append(pd.DataFrame(peak_dict))

    peaks_df = pd.concat(peak_dfs, ignore_index=True)

    return peaks_df




def main(args):

    fname = 'manager'

    logging.basicConfig(filename=fname+'.log', level=logging.DEBUG,\
                    format='%(asctime)s-' '%(levelname)s- %(message)s')

    f = open(fname+'.pid', 'w')
    f.write(str(os.getpid()))
    f.close()

    logging.info("Wrote pid: %d" %(os.getpid()))

    to = ['gzr5209@psu.edu','delauj2@gmail.com', 'aaron.tohu@gmail.com','jak51@psu.edu']
    subject = ' g3 BATML ' + args.GWname
    body = "Got data and starting analysis"
    try:
        send_email(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    t_0 = time.time()


    has_sky_map = False
    try:
        sky_map_fnames = [fname for fname in os.listdir('.') if\
                        'cWB.fits.gz' in fname or 'bayestar' in fname\
                        or 'skymap' in fname]
        sky_map = hp.read_map(sky_map_fnames[0], field=(0,), nest=True)
        logging.info('Opened sky map')
        perc_map = probm2perc(sky_map)
        logging.info('Made perc map')
        has_sky_map = True
    except Exception as E:
        logging.warning("problem reading skymap")
        logging.error(E)
        logging.error(traceback.format_exc())

    try:
        logging.info('Connecting to DB')
        if args.dbfname is None:
            db_fname = guess_dbfname()
            if isinstance(db_fname, list):
                db_fname = db_fname[0]
        else:
            db_fname = args.dbfname
        conn = get_conn(db_fname)
        info_tab = get_info_tab(conn)
        logging.info('Got info table')
        trigtime = info_tab['trigtimeMET'][0]
        files_tab = get_files_tab(conn)
        logging.info('Got files table')
        attfname = files_tab['attfname'][0]
        evfname = files_tab['evfname'][0]

    except Exception as E:
        logging.warning("problem getting files tab from DB")
        logging.error(E)
        logging.error(traceback.format_exc())
        attfname = 'attitude.fits'
        evfname = 'filter_evdata.fits'

    try:
        attfile = Table.read(attfname)
        logging.info('Opened att file')
    except Exception as E:
        logging.warning("Trouble openning attitude file")
        logging.error(E)
        logging.error(traceback.format_exc())

    try:
        GTI_pnt = Table.read(evfname, hdu='GTI_POINTING')
        logging.info('Opened GTI_pnt')
        logging.info(GTI_pnt)
        tot_exp = 0.0
        for row in GTI_pnt:
            tot_exp += row['STOP'] - row['START']
        logging.info("Total Exposure Time is %.3f seconds"%(tot_exp))
        if tot_exp < 1.0:
            logging.info("Total Pointing time is <1s")
            logging.info("Exiting now")
            body = "Total Pointing time is < 1s, only %.3f seconds. Exiting analysis." %(tot_exp)
            try:
                send_email(subject, body, to)
            except Exception as E:
                logging.error(E)
                logging.error("Trouble sending email")

            return


    except Exception as E:
        logging.warning("Trouble openning GTI file")
        logging.error(E)
        logging.error(traceback.format_exc())


    try:
        good_pix = np.load(args.pix_fname)
        Ngood_pix = len(good_pix)
        if Ngood_pix < 1:
            # stop here
            logging.info("Completely out of FoV")
            # logging.info("Exiting now")
            # body = "Completely out of FoV. Exiting analysis."
            # try:
            #     send_email(subject, body, to)
            # except Exception as E:
            #     logging.error(E)
            #     logging.error("Trouble sending email")
            #
            # return
        Nratejobs = 4
        if Ngood_pix > 5e4:
            Nratejobs = 16
        if Ngood_pix > 1e5:
            Nratejobs = 32
        if Ngood_pix > 2.5e5:
            Nratejobs = 48
        if Ngood_pix > 5e5:
            Nratejobs = 64
    except Exception as E:
        logging.warn("Trouble reading good pix file")
        Nratejobs = 64
        # if args.archive:
        #     Nratejobs = 108
    # just set this to 16 for now
    Nratejobs = 24


    if args.do_bkg:
        logging.info("Submitting bkg estimation job now")
        # try:
        if args.archive:
            extra_args = "--archive --twind %.3f"%(args.twind)
            if args.rhel7:
                sub_jobs(1, 'BKG_'+args.GWname, args.BKGpyscript,\
                        args.pbs_rhel7_fname, queue=args.queue, ppn=4,\
                        extra_args=extra_args, qos=None, rhel7=args.rhel7,q=args.q)
            else:
                sub_jobs(1, 'BKG_'+args.GWname, args.BKGpyscript,\
                        args.pbs_fname, queue=args.queue, ppn=4,\
                        extra_args=extra_args, qos=None, rhel7=args.rhel7,q=args.q)
        else:
            if args.rhel7:
                sub_jobs(1, 'BKG_'+args.GWname, args.BKGpyscript,\
                        args.pbs_rhel7_fname, queue=args.queue,\
                        ppn=4, qos=None, rhel7=args.rhel7,q=args.q)
            else:
                sub_jobs(1, 'BKG_'+args.GWname, args.BKGpyscript,\
                        args.pbs_fname,\
                        queue=args.queue,\
                        ppn=4, qos=None, rhel7=args.rhel7,q=args.q)
        logging.info("Job submitted")
        # except Exception as E:
        #     logging.warn(E)
        #     logging.warn("Might have been a problem submitting")



    #  Wait for bkg job to finish before submitting rates jobs
    dt = 0.0
    t_0 = time.time()
    bkg_fname = 'bkg_estimation.csv'
    while dt < 16*3600.0:
        if os.path.exists(bkg_fname):
            break
        else:
            time.sleep(10.0)
            dt = time.time() - t_0

    if not os.path.exists(bkg_fname):
        logging.info("Didn't do BKG for some reason")
        logging.info("Exiting now")
        body = "Didn't do BKG for some reason. Exiting analysis."
        try:
            send_email(subject, body, to)
        except Exception as E:
            logging.error(E)
            logging.error("Trouble sending email")
        return


    extra_args = "--min_pc %.4f"%(args.min_pc)

    if args.do_rates:
        logging.info("Submitting %d rates jobs now"%(Nratejobs))
        # try:
        if args.rhel7:
            sub_jobs(Nratejobs, 'RATES_'+args.GWname, args.RATEpyscript,\
                        args.pbs_rhel7_fname, queue=args.queue, qos=args.qos,\
                        extra_args=extra_args, rhel7=args.rhel7,q=args.q)
        else:
            sub_jobs(Nratejobs, 'RATES_'+args.GWname, args.RATEpyscript,\
                        args.pbs_fname, queue=args.queue, qos=args.qos,\
                        extra_args=extra_args, rhel7=args.rhel7,q=args.q)
        logging.info("Jobs submitted")
        # except Exception as E:
        #     logging.warn(E)
        #     logging.warn("Might have been a problem submitting")






    dt = 0.0
    t_0 = time.time()

    while (dt < 3600.0*36.0):

        rate_res_fnames = get_rate_res_fnames()
        logging.info("%d of %d rate jobs done" %(len(rate_res_fnames),Nratejobs))

        if args.skip_waiting:

            rate_res = get_merged_csv_df(rate_res_fnames)
            try:
                rate_res['dt'] = rate_res['time'] - trigtime
            except Exception:
                pass
            break


        elif len(rate_res_fnames) < Nratejobs:

            time.sleep(30.0)
            dt = time.time() - t_0


        else:
            rate_res = get_merged_csv_df(rate_res_fnames)
            try:
                rate_res['dt'] = rate_res['time'] - trigtime
            except Exception:
                pass
            break

    try:
        body = "Done with rates analysis\n"
        body += "Max TS is %.3f" %(np.max(rate_res['TS']))
        logging.info(body)
        send_email(subject, body, to)
        rate_res_tab_top = rate_res.sort_values("TS").tail(16)
        body = rate_res_tab_top.to_html()
        logging.info(body)
        # send_email(subject, body, to)
        send_email_wHTML(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")

    if args.archive:
        seed_in_tab = mk_in_seed_tab_archive(rate_res, args.twind*1.024,\
                                        TS_min=args.rateTScut)
    else:
        seed_in_tab = mk_in_seed_tab(rate_res, TS_min=args.rateTScut)
    Nmax_jobs = 96
    if args.queue == 'open':
        Nmax_jobs = 64
    seed_in_tab = assign_in_seeds2jobs(seed_in_tab, Nmax_jobs=Nmax_jobs)
    # sky_map_fnames = [fname for fname in os.listdir() if\
    #                 'cWB.fits.gz' in fname or 'bayestar' in fname\
    #                 or 'skymap' in fname]

    if has_sky_map:
        skfname = sky_map_fnames[0]
    else:
        skfname = None

    # good_map, good_hp_inds = pc_probmap2good_outFoVmap_inds(args.pcfname,\
    #                                     skfname, attfile, trigtime)
    # For now just do full sky
    good_map, good_hp_inds = pc_probmap2good_outFoVmap_inds(args.pcfname,\
                                        None, attfile, trigtime)

    attq = attfile['QPARAM'][np.argmin(np.abs(attfile['TIME'] - trigtime))]
    Nmax_jobs = 24
    if args.queue == 'open':
        Nmax_jobs = 16
    seed_out_tab = mk_out_seed_tab(seed_in_tab, good_hp_inds, attq, Nmax_jobs=Nmax_jobs)

    # if args.archive:
    #     seed_tab = mk_seed_tab(rate_res, TS_min=4.15)
    # else:
    #     seed_tab = mk_seed_tab(rate_res)

    seed_in_tab.to_csv('rate_seeds.csv', index=False)
    seed_out_tab.to_csv('out_job_table.csv', index=False)
    Njobs_in = np.max(seed_in_tab['proc_group']) + 1
    Njobs_out = np.max(seed_out_tab['proc_group']) + 1

    Nsquares = len(np.unique(seed_in_tab['squareID']))
    Nseeds_in = len(seed_in_tab)
    Nseeds_out = len(seed_out_tab)
    Nseeds = Nseeds_in + Nseeds_out

    Ntot_in_fnames = len(seed_in_tab.groupby(['squareID', 'proc_group']))
    Ntot_out_fnames = len(np.unique(seed_out_tab['hp_ind']))

    if Nseeds < 1:
        body = "No seeds. Exiting analysis."
        try:
            send_email(subject, body, to)
        except Exception as E:
            logging.error(E)
            logging.error("Trouble sending email")
        return


    # if Nsquares > 512:
    #     Njobs = 96
    # elif Nsquares > 128:
    #     Njobs = 64
    # else:
    #     Njobs = Nsquares/2
    # if (1.*Nseeds)/Nsquares >= 8:
    #     Njobs += Njobs/2
    # if (1.*Nseeds)/Nsquares >= 16:
    #     Njobs += Njobs/4
    # if (1.*Nseeds)/Nsquares >= 32:
    #     Njobs += Njobs/4
    #
    #
    # Njobs = max(Njobs,1)
    #
    # job_tab = mk_job_tab(seed_tab, Njobs)
    #
    # job_tab.to_csv('job_table.csv', index=False)

    # Now need to launch those jobs

    # then start the next while loop, curogating the results
    # get the best TSs or TSs > 6 and do whatever with them

    # maybe mk imgs and do batcelldetect on the good time windows
    # or if there's a decent TS from the full analysis

    # also maybe add some emails in here for progress and info and errors
    if args.do_llh:
        logging.info("Submitting %d in FoV Jobs now"%(Njobs_in))
        if args.rhel7:
            sub_jobs(Njobs_in, 'LLHin_'+args.GWname, args.LLHINpyscript,\
                        args.pbs_rhel7_fname, queue=args.queue, qos=args.qos,\
                        extra_args=extra_args, rhel7=args.rhel7,q=args.q)
            logging.info("Submitting %d out of FoV Jobs now"%(Njobs_out))
            sub_jobs(Njobs_out, 'LLHo_'+args.GWname, args.LLHOUTpyscript,\
                        args.pbs_rhel7_fname, queue=args.queue, qos=args.qos, rhel7=args.rhel7,q=args.q)
        else:
            sub_jobs(Njobs_in, 'LLHin_'+args.GWname, args.LLHINpyscript,\
                        args.pbs_fname, queue=args.queue, qos=args.qos,\
                        extra_args=extra_args, rhel7=args.rhel7,q=args.q)
            logging.info("Submitting %d out of FoV Jobs now"%(Njobs_out))
            sub_jobs(Njobs_out, 'LLHo_'+args.GWname, args.LLHOUTpyscript,\
                        args.pbs_fname, queue=args.queue, qos=args.qos, rhel7=args.rhel7,q=args.q)
        logging.info("Jobs submitted, now going to monitor progress")


    t_0 = time.time()
    dt = 0.0
    Ndone_in = 0
    Ndone_out = 0

    DoneIn = False
    DoneOut = False

    while (dt < 3600.0*40.0):

        # res_fnames = get_res_fnames()
        res_in_fnames = get_in_res_fnames()
        res_out_fnames = get_out_res_fnames()

        if args.skip_waiting:

            try:
                if has_sky_map:
                    res_in_tab = get_merged_csv_df_wpos(res_res_in_fnames, attfile, perc_map)
                else:
                    res_in_tab = get_merged_csv_df_wpos(res_in_fnames, attfile)
                logging.info("Got merged results with RA Decs")
            except Exception as E:
                logging.error(E)
                res_in_tab = get_merged_csv_df(res_in_fnames)
                logging.info("Got merged results without RA Decs")
            logging.info("Max TS: %.3f" %(np.max(res_tab['TS'])))
            break


        if len(res_in_fnames) != Ndone_in:
            Ndone_in = len(res_in_fnames)
            logging.info("%d of %d in files done" %(Ndone_in,Ntot_in_fnames))
            if Ndone_in < Ntot_in_fnames:
                res_in_tab = get_merged_csv_df(res_in_fnames)
            else:
                logging.info("Got all of the in results now")
                res_peak_fnames = get_peak_res_fnames()
                try:
                    if has_sky_map:
                        res_in_tab = get_merged_csv_df_wpos(res_in_fnames, attfile, perc_map)
                        res_peak_tab = get_merged_csv_df_wpos(res_peak_fnames, attfile, perc_map)
                    else:
                        res_in_tab = get_merged_csv_df_wpos(res_in_fnames, attfile)
                        res_peak_tab = get_merged_csv_df_wpos(res_peak_fnames, attfile)
                    logging.info("Got merged results with RA Decs")
                except Exception as E:
                    logging.error(E)
                    try:
                        res_in_tab = get_merged_csv_df(res_in_fnames)
                        res_peak_tab = get_merged_csv_df(res_peak_fnames)
                        logging.info("Got merged in results without RA Decs")
                    except Exception as E:
                        logging.error(E)
                    try:
                        res_peak_tab = get_merged_csv_df(res_peak_fnames)
                        logging.info("Got merged peak results without RA Decs")
                    except Exception as E:
                        logging.error(E)
                try:
                    res_in_tab['dt'] = res_in_tab['time'] - trigtime
                    res_peak_tab['dt'] = res_peak_tab['time'] - trigtime
                except Exception:
                    pass
                try:
                    logging.info("Max in TS: %.3f" %(np.max(res_in_tab['TS'])))
                    logging.info("Max peak TS: %.3f" %(np.max(res_peak_tab['TS'])))
                except Exception:
                    pass
                # try:
                #     res_in_tab_top = res_in_tab.sort_values("TS").tail(16)
                #     body = "LLH in FoV analysis results\n"
                #     body += res_in_tab_top.to_html()
                #     logging.info(body)
                #     # send_email(subject, body, to)
                #     send_email_wHTML(subject, body, to)
                # except Exception as E:
                #     logging.error(E)
                #     logging.error("Trouble sending email")
                try:
                    idx = res_peak_tab.groupby(['squareID','timeID'])['TS'].transform(max) == res_peak_tab['TS']
                    res_peak_maxSqTime_tab = res_peak_tab[idx]
                    res_peak_tab_top = res_peak_maxSqTime_tab.sort_values("TS").tail(16)
                    body = "LLH in FoV analysis results around Peaks\n"
                    body += res_peak_tab_top.to_html()
                    logging.info(body)
                    # send_email(subject, body, to)
                    send_email_wHTML(subject, body, to)
                except Exception as E:
                    logging.error(E)
                    logging.error("Trouble sending email")
                    try:
                        body = "LLH in FoV analysis results around Peaks\n"
                        body += "No results that pass cut"
                        logging.info(body)
                        send_email(subject, body, to)
                    except Exception as E:
                        logging.error(E)
                        logging.error("Trouble sending email")

                DoneIn = True

        if len(res_out_fnames) != Ndone_out:
            Ndone_out = len(res_out_fnames)
            logging.info("%d of %d out files done" %(Ndone_out,Ntot_out_fnames))
            if Ndone_out < Ntot_out_fnames:
                res_out_tab = get_merged_csv_df(res_out_fnames)
            else:
                logging.info("Got all of the out results now")
                try:
                    if has_sky_map:
                        res_out_tab = get_merged_csv_df_wpos(res_out_fnames, attfile, perc_map)
                    else:
                        res_out_tab = get_merged_csv_df_wpos(res_out_fnames, attfile)
                    logging.info("Got merged results with RA Decs")
                except Exception as E:
                    logging.error(E)
                    res_out_tab = get_merged_csv_df(res_out_fnames)
                    logging.info("Got merged in results without RA Decs")
                logging.info("Max TS: %.3f" %(np.max(res_out_tab['TS'])))
                DoneOut = True
                try:
                    res_out_tab['dt'] = res_out_tab['time'] - trigtime
                except Exception:
                    pass
                try:
                    res_out_tab_top = res_out_tab.sort_values("TS").tail(16)
                    body = "LLH out of FoV analysis results\n"
                    body += res_out_tab_top.to_html()
                    logging.info(body)
                    # send_email(subject, body, to)
                    send_email_wHTML(subject, body, to)
                except Exception as E:
                    logging.error(E)
                    logging.error("Trouble sending email")
        if DoneIn and DoneOut:
            break
        time.sleep(30.0)
        dt = time.time() - t_0
#Added on 4th Sept, 2022 from jjd repo

    dlogl_peak, dlogl_out = get_dlogl_peak_out(res_peak_maxSqTime_tab, res_out_tab)

    body = "DeltaLLH_peak = %.3f \nDeltaLLH_out = %.3f" %(dlogl_peak, dlogl_out)
    try:
        send_email(subject, body, to)
    except Exception as E:
        logging.error(E)
        logging.error("Trouble sending email")


    # logging.info("Saving full result table to: ")
    # save_fname = 'full_res_tab.csv'
    # logging.info(save_fname)
    # res_tab.to_csv(save_fname)

    # try:
    #     # body = "Done with LLH analysis\n"
    #     # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
    #     res_tab_top = res_tab.sort_values("TS").tail(16)
    #     body = "LLH analysis results\n"
    #     body += res_tab_top.to_html()
    #     logging.info(body)
    #     # send_email(subject, body, to)
    #     send_email_wHTML(subject, body, to)
    # except Exception as E:
    #     logging.error(E)
    #     logging.error("Trouble sending email")


    # Now need to find anything interesting and investigate it further
    # probably find each time bin with a TS>6 and scan around each
    # blip with a nllh that's within 5-10 or so

    # Should also probably do submit jobs for a full FoV scan
    # if a TS is found above something border line alert, like
    # TS ~>7-8


    # if np.nanmax(res_tab['TS']) < args.TSscan:
    #     return
    #
    # scan_seed_tab = mk_seed_tab4scans(res_tab, args.pcfname, seed_tab,\
    #                             TS_min=args.TSscan, pc_min=args.min_pc)
    #
    # Nscan_seeds = len(scan_seed_tab)
    # logging.info("%d scan seeds"%(Nscan_seeds))
    # Nscan_squares = len(np.unique(scan_seed_tab['squareID']))
    #
    #
    # Njobs = 64
    # if Nscan_squares < 64:
    #     Njobs = Nscan_squares/2
    # if Nscan_seeds > 1e3:
    #     Njobs = 72
    # if Nscan_seeds > 2.5e3:
    #     Njobs = 96
    # if Nscan_seeds > 5e3:
    #     Njobs = 128
    # if Nscan_seeds > 1e4:
    #     Njobs = 160
    #
    #
    # scan_job_tab = mk_job_tab(scan_seed_tab, Njobs)
    #
    # scan_seed_tab.to_csv('scan_seeds.csv', index=False)
    # scan_job_tab.to_csv('scan_job_table.csv', index=False)
    #
    # if args.do_scan:
    #     logging.info("Submitting %d Scan Jobs now"%(Njobs))
    #     sub_jobs(Njobs, 'SCAN_'+args.GWname, args.SCANpyscript,\
    #                 args.pbs_fname, queue=args.queue, qos=args.qos,\
    #                 extra_args=extra_args)
    #     logging.info("Jobs submitted, now going to monitor progress")
    #
    #
    #
    # t_0 = time.time()
    # dt = 0.0
    # Ndone = 0
    #
    # while (dt < 3600.0*40.0):
    #
    #     res_fnames = get_scan_res_fnames()
    #
    #     if args.skip_waiting:
    #         if len(res_fnames) < 1:
    #             scan_res_tab = pd.DataFrame()
    #             break
    #         try:
    #             if has_sky_map:
    #                 scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
    #             else:
    #                 scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
    #             logging.info("Got merged scan results with RA Decs")
    #         except Exception as E:
    #             logging.error(E)
    #             scan_res_tab = get_merged_csv_df(res_fnames)
    #             logging.info("Got merged scan results without RA Decs")
    #         logging.info("Max TS: %.3f" %(np.max(scan_res_tab['TS'])))
    #         break
    #
    #
    #
    #     if len(res_fnames) == Ndone:
    #         time.sleep(30.0)
    #         dt = time.time() - t_0
    #
    #     else:
    #         Ndone = len(res_fnames)
    #         logging.info("%d of %d squares scanned" %(Ndone,Nscan_squares))
    #
    #         if Ndone < Nscan_squares:
    #
    #             scan_res_tab = get_merged_csv_df(res_fnames)
    #
    #             time.sleep(30.0)
    #             dt = time.time() - t_0
    #
    #         else:
    #
    #             logging.info("Got all of the scan results now")
    #             try:
    #                 if has_sky_map:
    #                     scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
    #                 else:
    #                     scan_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
    #                 logging.info("Got merged scan results with RA Decs")
    #             except Exception as E:
    #                 logging.error(E)
    #                 scan_res_tab = get_merged_csv_df(res_fnames)
    #                 logging.info("Got merged scan results without RA Decs")
    #             logging.info("Max TS: %.3f" %(np.max(scan_res_tab['TS'])))
    #             break
    #
    #
    # try:
    #     scan_res_tab['dt'] = scan_res_tab['time'] - trigtime
    # except Exception:
    #     pass
    #
    # # logging.info("Saving full result table to: ")
    # # save_fname = 'full_scanRes_tab.csv'
    # # logging.info(save_fname)
    # # scan_res_tab.to_csv(save_fname)
    #
    # full_res_tab = pd.concat([res_tab,scan_res_tab], ignore_index=True)
    #
    # try:
    #     # body = "Done with LLH analysis\n"
    #     # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
    #     body = "All LLH analysis results\n"
    #     full_res_tab_top = full_res_tab.sort_values("TS").tail(16)
    #     body += full_res_tab_top.to_html()
    #     logging.info(body)
    #     # send_email(subject, body, to)
    #     send_email_wHTML(subject, body, to)
    # except Exception as E:
    #     logging.error(E)
    #     logging.error("Trouble sending email")
    #
    # if has_sky_map:
    #     if np.all(full_res_tab_top['cls']>.995):
    #         logging.info("None of the top 16 TSs are in the 0.995 credible region.")
    #
    # # Now need to put in the part where I find good candidates
    # # then do the integrated LLH
    #
    # logging.info("Making Peaks Table now")
    # peaks_tab = find_peaks2scan(full_res_tab, minTS=args.TSscan)
    #
    # Npeaks = len(peaks_tab)
    # logging.info("Found %d Peaks to scan"%(Npeaks))
    # Njobs = 96
    #
    # if Npeaks < Njobs:
    #     Njobs = Npeaks
    #     peaks_tab['jobID'] = np.arange(Njobs, dtype=np.int)
    # else:
    #     jobids = np.array([i%Njobs for i in range(Npeaks)])
    #     peaks_tab['jobID'] = jobids
    #
    # peaks_fname = 'peaks.csv'
    # peaks_tab.to_csv(peaks_fname)
    #
    #
    # logging.info("Submitting %d Jobs now"%(Njobs))
    # sub_jobs(Njobs, 'Peak_'+args.GWname, args.PEAKpyscript,\
    #             args.pbs_fname, queue=args.queue, qos=args.qos)
    # logging.info("Jobs submitted, now going to monitor progress")
    #
    #
    # t_0 = time.time()
    # dt = 0.0
    # Ndone = 0
    #
    # while (dt < 3600.0*20.0):
    #
    #     res_fnames = get_peak_res_fnames()
    #
    #     if len(res_fnames) == Ndone:
    #         time.sleep(30.0)
    #         dt = time.time() - t_0
    #
    #     else:
    #         Ndone = len(res_fnames)
    #         logging.info("%d of %d peaks scanned" %(Ndone,Npeaks))
    #
    #         if Ndone < Npeaks:
    #
    #             peak_res_tab = get_merged_csv_df(res_fnames)
    #
    #             time.sleep(30.0)
    #             dt = time.time() - t_0
    #
    #         else:
    #
    #             logging.info("Got all of the peak results now")
    #             try:
    #                 if has_sky_map:
    #                     peak_res_tab = get_merged_csv_df_wpos(res_fnames, attfile, perc_map)
    #                 else:
    #                     peak_res_tab = get_merged_csv_df_wpos(res_fnames, attfile)
    #                 logging.info("Got merged peak results with RA Decs")
    #             except Exception as E:
    #                 logging.error(E)
    #                 peak_res_tab = get_merged_csv_df(res_fnames)
    #                 logging.info("Got merged peak results without RA Decs")
    #             logging.info("Max TS: %.3f" %(np.max(peak_res_tab['TS'])))
    #             break
    #
    #
    # try:
    #     peak_res_tab['dt'] = peak_res_tab['time'] - trigtime
    # except Exception:
    #     pass
    #
    # idx = peak_res_tab.groupby(['timeID'])['TS'].transform(max) == peak_res_tab['TS']
    # peak_res_max_tab = peak_res_tab[idx]
    #
    # maxTS = np.max(peak_res_max_tab['TS'])
    # for timeID, df in peak_res_tab.groupby('timeID'):
    #     if has_sky_map:
    #         if (np.max(df['TS']) > 7.0) and (df['cls'].iloc[np.argmax(df['TS'])]<.995):
    #             try:
    #                 subject2 = subject + ' Possible Signal'
    #                 peak_res_tab_top = df.sort_values("TS").tail(16)
    #                 body = peak_res_tab_top.to_html()
    #                 send_email_wHTML(subject2, body, to)
    #             except Exception as E:
    #                 logging.error(E)
    #                 logging.error("Trouble sending email")
    #     else:
    #         if (np.max(df['TS']) > 7.0):
    #             try:
    #                 subject2 = subject + ' Possible Signal'
    #                 peak_res_tab_top = df.sort_values("TS").tail(16)
    #                 body = peak_res_tab_top.to_html()
    #                 send_email_wHTML(subject2, body, to)
    #             except Exception as E:
    #                 logging.error(E)
    #                 logging.error("Trouble sending email")
    #
    #
    # try:
    #     # body = "Done with LLH analysis\n"
    #     # body += "Max TS is %.3f\n\n" %(np.max(res_tab['TS']))
    #     peak_res_tab_top = peak_res_tab.sort_values("TS").tail(16)
    #     body = peak_res_tab_top.to_html()
    #     logging.info(body)
    #     # send_email(subject, body, to)
    #     send_email_wHTML(subject, body, to)
    # except Exception as E:
    #     logging.error(E)
    #     logging.error("Trouble sending email")
    #
    # if has_sky_map:
    #     if np.all(peak_res_tab_top['cls']>.995):
    #         logging.info("None of the top 16 TSs are in the 0.995 credible region.")





if __name__ == "__main__":

    args = cli()

    main(args)
