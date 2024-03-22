import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import os
import argparse
import logging, traceback
import pandas as pd
from copy import copy, deepcopy
import multiprocessing as mp

from ..analysis_seeds.bkg_rate_estimation import get_avg_lin_cub_rate_quad_obs
from ..config import (
    quad_dicts,
    EBINS0,
    EBINS1,
    solid_angle_dpi_fname,
    bright_source_table_fname,
)
from ..lib.sqlite_funcs import write_rate_fits_from_obj, get_conn
from ..lib.dbread_funcs import get_info_tab, guess_dbfname, get_files_tab
from ..lib.event2dpi_funcs import filter_evdata
from ..models.models import (
    Bkg_Model_wFlatA,
    CompoundModel,
    Point_Source_Model_Binned_Rates,
)
from ..llh_analysis.LLH import LLH_webins
from ..llh_analysis.minimizers import NLLH_ScipyMinimize, NLLH_ScipyMinimize_Wjacob
from ..response.ray_trace_funcs import RayTraces
from ..response.response import get_pc
from ..lib.coord_conv_funcs import convert_radec2imxy, imxy2theta_phi
from ..lib.gti_funcs import add_bti2gti, bti2gti, gti2bti, union_gtis
from ..lib.wcs_funcs import world2val


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evfname", type=str, help="Event data file", default=None)
    parser.add_argument("--dmask", type=str, help="Detmask fname", default=None)
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    parser.add_argument("--job_id", type=int, help="Job ID", default=0)
    parser.add_argument("--Njobs", type=int, help="Number of jobs", default=1)
    parser.add_argument(
        "--twind",
        type=float,
        help="Number of seconds to go +/- from the trigtime",
        default=20,
    )
    parser.add_argument(
        "--sig_twind",
        type=float,
        help="Size of the signal window in seconds",
        default=20 * 1.024,
    )
    parser.add_argument("--bkg_dur", type=float, help="bkg duration", default=60.0)
    parser.add_argument(
        "--archive", help="Adjust for longer event duration", action="store_true"
    )
    parser.add_argument(
        "--bkg_nopost",
        help="Don't use time after signal window for bkg",
        action="store_true",
    )
    parser.add_argument(
        "--bkg_nopre",
        help="Don't use time before signal window for bkg",
        action="store_true",
    )
    parser.add_argument(
        "--pcfname", type=str, help="partial coding file name", default=None
    )
    parser.add_argument(
        "--preset_bkg_log_file",
        help="Flag to tell the nitrates background estimation to print logging information into a log file that may have been created prior to calling the main function",
        action="store_true",
    )
    parser.add_argument("--disable_bkg_sourceloc_fit",
        help="Flag to tell the nitrates background estimation to not attempt to fit background sources",
        action="store_true"
    )

    args = parser.parse_args()
    return args


def ang_sep(ra0, dec0, ra1, dec1):
    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(
        np.cos(np.radians(90 - dec0)) * np.cos(np.radians(90 - dec1))
        + np.sin(np.radians(90 - dec0)) * np.sin(np.radians(90 - dec1)) * dcos
    )
    return np.rad2deg(angsep)


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot((imx1 - imx0), (imy1 - imy0))


def add_imxy2src_tab(src_tab, attfile, t0):
    att_ind = np.argmin(np.abs(attfile["TIME"] - t0))
    att_quat = attfile["QPARAM"][att_ind]
    pnt_ra, pnt_dec = attfile["POINTING"][att_ind, :2]
    imxs = np.zeros(len(src_tab))
    imys = np.zeros(len(src_tab))
    src_tab["PntSep"] = ang_sep(pnt_ra, pnt_dec, src_tab["RAJ2000"], src_tab["DEJ2000"])
    for i in range(len(imxs)):
        if src_tab["PntSep"][i] > 80.0:
            imxs[i], imys[i] = np.nan, np.nan
            continue
        imxs[i], imys[i] = convert_radec2imxy(
            src_tab["RAJ2000"][i], src_tab["DEJ2000"][i], att_quat
        )
    src_tab["imx"] = imxs
    src_tab["imy"] = imys
    return src_tab


def get_srcs_infov(attfile, t0, pcfname=None, bl_dmask=None, pcmin=1e-3):
    brt_src_tab = Table.read(bright_source_table_fname)
    add_imxy2src_tab(brt_src_tab, attfile, t0)
    bl_infov = (np.abs(brt_src_tab["imy"]) < 0.95) & (np.abs(brt_src_tab["imx"]) < 1.75)
    if pcfname is not None:
        try:
            PC = fits.open(pcfname)[0]
            pc = PC.data
            w_t = WCS(PC.header, key="T")
            pcvals = world2val(w_t, pc, brt_src_tab["imx"], brt_src_tab["imy"])
            bl_infov = bl_infov & (pcvals >= pcmin)
        except Exception as E:
            logging.warn("Trouble Using PC file")
            logging.error(E)
    elif bl_dmask is not None:
        try:
            theta, phi = imxy2theta_phi(brt_src_tab["imx"], brt_src_tab["imy"])
            pcvals = get_pc(bl_dmask, theta, phi)
            bl_infov = bl_infov & (pcvals >= pcmin)
        except Exception as E:
            logging.warn("Trouble Getting PC vals")
            logging.error(E)
            
    N_infov = np.sum(bl_infov)
    return brt_src_tab[bl_infov]


def min_by_ebin(miner, params_):
    nebins = miner.llh_obj.nebins
    params = copy(params_)
    bf_params = copy(params_)
    NLLH = 0.0

    for e0 in range(nebins):
        miner.set_fixed_params(list(params.keys()), values=list(params.values()))
        e0_pnames = []
        for pname in miner.param_names:
            try:
                if int(pname[-1]) == e0:
                    e0_pnames.append(pname)
            except:
                pass
        miner.set_fixed_params(e0_pnames, fixed=False)
        miner.llh_obj.set_ebin(e0)

        bf_vals, nllh, res = miner.minimize()
        NLLH += nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]

    return NLLH, bf_params


class Worker(mp.Process):
    def __init__(self, result_queue, miner, param_list):
        mp.Process.__init__(self)
        self.miner = miner
        self.param_list = param_list
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name

        for params in self.param_list:
            nllh, bf_params = min_by_ebin(self.miner, params)
            res_dict = bf_params
            res_dict["nllh"] = nllh
            self.result_queue.put(res_dict)

        self.result_queue.put(None)
        return


def bkg_withPS_fit(
    PS_tab, model, llh_obj, t0s, t1s, dimxy=2e-3, im_steps=5, test_null=False, Nprocs=1
):
    Nps = len(PS_tab)
    imax = np.linspace(-dimxy, dimxy, im_steps)
    if im_steps == 3:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
    elif im_steps == 2:
        imax = np.linspace(-dimxy / 2.0, dimxy / 2.0, im_steps)
        imax0 = [-dimxy, 0.0, dimxy]
        imax1 = [dimxy, 0.0, -dimxy]
    elif im_steps == 1:
        imax = np.array([0.0])

    imlist = []
    for i in range(Nps):
        if im_steps == 2:
            imlist += [imax0, imax1]
        else:
            imlist += [imax, imax]
    imgs = np.meshgrid(*imlist)
    Npnts = imgs[0].size
    if im_steps == 2:
        ind_grids = np.meshgrid(*(np.arange(3, dtype=np.int64) for i in range(Nps)))
        Npnts = ind_grids[0].size
    logging.info("Npnts: %d" % (Npnts))

    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_miner.set_llh(llh_obj)
    llh_obj.set_time(t0s, t1s)

    bf_params_list = []
    bkg_nllhs = np.zeros(Npnts)

    nebins = model.nebins
    param_list = []

    for i in range(Npnts):
        bf_params = {}
        im_names = []
        params_ = {
            pname: val["val"] for pname, val in bkg_miner.param_info_dict.items()
        }

        for j in range(Nps):
            row = PS_tab[j]
            psname = row["Name"]
            if im_steps == 2:
                params_[psname + "_imx"] = (
                    imlist[2 * j][ind_grids[j].ravel()[i]] + row["imx"]
                )
                params_[psname + "_imy"] = (
                    imlist[2 * j + 1][ind_grids[j].ravel()[i]] + row["imy"]
                )
            else:
                params_[psname + "_imx"] = imgs[2 * j].ravel()[i] + row["imx"]
                params_[psname + "_imy"] = imgs[2 * j + 1].ravel()[i] + row["imy"]
            im_names = [psname + "_imx", psname + "_imy"]
        #         im_vals = [bf_params[nm] for nm in im_names]
        if Nprocs > 1:
            param_list.append(params_)
            continue
        im_vals = [params_[nm] for nm in im_names]

        bkg_nllhs[i], bf_params = min_by_ebin(bkg_miner, params_)

        bf_params_list.append(bf_params)

    if Nprocs > 1:
        res_q = mp.Queue()
        workers = []
        Nper_worker = 1 + int(len(param_list) / (1.0 * Nprocs))
        logging.info("Nper_worker: %d" % (Nper_worker))
        for i in range(Nprocs):
            i0 = i * Nper_worker
            i1 = i0 + Nper_worker
            w = Worker(res_q, deepcopy(bkg_miner), param_list[i0:i1])
            workers.append(w)

        for w in workers:
            w.start()
        res_dicts = []
        Ndone = 0
        while True:
            res = res_q.get()
            if res is None:
                Ndone += 1
                logging.info("%d of %d done" % (Ndone, Nprocs))
                if Ndone >= Nprocs:
                    break
            else:
                res_dicts.append(res)
        for w in workers:
            w.join()

        df = pd.DataFrame(res_dicts)
        min_ind = np.argmin(df["nllh"])
        bf_df_row = df.iloc[min_ind]
        bf_nllh = bf_df_row["nllh"]
        bf_params = {
            name: bf_df_row[name] for name in bf_df_row.index if not "nllh" in name
        }

    else:
        bf_ind = np.argmin(bkg_nllhs)
        bf_params = bf_params_list[bf_ind]
        bf_nllh = bkg_nllhs[bf_ind]

    if test_null:
        TS_nulls = {}
        for i in range(Nps):
            params_ = copy(bf_params)
            row = PS_tab[i]
            psname = row["Name"]
            for j in range(nebins):
                params_[psname + "_rate_" + str(j)] = 0.0
            llh_obj.set_ebin(-1)
            nllh_null = -llh_obj.get_logprob(params_)
            TS_nulls[psname] = np.sqrt(2.0 * (nllh_null - bf_nllh))
            if np.isnan(TS_nulls[psname]):
                TS_nulls[psname] = 0.0
        return bf_nllh, bf_params, TS_nulls

    return bf_nllh, bf_params


def do_init_bkg_wPSs(
    bkg_mod,
    llh_obj,
    src_tab,
    rt_obj,
    GTI,
    sig_twind,
    TSmin=7.0,
    Nprocs=1,
    tmin=None,
    tmax=None, disable_bkg_sourceloc_fit=False
):
    if not tmin is None:
        bti = (-np.inf, tmin)
        GTI = add_bti2gti(bti, GTI)
    if not tmax is None:
        bti = (tmax, np.inf)
        GTI = add_bti2gti(bti, GTI)
    gti_bkg = add_bti2gti(sig_twind, GTI)
    bkg_t0s = gti_bkg["START"]
    bkg_t1s = gti_bkg["STOP"]
    exp = 0.0
    for i in range(len(bkg_t0s)):
        exp += bkg_t1s[i] - bkg_t0s[i]
    logging.info("exp: %.3f" % (exp))
    # if exp < 30.0:
    #     logging.info("Trying to expand sig_twind")
    #     sig_twind = (sig_twind[0] + 20.0, sig_twind[1] - 20.0)
    #     gti_bkg = add_bti2gti(sig_twind, GTI_)
    #     bkg_t0s = gti_bkg['START']
    #     bkg_t1s = gti_bkg['STOP']
    #     exp = 0.0
    #     for i in range(len(bkg_t0s)):
    #         exp += (bkg_t1s[i] - bkg_t0s[i])
    #     logging.info("exp: %.3f"%(exp))

    logging.info("bkg_t0s: ")
    logging.info(bkg_t0s)
    logging.info("bkg_t1s: ")
    logging.info(bkg_t1s)

    Nsrcs = len(src_tab)
    nebins = bkg_mod.nebins

    for ii in range(Nsrcs):
        mod_list = [bkg_mod]
        im_steps = 5
        TSmin_ = TSmin
        if Nsrcs >= 3:
            im_steps = 3
            TSmin_ = TSmin - 1.0
        if Nsrcs >= 5:
            im_steps = 2
            TSmin_ = TSmin - 1.5
        if Nsrcs >= 9:
            im_steps = 1
            TSmin_ = TSmin - 2.5

        ps_mods = []
        for i in range(Nsrcs):
            row = src_tab[i]
            mod = Point_Source_Model_Binned_Rates(
                row["imx"],
                row["imy"],
                0.1,
                [llh_obj.ebins0, llh_obj.ebins1],
                rt_obj,
                llh_obj.bl_dmask,
                use_deriv=True,
                name=row["Name"],
            )
            ps_mods.append(mod)

        mod_list += ps_mods
        comp_mod = CompoundModel(mod_list)

        llh_obj.set_model(comp_mod)

        ######Added here to test if we need to even do this type of operation. It doesnt seem to change the results that much. so keep this as an option for the mpi4py code
        if disable_bkg_sourceloc_fit:
            logging.info("Disabling the background source location fitting. ")
            im_steps=1
            Nprocs=1
        ###################

        bf_nllh, bf_params, TS_nulls = bkg_withPS_fit(
            src_tab,
            comp_mod,
            llh_obj,
            bkg_t0s,
            bkg_t1s,
            test_null=True,
            im_steps=im_steps,
            Nprocs=Nprocs,
        )

        logging.debug("TS_nulls: ")
        logging.debug(TS_nulls)

        bkg_rates = np.array(
            [bf_params["Background" + "_bkg_rate_" + str(j)] for j in range(nebins)]
        )
        min_rate = 1e-1 * bkg_rates
        logging.debug("min_rate: ")
        logging.debug(min_rate)
        PSs2keep = []
        for name, TS in TS_nulls.items():
            ps_rates = np.array(
                [bf_params[name + "_rate_" + str(j)] for j in range(nebins)]
            )
            logging.debug(name + " rates: ")
            logging.debug(ps_rates)
            if TS < TSmin_:
                # ps_rates = np.array([bf_params[name+'_rate_'+str(j)] for j in range(nebins)])
                # print ps_rates
                if np.all(ps_rates < min_rate):
                    continue
            if np.all(ps_rates < (min_rate / 20.0)):
                continue
            PSs2keep.append(name)

        # errs_dict = {}
        # corrs_dict = {}
        # for e0 in range(nebins):
        #     flat_pname = 'Background_flat_'+str(e0)
        #     comp_mod.param_dict[flat_pname]['fixed'] = True
        #     comp_mod.param_dict[flat_pname]['val'] = bf_params[flat_pname]
        #     err_dict, corr_dict = get_errs_corrs(llh_obj, comp_mod, copy(bf_params),\
        #                                          e0, pnames2skip=[flat_pname])
        #     for k, val in err_dict.iteritems():
        #         errs_dict[k] = val
        #     for k, val in corr_dict.iteritems():
        #         corrs_dict[k] = val

        if len(PSs2keep) == len(src_tab):
            break
        if len(PSs2keep) == 0:
            Nsrcs = 0
            src_tab = src_tab[np.zeros(len(src_tab), dtype=bool)]
            break
        bl = np.array([src_tab["Name"][i] in PSs2keep for i in range(Nsrcs)])
        src_tab = src_tab[bl]
        Nsrcs = len(src_tab)
        logging.debug("src_tab: ")
        logging.debug(src_tab)

    return bf_params, src_tab  # , errs_dict, corrs_dict


def get_info_mat_around_min(llh_obj, mod, params_, ebin):
    params = copy(params_)
    dt = llh_obj.dt
    mod_cnts = llh_obj.model.get_rate_dpi(params, ebin) * dt
    data_cnts = llh_obj.data_dpis[ebin]

    dR_dparams = mod.get_dr_dp(params, ebin)

    cov_ndim = len(dR_dparams)

    info_mat = np.zeros((cov_ndim, cov_ndim))

    for i in range(cov_ndim):
        for j in range(cov_ndim):
            info_mat[i, j] = np.sum(
                ((dR_dparams[j] * dt) * (dR_dparams[i] * dt) * data_cnts)
                / np.square(mod_cnts)
            )
    #             cov_mat[i,j] = np.sum(np.square(mod_cnts)/\
    #                         ((dR_dparams[j]*dt)*(dR_dparams[i]*dt)*data_cnts))

    return info_mat


def get_errs_corrs(llh_obj, model, params, e0, pnames2skip=[]):
    imat = get_info_mat_around_min(llh_obj, model, copy(params), e0)
    cov_mat = np.linalg.inv(imat)
    e0_pnames = []
    for pname in model.param_names:
        try:
            if int(pname[-1]) == e0 and not (pname in pnames2skip):
                e0_pnames.append(pname)
        except:
            pass

    err_dict = {}
    corr_dict = {}
    errs = np.sqrt(np.diag(cov_mat))
    for i, pname in enumerate(e0_pnames):
        k = "err_" + pname
        err_dict[k] = errs[i]
    Npars = len(e0_pnames)
    for i in range(Npars - 1):
        pname0 = e0_pnames[i]
        for j in range(i + 1, Npars):
            pname1 = e0_pnames[j]
            k = "corr_" + pname0 + "_" + pname1
            corr_dict[k] = cov_mat[i, j] / (errs[i] * errs[j])

    return err_dict, corr_dict


def bkg_withPS_fit_fiximxy(
    PS_tab, model, llh_obj, t0s, t1s, params_, fixed_pnames=None
):
    Nps = len(PS_tab)
    params = copy(params_)

    llh_obj.set_model(model)
    bkg_miner = NLLH_ScipyMinimize_Wjacob("")
    bkg_miner.set_llh(llh_obj)
    llh_obj.set_time(t0s, t1s)

    nllh = 0.0
    #     bf_params = {fixed_pars[i]:fixed_vals[i] for i in range(len(fixed_pars))}
    bf_params = copy(params)
    fixed_vals = [bf_params[pname] for pname in fixed_pnames]
    errs_dict = {}
    corrs_dict = {}

    for e0 in range(llh_obj.nebins):
        bkg_miner.set_fixed_params(bkg_miner.param_names)
        if fixed_pnames is not None:
            bkg_miner.set_fixed_params(fixed_pnames, values=fixed_vals)
        e0_pnames = []
        for pname in bkg_miner.param_names:
            try:
                if int(pname[-1]) == e0 and not (pname in fixed_pnames):
                    e0_pnames.append(pname)
            except:
                pass
        bkg_miner.set_fixed_params(e0_pnames, fixed=False)
        llh_obj.set_ebin(e0)

        bf_vals, bkg_nllh, res = bkg_miner.minimize()
        nllh += bkg_nllh[0]
        for ii, pname in enumerate(e0_pnames):
            bf_params[pname] = bf_vals[0][ii]

        print(bf_params)

        err_dict, corr_dict = get_errs_corrs(
            llh_obj, model, copy(bf_params), e0, pnames2skip=fixed_pnames
        )
        for k, val in err_dict.items():
            errs_dict[k] = val
        for k, val in corr_dict.items():
            corrs_dict[k] = val

    return nllh, bf_params, errs_dict, corrs_dict


def main(args):
    if not args.preset_bkg_log_file:
        logging.basicConfig(
            filename="bkg_rate_estimation_wPSs.log",
            level=logging.DEBUG,
            format="%(asctime)s-" "%(levelname)s- %(message)s",
        )

    if args.bkg_nopost and args.bkg_nopre:
        raise Exception("Can't have no pre and no post")

    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    logging.info("Connecting to DB")
    conn = get_conn(db_fname)

    info_tab = get_info_tab(conn)
    logging.info("Got info table")

    files_tab = get_files_tab(conn)
    logging.info("Got files table")

    trigtime = info_tab["trigtimeMET"][0]
    tstart = trigtime - args.twind
    tstop = trigtime + args.twind

    evfname = files_tab["evfname"][0]
    dmfname = files_tab["detmask"][0]
    attfname = files_tab["attfname"][0]
    ev_data = fits.open(evfname)[1].data
    try:
        GTI = Table.read(evfname, hdu="GTI_POINTING")
    except:
        GTI = Table.read(evfname, hdu="GTI")
    dmask = fits.open(dmfname)[0].data
    bl_dmask = dmask == 0.0
    attfile = fits.open(attfname)[1].data
    logging.debug("Opened up event, detmask, and att files")

    logging.debug("trigtime: %.3f" % (trigtime))
    gti_bl = (GTI["STOP"] > (trigtime - 2e3)) & (GTI["START"] < (trigtime + 2e3))
    logging.debug("Full GTI_pnt: ")
    logging.debug(GTI)
    logging.debug("GTI_pnt to use: ")
    logging.debug(GTI[gti_bl])
    GTI = GTI[gti_bl]
    tot_exp = 0.0
    for row in GTI:
        tot_exp += row["STOP"] - row["START"]
    logging.info("Tot_Exp: ")
    logging.info(tot_exp)

    tmin = GTI["START"][0]
    tmax = GTI["STOP"][-1]
    tmids = (GTI["START"] + GTI["STOP"]) / 2.0
    tmid = tmids[np.argmin(np.abs(tmids - trigtime))]
    logging.info("tmid: %.3f" % (tmid))
    # sig_twind = (tmid - 40.0, tmid+40.0)
    sig_dtwind = (-10 * 1.024, 20 * 1.024)
    sig_twind = (trigtime + sig_dtwind[0], trigtime + sig_dtwind[1])
    if args.archive:
        sig_twind = (tmin + 40.0, tmax - 40.0)
    logging.info("sig_twind: ")
    logging.info(sig_twind)

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)
    ebins0 = np.array([15.0, 24.0, 35.0, 48.0, 64.0])
    ebins0 = np.append(ebins0, np.logspace(np.log10(84.0), np.log10(500.0), 5 + 1))[:-1]
    ebins0 = np.round(ebins0, decimals=1)[:-1]
    ebins1 = np.append(ebins0[1:], [350.0])
    nebins = len(ebins0)
    logging.debug("ebins0")
    logging.debug(ebins0)
    logging.debug("ebins1")
    logging.debug(ebins1)

    solid_angle_dpi = np.load(solid_angle_dpi_fname)

    src_tab = get_srcs_infov(attfile, tmid, pcfname=args.pcfname, bl_dmask=bl_dmask)
    Nsrcs = len(src_tab)
    logging.info("src_tab: ")
    logging.info(src_tab)

    bkg_mod = Bkg_Model_wFlatA(bl_dmask, solid_angle_dpi, nebins, use_deriv=True)

    llh_obj = LLH_webins(ev_data, ebins0, ebins1, bl_dmask)

    # add in stuff later for if there's no srcs

    if Nsrcs > 0:
        rt_dir = files_tab["rtDir"][0]
        rt_obj = RayTraces(rt_dir)

        Nprocs = 1
        if Nsrcs > 1:
            Nprocs = 4
        logging.info("Using %d Procs" % (Nprocs))

        if args.archive:
            tmin_ = trigtime - 2e3
            tmax_ = trigtime + 2e3
        else:
            tmin_ = trigtime - 5e2
            tmax_ = trigtime + 5e2
            
        if args.disable_bkg_sourceloc_fit:
            logging.info("Disabling the background source location fitting.")
        
        init_bf_params, src_tab = do_init_bkg_wPSs(
            bkg_mod,
            llh_obj,
            src_tab,
            rt_obj,
            GTI,
            sig_twind,
            Nprocs=Nprocs,
            tmin=tmin_,
            tmax=tmax_,
            disable_bkg_sourceloc_fit=args.disable_bkg_sourceloc_fit
        )

        Nsrcs = len(src_tab)

        logging.debug("Final src_tab:")
        logging.debug(src_tab)

        # Now need to do each time, with these PSs and these imxys

    else:
        init_bf_params = {k: bkg_mod.param_dict[k]["val"] for k in bkg_mod.param_names}

    if Nsrcs > 0:
        fixed_pars = [
            pname
            for pname in list(init_bf_params.keys())
            if "_flat_" in pname or "_imx" in pname or "_imy" in pname
        ]

        mod_list = [bkg_mod]
        ps_mods = []
        for i in range(Nsrcs):
            row = src_tab[i]
            mod = Point_Source_Model_Binned_Rates(
                row["imx"],
                row["imy"],
                0.1,
                [ebins0, ebins1],
                rt_obj,
                bl_dmask,
                use_deriv=True,
                name=row["Name"],
            )
            ps_mods.append(mod)

        mod_list += ps_mods
        mod = CompoundModel(mod_list)

    else:
        init_bf_params = {k: bkg_mod.param_dict[k]["val"] for k in bkg_mod.param_names}
        mod = bkg_mod
        fixed_pars = []

    bkg_dur = args.bkg_dur * 1.024
    if args.archive:
        twind = (1.5e3) * 1.024
        twind = args.twind * 1.024
    else:
        twind = args.twind * 1.024
    bkg_tstep = 1 * 1.024
    dt_ax = np.arange(-twind, twind + 1, bkg_tstep)
    t_ax = dt_ax + trigtime
    Ntpnts = len(dt_ax)
    logging.info("Ntpnts: %d" % (Ntpnts))
    logging.info("min(dt_ax): %.3f" % (np.min(dt_ax)))
    logging.info("max(dt_ax): %.3f" % (np.max(dt_ax)))
    sig_wind = 20 * 1.024
    sig_wind = args.sig_twind
    # sig_twind = (trigger_time + sig_dtwind[0], trigger_time + sig_dtwind[1])

    bkg_bf_dicts = []

    for i in range(Ntpnts):
        tmid = t_ax[i]
        sig_twind = (-sig_wind / 2.0 + tmid, sig_wind / 2.0 + tmid)
        sig_twind = (-sig_wind / 4.0 + tmid, 3.0 * sig_wind / 4.0 + tmid)
        gti_ = add_bti2gti(sig_twind, GTI)
        # bkg_t0 = tmid - sig_wind/2. - bkg_dur/2.
        # bkg_t1 = tmid + sig_wind/2. + bkg_dur/2.
        bkg_t0 = tmid - sig_wind / 4.0 - bkg_dur / 2.0
        bkg_t1 = tmid + 3.0 * sig_wind / 4.0 + bkg_dur / 2.0
        bkg_bti = Table(
            data=([-np.inf, bkg_t1], [bkg_t0, np.inf]), names=("START", "STOP")
        )
        gti_ = add_bti2gti(bkg_bti, gti_)
        print(tmid - trigtime)
        print(gti_)
        t0s = gti_["START"]
        t1s = gti_["STOP"]
        exp = 0.0
        for ii in range(len(t0s)):
            exp += t1s[ii] - t0s[ii]
        if exp < 10.0:
            continue

        nllh, params, errs_dict, corrs_dict = bkg_withPS_fit_fiximxy(
            src_tab,
            mod,
            llh_obj,
            t0s,
            t1s,
            copy(init_bf_params),
            fixed_pnames=fixed_pars,
        )

        params.update(errs_dict)
        params.update(corrs_dict)
        params["nllh"] = nllh
        params["time"] = tmid
        params["dt"] = tmid - trigtime
        bkg_bf_dicts.append(params)
        params["exp"] = llh_obj.dt

    bkg_df = pd.DataFrame(bkg_bf_dicts)

    save_fname = "bkg_estimation.csv"
    logging.info("Saving results in a DataFrame to file: ")
    logging.info(save_fname)
    bkg_df.to_csv(save_fname, index=False)


if __name__ == "__main__":
    args = cli()

    main(args)
