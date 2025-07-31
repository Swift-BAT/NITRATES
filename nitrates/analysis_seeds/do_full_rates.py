import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import os
import argparse
import logging
import json


from ..lib.sqlite_funcs import (
    get_conn,
    setup_tab_twinds,
    setup_files_tab,
    setup_tab_twind_status,
    make_timeIDs,
)
from ..lib.dbread_funcs import (
    get_info_tab,
    get_twinds_tab,
    get_files_tab,
    get_full_sqlite_table_as_df,
    guess_dbfname,
)
from ..analysis_seeds.bkg_rate_estimation import cov2err, lin_func, get_chi2
from scipy import optimize
from ..lib.gti_funcs import check_if_in_GTI, add_bti2gti
from ..config import EBINS0, EBINS1

from ..lib.search_config import Config

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
        default=20 * 1.024,
    )
    parser.add_argument(
        "--min_dt", type=float, help="Min time from trigger to do", default=None
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
        "--min_tbin", type=float, help="Smallest tbin size to use", default=0.256
    )
    parser.add_argument(
        "--max_tbin",
        type=float,
        help="Largest tbin size to use",
        default=0.256 * (2**6),
    )
    parser.add_argument(
        "--snr_min", type=float, help="Min snr cut for time seeds", default=2.5
    )
    parser.add_argument(
        "--api_token",
        type=str,
        help="EchoAPI key for interactions.",
        default=None
    )

    args = parser.parse_args()
    return args


class Linear_Rates(object):
    def __init__(
        self,
        ev_data,
        tmin,
        tmax,
        trig_time,
        GTI,
        t_poly_step=1.024,
        bkg_post=True,
        poly_trng=15,
        sig_clip=None,
        tbin_size=0.256,
    ):
        self.t_bins0 = np.arange(tmin, tmax, tbin_size)
        self.t_bins1 = self.t_bins0 + tbin_size
        self.bin_size = tbin_size
        self.n_tbins = len(self.t_bins0)
        self.ev_data = ev_data
        self.cnts_per_tbin = get_cnts_tbins_fast(
            self.t_bins0, self.t_bins1, self.ev_data
        )

        self.GTI = GTI
        self.gti_tbin_bl = np.zeros(self.n_tbins, dtype=bool)
        for i in range(self.n_tbins):
            self.gti_tbin_bl[i] = check_if_in_GTI(GTI, self.t_bins0[i], self.t_bins1[i])
        print(len(self.gti_tbin_bl), np.sum(self.gti_tbin_bl))

        #         self.t_bins0 = t_bins0
        #         self.t_bins1 = t_bins1
        #         self.tstep = t_bins0[1] - t_bins0[0]
        #         self.bin_size = t_bins1[0] - t_bins0[0]
        self.sig_window = (-5.0 * 1.024, 10.0 * 1.024)
        self.sig_exp = self.sig_window[1] - self.sig_window[0]
        self.post = bkg_post
        self.deg = 1
        if bkg_post:
            self.bkg_window = (-30.0 * 1.024, 30.0 * 1.024)
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0] - self.sig_exp
        else:
            self.bkg_window = (-30.0 * 1.024, self.sig_window[0])
            self.bkg_exp = self.bkg_window[1] - self.bkg_window[0]
        self.trig_time = trig_time

        self.t_poly_step = t_poly_step

        self.t0 = trig_time - poly_trng * self.t_poly_step
        self.t1 = trig_time + poly_trng * self.t_poly_step

        self.t_poly_ax = np.arange(self.t0, self.t1, self.t_poly_step)
        self.n_lin_pnts = len(self.t_poly_ax)

        self.slopes = np.zeros(self.n_lin_pnts)
        self.ints = np.zeros_like(self.slopes)
        self.errs = np.zeros_like(self.slopes)
        self.chi2s = np.zeros_like(self.errs)
        self.dof = np.zeros_like(self.slopes, dtype=np.int64)
        self.sig_clip = sig_clip

        self.npars = self.deg + 1
        # self.dof = int(self.bkg_exp/self.bin_size) - self.npars

    def do_fits(self):
        for i in range(self.n_lin_pnts):
            t_mid = self.t_poly_ax[i]

            t_0 = t_mid + self.bkg_window[0]
            t_1 = t_mid + self.bkg_window[1]

            t_sig0 = t_mid + self.sig_window[0]
            t_sig1 = t_mid + self.sig_window[1]

            #             sig_twind = (-sig_wind/2. + tmid, sig_wind/2. + tmid)
            sig_twind = (t_sig0, t_sig1)
            gti_ = add_bti2gti(sig_twind, self.GTI)
            #             bkg_t0 = tmid - sig_wind/2. - bkg_dur/2.
            #             bkg_t1 = tmid + sig_wind/2. + bkg_dur/2.
            bkg_bti = Table(
                data=([-np.inf, t_1], [t_0, np.inf]), names=("START", "STOP")
            )
            #             bkg_bti = Table(data=([-np.inf, t_0], [t_1, np.inf]), names=('START', 'STOP'))
            gti_ = add_bti2gti(bkg_bti, gti_)
            tbl = np.zeros(self.n_tbins, dtype=bool)
            for ii in range(self.n_tbins):
                tbl[ii] = check_if_in_GTI(gti_, self.t_bins0[ii], self.t_bins1[ii])

            #             ind0 = np.argmin(np.abs(self.t_bins0 - t_0))
            #             ind1 = np.argmin(np.abs(self.t_bins1 - t_1))

            #             ind0_sig = np.argmin(np.abs(self.t_bins1 - t_sig0))
            #             ind1_sig = np.argmin(np.abs(self.t_bins0 - t_sig1))

            #             _t_ax0 = ((self.t_bins0 + self.t_bins1)/2.)[ind0:ind0_sig]
            #             _t_ax1 = ((self.t_bins0 + self.t_bins1)/2.)[ind1_sig:ind1]
            #             _t_ax = np.append(_t_ax0, _t_ax1) - self.trig_time

            #             _cnts = np.append(self.cnts_per_tbin[ind0:ind0_sig],\
            #                                   self.cnts_per_tbin[ind1_sig:ind1], axis=0)

            t_ax = ((self.t_bins0 + self.t_bins1) / 2.0)[tbl] - self.trig_time
            cnts = self.cnts_per_tbin[tbl]

            try:
                bl = np.ones(len(cnts), dtype=bool)
                if self.sig_clip is not None:
                    avg = np.mean(cnts)
                    std = np.std(cnts)
                    std_res = np.abs(cnts - avg) / std
                    while np.any(std_res[bl] > self.sig_clip):
                        bl[np.argmax(std_res)] = False
                        avg = np.mean(cnts[bl])
                        std = np.std(cnts[bl])
                        std_res = np.zeros_like(cnts)
                        std_res[bl] = np.abs(cnts[bl] - avg) / std
                        if (np.sum(bl) / float(len(bl)) < 0.7) or (np.sum(bl) < 10):
                            break

                res_lin = optimize.curve_fit(
                    lin_func,
                    t_ax[bl],
                    cnts[bl],
                    sigma=np.sqrt(cnts[bl]),
                    absolute_sigma=False,
                )
            except Exception as E:
                print(E)
                print("_cnts[:,j].shape: ", cnts.shape)
                print("_t_ax.shape: ", t_ax.shape)
                raise E

            tot_cnts = np.sum(cnts[bl])
            cnt_err = np.sqrt(tot_cnts) / (len(cnts[bl]))

            fit_err = cov2err(np.array(res_lin[1]), t_mid - self.trig_time)

            err = np.hypot(cnt_err, fit_err)

            self.slopes[i] = res_lin[0][0]
            self.ints[i] = res_lin[0][1]
            self.errs[i] = err

            preds = lin_func(t_ax[bl], self.slopes[i], self.ints[i])
            self.chi2s[i] = get_chi2(cnts[bl], preds)
            self.dof[i] = len(cnts[bl]) - self.npars

    def get_rate(self, t, chi2=False):
        ind = np.argmin(np.abs(t - self.t_poly_ax))

        rate = (
            lin_func(t - self.trig_time, self.slopes[ind], self.ints[ind])
            / self.bin_size
        )

        error = self.errs[ind] / self.bin_size
        if chi2:
            chi2 = self.chi2s[ind]
            dof = self.dof[ind]
            return rate, error, chi2 / dof

        return rate, error


def get_cnts_tbins_fast(tbins0, tbins1, evd):
    ntbins = len(tbins0)
    #     nebins = len(ebins0)
    #     quadIDs = ev2quad_ids(evd)
    tstep = tbins0[1] - tbins0[0]
    tbin_size = tbins1[0] - tbins0[0]
    tfreq = int(np.rint(tbin_size / tstep))
    t_add = [tbins0[-1] + (i + 1) * tstep for i in range(tfreq)]
    tbins = np.append(tbins0, t_add)
    #     ebins = np.append(ebins0, [ebins1[-1]])
    #     qbins = np.arange(5) - .5

    #     h = np.histogramdd([evd['TIME'],evd['ENERGY'],quadIDs],
    #                    bins=[tbins,ebins,qbins])[0]
    h = np.histogram(evd["TIME"], bins=tbins)[0]

    if tfreq <= 1:
        return h
    #     h2 = np.zeros((h.shape[0]-(tfreq-1),h.shape[1],h.shape[2]))
    h2 = np.zeros(h.size - (tfreq - 1))
    for i in range(tfreq):
        i0 = i
        i1 = -tfreq + 1 + i
        if i1 < 0:
            h2 += h[i0:i1]
        else:
            h2 += h[i0:]
    return h2


def calc_rate_snrs(tbins0, tbins1, tcnts, bkg_obj):
    ntbins = len(tbins0)
    snrs = np.zeros(ntbins)
    for i in range(ntbins):
        dur = tbins1[i] - tbins0[i]
        tmid = (tbins1[i] + tbins0[i]) / 2.0
        bkg_rate, bkg_rate_err = bkg_obj.get_rate(tmid)
        sig2_bkg = (bkg_rate_err * dur) ** 2 + (bkg_rate * dur)
        snrs[i] = (tcnts[i] - bkg_rate * dur) / np.sqrt(sig2_bkg)
    return snrs


def do_rates_analysis4dur(dtmin, dtmax, trig_time, dur, ev_data, bkg_obj):
    tstep = dur / 4.0
    tbins0 = np.arange(dtmin, dtmax, tstep) + trig_time
    tbins1 = tbins0 + dur
    tcnts = get_cnts_tbins_fast(tbins0, tbins1, ev_data)
    timeIDs = make_timeIDs(tbins0, dur * np.ones_like(tbins0), trig_time)

    #     rate_trig_scores = calc_rate_trig_scores(tbins0, tbins1, tcnts, bkg_obj)
    snrs = calc_rate_snrs(tbins0, tbins1, tcnts, bkg_obj)

    res_dict = {
        "time": tbins0,
        "duration": dur,
        "snr": snrs,
        "timeID": timeIDs,
        "dt": tbins0 - trig_time,
    }

    df = pd.DataFrame(data=res_dict)

    return df


def do_rates_analysis(dtmin, dtmax, trig_time, durs2do, ev_data, bkg_obj):
    dfs = []
    for dur in durs2do:
        dfs.append(
            do_rates_analysis4dur(dtmin, dtmax, trig_time, dur, ev_data, bkg_obj)
        )
    df = pd.concat(dfs, ignore_index=True)
    return df


def choose_tbins_archive(
    snr_res_tab, twind_size, tbin_size=60.0, snr_min=2.5, GTI=None
):
    tbins0 = np.arange(-twind_size - 1.0, twind_size + 1.0, tbin_size)
    tbins1 = tbins0 + tbin_size

    Ntbins = len(tbins0)

    seed_tabs = []

    for i in range(Ntbins):
        tbl = (snr_res_tab["dt"] >= tbins0[i]) & (snr_res_tab["dt"] < tbins1[i])
        df = choose_tbins(snr_res_tab[tbl], snr_min=snr_min, GTI=GTI)
        if len(df) > 0:
            seed_tabs.append(df)

    seed_tab = pd.concat(seed_tabs, ignore_index=True).drop_duplicates("timeID")
    logging.debug("full archive len(seed_tab): %d" % (len(seed_tab)))
    return seed_tab


def choose_tbins(snr_res_tab, snr_min=2.5, GTI=None):
    # probably should dur by dur and pick tbins
    # that go above the min
    # then also are the max or close to the max snr
    # within a certain time
    # maybe something like tbins that are within +/- 2dur
    # then keep all tbins that are above 0.75*max_snr (max of that group)
    # or at most half of the tbins in that group
    # possibly also put a stricter snr cut on smaller tbins

    dur_bins = [0.0, 0.2, 0.5, 100.0]
    # snr_mins = [2.5, 2.25, 2.0]
    snr_min_add = np.array([0.5, 0.25, 0.0])
    # snr_min = 2.0
    snr_mins = snr_min + snr_min_add

    df = snr_res_tab[(snr_res_tab["snr"] >= snr_min)]

    min_dt = np.min(df["dt"])
    max_dt = np.max(df["dt"])

    tbins = np.arange(min_dt, max_dt + np.max(df["duration"]) / 2.0, 2.048)

    rows2keep = []

    for dur, dur_df in df.groupby("duration"):
        # step of tbins is 2*dur
        # size of tbin is 4*dur, inside next loop +/- 2*dur
        tbins = np.arange(min_dt - dur, max_dt + 2 * dur, 2 * dur)
        tax = (tbins[:-1] + tbins[1:]) / 2.0
        res_dicts = []

        snr_thresh0 = snr_mins[np.digitize(dur, dur_bins) - 1]
        logging.debug("dur, snr_thresh0: %.3f, %.2f" % (dur, snr_thresh0))
        logging.debug(dur_df)

        if GTI is not None:
            gti_bl = []
            for ind, row in dur_df.iterrows():
                if check_if_in_GTI(GTI, row["time"], row["time"] + row["duration"]):
                    gti_bl.append(True)
                else:
                    gti_bl.append(False)
            gti_bl = np.array(gti_bl)

            # for ind, row in dur_df.sort_values('snr', ascending=False).iterrows():
            #     if check_if_in_GTI(GTI, row['time'], row['time']+row['duration']):
            #         if not np.isfinite(row['snr']):
            #             continue
            #         snr_max = row['snr']
            #         break
            #     else:
            #         continue
        else:
            gti_bl = np.ones(len(dur_df), dtype=bool)

        snr_max = np.max(dur_df[gti_bl]["snr"])

        logging.debug("dur, snr_max: %.3f, %.3f" % (dur, snr_max))

        for i in range(len(tax)):
            # size of tbin is 4*dur
            t0 = tax[i] - 2 * dur
            t1 = tax[i] + 2 * dur
            logging.debug("t0, t1: %.3f, %.3f" % (t0, t1))

            bl = (dur_df["dt"] >= t0) & (dur_df["dt"] < t1) & gti_bl
            if np.sum(bl) < 0:
                logging.debug("None between t0 and t1")
                continue

            logging.debug(
                "sum(isnan(dur_df[bl][snr])), sum(bl): %d, %d"
                % (np.sum(np.isnan(dur_df[bl]["snr"])), np.sum(bl))
            )

            if np.sum(np.isfinite(dur_df[bl]["snr"])) < 1:
                continue

            snr_max0 = np.nanmax(dur_df[bl]["snr"])
            logging.debug("snr_max0: %.3f" % (snr_max0))

            if snr_max0 < snr_thresh0:
                continue

            # snr_thresh = max(0.75*snr_max, snr_thresh0)
            snr_thresh = max(0.75 * snr_max0, snr_thresh0)

            df_sort = dur_df[bl].sort_values("snr", ascending=False)
            logging.debug("len(df_sort): %d" % (len(df_sort)))

            Nbin_max = 3  # max time seeds in tbin
            N = 0
            for row_ind, row in df_sort.iterrows():
                if row["snr"] >= snr_thresh:
                    if GTI is not None:
                        if not check_if_in_GTI(
                            GTI, row["time"], row["time"] + row["duration"]
                        ):
                            continue
                    rows2keep.append(row)
                    N += 1
                    if N >= Nbin_max:
                        break

    df2keep = pd.DataFrame(data=rows2keep)
    logging.debug("len(df2keep): %d" % (len(df2keep)))

    return df2keep.drop_duplicates("timeID")


def main(args):
    logging.basicConfig(
        filename="full_rates.log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )


    if args.api_token is not None:
        try:
            from EchoAPI import API
        except ImportError:
            return print("EchoAPI required, exiting.")
        #look for file called 'config.json' in working directory
        #if not present, use cli args
        config_filename= 'config.json'
        if os.path.exists(config_filename):
            search_config = Config(config_filename)
            args.min_tbin = search_config.MinDur
            args.max_tbin = search_config.MaxDur
            args.twind = search_config.MaxDT
            args.min_dt = search_config.MinDT
            args.snr_min = search_config.minSNR
            args.bkg_nopre = not search_config.BkgPre
            args.bkg_nopost = not search_config.BkgPost
            api = API(api_token = args.api_token)
        else:
            logging.error('Api_token passed but no config.json file found. Exiting.')
            return False

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

    evfname = files_tab["evfname"][0]
    ev_data = fits.open(evfname)[1].data
    try:
        GTI = Table.read(evfname, hdu="GTI_POINTING")
    except:
        GTI = Table.read(evfname, hdu="GTI")

    ebins0 = np.array(EBINS0)
    ebins1 = np.array(EBINS1)

    emin = ebins0[0]
    emax = ebins1[-1]

    emin = 15.0
    emax = 350.0

    ebl = (ev_data["ENERGY"] >= emin) & (ev_data["ENERGY"] < emax)
    ev_data = ev_data[ebl]

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

    poly_trng = int(args.twind)
    try:
        bkg_obj = Linear_Rates(
            ev_data, tmin, tmax, trigtime, GTI, sig_clip=4.0, poly_trng=poly_trng
        )
    except Exception as e:
        logging.error(e)
        if args.api_token is not None:
            try:
                api.report(search_config.queueID,complete=True)
            except Exception as e:
                logging.error(e)
                logging.error('Could not report complete to Queue via EchoAPI.')
        
        return -1

    logging.info("Inited bkg_obj, now starting fits")

    try:
        bkg_obj.do_fits()
    except Exception as e:
        logging.error(e)
        if args.api_token is not None:
            try:
                api.report(search_config.queueID,complete=False)
            except Exception as e:
                logging.error(e)
                logging.error('Could not report error to Queue via EchoAPI.')   

    logging.info("Done doing bkg linear fits")

    dur = args.min_tbin
    durs2do = [dur]
    while dur < args.max_tbin:
        dur = 2 * dur
        durs2do.append(dur)
    logging.info("durs to use: ")
    logging.info(durs2do)

    if args.min_dt is None:
        min_dt = -args.twind
    else:
        min_dt = max(-args.twind, args.min_dt)
    sig_dtmin = min_dt
    sig_dtmax = args.twind

    snr_df = do_rates_analysis(
        sig_dtmin, sig_dtmax, trigtime, durs2do, ev_data, bkg_obj
    )

    save_fname = "time_seeds.csv"
    logging.info("max snr: %.3f" % (np.nanmax(snr_df["snr"])))
    logging.debug("snr_min: %.3f" % (args.snr_min))
    logging.debug(snr_df.head())

    if np.nanmax(snr_df["snr"]) >= args.snr_min:
        if args.archive:
            df2keep = choose_tbins_archive(
                snr_df, args.twind, snr_min=args.snr_min, GTI=GTI
            )
        else:
            df2keep = choose_tbins(snr_df, snr_min=args.snr_min, GTI=GTI)
        logging.info("%d time seeds" % (len(df2keep)))
        logging.info("Saving results in a DataFrame to file: ")
        logging.info(save_fname)
        df2keep.to_csv(save_fname, index=False)

        if args.api_token is not None:
            from ..post_process.nitrates_reader import grab_full_rate_results
            from UtilityBelt.llhplot import plotly_waterfall_seeds

            #for now reading this file back in again. Really should just use the DataFrame defined above and add the 2 necessary columns...
            fullrate = grab_full_rate_results(os.getcwd(),search_config.triggerID, config_id=search_config.id)

            try:
                plot = plotly_waterfall_seeds(fullrate, search_config.triggerID, config_id = search_config.id)
            except Exception as e:
                logging.error(e)
                logging.error('Could not make rates waterfall plot.')

            try:
                api.post_nitrates_results(trigger=search_config.triggerID,config_id=search_config.id,result_type='n_FULLRATE',result_data=fullrate)
                with open(plot) as f:
                    api.post_nitrates_plot(trigger=search_config.triggerID,config_id=search_config.id,result_type='n_FULLRATE',plot_data=json.load(f))
                os.remove(plot)
            except Exception as e:
                logging.error(e)
                logging.error('Could not post to rates results via EchoAPI.')

    else:
        logging.info("0 time seeds")
        f = open(save_fname, "w")
        f.write("NONE")
        f.close()


if __name__ == "__main__":
    args = cli()

    main(args)
