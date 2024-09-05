import time
import numpy as np
import os
import argparse
import logging, traceback


from ..lib.sqlite_funcs import get_conn, write_seeds
from ..lib.dbread_funcs import (
    get_rates_tab,
    guess_dbfname,
    get_top_rate_timeIDs,
    query_blips_tab,
    get_twind_status_tab,
)
from ..config import EBINS0, EBINS1, quad_dicts


def find_and_write_seeds(
    args, db_fname, timeIDs2assign, job_id_start, min_priority=1, max_priority=6
):
    logging.info("Connecting to DB")
    conn = get_conn(db_fname)

    NtimeIDs = len(timeIDs2assign)

    top_timeIDs, rates_df = get_top_rate_timeIDs(conn, N=100, ret_rate_tab=True)

    TS_vals = rates_df["TS"].values
    percs = [50, 75, 90, 95, 98]
    TS_quants = np.nanpercentile(TS_vals, percs)
    logging.info("percs: " + str(percs))
    logging.info("TS_quants: " + str(TS_quants))
    inds_sort = np.argsort(TS_vals)[::-1]

    all_pc_min = 0.4
    half_imx = 0.6
    half_imy = 0.4
    snr_break = 4.0
    snr_high_break = 8.0
    N_jobs2twinds = 4

    half_names = [
        name
        for name in list(quad_dicts.keys())
        if ("quad" not in name) and (name != "all")
    ]
    quad_names = [name for name in list(quad_dicts.keys()) if "quad" in name]

    job_ids = None

    rates_twind_groups = rates_df.groupby("timeID")

    for i in range(NtimeIDs):
        # timeID = top_timeIDs[i]
        timeID = timeIDs2assign[i]
        blips = query_blips_tab(conn, timeID=timeID)
        rates_group = rates_twind_groups.get_group(timeID)

        job_ids = (
            i * N_jobs2twinds + np.arange(N_jobs2twinds, dtype=np.int64) + job_id_start
        )
        for ii in range(len(job_ids)):
            job_ids[ii] = job_ids[ii] % args.njobs
        logging.debug("job_ids: " + str(job_ids))

        priority = max_priority * np.ones(len(blips), dtype=np.int64)
        maxp = max_priority

        # blips['priority'] = priority

        # look at TSs for the different quadrants/havles/all
        # find if all of them are high
        # probably at least 2 or 3 will be high
        # give each blip a priority based on the rate TS
        # that's near it, priority =1 is the most urgent
        # possibly decide by where the TS falls in the cdf of all of them

        max_TS = np.max(rates_group["TS"])
        if max_TS > TS_quants[-1]:
            maxp = 3
        elif max_TS > TS_quants[-2]:
            maxp = 4
        elif max_TS > TS_quants[-3]:
            maxp = 5

        all_row = rates_group[(rates_group["quadID"] == 0)]

        all_TS = all_row["TS"].values[0]
        logging.debug("all_TS is " + str(all_TS))
        prior_all = 1
        for TSq in TS_quants:
            if TSq > all_TS:
                prior_all += 1
        prior_all = min(prior_all, maxp)

        blips_middle = blips["pc"] > all_pc_min

        priority[blips_middle] = prior_all

        for name in list(quad_dicts.keys()):
            if name == "all":
                continue
            qid = quad_dicts[name]["id"]
            imx = quad_dicts[name]["imx"]
            imy = quad_dicts[name]["imy"]
            qbl = rates_group["quadID"] == qid
            if np.sum(qbl) < 1:
                continue
            TS = rates_group[(rates_group["quadID"] == qid)]["TS"].values[0]
            prior = 1
            for TSq in TS_quants:
                if TSq > TS:
                    prior += 1
            prior = min(prior, maxp)
            if name in ["left", "right"]:
                blips_bl = np.sign(imx) * blips["imx"] > half_imx
            elif name in ["top", "bottom"]:
                blips_bl = np.sign(imy) * blips["imy"] > half_imy
            else:
                blips_bl = (np.sign(imy) * blips["imy"] > np.abs(imy) - 0.2) & (
                    np.sign(imx) * blips["imx"] > np.abs(imx) - 0.25
                )
            Nblips = np.sum(blips_bl)
            for ii in range(Nblips):
                p0 = priority[blips_bl][ii]
                if prior < p0:
                    priority[blips_bl][ii] = prior

        proc_group = -1 * np.ones_like(priority)

        priority[(priority > maxp)] = maxp
        snr_bl = blips["snr"] < snr_break
        priority[snr_bl] += 1
        snr_high_bl = blips["snr"] >= snr_high_break
        priority[snr_high_bl] = 1

        for ii in range(max_priority + 1):
            bl = priority == (ii + 1)
            n_p = np.sum(bl)
            logging.debug("%d seeds with priority %d" % (n_p, ii + 1))
            n_per_job = 2 + n_p / N_jobs2twinds
            if n_p > 0:
                p_inds = np.where(bl)[0]
                for j in range(n_p):
                    proc_group[p_inds[j]] = job_ids[j % N_jobs2twinds]
                # for j in xrange(N_jobs2twinds):
                #    j0 = j*n_per_job
                #    j1 = j0 + n_per_job
                #    inds = np.arange(j0, j1+1, dtype=np.int64)
                #    logging.debug("assigning %d seeds job_id %d" %(j1-j0, job_ids[j]))
                #    proc_group[bl][inds] = job_ids[j]*np.ones(len(inds), dtype=np.int64)
                #    proc_group[bl][j0:j1] = job_ids[j]

        logging.debug("min(proc_group): %d" % (np.min(proc_group)))
        logging.debug("max(proc_group): %d" % (np.max(proc_group)))

        blips["proc_group"] = proc_group
        blips["priority"] = priority

        try:
            write_seeds(conn, blips)
        except Exception as E:
            logging.error(E)
            logging.error(traceback.format_exc())
            logging.error("Failed writing seeds, trying again")
            try:
                conn.close()
                conn = get_conn(db_fname)
                write_seeds(conn, blips)
            except Exception as E:
                logging.error(E)
                logging.error(traceback.format_exc())
                logging.error("Failed to write seeds")

    conn.close()
    return job_ids


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--njobs", type=int, help="Number of Jobs being used", default=20
    )
    parser.add_argument(
        "--dbfname", type=str, help="Name to save the database to", default=None
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.dbfname is None:
        db_fname = guess_dbfname()
        if isinstance(db_fname, list):
            db_fname = db_fname[0]
    else:
        db_fname = args.dbfname

    fname = "assigning_seeds"

    logging.basicConfig(
        filename=fname + ".log",
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )

    f = open(fname + ".pid", "w")
    f.write(str(os.getpid()))
    f.close()

    logging.info("Wrote pid: %d" % (os.getpid()))

    time_starting = time.time()
    job_id_start = 0
    dt = 0.0
    while dt < (24.0 * 3600.0):
        conn = get_conn(db_fname)
        status_df = get_twind_status_tab(conn)
        conn.close()

        blips_found = status_df["BlipsFounds"] == 1
        SeedsNotAssigned = status_df["SeedsAssigned"] == 0
        seeds_to_assign = blips_found & SeedsNotAssigned
        N_toassign = np.sum(seeds_to_assign)
        logging.info(str(N_toassign) + " time windows with seeds to assign")
        if N_toassign > 0:
            timeIDs2assign = (status_df["timeID"][seeds_to_assign]).values
            last_job_ids = find_and_write_seeds(
                args, db_fname, timeIDs2assign, job_id_start
            )
            if last_job_ids is not None:
                job_id_start = (last_job_ids[-1] + 1) % args.njobs
        time.sleep(30.0)
        dt = time.time() - time_starting


if __name__ == "__main__":
    args = cli()

    main(args)
