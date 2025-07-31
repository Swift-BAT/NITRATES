import re
import os
import pandas as pd
import subprocess
import json
import numpy as np
import sqlite3

from math import floor
from datetime import datetime, timedelta, timezone
import pytz
from astropy.io import fits
from ..lib.coord_conv_funcs import imxy2theta_phi, convert_theta_phi2radec
from ..lib.sqlite_funcs import get_conn
from ..lib.dbread_funcs import get_info_tab
from ..lib.search_config import Config

import argparse
from pathlib import Path

# Assume all nitrates archival jobs are running on computers with US/Eastern timestamps
tzlocal = pytz.timezone("US/Eastern")
utc = pytz.timezone("UTC")

def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, help="Results directory", default= Path.cwd())
    parser.add_argument("--api_token", type=str, help="api token", default= None)
    parser.add_argument("--mpi4py", action='store_true', help="This switch, when set, will redirect some of the defaults to use the logs output from the mpi4py version of NITRATES.")

    args = parser.parse_args()

    return args


def read_manager_log(path, ismpi4py=False):

    if ismpi4py:
        #if this is set to true, we want to look at the nitrataes_0.log file
        logfile="nitrates_0.log"
    else:
        logfile="manager.log"
        
    print(f"Reading from the log file {logfile}")
        
    start = None
    bkgstart = None
    splitstart = None
    splitdone = None
    Nsquares = None
    Ntbins = None
    Ntotseeds = None
    OFOVfilesTot = None
    IFOVfilesTot = None
    submitIFOVstamp = None
    NjobsIFOV = None
    submitOFOVstamp = None
    NjobsOFOV = None
    OFOVDone = None
    IFOVDone = None
    OFOVfilesDone = None
    IFOVfilesDone = None

    try:
        with open(os.path.join(path, logfile)) as fob:
            x = fob.read()
            
            #get the start time of NITRATES
            if not ismpi4py:
                startlocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Wrote pid:", x
                    )[1]
                )
            else:
                startlocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),*", x.split("\n")[0])[1])
                
            start = tzlocal.localize(startlocal).astimezone(utc)
            
            
            # print(f'start:{start}')
            #get the start time of the background estimation
            if not ismpi4py:
                bkgstartlocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Job submitted", x
                    )[1]
                )
            else:
                bkgstartlocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Conducting the background estimation", x)[1] )
                bkgendlocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- bkg_estimation.csv", x)[1] )
                
            bkgstart = tzlocal.localize(bkgstartlocal).astimezone(utc)
            bkgend = tzlocal.localize(bkgendlocal).astimezone(utc)
            # print(f'bkg start: {bkgstart}')
            
            #get the start of the splitrates analysis
            if not ismpi4py:
                splitstartlocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Jobs submitted", x
                    )[1]
                )
            else:
                splitstartlocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-ERROR- .*SplitRatesStart *", x)[1] )
            splitstart = tzlocal.localize(splitstartlocal).astimezone(utc)
            # print(f'split start: {splitstart}')
            
            #get the end time of the splitrates analysis
            if not ismpi4py:
                splitdonelocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO-\s+Done with rates analysis",
                        x,
                    )[1]
                )
            else:
                splitdonelocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-ERROR- .*SplitRatesDone *", x)[1] )
            splitdone = tzlocal.localize(splitdonelocal).astimezone(utc)
            # print(f'split done: {splitdone}')
            
            #get stats of how many seeds there are in total to analyze
            Nsquares = re.search(r"Nsquares: ([0-9]+)\n", x)[1]
            Ntbins = re.search(r"Ntbins: ([0-9]+)\n", x)[1]
            Ntotseeds = re.search(r"Ntot Seeds: ([0-9]+)\n", x)[1]
            
            #get how many have been completed already
            if not ismpi4py:
                OFOVfilesTot = re.search(r"of ([0-9]+) out files done\n", x[1])
                IFOVfilesTot = re.search(r"of ([0-9]+) in files done\n", x[1])
            else:
                OFOVfilesTot = re.search("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- .*OFOVfilesTot .* (\d+)", x)
                IFOVfilesTot = re.search("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- .*IFOVfilesTot .* (\d+)", x)
            
            #get when the IFOV job has been submitted, this isnt applicable to the mpi4py code. Instead Look a when this portion of the analysis started.
            if not ismpi4py:
                submitIFOV = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Submitting (\d+) in FoV Jobs now",
                    x,
                )
            else:
                submitIFOV = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- .*IFOV analysis *", x)
            submitIFOVlocal = datetime.fromisoformat(submitIFOV[1])
            submitIFOVstamp = tzlocal.localize(submitIFOVlocal).astimezone(utc)
            
            #get the number of IFOV jobs submitted
            if not ismpi4py:
                NjobsIFOV = submitIFOV[2]
            else:
                try:
                    #get the number of requested mpi processes
                    NjobsIFOV = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-DEBUG- Njobs, job_iter: (\d+), (\d+)", x)[2]
                except TypeError as e:
                    #sometimes the underlying nitrates function doesnt print the job_iter so can just look for Njobs
                    print(e)
                    NjobsIFOV = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-DEBUG- Njobs*: (\d+)*", x)[2]
            # print(f'IFOV start: {submitIFOVstamp}')
            
            #get when the OFOV job has been submitted, this isnt applicable to the mpi4py code
            if not ismpi4py:
                submitOFOV = re.search(
                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Submitting (\d+) out of FoV Jobs now",
                    x,
                )
            else:
                submitOFOV = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- .*OFOV analysis *", x)
            submitOFOVlocal = datetime.fromisoformat(submitOFOV[1])
            submitOFOVstamp = tzlocal.localize(submitOFOVlocal).astimezone(utc)
            
            #get the number of IFOV jobs submitted. For mpi this is the same as the number of mpi processes which we got above
            if not ismpi4py:
                NjobsOFOV = submitOFOV[2]
            else:
                NjobsOFOV = NjobsIFOV
            # print(f'OFOV start: {submitOFOVstamp}')
            # print(Nsquares, Ntbins, Ntotseeds, NjobsIFOV, NjobsOFOV)
            
            #get the number of OFOV and IFOV jobs that have been completed
            if not ismpi4py:
                OFOVfilesTot = int(re.search(r"of ([0-9]+) out files done", x)[1])
                IFOVfilesTot = int(re.search(r"of ([0-9]+) in files done", x)[1])
            else:
                OFOVfilesTot = int(OFOVfilesTot[2])
                IFOVfilesTot = int(IFOVfilesTot[2])

            # print(OFOVfilesTot, IFOVfilesTot)
            
            #get the portion of each IFOV and OFOV jobs that are done
            if not ismpi4py:
                OFOVfilesDone = int(re.findall("(\d+) of (\d+) out files done", x)[-1][0])
                IFOVfilesDone = int(re.findall("(\d+) of (\d+) in files done", x)[-1][0])
            else:
                #the mpi code has to all finish at the same time
                OFOVfilesDone = OFOVfilesTot
                IFOVfilesDone = IFOVfilesTot
            
            #get when the OFOV analysis is complete
            if not ismpi4py:
                OFOVdonelocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Got all of the out results now",
                        x,
                    )[1]
                )
            else:
                OFOVdonelocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-DEBUG- Completed the OFOV analysis", x)[1] )
            OFOVDone = tzlocal.localize(OFOVdonelocal).astimezone(utc)
            # print(f'OFOV done: {OFOVdone}')
            
            #get when the IFOV analysis is complete
            if not ismpi4py:
                IFOVdonelocal = datetime.fromisoformat(
                    re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO- Got all of the in results now",
                        x,
                    )[1]
                )
            else:
                IFOVdonelocal = datetime.fromisoformat(re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-DEBUG- Completed the IFOV analysis", x)[1] )
            IFOVDone = tzlocal.localize(IFOVdonelocal).astimezone(utc)
            # print(f'IFOV done: {IFOVdone}')
    except Exception as e:
        print(e)

    return (
        start,
        bkgstart,
        splitstart,
        splitdone,
        Nsquares,
        Ntbins,
        Ntotseeds,
        OFOVfilesTot,
        IFOVfilesTot,
        submitIFOVstamp,
        NjobsIFOV,
        submitOFOVstamp,
        NjobsOFOV,
        OFOVDone,
        IFOVDone,
        OFOVfilesDone,
        IFOVfilesDone,
    )


def grab_full_rate_results(path, trig_id, config_id=0):
    full_rate_results = pd.read_csv(os.path.join(path, "time_seeds.csv"))
    full_rate_results["trigger_id"] = trig_id
    full_rate_results["config_id"] = config_id

    return full_rate_results


def grab_split_rate_results(
    path, trig_id, att_q, trigtime, top_n=64, notDB=False, config_id=0
):
    # Read in all csvs in the form rates_llh_res_*_.csv where * is replaced with an integer
    split_rate_regex = re.compile(r"rates_llh_(out_)?res_\d+_.csv")
    files=os.listdir(path)
    dfs = [
        pd.read_csv(os.path.join(path, file))
        for file in filter(split_rate_regex.fullmatch, files)
    ]

    # Various sources of x-ray radiation I think? that pop up in split rate results CSVs when they are within bat FOV
    sources = [
        "1A 0535-262j",
        "Sco X-1",
        "GX 304-1",
        "Vela X-1",
        "Cyg X-1",
        "GX 301-2",
        "4U 1700-377",
        "Crab Nebula",
        "EXO 2030+375",
        "GRO J1008-57",
        "Swift J1745.1-2624",
        "XTE J1752-223",
        "GRO J1655-40",
        "GX 339-4",
        "GRS 1915+105",
        "1A 1118-61",
        "Cyg X-3",
    ]

    # Replaces spaces, dots, pluses, and hyphens with an underscore and adds src_ to the beginning of the col so they're
    # valid var names. The _rt_sum is a suffix present in the CSVs and is preserved in the SQL table.
    src_renaming = {
        x + "_rt_sum": "src_" + re.sub(r"[ \+\.-]", "_", x) + "_rt_sum" for x in sources
    }

    split_rate_results = pd.concat(dfs, axis=0, ignore_index=True).rename(
        mapper=src_renaming, axis=1
    )

    # Grabbing top_n results by test statistic
    split_rate_results = split_rate_results.iloc[
        split_rate_results["TS"].nlargest(top_n).index
    ]
    split_rate_results["dt"] = split_rate_results["time"] - trigtime
    split_rate_results["trigger_id"] = trig_id
    split_rate_results["config_id"] = config_id

    thetas, phis = imxy2theta_phi(split_rate_results.imx, split_rate_results.imy)
    split_rate_results["theta"].fillna(thetas, inplace=True)
    split_rate_results["phi"].fillna(phis, inplace=True)
    ras, decs = convert_theta_phi2radec(
        split_rate_results.theta, split_rate_results.phi, att_q
    )
    ra_dec = [
        f"POINT({ra} {dec})" if not np.isnan(ra) else None for ra, dec in zip(ras, decs)
    ]
    split_rate_results["ra_dec"] = ra_dec

    return split_rate_results


def grab_out_fov_results(
    path,
    trig_id,
    att_q,
    trigtime,
    top_n=1000000,
    notDB=False,
    cluster=True,
    config_id=0,
):
    # Read in all csvs in the form res_hpind_*_.csv where * is replaced with an integer
    out_fov_regex = re.compile(r"res_hpind_\d+_.csv")
    files=os.listdir(path)
    if cluster:
        dfs = [
            pd.read_csv(os.path.join(path, file), index_col=0)
            for file in filter(out_fov_regex.fullmatch, files)
        ]
        out_fov_results = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        files = [file for file in files if out_fov_regex.fullmatch(file)]
        subprocess.run(f"sed -n 1p {files[0]} > merged.csv", cwd=path, shell=True)
        subprocess.run(
            f"tail -q -n +2 res_hpind*.csv >> merged.csv", cwd=path, shell=True
        )
        subprocess.run("rm merged.csv", cwd=path, shell=True)
        out_fov_results = pd.read_csv(
            os.path.join(path, "merged.csv"), index_col=None, engine="pyarrow"
        )
        out_fov_results = out_fov_results.iloc[:, 1:]

    out_fov_results = out_fov_results.drop_duplicates()
    out_fov_results = out_fov_results.loc[out_fov_results["TS"].nlargest(top_n).index]

    out_fov_results["dt"] = out_fov_results["time"] - trigtime
    out_fov_results["trigger_id"] = trig_id
    out_fov_results["config_id"] = config_id

    ras, decs = convert_theta_phi2radec(
        out_fov_results.theta, out_fov_results.phi, att_q
    )
    ra_dec = [f"POINT({ra} {dec})" for ra, dec in zip(ras, decs)]
    out_fov_results["ra_dec"] = ra_dec
    phi_theta = [
        f"POINT({phi} {theta-90})"
        for theta, phi in zip(out_fov_results.theta, out_fov_results.phi)
    ]
    out_fov_results["phi_theta"] = phi_theta

    return out_fov_results


def grab_in_fov_results(
    path,
    trig_id,
    att_q,
    trigtime,
    top_n=1000000,
    notDB=False,
    cluster=True,
    config_id=0,
):
    # Read in all csvs in the form peak_res_*_*_.csv (if TS > 6) where * is replaced with an integer, optionally
    # omitting peak_ for the files in which TS <= 6
    in_fov_regex = re.compile(r"(peak_)?res_\d+_\d+_.csv")
    files=os.listdir(path)
    
    if cluster:
        dfs = [
            pd.read_csv(os.path.join(path, file), index_col=0)
            for file in filter(in_fov_regex.fullmatch, files)
        ]
        in_fov_results = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        files = [file for file in files if in_fov_regex.fullmatch(file)]
        if any(file.startswith("peak") for file in files):
            peak = True
            filepeak = next(file for file in files if file.startswith("peak"))
        else:
            peak = False
        fileres = next(file for file in files if file.startswith("res"))

        subprocess.run(f"mkdir hpind", cwd=path, shell=True)
        subprocess.run(f"mv res_hpind* ./hpind", cwd=path, shell=True)
        subprocess.run(f"mkdir rates", cwd=path, shell=True)
        subprocess.run(f"mv rates_llh* ./rates", cwd=path, shell=True)
        subprocess.run(f"sed -n 1p {fileres} > resmerged.csv", cwd=path, shell=True)
        subprocess.run(
            f"tail -q -n +2 res_*.csv >> resmerged.csv", cwd=path, shell=True
        )

        in_fov_results_res = pd.read_csv(
            os.path.join(path, "resmerged.csv"), index_col=None, engine="pyarrow"
        )
        subprocess.run("rm resmerged.csv", cwd=path, shell=True)
        if peak:
            subprocess.run(
                f"sed -n 1p {filepeak} > peakmerged.csv", cwd=path, shell=True
            )
            subprocess.run(
                f"tail -q -n +2 peak_*.csv >> peakmerged.csv", cwd=path, shell=True
            )
            in_fov_results_peak = pd.read_csv(
                os.path.join(path, "peakmerged.csv"), index_col=None, engine="pyarrow"
            )
            subprocess.run("rm peakmerged.csv", cwd=path, shell=True)
            in_fov_results = pd.concat(
                [in_fov_results_res, in_fov_results_peak], axis=0, ignore_index=True
            )
        else:
            in_fov_results = in_fov_results_res

        subprocess.run("mv ./hpind/* .", cwd=path, shell=True)
        subprocess.run("rm -r ./hpind", cwd=path, shell=True)
        subprocess.run("mv ./rates/* .", cwd=path, shell=True)
        subprocess.run("rm -r ./rates", cwd=path, shell=True)

        in_fov_results = in_fov_results.iloc[:, 1:]

    in_fov_results = in_fov_results.drop_duplicates()

    # Filtering to only the top result for each squareID and timeID combo
    inds = (
        in_fov_results.groupby(["squareID", "timeID"])["TS"].transform(max)
        == in_fov_results["TS"]
    )
    in_fov_results = in_fov_results[inds]

    # Grabbing top_n results by test statistic, leaving df as is if there are top_n or fewer rows
    if inds.sum() > top_n:
        in_fov_results = in_fov_results.loc[in_fov_results["TS"].nlargest(top_n).index]

    # Initializing some columns
    in_fov_results["dt"] = in_fov_results["time"] - trigtime
    in_fov_results["trigger_id"] = trig_id
    in_fov_results["config_id"] = config_id

    ras, decs = convert_theta_phi2radec(in_fov_results.theta, in_fov_results.phi, att_q)
    ra_dec = [f"POINT({ra} {dec})" for ra, dec in zip(ras, decs)]
    in_fov_results["ra_dec"] = ra_dec
    phi_theta = [
        f"POINT({phi} {theta-90})"
        for theta, phi in zip(in_fov_results.theta, in_fov_results.phi)
    ]
    in_fov_results["phi_theta"] = phi_theta

    return in_fov_results


def im_dist(imx0, imy0, imx1, imy1):
    return np.hypot((imx1 - imx0), (imy1 - imy0))


def get_dlogls_inout(res_tab, res_out_tab, trigger_id, config_id=0, imdistthresh=0.012):
    """
    returns DeltaLLH_peak, DeltaLLH_out, timeID for each time bin
    """

    TSs = []
    dlogls = []
    dlogls_in_out = []
    timeIDs = []
    radecs = []
    phithetas = []
    As = []
    Epeaks = []
    gammas = []
    dts = []
    durs = []
    for timeID, df in res_tab.groupby("timeID"):
        idx = df["TS"].idxmax()
        row = df.loc[idx]
        imdists = im_dist(row["imx"], row["imy"], df["imx"], df["imy"])
        bld = imdists > imdistthresh
        try:
            dlogls.append(np.nanmin(df[bld]["nllh"]) - row["nllh"])
        except Exception as E:
            print(E)
            dlogls.append(np.nan)
        blo = np.isclose(res_out_tab["timeID"], timeID, rtol=1e-9, atol=1e-3)
        dlogls_in_out.append(np.nanmin(res_out_tab[blo]["nllh"]) - row["nllh"])
        timeIDs.append(timeID)
        TSs.append(row["TS"])
        radecs.append(row["ra_dec"])
        As.append(row["A"])
        Epeaks.append(row["Epeak"])
        gammas.append(row["gamma"])
        dts.append(row["dt"])
        durs.append(row["dur"])
        phithetas.append(row["phi_theta"])

    tuples = list(
        zip(
            TSs,
            dlogls,
            dlogls_in_out,
            radecs,
            phithetas,
            As,
            Epeaks,
            gammas,
            timeIDs,
            dts,
            durs,
        )
    )
    df = pd.DataFrame(
        tuples,
        columns=[
            "maxTS",
            "DeltaLLHPeak",
            "DeltaLLHOut",
            "ra_dec",
            "phi_theta",
            "A",
            "Epeak",
            "gamma",
            "timeID",
            "dt",
            "dur",
        ],
    )
    df["trigger_id"] = trigger_id
    df["config_id"] = config_id
    df = df.sort_values(by=["maxTS"], ascending=False)
    return df


def read_results_dirs(paths, api_token, figures=True, ismpi4py=False):
    try:
        from swifttools.swift_too import Clock
        from EchoAPI import API
        from UtilityBelt.llhplot import plotly_waterfall_seeds, plotly_splitrates, plotly_dlogl_sky
    except ImportError:
        return print("swiftools, EchoAPI, and UtilityBelt required, exiting.")

    config_values = [0, 1, 2, 99]  # see https://guano.swift.psu.edu/configs

    api = API(api_token=api_token)

    if isinstance(paths, str):
        paths = [paths]

    # Number of triggers to process
    n_dirs = len(paths)

    trig_ids = [0] * n_dirs
    datetimes = [0] * n_dirs
    sctimes = [0] * n_dirs
    files = [[]] * n_dirs
    att_qs = [0] * n_dirs

    failed = {}
    for i, path in enumerate(paths):
        try:
            print("***********************************************")
            print(f"Starting {path}")


            #look for file called 'config.json' in working directory
            #if not present, use default
            config_filename= os.path.join(path, 'config.json')
            if os.path.exists(config_filename):
                search_config = Config(config_filename)
                config_id = search_config.id
            else:
                logging.error('Api_token passed but no config.json file found. Assuming config_id=0')
                config_id=0

            if config_id in config_values:
                print(f"Config: {config_id}")
            else:
                raise ValueError(
                    f"The value {config_id} is not a valid configuration id value. See https://guano.swift.psu.edu/configs for more information."
                )

            # Grabbing trigger time from results.db
            try:
                conn = get_conn(os.path.join(path, "results.db"))
                info_tab = get_info_tab(conn)
                datetimes[i] = datetime.fromisoformat(info_tab["trigtimeUTC"][0])
                datetimes[i] = datetimes[i].replace(tzinfo=timezone.utc)
                sctimes[i] = Clock(utctime=datetimes[i]).met
            except Exception as e:
                print("Can not find triggertime in results.db. Exiting")
                failed[path] = "Can not find triggertime in results.db."
                print(e)
                continue

            # Grabbing timestamps from data log
            try:
                with open(os.path.join(path, "data_setup.log")) as fob:
                    x = fob.read()
                    evdatafound = None
                    try:
                        evdatafoundlocal = datetime.fromisoformat(
                            re.search(
                                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO-\s+[0-9]+ event files found",
                                x,
                            )[1]
                        )
                        evdatafound = tzlocal.localize(evdatafoundlocal).astimezone(utc)
                    except Exception as e:
                        print("No EvData found timestamp in log. Trying to continue ")
                        print(e)

                    alldatafound = None
                    try:
                        alldatafoundlocal = datetime.fromisoformat(
                            re.search(
                                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+-INFO-\s+Finally got all the data",
                                x,
                            )[1]
                        )
                        alldatafound = tzlocal.localize(alldatafoundlocal).astimezone(
                            utc
                        )
                    except Exception as e:
                        print("No All Data found timestamp in log. Trying to continue")
                        print(e)
            except Exception as e:
                print(
                    "No data setup log found! Or issue with data setup log. Non-critical, trying to continue"
                )
                print(e)

            # datetimes[i] = datetime(2023,5,3,1,1,1,tzinfo=timezone.utc)
            try:
                # If a trigtime within 1 second of this time is in the database, associate the file with the trigger
                response = api.get_trig(datetimes[i].isoformat())
            except Exception as e:
                print("Failed to query triggers from API.")
                print(e)
                failed[path] = "Failed to query triggers from API."
                continue

            try:
                existing_trig = response["trigger_id"]
            except Exception as e:
                print(e)
                print(response)
                failed[path] = response
                continue

            if existing_trig:
                print("Already existing trigger!")
                trig_ids[i] = existing_trig
            # If not, create a new entry in the triggers table
            else:
                trig_ids[i] = floor(Clock(utctime=datetimes[i]).met)
                dir_name = os.path.basename(os.path.normpath(path))

                index = next(
                    (i for i, c in enumerate(dir_name) if c.isdigit()), len(dir_name)
                )
                instrument_code = dir_name[:index]

                # instrument_code = re.sub(r'\d', '', dir_name)
                event_numbers = dir_name[index:]

                instrument = {
                    "F": "Fermi/GBM",
                    "C": "CALET",
                    "IC": "IceCube",
                    "I": "INTEGRAL",
                    "FRB": "CHIME",
                    "S": "IGWN",
                    "H": "HAWC",
                    "Gutn": "GECAM",
                }.get(instrument_code)
                trigger_type = {
                    "Fermi/GBM": "GRB",
                    "IceCube": "neutrino",
                    "INTEGRAL": "GRB",
                    "CALET": "GRB",
                    "CHIME": "FRB",
                    "IGWN": "GW",
                    "HAWC": "GRB",
                    "GECAM": "GRB",
                }.get(instrument)

                if instrument == None:
                    instrument = "Unknown"
                if trigger_type == None:
                    trigger_type = "GRB"

                response = api.post_trig(
                    trigger_time=datetimes[i].isoformat(),
                    trigger_instrument=instrument,
                    trigger_name=f"{instrument} {event_numbers}",
                    trigger_type=trigger_type,
                )

                trig_ids[i] = response["trigger_id"]

            print(f"Trigger: {trig_ids[i]}")

            # trigger_type is undefined if trigger already exists in DB, need to do this differently.
            # if trigger_type == 'GW':
            #     try:
            #         newfilename = os.path.join(outdir, f'{trig_ids[i]}__skymap.fits')
            #         oldfilename=os.path.join(path, 'skymap.fits')
            #         shutil.copy(oldfilename,newfilename)
            #         upload_file(BUCKET, newfilename)
            #         os.remove(newfilename)
            #         print('Uploaded skymap')
            #     except:
            #         print('No skymap found')

            # log it!
            (
                start,
                bkgstart,
                splitstart,
                splitdone,
                Nsquares,
                Ntbins,
                Ntotseeds,
                OFOVfilesTot,
                IFOVfilesTot,
                submitIFOVstamp,
                NjobsIFOV,
                submitOFOVstamp,
                NjobsOFOV,
                OFOVDone,
                IFOVDone,
                OFOVfilesDone,
                IFOVfilesDone,
            ) = read_manager_log(path, ismpi4py=ismpi4py)
            
            try:
                api.post_log(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    EvDataFound=evdatafound.isoformat() if evdatafound else None,
                    AllDataFound=alldatafound.isoformat() if alldatafound else None,
                    NITRATESstart=start.isoformat(),
                    BkgStart=bkgstart.isoformat(),
                    SplitRatesStart=splitstart.isoformat(),
                    SplitRatesDone=splitdone.isoformat(),
                    SquareSeeds=Nsquares,
                    TimeBins=Ntbins,
                    TotalSeeds=Ntotseeds,
                    IFOVStart=submitIFOVstamp.isoformat(),
                    IFOVjobs=NjobsIFOV,
                    IFOVDone=IFOVDone.isoformat(),
                    OFOVStart=submitOFOVstamp.isoformat(),
                    OFOVjobs=NjobsOFOV,
                    OFOVDone=OFOVDone.isoformat(),
                    OFOVfilesTot=OFOVfilesTot,
                    OFOVfilesDone=OFOVfilesDone,
                    IFOVfilesTot=IFOVfilesTot,
                    IFOVfilesDone=IFOVfilesDone,
                )
                print("Posted log")
            except Exception as e:
                print("Failed to post log")
                print(e)

            # Grab attitude quaternion at trigtime for later use to convert imx and imy to ra dec
            try:
                attfile = fits.open(os.path.join(path, "attitude.fits"))[1].data
                att_qs[i] = attfile["QPARAM"][
                    np.argmin(np.abs(attfile["TIME"] - sctimes[i]))
                ]
            except Exception as e:
                print("No attitude file found. Exiting!")
                failed[path] = "No attitude file found."
                continue

            # Full Rate Results
            try:
                dfs = [
                    grab_full_rate_results(paths[i], trig_ids[i], config_id=config_id)
                ]
                full_rate_results = pd.concat(dfs, axis=0, ignore_index=True)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_FULLRATE",
                    result_data=full_rate_results,
                )
                print(result)
                if figures:
                    rates = grab_full_rate_results(paths[i], trig_ids[i])
                    plot = plotly_waterfall_seeds(rates, trig_ids[i], config_id=config_id)
                    print(f"Made {plot}")
                    with open(plot) as f:
                        api.post_nitrates_plot(trig_ids[i], config_id, 'n_FULLRATE', json.load(f))
                    os.remove(plot)
                    print("Uploaded rates seeds plot :)")

            except Exception as e:
                print("Failed to ingest full rates results :(")
                print(e)
                failed[path] = "Failed to ingest full rates results."

            # Split Rate Results
            try:
                dfs = [
                    grab_split_rate_results(
                        paths[i],
                        trig_ids[i],
                        att_qs[i],
                        sctimes[i],
                        config_id=config_id,
                    )
                ]
                split_rate_results = pd.concat(dfs, axis=0, ignore_index=True)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_SPLITRATE",
                    result_data=split_rate_results,
                )
                print(result)

                if figures:
                    splitrates = grab_split_rate_results(
                        paths[i],
                        trig_ids[i],
                        att_qs[i],
                        sctimes[i],
                        top_n=1000000,
                        notDB=True,
                    )
                    plot = plotly_splitrates(trig_ids[i], splitrates, config_id=config_id)
                    print(f"Made {plot}")
                    with open(plot) as f:
                        api.post_nitrates_plot(trig_ids[i], config_id, 'n_SPLITRATE', json.load(f))
                    os.remove(plot)
                    print("Uploaded split rates plot :)")
            except Exception as e:
                print("Failed to ingest split rates results")
                print(e)
                failed[path] = "Failed to ingest split rates results."

            # Out of FoV Results
            try:
                res_out_tab = grab_out_fov_results(
                    paths[i],
                    trig_ids[i],
                    att_qs[i],
                    sctimes[i],
                    notDB=True,
                    config_id=config_id,
                )
                table64 = res_out_tab.loc[res_out_tab["TS"].nlargest(64).index]
                dfos = [table64]
                out_fov_results = pd.concat(dfos, axis=0, ignore_index=True)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_OUTFOV",
                    result_data=out_fov_results,
                )
                print(result)
                if figures:
                    # make the plot, and upload it!
                    plot = plotly_dlogl_sky(trig_ids[i], res_out_tab, config_id=config_id)
                    print(f"Made {plot}")
                    with open(plot) as f:
                        api.post_nitrates_plot(trig_ids[i], config_id, 'n_OUTFOV', json.load(f))
                    os.remove(plot)
                    print("Uploaded OFOV plot :)")
            except Exception as e:
                print("Failed to ingest OFOV results :(")
                print(e)
                failed[path] = "Failed to ingest OFOV results."

            # In FoV Results
            try:
                res_in_tab = grab_in_fov_results(
                    paths[i],
                    trig_ids[i],
                    att_qs[i],
                    sctimes[i],
                    config_id=config_id,
                )
                table64 = res_in_tab.loc[res_in_tab["TS"].nlargest(64).index]
                dfis = [table64]
                in_fov_results = pd.concat(dfis, axis=0, ignore_index=True)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_INFOV",
                    result_data=in_fov_results,
                )
                print(result)
            except Exception as e:
                print("Failed to ingest IFOV results :(")
                print(e)
                failed[path] = "Failed to ingest IFOV results"

            # Max TS results
            try:
                out_fov_results = res_out_tab
                in_fov_results = res_in_tab
                dfs = [
                    get_dlogls_inout(
                        in_fov_results,
                        out_fov_results,
                        trig_ids[i],
                        config_id=config_id,
                    )
                ]
                top_results = pd.concat(dfs, axis=0, ignore_index=True)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_TOP",
                    result_data=top_results,
                )
                print(result)
            except Exception as e:
                print("Failed to ingest top results :(")
                print(e)
                failed[path] = "Failed to ingest top results."

            # ProbSkymap
            try:
                fname = [fname for fname in os.listdir(paths[i]) if 'moc_prob_map.fits' in fname][0]
                skymap_fname = os.path.join(paths[i], fname)
                result = api.post_nitrates_results(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_PROBMAP",
                    result_data=skymap_fname,
                )
                print(result)
            except Exception as e:
                print("Failed to ingest prob skymap :(")
                print(e)
                failed[path] = "Failed to ingest prob skymap."

            # ProbSkymap Figures
            try:
                moll_fname = os.path.join(paths[i],[fname for fname in os.listdir(paths[i]) if 'mollview_plot.png' in fname][0])
                zoom_fname = os.path.join(paths[i],[fname for fname in os.listdir(paths[i]) if 'zoom_plot.png' in fname][0])
                plot_data = {'moll_fname':moll_fname, 'zoom_fname':zoom_fname}
                result = api.post_nitrates_plot(
                    trigger=trig_ids[i],
                    config_id=config_id,
                    result_type="n_PROBMAP",
                    plot_data=plot_fname,
                )
                print(result)
            except Exception as e:
                print("Failed to ingest prob skymap plots :(")
                print(e)
                failed[path] = "Failed to ingest prob skymap plots."


            print("Successfully loaded into database.")
        except Exception as e:
            print("Failed for untested reason")
            failed[path] = "Failed for untested reason, likely integrity error"

    if len(failed) > 0:
        print("Failed to ingest the following triggers:")
    if len(failed) < 20:
        for i in failed:
            print(str(i) + ":       " + str(failed[i]))
    else:
        print(f"{len(failed)} failures, here are first 10:")
        for i in list(failed.keys())[0:10]:
            print(str(i) + ":       " + str(failed[i]))

    with open("failures.json", "a") as fob:
        json.dump(failed, fob)


if __name__ == "__main__":

    args = cli()
        
    read_results_dirs(args.work_dir, api_token=args.api_token, ismpi4py=args.mpi4py)
