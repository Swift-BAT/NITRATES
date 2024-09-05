from astropy.table import Table, vstack
import numpy as np
import logging


def union_gtis(gti_tabs):
    """Union of overlapping time intervals.

    Returns a new `~gammapy.data.GTI` object.

    Intervals that touch will be merged, e.g.
    ``(1, 2)`` and ``(2, 3)`` will result in ``(1, 3)``.
    """
    # Algorithm to merge overlapping intervals is well-known,
    # see e.g. https://stackoverflow.com/a/43600953/498873

    table = vstack(gti_tabs)

    table.sort("START")

    # We use Python dict instead of astropy.table.Row objects,
    # because on some versions modifying Row entries doesn't behave as expected
    merged = [{"START": table[0]["START"], "STOP": table[0]["STOP"]}]
    for row in table[1:]:
        interval = {"START": row["START"], "STOP": row["STOP"]}
        if merged[-1]["STOP"] <= interval["START"]:
            merged.append(interval)
        else:
            merged[-1]["STOP"] = max(interval["STOP"], merged[-1]["STOP"])

    merged = Table(rows=merged, names=["START", "STOP"], meta=table.meta)
    return merged


def gti2bti(gti):
    starts = [-np.inf]
    stops = [np.min(gti["START"])]

    for i in range(len(gti)):
        starts.append(gti["STOP"][i])
        if i + 1 >= len(gti):
            stops.append(np.inf)
        else:
            stops.append(gti["START"][i + 1])

    bti = Table({"START": starts, "STOP": stops})

    return bti


def bti2gti(bti):
    starts = []
    stops = []

    for i in range(1, len(bti)):
        starts.append(bti["STOP"][i - 1])
        stops.append(bti["START"][i])

    gti = Table({"START": starts, "STOP": stops})

    return gti


def add_bti2gti(bti, gti):
    if type(bti) is tuple:
        bti = Table({"START": [bti[0]], "STOP": [bti[1]]})
    bti_orig = gti2bti(gti)
    bti_merg = union_gtis([bti_orig, bti])
    gti_new = bti2gti(bti_merg)
    return gti_new


def check_if_in_GTI(GTI, t0, t1):
    for row in GTI:
        if t0 >= row["START"] and t1 <= row["STOP"]:
            return True
    return False


def flags2gti(times, flags):
    # make it seem that it's changing to bad at the end
    Ntimes = len(times)
    diffs = np.append(np.diff(flags.astype(np.int64)), [-1])
    if not flags[-1]:
        diffs[-1] = 0

    chngs = times[(np.abs(diffs) > 0)]
    Nchngs = len(chngs)

    starts = [np.min(times[flags])]
    start_ind = np.argmin(np.abs(times - starts[0]))

    chngs2good_bl = diffs > 0.1
    Nchngs2good = np.sum(chngs2good_bl)
    chngs2good = times[chngs2good_bl]
    if start_ind == 0:
        chngs2good = np.append(times[0], chngs2good)
    print(Nchngs2good)

    chngs2bad_bl = diffs < -0.1
    Nchngs2bad = np.sum(chngs2bad_bl)
    chngs2bad = times[chngs2bad_bl]
    print(Nchngs2bad)

    #     if Nchngs2bad > 0:
    stops = [chngs2bad[0]]
    first_bad_ind = np.argmin(np.abs(times - stops[0]))

    for i in range(1, Nchngs2bad):
        starts.append(chngs2good[i])
        stops.append(chngs2bad[i])

    GTI = Table(data=(starts, stops), names=("START", "STOP"))

    return GTI


def mk_gti_bl(times, GTI, time_pad=0.0):
    bls = []
    for row in GTI:
        bls.append(
            (times >= (row["START"] - time_pad)) & (times < (row["STOP"] + time_pad))
        )
    bl = bls[0]
    for bl_ in bls[1:]:
        bl = bl | bl_
    return bl


def get_btis_for_glitches(evdata, tstart, tstop, tbin_size=16e-3):
    bins = np.arange(tstart, tstop + tbin_size / 2.0, tbin_size)
    ebl = evdata["ENERGY"] <= 25.0
    ebl2 = evdata["ENERGY"] > 50.0

    h = np.histogram(evdata["TIME"][ebl], bins=bins)[0]
    h2 = np.histogram(evdata["TIME"][ebl2], bins=bins)[0]
    stds = (h - np.mean(h)) / np.std(h)
    stds2 = (h2 - np.mean(h2)) / np.std(h2)

    # bl_bad = (stds>10.0)&(stds2<2.5)
    bl_lowE_highSNR = stds > 10.0
    # bl_highE_lowSNR = (stds2<2.5)|((stds/stds2)>5)
    bl_highE_lowSNR = (stds / np.abs(stds2)) > 3
    logging.debug("N_lowE_highSNR: " + str(np.sum(bl_lowE_highSNR)))
    if np.sum(bl_lowE_highSNR) > 0:
        logging.debug("LowE highSNRs: ")
        logging.debug(stds[bl_lowE_highSNR])
        logging.debug("HighE SNRs at lowE highSNRs: ")
        logging.debug(stds2[bl_lowE_highSNR])
    bl_bad = bl_highE_lowSNR & bl_lowE_highSNR
    bad_twinds = []
    Nbad = np.sum(bl_bad)
    logging.debug("Nbad: " + str(Nbad))
    for i in range(np.sum(bl_bad)):
        t0 = bins[:-1][bl_bad][i]
        t1 = t0 + tbin_size

        bad_twind = (
            bins[:-1][bl_bad][i] - tbin_size / 2.0,
            bins[:-1][bl_bad][i] + 1.5 * tbin_size,
        )
        bad_twinds.append(bad_twind)

    return bad_twinds


def find_cr_glitch_times(ev_data, tmin, tmax, tbin_size=5e-5, emin=50, max_cnts=20):
    bad_times = []

    t0 = tmin
    t1 = tmax

    tbins0 = np.arange(t0, t1 - 1.0, 20.0)
    tbins1 = tbins0 + 20.0

    Ntbins = len(tbins0)
    ebl = ev_data["ENERGY"] >= emin

    for i in range(Ntbins):
        logging.debug("t0: %.3f" % (tbins0[i]))

        tbins = np.arange(tbins0[i], tbins1[i] + tbin_size, tbin_size)

        h = np.histogram(ev_data["TIME"][ebl], tbins)[0]

        logging.debug("max(h): %d" % (np.max(h)))

        bl = h > max_cnts

        logging.debug("sum(h>max_cnts): %d" % (np.sum(bl)))

        if np.sum(bl) > 0:
            times = tbins[:-1][bl]

            for j, time in enumerate(times):
                cnts = h[bl][j]
                logging.debug("cnts: %d" % (cnts))
                tbins_ = np.arange(time - 0.5, time + 0.5, tbin_size)
                h0 = np.histogram(ev_data["TIME"][ebl], tbins_)[0]
                avg = np.mean(h0)
                logging.debug("avg: %.2f" % (avg))
                if cnts > (avg * 10):
                    logging.debug("found bad time: %.4f" % (time))
                    bad_times.append(time)

    return bad_times


def find_and_remove_cr_glitches(ev_data, GTI, tbin_size=5e-5):
    bad_times = []

    for row in GTI:
        bad_times += find_cr_glitch_times(
            ev_data, row["START"], row["STOP"], tbin_size=tbin_size
        )

    logging.debug("Got all bad times")
    logging.debug(bad_times)

    bad_time_bl = np.zeros(len(ev_data), dtype=bool)

    for bad_time in bad_times:
        tmid = bad_time + tbin_size / 2.0
        t0 = tmid - tbin_size
        t1 = tmid + tbin_size
        bl = (ev_data["TIME"] >= t0) & (ev_data["TIME"] <= t1)
        logging.debug("t0, t1: %.4f, %.4f" % (t0, t1))
        logging.debug("sum(bl): %d" % (np.sum(bl)))

        bad_time_bl[bl] = True

    bl = ~bad_time_bl

    ev_data = ev_data[bl]

    return ev_data
