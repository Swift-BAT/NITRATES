import numpy as np
from scipy import stats
import logging

from ..lib.event2dpi_funcs import det2dpi
from ..lib.gti_funcs import check_if_in_GTI


def combine_detmasks(detmask_list):
    dmask = np.ones(detmask_list[0].shape)
    bl = np.ones(detmask_list[0].shape, dtype=bool)

    for detmask in detmask_list:
        bl = bl & (detmask == 0)
    dmask[bl] = 0.0

    return dmask


def get_hotpix_map(ev_data, bl_dmask, mask_zeros=True):
    dpi = det2dpi(ev_data)
    cnts = dpi[bl_dmask]
    ndets = np.sum(bl_dmask)
    if mask_zeros and (np.median(cnts) > 20):
        cnts = cnts[(cnts > 0)]
    perc01, perc99 = np.percentile(cnts, [1.0, 99.0])
    # lower_limit = perc01 - .25*np.median(cnts) - 1.
    lower_limit = stats.poisson.ppf(0.1 / ndets, 0.85 * np.median(cnts)) - 1.0
    if mask_zeros and lower_limit < 1:
        lower_limit = 0.5
    upper_limit = perc99 + 0.25 * np.median(cnts) + 5.0
    hotpix_map = np.ones(dpi.shape)
    hotpix_bl = (dpi > lower_limit) & (dpi < upper_limit)
    hotpix_map[hotpix_bl] = 0
    return hotpix_map


def find_rate_spike_dets2mask(evdata, GTI=None, emax=50, tbin_size=20e-3, max_cnts=10):
    tmin = np.min(evdata["TIME"])
    tmax = np.max(evdata["TIME"])
    tot_exp = tmax - tmin

    tstep = 16e0

    tsteps = 1 + int(tot_exp / tstep)
    t0 = tmin

    detx_bins = np.arange(np.max(evdata["DETX"]) + 1) - 0.5
    dety_bins = np.arange(np.max(evdata["DETY"]) + 1) - 0.5

    bad_times = []
    bad_dets = []
    ebl = evdata["ENERGY"] < emax

    for i in range(tsteps):
        t1 = t0 + tstep
        tbins = np.arange(t0, t1 + tbin_size / 2.0, tbin_size)
        h = np.histogramdd(
            [evdata["TIME"][ebl], evdata["DETX"][ebl], evdata["DETY"][ebl]],
            bins=[tbins, detx_bins, dety_bins],
        )[0]
        # print len(h)
        print(np.shape(h))
        print(np.max(h), np.mean(h))
        # print tbins[np.argmax(h)]
        max_inds = np.unravel_index(np.argmax(h), h.shape)
        print(max_inds)
        print(tbins[max_inds[0]], detx_bins[max_inds[1]], dety_bins[max_inds[2]])
        print(np.mean(h[max_inds[0]]), np.sum(h[max_inds[0]]))
        print()
        if np.max(h) > max_cnts:
            bl = h > max_cnts
            tbin_inds, detx_inds, dety_inds = np.where(bl)
            for tbin_ind, detx, dety in zip(tbin_inds, detx_inds, dety_inds):
                if not GTI is None:
                    t_start = tbins[:-1][tbin_ind]
                    if not check_if_in_GTI(GTI, t_start, t_start + tbin_size):
                        continue
                bad_times.append(tbins[:-1][tbin_ind])
                bad_dets.append((dety, detx))
        t0 = t1
    logging.debug("Bad Times: ")
    logging.debug(bad_times)

    return bad_dets
