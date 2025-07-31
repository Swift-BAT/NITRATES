import numpy as np
from ..lib.event2dpi_funcs import det2dpis, mask_detxy


def dpi2cnts_perquad(dpi):
    x_mid = 142
    y_mid = 86

    quads = []
    quads.append(np.sum(dpi[:y_mid, :x_mid]))
    quads.append(np.sum(dpi[y_mid:, :x_mid]))
    quads.append(np.sum(dpi[y_mid:, x_mid:]))
    quads.append(np.sum(dpi[:y_mid, x_mid:]))
    return quads


def ev2quad_cnts(ev):
    x_mid = 142
    y_mid = 86

    quads = [np.sum((ev["DETX"] < x_mid) & (ev["DETY"] < y_mid))]
    quads.append(np.sum((ev["DETX"] < x_mid) & (ev["DETY"] > y_mid)))
    quads.append(np.sum((ev["DETX"] > x_mid) & (ev["DETY"] > y_mid)))
    quads.append(np.sum((ev["DETX"] > x_mid) & (ev["DETY"] < y_mid)))
    return np.array(quads)


def ev2quad_ids(ev):
    x_mid = 142
    y_mid = 86

    quad_ids = -1 * np.ones(len(ev), dtype=np.int64)

    bl0 = (ev["DETX"] < x_mid) & (ev["DETY"] < y_mid)
    quad_ids[bl0] = 0
    bl1 = (ev["DETX"] < x_mid) & (ev["DETY"] > y_mid)
    quad_ids[bl1] = 1
    bl2 = (ev["DETX"] > x_mid) & (ev["DETY"] > y_mid)
    quad_ids[bl2] = 2
    bl3 = (ev["DETX"] > x_mid) & (ev["DETY"] < y_mid)
    quad_ids[bl3] = 3
    return quad_ids


def dmask2ndets_perquad(dmask):
    quads = dpi2cnts_perquad((dmask == 0).reshape(dmask.shape))

    return quads


def quads2drm_imxy():
    # bottom left, top left, top right, bottom right
    quads_imxy = [(1.0, 0.5), (0.8, -0.4), (-0.75, -0.45), (-1.1, 0.5)]

    return quads_imxy


def halves2drm_imxy():
    # left, top, right, bottom
    halves_imxy = [(1.0, 0.15), (0.0, -0.5), (-1.0, 0.15), (0.0, 0.45)]

    return halves_imxy


def get_cnts_per_tbins(t_bins0, t_bins1, ebins0, ebins1, ev_data, dmask):
    ntbins = len(t_bins0)
    nebins = len(ebins0)
    cnts_per_tbin = np.zeros((ntbins, nebins))

    for i in range(ntbins):
        sig_bl = (ev_data["TIME"] >= t_bins0[i]) & (ev_data["TIME"] < (t_bins1[i]))
        sig_data = ev_data[sig_bl]

        sig_data_dpis = det2dpis(sig_data, ebins0, ebins1)
        if dmask is None:
            cnts_per_tbin[i] = np.array([np.sum(dpi) for dpi in sig_data_dpis])
        else:
            cnts_per_tbin[i] = np.array(
                [np.sum(dpi[(dmask == 0)]) for dpi in sig_data_dpis]
            )

    return cnts_per_tbin


def get_quad_cnts_tbins(tbins0, tbins1, ebins0, ebins1, evd):
    ntbins = len(tbins0)
    nebins = len(ebins0)

    cnts_mat = np.zeros((ntbins, nebins, 4))

    for i in range(ntbins):
        sig_bl = (evd["TIME"] >= tbins0[i]) & (evd["TIME"] < (tbins1[i]))
        sig_data = evd[sig_bl]

        for j in range(nebins):
            e_bl = (sig_data["ENERGY"] >= ebins0[j]) & (
                sig_data["ENERGY"] < (ebins1[j])
            )

            cnts_mat[i, j] = ev2quad_cnts(sig_data[e_bl])

    return cnts_mat


def get_quad_cnts_tbins_fast(tbins0, tbins1, ebins0, ebins1, evd):
    ntbins = len(tbins0)
    nebins = len(ebins0)
    quadIDs = ev2quad_ids(evd)
    tstep = tbins0[1] - tbins0[0]
    tbin_size = tbins1[0] - tbins0[0]
    tfreq = int(np.rint(tbin_size / tstep))
    t_add = [tbins0[-1] + (i + 1) * tstep for i in range(tfreq)]
    tbins = np.append(tbins0, t_add)
    ebins = np.append(ebins0, [ebins1[-1]])
    qbins = np.arange(5) - 0.5

    h = np.histogramdd(
        [evd["TIME"], evd["ENERGY"], quadIDs], bins=[tbins, ebins, qbins]
    )[0]

    if tfreq <= 1:
        return h
    h2 = np.zeros((h.shape[0] - (tfreq - 1), h.shape[1], h.shape[2]))
    for i in range(tfreq):
        i0 = i
        i1 = -tfreq + 1 + i
        if i1 < 0:
            h2 += h[i0:i1]
        else:
            h2 += h[i0:]
    return h2

    cnts_mat = np.zeros((ntbins, nebins, 4))

    for i in range(ntbins):
        sig_bl = (evd["TIME"] >= tbins0[i]) & (evd["TIME"] < (tbins1[i]))
        sig_data = evd[sig_bl]

        for j in range(nebins):
            e_bl = (sig_data["ENERGY"] >= ebins0[j]) & (
                sig_data["ENERGY"] < (ebins1[j])
            )

            cnts_mat[i, j] = ev2quad_cnts(sig_data[e_bl])

    return cnts_mat
