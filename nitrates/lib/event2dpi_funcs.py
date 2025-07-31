import numpy as np


def det2dpi(tab, weights=None):
    xbins = np.arange(286 + 1) - 0.5
    ybins = np.arange(173 + 1) - 0.5

    dpi = np.swapaxes(
        np.histogram2d(tab["DETX"], tab["DETY"], bins=[xbins, ybins], weights=weights)[
            0
        ],
        0,
        1,
    )

    return dpi


# def det2dpis(tab, ebins0, ebins1, bl_dmask=None):
#
#     dpis = []
#     for i in xrange(len(ebins0)):
#
#         bl = (tab['ENERGY']>=ebins0[i])&(tab['ENERGY']<ebins1[i])
#
#         if bl_dmask is None:
#             dpis.append(det2dpi(tab[bl]))
#         else:
#             dpis.append(det2dpi(tab[bl])[bl_dmask])
#
#     return dpis


def det2dpis(tab, ebins0, ebins1, bl_dmask=None):
    xbins = np.arange(286 + 1) - 0.5
    ybins = np.arange(173 + 1) - 0.5
    ebins = np.append(ebins0, [ebins1[-1]])

    dpis = np.histogramdd(
        [tab["ENERGY"], tab["DETY"], tab["DETX"]], bins=[ebins, ybins, xbins]
    )[0]

    if bl_dmask is None:
        return dpis

    return dpis[:, bl_dmask]


def det2dpis_tbins(tab, ebins0, ebins1, tbins0, tbins1, bl_dmask=None):
    dpi_list = []
    ntbins = len(tbins0)
    for ii in range(ntbins):
        blt = (tab["TIME"] >= tbins0[ii]) & (tab["TIME"] < tbins1[ii])
        dpis = []
        for i in range(len(ebins0)):
            bl = (tab["ENERGY"] >= ebins0[i]) & (tab["ENERGY"] < ebins1[i]) & blt

            if bl_dmask is None:
                dpis.append(det2dpi(tab[bl]))
            else:
                dpis.append(det2dpi(tab[bl])[bl_dmask])
        dpi_list.append(np.array(dpis))
    return dpi_list


def mask_detxy(dmask, tab):
    mask_vals = dmask[tab["DETY"], tab["DETX"]]
    return mask_vals


def filter_evdata(evdata, dmask, emin, emax, tmin, tmax):
    if dmask is not None:
        mask_vals = mask_detxy(dmask, evdata)
        bl_mask = mask_vals == 0
    else:
        bl_mask = np.ones(len(evdata["TIME"]), dtype=bool)
    bl_ev = (
        (evdata["TIME"] >= tmin)
        & (evdata["TIME"] < tmax)
        & (evdata["EVENT_FLAGS"] < 1)
        & (evdata["ENERGY"] <= emax)
        & (evdata["ENERGY"] >= emin)
        & (bl_mask)
    )
    return evdata[bl_ev]
