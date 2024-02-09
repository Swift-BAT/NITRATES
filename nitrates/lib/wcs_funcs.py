import numpy as np
from astropy.wcs import WCS


def world2val(w, img, imxs, imys):
    #     pnts = np.vstack([imxs, imys]).T
    xinds, yinds = w.wcs_world2pix(imxs, imys, 0)
    int0_xinds = np.floor(xinds).astype(np.int64)
    int0_yinds = np.floor(yinds).astype(np.int64)
    if np.any(int0_xinds <= 0):
        int0_xinds[(int0_xinds <= 0)] = 0
    if np.any(int0_yinds <= 0):
        int0_yinds[(int0_yinds <= 0)] = 0
    if np.any(int0_xinds >= (img.shape[1] - 2)):
        int0_xinds[(int0_xinds >= (img.shape[1] - 2))] = img.shape[1] - 2
    if np.any(int0_yinds >= (img.shape[0] - 2)):
        int0_yinds[int0_yinds >= (img.shape[0] - 2)] = img.shape[0] - 2
    int1_xinds = int0_xinds + 1
    int1_yinds = int0_yinds + 1

    dx1inds = xinds - int0_xinds
    dy1inds = yinds - int0_yinds
    dx0inds = 1.0 - dx1inds
    dy0inds = 1.0 - dy1inds

    vals = (
        dx0inds * dy0inds * img[int0_yinds, int0_xinds]
        + dx1inds * dy0inds * img[int0_yinds, int1_xinds]
        + dx1inds * dy1inds * img[int1_yinds, int1_xinds]
        + dx0inds * dy1inds * img[int1_yinds, int0_xinds]
    )

    return vals
