import numpy as np
from astropy.io import fits
from scipy import stats
import os
from scipy import interpolate, optimize
import logging, traceback
import time
import gc


def get_rt_arr(rt_dir, ident="fwd_ray_trace"):
    ray_trace_fnames = np.array([fn for fn in os.listdir(rt_dir) if ident in fn])

    # code to filterout weird "._*" filenames too
    ray_trace_fnames = np.array([i for i in ray_trace_fnames if "._" + ident not in i])

    i0 = len(ident.split("_"))

    imx0s = np.array([float(fn.split("_")[i0]) for fn in ray_trace_fnames])
    imy0s = np.array([float(fn.split("_")[i0 + 1]) for fn in ray_trace_fnames])
    imx1s = np.array([float(fn.split("_")[i0 + 2]) for fn in ray_trace_fnames])
    imy1s = np.array([float(fn.split("_")[i0 + 3]) for fn in ray_trace_fnames])

    dtp = [
        ("imx0", np.float64),
        ("imy0", np.float64),
        ("imx1", np.float64),
        ("imy1", np.float64),
        ("fname", ray_trace_fnames.dtype),
        ("time", np.float64),
    ]
    ray_trace_arr = np.empty(len(imx0s), dtype=dtp)
    ray_trace_arr["imx0"] = imx0s
    ray_trace_arr["imy0"] = imy0s
    ray_trace_arr["imx1"] = imx1s
    ray_trace_arr["imy1"] = imy1s
    ray_trace_arr["time"][:] = np.nan
    ray_trace_arr["fname"] = ray_trace_fnames

    return ray_trace_arr


def intp_1d(x, vals, xs):
    m = (vals[1] - vals[0]) / (xs[1] - xs[0])

    y = m * (x - xs[0]) + vals[0]

    return y


def intp_corners2d_derivs(dxs, dys, vals, dx=0.002):
    dy = dx

    A = 1.0 / (dx * dy)
    dxs_dx = np.array([[[[-1.0, 1.0]]]])
    dys_dy = np.array([[[[-1.0], [1.0]]]])

    dz_dx = A * dxs_dx.T * vals * (dys.T)
    dz_dy = A * dxs.T * vals * (dys_dy.T)

    #     z = vals[0]*(xs[1]-x)*(ys[1]-y) +\
    #         vals[1]*(x-xs[0])*(ys[1]-y) +\
    #         vals[2]*(xs[1]-x)*(y-ys[0]) +\
    #         vals[3]*(x-xs[0])*(y-ys[0])

    return [A * np.sum(dz_dx, axis=(0, 1)), A * np.sum(dz_dy, axis=(0, 1))]


def get_intp_corners_coefs(vals, xs, ys, dx=2e-3):
    dy = dx

    A = 1.0 / (dx * dy)

    a0 = A * np.sum(
        vals
        * np.array(
            [[[[xs[1] * ys[1], -xs[0] * ys[1]], [-xs[1] * ys[0], xs[0] * ys[0]]]]]
        ).T,
        axis=(0, 1),
    )
    a1 = A * np.sum(
        vals * np.array([[[[-ys[1], ys[1]], [ys[0], -ys[0]]]]]).T, axis=(0, 1)
    )
    a2 = A * np.sum(
        vals * np.array([[[[-xs[1], xs[0]], [xs[1], -xs[0]]]]]).T, axis=(0, 1)
    )
    a3 = A * np.sum(vals * np.array([[[[1, -1], [-1, 1]]]]).T, axis=(0, 1))

    return a0, a1, a2, a3


def intp_corners2d3(x, y, a0, a1, a2, a3):
    z = a0 + a1 * x + a2 * y + a3 * x * y

    return z


def intp_corners2d_deriv2(x, y, a0, a1, a2, a3):
    dz_dx = a1 + a3 * y
    dz_dy = a2 + a3 * x

    return dz_dx, dz_dy


def intp_corners2d2(dxs, dys, vals, dx=0.002):
    dy = dx

    A = 1.0 / (dx * dy)

    # z0 = dxs*vals#*(dys.T)
    z0 = dxs.T * vals
    z = z0 * (dys.T)

    #     z = vals[0]*(xs[1]-x)*(ys[1]-y) +\
    #         vals[1]*(x-xs[0])*(ys[1]-y) +\
    #         vals[2]*(xs[1]-x)*(y-ys[0]) +\
    #         vals[3]*(x-xs[0])*(y-ys[0])

    return A * np.sum(z, axis=(0, 1))


def intp_corners2d(x, y, vals, xs, ys):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    A = 1.0 / (dx * dy)

    z = (
        vals[0] * (xs[1] - x) * (ys[1] - y)
        + vals[1] * (x - xs[0]) * (ys[1] - y)
        + vals[2] * (xs[1] - x) * (y - ys[0])
        + vals[3] * (x - xs[0]) * (y - ys[0])
    )

    return A * z


class ray_trace_file(object):
    # maybe add the update im_dist thing here too
    # and save the last ray trace made

    def __init__(self, rt_arr0, rt_dir, rng=0.02, rtstep=0.002, mmap=False):
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        imx_ax = (imxs[1:] + imxs[:-1]) / 2.0
        imy_ax = (imys[1:] + imys[:-1]) / 2.0
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.nbytes = 0.0

        self.rng = rng
        self.rtstep = rtstep
        self.imx0 = rt_arr0["imx0"]
        self.imy0 = rt_arr0["imy0"]
        self.imx1 = rt_arr0["imx1"]
        self.imy1 = rt_arr0["imy1"]
        self.fname = os.path.join(rt_dir, rt_arr0["fname"])
        logging.debug("opening file: " + rt_arr0["fname"])
        try:
            self.file = fits.open(self.fname, memmap=mmap, ignore_missing_end=True)
        except Exception as E:
            logging.warning("couldn't open file " + self.fname)
            logging.error(traceback.format_exc())
            self.file.close()
            # raise E
        try:
            self.datas = []
            for i, f in enumerate(self.file):
                self.datas.append(np.copy(f.data))
                self.nbytes += self.datas[i].nbytes
        except Exception as E:
            logging.warning(
                "trouble getting data from file " + self.fname + " on index " + str(i)
            )
            logging.error(traceback.format_exc())
            self.file.close()
            return False

            # raise E
        self.file.close()
        # self.data = self.file.

        self.imxs = _imxs_rt_file + self.imx0
        self.imys = _imys_rt_file + self.imy0
        self.grid_shape = grids[0].shape
        self.imx_ax = imx_ax + self.imx0
        self.imy_ax = imy_ax + self.imy0
        self.setup_intp()
        self.nbytes *= 4
        self.last_used = time.time()

    def get_rt_from_ind(self, ind):
        # return self.file[ind].data
        return self.datas[ind]

    def get_rt(self, imx, imy):
        ind = np.argmin(np.hypot(imx - self.imxs, imy - self.imys))
        # return self.file[ind].data
        return self.datas[ind]

    def get_rt_corners2(self, imx, imy):
        x0_ind = np.argmin(np.abs(imx - self.imx_ax))
        x1_ind = x0_ind + 1
        y0_ind = np.argmin(np.abs(imy - self.imy_ax))
        y1_ind = y0_ind + 1
        ravel_ind00 = np.ravel_multi_index((x0_ind, y0_ind), self.grid_shape)
        ravel_ind10 = np.ravel_multi_index((x1_ind, y0_ind), self.grid_shape)
        ravel_ind01 = np.ravel_multi_index((x0_ind, y1_ind), self.grid_shape)
        ravel_ind11 = np.ravel_multi_index((x1_ind, y1_ind), self.grid_shape)

        inds = [ravel_ind00, ravel_ind10, ravel_ind01, ravel_ind11]

        ray_traces = [
            [self.get_rt_from_ind(inds[0]), self.get_rt_from_ind(inds[2])],
            [self.get_rt_from_ind(inds[1]), self.get_rt_from_ind(inds[3])],
        ]
        imxs = []
        imys = []

        for ind in inds:
            #             ray_traces.append(self.get_rt_from_ind(ind))
            imxs.append(self.imxs[ind])
            imys.append(self.imys[ind])

        return ray_traces, imxs, imys

    def get_rt_corners(self, imx, imy):
        # 00 lower left
        # 10 lower right
        # 01 upper left
        # 11 upper right

        bl_0_ = (np.abs(imx - self.imxs) <= (1.0 * self.rtstep)) & (
            (imx - self.imxs) >= 0
        )
        bl__0 = (np.abs(imy - self.imys) <= (1.0 * self.rtstep)) & (
            (imy - self.imys) >= 0
        )
        bl_1_ = (np.abs(imx - self.imxs) <= (1.0 * self.rtstep)) & (
            (imx - self.imxs) < 0
        )
        bl__1 = (np.abs(imy - self.imys) <= (1.0 * self.rtstep)) & (
            (imy - self.imys) < 0
        )

        for bl in [bl_0_, bl__0, bl_1_, bl__1]:
            if np.sum(bl) < 1:
                logging.warning(
                    "couldn't find imx: %.3f, imy: %.3f in file" % (imx, imy)
                )
                return None, None, None

        try:
            ind_00 = np.where(bl_0_ & bl__0)[0][0]
            ind_10 = np.where(bl_1_ & bl__0)[0][0]
            ind_01 = np.where(bl_0_ & bl__1)[0][0]
            ind_11 = np.where(bl_1_ & bl__1)[0][0]
        except:
            logging.warning("couldn't find imx: %.3f, imy: %.3f in file" % (imx, imy))
            logging.error(traceback.format_exc())
            return None, None, None

        inds = [ind_00, ind_10, ind_01, ind_11]

        ray_traces = []
        imxs = []
        imys = []

        for ind in inds:
            ray_traces.append(self.get_rt_from_ind(ind))
            imxs.append(self.imxs[ind])
            imys.append(self.imys[ind])

        return ray_traces, imxs, imys

    def get_closest_ind(self, imx, imy):
        return np.argmin(np.hypot(imx - self.imxs, imy - self.imys))

    def get_closest_ax_inds(self, imx, imy):
        xind = np.argmin(np.abs(imx - self.imx_ax))
        yind = np.argmin(np.abs(imy - self.imy_ax))
        return xind, yind

    def setup_intp(self):
        self.a0_grid = np.zeros((len(self.imx_ax), len(self.imy_ax), 173, 286))
        self.a1_grid = np.zeros_like(self.a0_grid)
        self.a2_grid = np.zeros_like(self.a0_grid)
        self.a3_grid = np.zeros_like(self.a0_grid)

        for i in range(len(self.imx_ax)):
            imx = self.imx_ax[i]
            for j in range(len(self.imy_ax)):
                imy = self.imy_ax[j]
                rts, imxs, imys = self.get_rt_corners2(imx, imy)
                xs = [min(imxs), max(imxs)]
                ys = [min(imys), max(imys)]
                a0, a1, a2, a3 = get_intp_corners_coefs(np.array(rts), xs, ys)
                self.a0_grid[i, j] = a0
                self.a1_grid[i, j] = a1
                self.a2_grid[i, j] = a2
                self.a3_grid[i, j] = a3

    def intp_ray_traces(self, imx, imy, min_diff=5e-6, get_deriv=False):
        self.last_used = time.time()

        xind, yind = self.get_closest_ax_inds(imx, imy)
        a0 = self.a0_grid[xind, yind]
        a1 = self.a1_grid[xind, yind]
        a2 = self.a2_grid[xind, yind]
        a3 = self.a3_grid[xind, yind]

        rt = intp_corners2d3(imx, imy, a0, a1, a2, a3)
        rt = np.clip(rt, 0.0, None)

        if get_deriv:
            drt_dx, drt_dy = intp_corners2d_deriv2(imx, imy, a0, a1, a2, a3)
            return rt, drt_dx, drt_dy
        else:
            return rt


#     def intp_ray_traces(self, imx, imy, min_diff=5e-6, get_deriv=False):

#         self.last_used = time.time()

#         if np.min(np.abs(imx - self.imxs)) < min_diff:
#             if np.min(np.abs(imy - self.imys)) < min_diff:
#                 return self.get_rt(imx, imy)

#             ind0 = self.get_closest_ind(imx, imy)
#             ind1 = self.get_closest_ind(imx, imy - self.rtstep) if imy < self.imys[ind0] else\
#                     self.get_closest_ind(imx, imy + self.rtstep)
#             if ind0 == ind1:
#                 return self.get_rt(imx, imy)
#             inds = [ind0, ind1]
#             ys = []; rts = []
#             for ind in inds:
#                 ys.append(self.imys[ind])
#                 rts.append(self.get_rt_from_ind(ind))

#             return intp_1d(imy, rts, ys)


#         elif np.min(np.abs(imy - self.imys)) < min_diff:

#             ind0 = self.get_closest_ind(imx, imy)
#             ind1 = self.get_closest_ind(imx - self.rtstep, imy) if imx < self.imxs[ind0] else\
#                     self.get_closest_ind(imx + self.rtstep, imy)
#             if ind0 == ind1:
#                 return self.get_rt(imx, imy)
#             inds = [ind0, ind1]
#             xs = []; rts = []
#             for ind in inds:
#                 xs.append(self.imxs[ind])
#                 rts.append(self.get_rt_from_ind(ind))

#             return intp_1d(imx, rts, xs)

#         else:

#             try:
#                 rts, imxs, imys = self.get_rt_corners2(imx, imy)
#                 if rts is None:
#                     return None
#             except:
#                 logging.warning("couldn't find imx: %.3f, imy: %.3f in file"\
#                            %(imx, imy))
#                 logging.error(traceback.format_exc())
#                 return None

#             xs = [min(imxs),max(imxs)]
#             ys = [min(imys),max(imys)]

#             dxs = np.array([[[[max(imxs) - imx, imx - min(imxs)]]]])
#             dys = np.array([[[[max(imys) - imy], [imy - min(imys)]]]])
#             print "dxs: ", dxs
#             print "dys: ", dys

#             a0, a1, a2, a3 = get_intp_corners_coefs(np.array(rts), xs, ys)

#             rt = intp_corners2d3(imx, imy, a0, a1, a2, a3)

#             if get_deriv:
#                 drt_dx, drt_dy = intp_corners2d_deriv2(imx, imy, a0, a1, a2, a3)
#                 return rt, drt_dx, drt_dy
#             else:
#                 return rt
# #             return intp_corners2d(imx, imy, rts, xs, ys)


class ray_trace_file_npy(object):
    # maybe add the update im_dist thing here too
    # and save the last ray trace made

    def __init__(self, rt_arr0, rt_dir, rng=0.02, rtstep=0.002, mmap=False):
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        imx_ax = (imxs[1:] + imxs[:-1]) / 2.0
        imy_ax = (imys[1:] + imys[:-1]) / 2.0
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.nbytes = 0.0

        self.rng = rng
        self.rtstep = rtstep
        self.imx0 = rt_arr0["imx0"]
        self.imy0 = rt_arr0["imy0"]
        self.imx1 = rt_arr0["imx1"]
        self.imy1 = rt_arr0["imy1"]
        self.fname = os.path.join(rt_dir, rt_arr0["fname"])
        logging.debug("opening file: " + rt_arr0["fname"])
        try:
            self.datas = np.load(self.fname)
        except Exception as E:
            logging.warning("couldn't open file " + self.fname)
            logging.error(traceback.format_exc())
            return False

        self.nbytes = self.datas.nbytes

        self.imxs = _imxs_rt_file + self.imx0
        self.imys = _imys_rt_file + self.imy0
        self.grid_shape = grids[0].shape
        self.imx_ax = imx_ax + self.imx0
        self.imy_ax = imy_ax + self.imy0
        self.setup_intp()
        self.nbytes *= 4
        self.last_used = time.time()

    def get_rt_from_ind(self, ind):
        # return self.file[ind].data
        return self.datas[ind]

    def get_rt(self, imx, imy):
        ind = np.argmin(np.hypot(imx - self.imxs, imy - self.imys))
        # return self.file[ind].data
        return self.datas[ind]

    def get_rt_corners(self, imx, imy):
        x0_ind = np.argmin(np.abs(imx - self.imx_ax))
        x1_ind = x0_ind + 1
        y0_ind = np.argmin(np.abs(imy - self.imy_ax))
        y1_ind = y0_ind + 1
        ravel_ind00 = np.ravel_multi_index((x0_ind, y0_ind), self.grid_shape)
        ravel_ind10 = np.ravel_multi_index((x1_ind, y0_ind), self.grid_shape)
        ravel_ind01 = np.ravel_multi_index((x0_ind, y1_ind), self.grid_shape)
        ravel_ind11 = np.ravel_multi_index((x1_ind, y1_ind), self.grid_shape)

        inds = [ravel_ind00, ravel_ind10, ravel_ind01, ravel_ind11]

        ray_traces = [
            [self.get_rt_from_ind(inds[0]), self.get_rt_from_ind(inds[2])],
            [self.get_rt_from_ind(inds[1]), self.get_rt_from_ind(inds[3])],
        ]
        imxs = []
        imys = []

        for ind in inds:
            #             ray_traces.append(self.get_rt_from_ind(ind))
            imxs.append(self.imxs[ind])
            imys.append(self.imys[ind])

        return ray_traces, imxs, imys

    def get_closest_ind(self, imx, imy):
        return np.argmin(np.hypot(imx - self.imxs, imy - self.imys))

    def get_closest_ax_inds(self, imx, imy):
        xind = np.argmin(np.abs(imx - self.imx_ax))
        yind = np.argmin(np.abs(imy - self.imy_ax))
        return xind, yind

    def setup_intp(self):
        self.a0_grid = np.zeros(
            (len(self.imx_ax), len(self.imy_ax), 173, 286), dtype=np.float32
        )
        self.a1_grid = np.zeros_like(self.a0_grid)
        self.a2_grid = np.zeros_like(self.a0_grid)
        self.a3_grid = np.zeros_like(self.a0_grid)

        for i in range(len(self.imx_ax)):
            imx = self.imx_ax[i]
            for j in range(len(self.imy_ax)):
                imy = self.imy_ax[j]
                rts, imxs, imys = self.get_rt_corners(imx, imy)
                xs = [min(imxs), max(imxs)]
                ys = [min(imys), max(imys)]
                a0, a1, a2, a3 = get_intp_corners_coefs(np.array(rts), xs, ys)
                self.a0_grid[i, j] = a0
                self.a1_grid[i, j] = a1
                self.a2_grid[i, j] = a2
                self.a3_grid[i, j] = a3

    def intp_ray_traces(self, imx, imy, min_diff=5e-6, get_deriv=False):
        self.last_used = time.time()

        xind, yind = self.get_closest_ax_inds(imx, imy)
        a0 = self.a0_grid[xind, yind]
        a1 = self.a1_grid[xind, yind]
        a2 = self.a2_grid[xind, yind]
        a3 = self.a3_grid[xind, yind]

        rt = intp_corners2d3(imx, imy, a0, a1, a2, a3)
        rt = np.clip(rt, 0.0, None)

        if get_deriv:
            drt_dx, drt_dy = intp_corners2d_deriv2(imx, imy, a0, a1, a2, a3)
            return rt, drt_dx, drt_dy
        else:
            return rt


class ray_trace_square(object):
    def __init__(
        self, imx0, imx1, imy0, imy1, rt_dir, rng=0.02, rtstep=0.002, mmap=False
    ):
        print(rng)
        print(rtstep)
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        ray_trace_arr = get_rt_arr(rt_dir)

        bl = (
            (ray_trace_arr["imx0"] < imx1)
            & (ray_trace_arr["imx1"] > imx0)
            & (ray_trace_arr["imy0"] < imy1)
            & (ray_trace_arr["imy1"] > imy0)
        )

        print((np.sum(bl)))

        self.rt_arr = ray_trace_arr[bl]

        Nfiles = len(self.rt_arr)

        self.rt_files = []

        for i in range(Nfiles):
            self.rt_files.append(ray_trace_file(self.rt_arr[i], rt_dir, mmap=mmap))

    def get_rt_file_obj(self, imx, imy):
        bl = (
            (imx >= self.rt_arr["imx0"])
            & (imx <= self.rt_arr["imx1"])
            & (imy >= self.rt_arr["imy0"])
            & (imy <= self.rt_arr["imy1"])
        )

        if np.sum(bl) < 1:
            logging.warning("No ray trace files for this imx, imy")

        ind = np.where(bl)[0][0]

        return self.rt_files[ind]

    def get_intp_rt(self, imx, imy):
        rt_file = self.get_rt_file_obj(imx, imy)

        return rt_file.intp_ray_traces(imx, imy)


class RayTraces(object):
    def __init__(
        self,
        rt_dir,
        rng=0.02,
        rtstep=0.002,
        mmap=False,
        im_dist_update=1e-6,
        max_nbytes=5e9,
        npy=True,
        ident="fwd_ray_trace",
    ):
        logging.info(rng)
        logging.info(rtstep)
        self.rng = rng
        self.rtstep = rtstep
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.rt_dir = rt_dir
        self.mmap = mmap
        self.rt_arr = get_rt_arr(rt_dir, ident=ident)
        self.nbytes = 0.0
        self.max_nbytes = max_nbytes
        self.npy = npy

        # use fnames as the dictionary keys
        self.rt_files = {}

        self._last_imx = 10.0
        self._last_imy = 10.0
        self._last_dimx = 10.0
        self._last_dimy = 10.0
        self._update_im_dist = im_dist_update

    def close_rt_file_obj(self, rt_arr_ind):
        k = self.rt_arr[rt_arr_ind]["fname"]
        logging.debug("nbytes=" + str(self.nbytes))
        self.nbytes -= self.rt_files[k].nbytes
        logging.debug("closing file " + k)
        del self.rt_files[k]
        self.rt_arr["time"][rt_arr_ind] = np.nan
        gc.collect()

    def mem_check(self):
        if self.nbytes > self.max_nbytes:
            ind2close = np.nanargmin(self.rt_arr["time"])
            self.close_rt_file_obj(ind2close)

    def open_rt_file_obj(self, rt_arr0):
        if self.npy:
            rt_file = ray_trace_file_npy(
                rt_arr0, self.rt_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        else:
            rt_file = ray_trace_file(
                rt_arr0, self.rt_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        self.nbytes += rt_file.nbytes
        logging.debug("nbytes_total=" + str(self.nbytes))
        self.mem_check()
        self.rt_files[rt_arr0["fname"]] = rt_file

    def get_rt_file_obj(self, imx, imy):
        ind = np.argmin(
            np.hypot(
                imx - (self.rt_arr["imx0"] + self.rt_arr["imx1"]) / 2.0,
                imy - (self.rt_arr["imy0"] + self.rt_arr["imy1"]) / 2.0,
            )
        )

        rt_arr0 = self.rt_arr[ind]

        if not (
            (rt_arr0["imx0"] <= imx <= rt_arr0["imx1"])
            and (rt_arr0["imy0"] <= imy <= rt_arr0["imy1"])
        ):
            logging.warning("No ray trace files for this imx, imy")

        fname = rt_arr0["fname"]

        if fname not in list(self.rt_files.keys()):
            self.open_rt_file_obj(rt_arr0)
        self.rt_arr["time"][ind] = time.time()
        return self.rt_files[fname]

    def check_im_dist(self, imx, imy, deriv=False):
        if deriv:
            return (
                np.hypot(imx - self._last_dimx, imy - self._last_dimy)
                < self._update_im_dist
            )
        return (
            np.hypot(imx - self._last_imx, imy - self._last_imy) < self._update_im_dist
        )

    def get_intp_rt(self, imx, imy, get_deriv=False):
        if get_deriv:
            if (not self.check_im_dist(imx, imy)) or (
                not self.check_im_dist(imx, imy, deriv=True)
            ):
                rt_file = self.get_rt_file_obj(imx, imy)
                (
                    self._last_ray_trace,
                    self._last_drdx,
                    self._last_drdy,
                ) = rt_file.intp_ray_traces(imx, imy, get_deriv=get_deriv)
                self._last_imx = imx
                self._last_imy = imy
                self._last_dimx = imx
                self._last_dimy = imy

            return self._last_ray_trace, self._last_drdx, self._last_drdy

        if not self.check_im_dist(imx, imy):
            rt_file = self.get_rt_file_obj(imx, imy)
            self._last_ray_trace = rt_file.intp_ray_traces(
                imx, imy, get_deriv=get_deriv
            )
            self._last_imx = imx
            self._last_imy = imy

        return self._last_ray_trace


class footprint_file_npy(object):
    # maybe add the update im_dist thing here too
    # and save the last ray trace made

    def __init__(self, fp_arr0, fp_dir, rng=0.02, rtstep=0.002, mmap=False):
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        imx_ax = (imxs[1:] + imxs[:-1]) / 2.0
        imy_ax = (imys[1:] + imys[:-1]) / 2.0
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.nbytes = 0.0

        self.rng = rng
        self.rtstep = rtstep
        self.imx0 = fp_arr0["imx0"]
        self.imy0 = fp_arr0["imy0"]
        self.imx1 = fp_arr0["imx1"]
        self.imy1 = fp_arr0["imy1"]
        self.fname = os.path.join(fp_dir, fp_arr0["fname"])
        logging.debug("opening file: " + fp_arr0["fname"])
        try:
            self.datas = np.load(self.fname)
        except Exception as E:
            logging.warning("couldn't open file " + self.fname)
            logging.error(traceback.format_exc())
            return False

        self.nbytes = self.datas.nbytes

        self.imxs = _imxs_rt_file + self.imx0
        self.imys = _imys_rt_file + self.imy0
        self.grid_shape = grids[0].shape
        self.imx_ax = imx_ax + self.imx0
        self.imy_ax = imy_ax + self.imy0
        #         self.setup_intp()
        #         self.nbytes *= 4
        self.last_used = time.time()

    def get_fp_from_ind(self, ind):
        # return self.file[ind].data
        return self.datas[ind]

    def get_fp(self, imx, imy):
        ind = np.argmin(np.hypot(imx - self.imxs, imy - self.imys))
        # return self.file[ind].data
        return self.datas[ind]

    def get_fp_corners(self, imx, imy):
        x0_ind = np.argmin(np.abs(imx - self.imx_ax))
        x1_ind = x0_ind + 1
        y0_ind = np.argmin(np.abs(imy - self.imy_ax))
        y1_ind = y0_ind + 1
        ravel_ind00 = np.ravel_multi_index((x0_ind, y0_ind), self.grid_shape)
        ravel_ind10 = np.ravel_multi_index((x1_ind, y0_ind), self.grid_shape)
        ravel_ind01 = np.ravel_multi_index((x0_ind, y1_ind), self.grid_shape)
        ravel_ind11 = np.ravel_multi_index((x1_ind, y1_ind), self.grid_shape)

        inds = [ravel_ind00, ravel_ind10, ravel_ind01, ravel_ind11]

        fps = [
            [self.get_fp_from_ind(inds[0]), self.get_fp_from_ind(inds[2])],
            [self.get_fp_from_ind(inds[1]), self.get_fp_from_ind(inds[3])],
        ]
        imxs = []
        imys = []

        for ind in inds:
            #             ray_traces.append(self.get_rt_from_ind(ind))
            imxs.append(self.imxs[ind])
            imys.append(self.imys[ind])

        return fps, imxs, imys

    def get_closest_ind(self, imx, imy):
        return np.argmin(np.hypot(imx - self.imxs, imy - self.imys))

    def get_closest_ax_inds(self, imx, imy):
        xind = np.argmin(np.abs(imx - self.imx_ax))
        yind = np.argmin(np.abs(imy - self.imy_ax))
        return xind, yind


class FootPrints(object):
    def __init__(
        self,
        fp_dir,
        rng=0.02,
        rtstep=0.002,
        mmap=False,
        im_dist_update=1e-4,
        max_nbytes=1e9,
        npy=True,
        ident="footprint",
    ):
        logging.info(rng)
        logging.info(rtstep)
        self.rng = rng
        self.rtstep = rtstep
        imxs = np.linspace(0, rng, int(rng / rtstep) + 1)
        imys = np.linspace(0, rng, int(rng / rtstep) + 1)
        grids = np.meshgrid(imxs, imys, indexing="ij")
        _imxs_rt_file = grids[0].ravel()
        _imys_rt_file = grids[1].ravel()

        self.fp_dir = fp_dir
        self.mmap = mmap
        self.fp_arr = get_rt_arr(self.fp_dir, ident=ident)
        self.nbytes = 0.0
        self.max_nbytes = max_nbytes
        self.npy = npy

        # use fnames as the dictionary keys
        self.fp_files = {}

        self._last_imx = 10.0
        self._last_imy = 10.0
        self._last_dimx = 10.0
        self._last_dimy = 10.0
        self._update_im_dist = im_dist_update

    def close_fp_file_obj(self, fp_arr_ind):
        k = self.fp_arr[fp_arr_ind]["fname"]
        logging.debug("nbytes=" + str(self.nbytes))
        self.nbytes -= self.fp_files[k].nbytes
        logging.debug("closing file " + k)
        del self.fp_files[k]
        self.fp_arr["time"][fp_arr_ind] = np.nan
        gc.collect()

    def mem_check(self):
        if self.nbytes > self.max_nbytes:
            ind2close = np.nanargmin(self.fp_arr["time"])
            self.close_fp_file_obj(ind2close)

    def open_fp_file_obj(self, fp_arr0):
        if self.npy:
            fp_file = footprint_file_npy(
                fp_arr0, self.fp_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        else:
            rt_file = ray_trace_file(
                rt_arr0, self.rt_dir, mmap=self.mmap, rtstep=self.rtstep, rng=self.rng
            )
        self.nbytes += fp_file.nbytes
        logging.debug("nbytes_total=" + str(self.nbytes))
        self.mem_check()
        self.fp_files[fp_arr0["fname"]] = fp_file

    def get_fp_file_obj(self, imx, imy):
        ind = np.argmin(
            np.hypot(
                imx - (self.fp_arr["imx0"] + self.fp_arr["imx1"]) / 2.0,
                imy - (self.fp_arr["imy0"] + self.fp_arr["imy1"]) / 2.0,
            )
        )

        fp_arr0 = self.fp_arr[ind]

        if not (
            (fp_arr0["imx0"] <= imx <= fp_arr0["imx1"])
            and (fp_arr0["imy0"] <= imy <= fp_arr0["imy1"])
        ):
            logging.warning(
                "No footprint files for this imx, imy: %.3f, %.3f" % (imx, imy)
            )

        fname = fp_arr0["fname"]

        if fname not in list(self.fp_files.keys()):
            self.open_fp_file_obj(fp_arr0)
        self.fp_arr["time"][ind] = time.time()
        return self.fp_files[fname]

    def check_im_dist(self, imx, imy, deriv=False):
        return (
            np.hypot(imx - self._last_imx, imy - self._last_imy) < self._update_im_dist
        )

    def get_fp(self, imx, imy):
        if not self.check_im_dist(imx, imy):
            fp_file = self.get_fp_file_obj(imx, imy)
            self._last_footprint = fp_file.get_fp(imx, imy)
            self._last_imx = imx
            self._last_imy = imy

        return self._last_footprint
