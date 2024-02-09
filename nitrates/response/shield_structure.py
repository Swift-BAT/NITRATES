import numpy as np
import matplotlib.path as mpltPath
from numba import njit


@njit(cache=True, fastmath=True)
def normalize_vec(v):
    return v / np.sqrt(np.sum(v**2))


@njit(cache=True, fastmath=True)
def get_norm_vec_from_wall_bnds(wall_bnds):
    #     x0 = np.array(wall_bnds[0])
    #     x1 = np.array(wall_bnds[1])
    #     x2 = np.array(wall_bnds[2])
    x0 = wall_bnds[0]
    x1 = wall_bnds[1]
    x2 = wall_bnds[2]
    v01 = x1 - x0
    v02 = x2 - x0
    norm_vec = normalize_vec(np.cross(v01, v02))
    return norm_vec


@njit(cache=True)
def get_intersection_pnt(wall_bnds, gam_theta, gam_phi, batx, baty, batz):
    wall_norm_vec = get_norm_vec_from_wall_bnds(wall_bnds)
    wall_x0, wall_y0, wall_z0 = wall_bnds[0]
    denom = (
        wall_norm_vec[0] * np.sin(gam_theta) * np.cos(gam_phi)
        + wall_norm_vec[1] * np.sin(gam_theta) * np.sin(-gam_phi)
        + wall_norm_vec[2] * np.cos(gam_theta)
    )
    if denom == 0.0:
        return np.nan, np.nan, np.nan
    r = -(
        wall_norm_vec[0] * (batx - wall_x0)
        + wall_norm_vec[1] * (baty - wall_y0)
        + wall_norm_vec[2] * (batz - wall_z0)
    ) / (
        wall_norm_vec[0] * np.sin(gam_theta) * np.cos(gam_phi)
        + wall_norm_vec[1] * np.sin(gam_theta) * np.sin(-gam_phi)
        + wall_norm_vec[2] * np.cos(gam_theta)
    )
    x = r * np.sin(gam_theta) * np.cos(gam_phi) + batx
    y = r * np.sin(gam_theta) * np.sin(-gam_phi) + baty
    z = r * np.cos(gam_theta) + batz
    return x, y, z


@njit(fastmath=True, cache=True)
def rot_pnts(xs, ys, zs, rot_mat):
    x_news = np.zeros_like(xs)
    y_news = np.zeros_like(xs)
    z_news = np.zeros_like(xs)

    for i in range(len(x_news)):
        x_news[i] = (
            xs[i] * rot_mat[0, 0] + ys[i] * rot_mat[0, 1] + zs[i] * rot_mat[0, 2]
        )
        y_news[i] = (
            xs[i] * rot_mat[1, 0] + ys[i] * rot_mat[1, 1] + zs[i] * rot_mat[1, 2]
        )
        z_news[i] = (
            xs[i] * rot_mat[2, 0] + ys[i] * rot_mat[2, 1] + zs[i] * rot_mat[2, 2]
        )

    return x_news, y_news, z_news


@njit(cache=True)
def get_intersection_pnts(wall_bnds, gam_theta, gam_phi, batxs, batys, batzs):
    wall_norm_vec = get_norm_vec_from_wall_bnds(wall_bnds)
    wall_x0, wall_y0, wall_z0 = wall_bnds[0]
    Npnts = len(batxs)
    xs = np.zeros_like(batxs)
    ys = np.zeros_like(batxs)
    zs = np.zeros_like(batxs)
    for i in range(Npnts):
        denom = (
            wall_norm_vec[0] * np.sin(gam_theta) * np.cos(gam_phi)
            + wall_norm_vec[1] * np.sin(gam_theta) * np.sin(-gam_phi)
            + wall_norm_vec[2] * np.cos(gam_theta)
        )
        if denom == 0.0:
            xs[i], ys[i], zs[i] = (np.nan, np.nan, np.nan)
            continue
        r = (
            -(
                wall_norm_vec[0] * (batxs[i] - wall_x0)
                + wall_norm_vec[1] * (batys[i] - wall_y0)
                + wall_norm_vec[2] * (batzs[i] - wall_z0)
            )
            / denom
        )

        xs[i] = r * np.sin(gam_theta) * np.cos(gam_phi) + batxs[i]
        ys[i] = r * np.sin(gam_theta) * np.sin(-gam_phi) + batys[i]
        zs[i] = r * np.cos(gam_theta) + batzs[i]
    return xs, ys, zs


def get_RotMatX(ang):
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = 1.0
    rot_mat[1, 1] = np.cos(ang)
    rot_mat[2, 2] = np.cos(ang)
    rot_mat[1, 2] = -np.sin(ang)
    rot_mat[2, 1] = np.sin(ang)
    return rot_mat


def get_RotMatY(ang):
    rot_mat = np.zeros((3, 3))
    rot_mat[1, 1] = 1.0
    rot_mat[0, 0] = np.cos(ang)
    rot_mat[2, 2] = np.cos(ang)
    rot_mat[0, 2] = np.sin(ang)
    rot_mat[2, 0] = -np.sin(ang)
    return rot_mat


def get_RotMatZ(ang):
    rot_mat = np.zeros((3, 3))
    rot_mat[2, 2] = 1.0
    rot_mat[0, 0] = np.cos(ang)
    rot_mat[1, 1] = np.cos(ang)
    rot_mat[0, 1] = -np.sin(ang)
    rot_mat[1, 0] = np.sin(ang)
    return rot_mat


def rot_list2rot_mat(rot_dicts):
    rot_func_dict = {"X": get_RotMatX, "Y": get_RotMatY, "Z": get_RotMatZ}

    Nrots = len(rot_dicts)
    if Nrots < 2:
        return rot_func_dict[rot_dicts[0]["axis"]](rot_dicts[0]["angle"])
    rot_mat0 = rot_func_dict[rot_dicts[-1]["axis"]](rot_dicts[-1]["angle"])
    rot_mat1 = rot_func_dict[rot_dicts[-2]["axis"]](rot_dicts[-2]["angle"])
    rot_mat = np.matmul(rot_mat0, rot_mat1)
    for i in range(-3, -Nrots - 1, -1):
        rmat = rot_func_dict[rot_dicts[i]["axis"]](rot_dicts[i]["angle"])
        rot_mat = np.matmul(rot_mat, rmat)
    return rot_mat


class shield_polygon(object):
    def __init__(self, orig_verts, trans_vec, rotations):
        """
        rotations is a list of rot_dicts
        rot_dict = {'axis':'X', 'angle':np.pi}
        """
        self.orig_verts = orig_verts
        self.poly_pnts = [(vert[0], vert[1]) for vert in self.orig_verts]
        self.path = mpltPath.Path(self.poly_pnts)
        self.trans_vec = trans_vec
        self.Nrots = len(rotations)
        if self.Nrots > 0:
            self.rot_mat = rot_list2rot_mat(rotations)
            self.inv_rot_mat = np.linalg.inv(self.rot_mat)

        if self.Nrots > 0:
            self.verts = [
                np.matmul(self.rot_mat, vert) + self.trans_vec
                for vert in self.orig_verts
            ]
        else:
            self.verts = [vert + self.trans_vec for vert in self.orig_verts]

        self.norm_vec = get_norm_vec_from_wall_bnds(self.verts)

    def coord_conv2orig(self, x, y, z):
        if np.isscalar(x):
            x_ = [x - self.trans_vec[0]]
            y_ = [y - self.trans_vec[1]]
            z_ = [z - self.trans_vec[2]]
            if self.Nrots > 0:
                x_, y_, z_ = rot_pnts(x_, y_, z_, self.inv_rot_mat)
            return x_[0], y_[0], z_[0]
        else:
            x_ = x - self.trans_vec[0]
            y_ = y - self.trans_vec[1]
            z_ = z - self.trans_vec[2]
            if self.Nrots > 0:
                x_, y_, z_ = rot_pnts(x_, y_, z_, self.inv_rot_mat)
            return x_, y_, z_

    def coord_conv2actual(self, x, y, z):
        if np.isscalar(x):
            if self.Nrots > 0:
                x, y, z = rot_pnts([x], [y], [z], self.rot_mat)
                x = x[0]
                y = y[0]
                z = z[0]
            x_ = x + self.trans_vec[0]
            y_ = y + self.trans_vec[1]
            z_ = z + self.trans_vec[2]
            return x_, y_, z_
        else:
            if self.Nrots > 0:
                x, y, z = rot_pnts(x, y, z, self.rot_mat)
            x_ = x + self.trans_vec[0]
            y_ = y + self.trans_vec[1]
            z_ = z + self.trans_vec[2]
            return x_, y_, z_

    def get_grid_pnts(self, dx=1.0):
        poly_xs = [vert[0] for vert in self.poly_pnts]
        poly_ys = [vert[1] for vert in self.poly_pnts]
        x0, x1 = np.min(poly_xs), np.max(poly_xs)
        y0, y1 = np.min(poly_ys), np.max(poly_ys)

        xs_ = np.arange(x0 - dx, x1 + dx, dx)
        ys_ = np.arange(y0 - dx, y1 + dx, dx)

        xgrid, ygrid = np.meshgrid(xs_, ys_)

        path_pnts = np.swapaxes(np.array([xgrid.ravel(), ygrid.ravel()]), 0, 1)
        bl_inpoly = self.path.contains_points(path_pnts)

        xs = xgrid.ravel()[bl_inpoly]
        ys = ygrid.ravel()[bl_inpoly]
        zs = np.zeros_like(xs)

        x, y, z = self.coord_conv2actual(xs, ys, zs)

        return x, y, z

    def calc_intersect_point(self, src_theta, src_phi, detx, dety, detz):
        if np.isscalar(detx):
            return get_intersection_pnt(
                self.verts, src_theta, src_phi, detx, dety, detz
            )
        else:
            return get_intersection_pnts(
                self.verts, src_theta, src_phi, detx, dety, detz
            )

    def does_intersect(self, src_theta, src_phi, detx, dety, detz):
        x, y, z = self.calc_intersect_point(src_theta, src_phi, detx, dety, detz)
        if np.isscalar(detx):
            if src_theta < np.pi / 2.0:
                if (z - detz) < -0.2:
                    return False
            else:
                if (z - detz) > 0.0:
                    return False
            pos = np.array([x, y, z]) - self.trans_vec
            if self.Nrots > 0:
                pos = np.matmul(pos, self.rot_mat)
            return self.path.contains_point(pos[:2])
        else:
            bl_goodz = np.ones(len(detx), dtype=bool)
            if src_theta < np.pi / 2.0:
                bl_goodz[((z - detz) < -0.2)] = False
            else:
                bl_goodz[((z - detz) > 0.0)] = False
            xs = x - self.trans_vec[0]
            ys = y - self.trans_vec[1]
            zs = z - self.trans_vec[2]
            if self.Nrots > 0:
                xs, ys, zs = rot_pnts(xs, ys, zs, self.inv_rot_mat)
            path_pnts = np.swapaxes(np.array([xs, ys]), 0, 1)
            return self.path.contains_points(path_pnts) & bl_goodz


class Shield_Structure(object):
    def __init__(self):
        self.shield_dict = {}
        self.shield_names = []

        Zoffset = 3.187
        extra2add = 0.1

        # SS00
        y_hlength = 46.990
        x0_hlength = 81.915
        x1_hlength = 81.915
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 0.0
        y_mid = 0.0
        z_mid = -7.8 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_00 = shield_polygon(orig_verts, trans_vec, [])
        name = "00"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_00

        # section A
        # SS01
        y_hlength = 17.647 + extra2add
        x0_hlength = 81.915 + extra2add
        x1_hlength = 94.904 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]
        rotate_x0 = 90.0  # deg
        rotate_x1 = -7.384  # deg
        rot_dicts_secA = [{"axis": "X", "angle": np.radians(rotate_x0 + rotate_x1)}]
        x_mid = 0.0
        y_mid = 49.258
        z_mid = 9.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_A01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secA)
        name = "A01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_A01

        # SS02

        y_hlength = 7.563 + extra2add
        x0_hlength = 94.904 + extra2add
        x1_hlength = 100.471 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]
        trans_vec = np.array([0.0, 52.495, 34.7 + Zoffset])

        poly_A02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secA)
        name = "A02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_A02

        # SS03

        y_hlength = 29.142 + extra2add
        x0_hlength = 100.471 + extra2add
        x1_hlength = 121.920 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]
        trans_vec = np.array([0.0, 57.205, 71.1 + Zoffset])

        poly_A03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secA)
        name = "A03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_A03

        # section Ba
        # SS01
        y_hlength = 18.654 + extra2add
        x0_hlength = 39.367 + extra2add
        x1_hlength = 39.367 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]
        rot_dicts_secB = [
            {"axis": "X", "angle": np.radians(20.36)},
            {"axis": "Y", "angle": np.radians(-90.0)},
            {"axis": "X", "angle": np.radians(-90.0)},
        ]
        rot_dicts_secBFlip = [
            {"axis": "X", "angle": np.radians(-20.36)},
            {"axis": "Y", "angle": np.radians(-90.0)},
            {"axis": "X", "angle": np.radians(-90.0)},
        ]
        x_mid = 88.405
        y_mid = 7.623
        z_mid = 9.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Ba01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Ba01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Ba01

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBa01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Ba01flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBa01

        # SS02
        y_hlength = 8.0 + extra2add
        x0_hlength = 38.364 + extra2add
        x1_hlength = 38.364 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 97.679
        y_mid = 13.158
        z_mid = 34.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Ba02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Ba02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Ba02

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBa02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Ba02flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBa02

        # SS03
        y_hlength = 30.826 + extra2add
        x0_hlength = 27.826 + extra2add
        x1_hlength = 27.826 + extra2add
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 111.186
        y_mid = 26.413
        z_mid = 71.1 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Ba03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Ba03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Ba03

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBa03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Ba03flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBa03

        # section Bb
        # SS01
        y_hlength = 18.654 + extra2add
        x0_hlength = 2.266 + extra2add
        x1_hlength = 0.0
        x_offset = y_hlength * np.tan(np.radians(-3.476))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            #                   np.array([-x1_hlength+x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 88.405
        y_mid = 48.124
        z_mid = 9.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bb01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bb01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bb01

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBb01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bb01flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBb01

        # SS02
        y_hlength = 8.0 + extra2add
        x0_hlength = 0.972 + extra2add
        x1_hlength = 0.0
        x_offset = y_hlength * np.tan(np.radians(-3.476))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            #                   np.array([-x1_hlength+x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 97.679
        y_mid = 52.0
        z_mid = 34.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bb02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bb02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bb02

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBb02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bb02flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBb02

        # SS03
        y_hlength = 30.826 + extra2add
        x0_hlength = 3.744 + extra2add
        x1_hlength = 0.0
        x_offset = y_hlength * np.tan(np.radians(-3.476))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            #                   np.array([-x1_hlength+x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 111.186
        y_mid = 55.368
        z_mid = 71.020 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bb03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bb03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bb03

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBb03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bb03flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBb03

        # section Bc
        # SS01
        y_hlength = 18.654 + extra2add
        x0_hlength = 0.0
        x1_hlength = 7.623 + extra2add
        x_offset = y_hlength * np.tan(np.radians(-11.548))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            #                       np.array([-x0_hlength-x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 88.405
        y_mid = -35.555
        z_mid = 9.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bc01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bc01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bc01

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBc01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bc01flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBc01

        # SS02
        y_hlength = 8.0 + extra2add
        x0_hlength = 0.0
        x1_hlength = 3.27 + extra2add
        x_offset = y_hlength * np.tan(np.radians(-11.548))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            #                       np.array([-x0_hlength-x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 97.661
        y_mid = -26.841
        z_mid = 34.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bc02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bc02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bc02

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBc02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bc02flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBc02

        # SS03
        y_hlength = 30.826 + extra2add
        x0_hlength = 0.0
        x1_hlength = 12.382 + extra2add
        x_offset = y_hlength * np.tan(np.radians(-11.548))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            #                       np.array([-x0_hlength-x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 111.186
        y_mid = -6.832
        z_mid = 71.1 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_Bc03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secB)
        name = "Bc03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_Bc03

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipBc03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secBFlip)
        name = "Bc03flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipBc03

        # section C
        # SS01
        y_hlength = 17.647 + extra2add
        x0_hlength = 81.915 + 0.5
        x1_hlength = 75.317 + 0.5
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]
        rotate_x0 = 90.0  # deg
        rotate_x1 = 7.384  # deg
        rot_dicts_secC = [{"axis": "X", "angle": np.radians(rotate_x0 + rotate_x1)}]
        x_mid = 0.0
        y_mid = -49.258
        z_mid = 9.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_C01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secC)
        name = "C01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_C01

        # SS02
        y_hlength = 7.563 + extra2add
        x0_hlength = 75.317 + 0.5
        x1_hlength = 72.490 + 0.5
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 0.0
        y_mid = -52.493
        z_mid = 34.7 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_C02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secC)
        name = "C02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_C02

        # SS03
        y_hlength = 29.142 + extra2add
        x0_hlength = 72.490 + 0.5
        x1_hlength = 61.594 + 1.0
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 0.0
        y_mid = -57.205
        z_mid = 71.1 + Zoffset
        trans_vec = np.array([x_mid, y_mid, z_mid])

        poly_C03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secC)
        name = "C03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_C03

        # section D
        # SS01
        y_hlength = 17.528 + extra2add
        #     x0_hlength = 81.915
        x1_hlength = 13.845 + 0.5
        x_offset = y_hlength * np.tan(np.radians(-9.844))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x_offset, -y_hlength, 0.0]),
        ]

        rot_dicts_secD = [
            {"axis": "X", "angle": np.radians(-2.385)},
            {"axis": "Y", "angle": np.radians(-45.0)},
            {"axis": "X", "angle": np.radians(90.0)},
        ]

        rot_dicts_secDflip = [
            {"axis": "X", "angle": np.radians(2.385)},
            {"axis": "Y", "angle": np.radians(-135.0)},
            {"axis": "X", "angle": np.radians(90.0)},
        ]
        x_mid = -83.55
        y_mid = -44.323
        z_mid = 9.7 + Zoffset

        trans_vec = np.array([x_mid, y_mid, z_mid])
        poly_D01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secD)
        name = "D01"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_D01

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipD01 = shield_polygon(orig_verts, trans_vec, rot_dicts_secDflip)
        name = "D01flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipD01

        # SS02
        y_hlength = 7.492 + extra2add
        x0_hlength = 13.845 + 0.5
        x1_hlength = 19.779 + 0.5
        x_offset = y_hlength * np.tan(np.radians(-9.844))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = -85.855
        y_mid = -40.544
        z_mid = 34.7 + Zoffset

        trans_vec = np.array([x_mid, y_mid, z_mid])
        poly_D02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secD)
        name = "D02"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_D02

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipD02 = shield_polygon(orig_verts, trans_vec, rot_dicts_secDflip)
        name = "D02flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipD02

        # SS03
        y_hlength = 27.746 + 1.5 + 0.2
        x0_hlength = 19.779 + 0.5
        x1_hlength = 42.188 + 0.5
        x_offset = y_hlength * np.tan(np.radians(-9.844))
        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x1_hlength + x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = -89.233
        y_mid = -35.018
        z_mid = 70.22 + Zoffset + 0.55

        trans_vec = np.array([x_mid, y_mid, z_mid])
        poly_D03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secD)
        name = "D03"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_D03

        trans_vec = np.array([-x_mid, y_mid, z_mid])
        poly_flipD03 = shield_polygon(orig_verts, trans_vec, rot_dicts_secDflip)
        name = "D03flip"
        self.shield_names.append(name)
        self.shield_dict[name] = poly_flipD03


class Shield_Interactions(object):
    def __init__(self):
        self.shield_struct = Shield_Structure()
        self.Npolys = len(self.shield_struct.shield_names)
        self.shield_layer = np.zeros(self.Npolys, dtype=np.int64)
        for i, name in enumerate(self.shield_struct.shield_names):
            if "00" in name:
                self.shield_layer[i] = 0
                continue
            elif "01" in name:
                self.shield_layer[i] = 1
            elif "02" in name:
                self.shield_layer[i] = 2
            elif "03" in name:
                self.shield_layer[i] = 3
            poly = self.get_poly(i)
            if poly.norm_vec[2] < 0 and (not "D" in name):
                poly.norm_vec *= -1

    def get_poly(self, ident):
        """
        ident is either the name as a string or
        the id (the list index) as an int
        """

        if (
            isinstance(ident, int)
            or isinstance(ident, float)
            or isinstance(ident, np.integer)
        ):
            name = self.shield_struct.shield_names[int(ident)]
        else:
            name = ident
        return self.shield_struct.shield_dict[name]

    def does_intersect_poly(self, theta, phi, x, y, z, ident):
        poly = self.get_poly(ident)
        return poly.does_intersect(theta, phi, x, y, z)

    def which_poly_it_intersects(self, theta, phi, x, y, z, polyIDs2ignore=[]):
        poly_idents = -1 * np.ones(len(x), dtype=np.int64)
        for ident in range(self.Npolys):
            if ident in polyIDs2ignore:
                continue
            no_ints_yet = poly_idents == -1
            does_int = np.zeros(len(x), dtype=bool)
            does_int[no_ints_yet] = self.does_intersect_poly(
                theta, phi, x[no_ints_yet], y[no_ints_yet], z[no_ints_yet], ident
            )
            if np.sum(does_int) < 1:
                continue
            poly_idents[does_int] = ident
            if np.sum(poly_idents == -1) == 0:
                break
        return poly_idents

    def get_name_from_id(self, ID):
        return self.shield_struct.shield_names[ID]

    def get_shield_layer(self, ident):
        if isinstance(ident, str):
            ID = self.shield_struct.shield_names.index(ident)
        else:
            ID = ident
        return self.shield_layer[ID]

    def get_angs2norm(self, theta, phi):
        gamma_vec = np.array(
            [np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        angs2norm = np.zeros(self.Npolys)
        for ident in range(self.Npolys):
            poly = self.get_poly(ident)
            angs2norm[ident] = np.arccos(np.dot(gamma_vec, poly.norm_vec))
        return angs2norm


class Sun_Shield_Structure(object):
    def __init__(self):
        self.shield_dict = {}
        self.shield_names = []

        # Sun Shield 1
        y_hlength = 107.45
        x0_hlength = 70.575
        x1_hlength = 70.575
        orig_verts = [
            np.array([x1_hlength, y_hlength, 0.0]),
            np.array([-x1_hlength, y_hlength, 0.0]),
            np.array([-x0_hlength, -y_hlength, 0.0]),
            np.array([x0_hlength, -y_hlength, 0.0]),
        ]

        x_mid = 0.0
        y_mid = 130.0
        z_mid = 71.249
        trans_vec = np.array([x_mid, y_mid, z_mid])

        rot_dicts1 = [{"axis": "X", "angle": np.radians(90.0)}]

        poly1 = shield_polygon(orig_verts, trans_vec, rot_dicts1)
        name = "Shield1"
        self.shield_names.append(name)
        self.shield_dict[name] = poly1

        # Sun Shield 2
        y_hlength = 107.45
        x0_hlength = 39.995
        x1_hlength = 0.0
        x_offset = y_hlength * np.tan(np.radians(-10.532))

        orig_verts = [
            np.array([x1_hlength + x_offset, y_hlength, 0.0]),
            #                   np.array([-x1_hlength+x_offset, y_hlength, 0.0]),
            np.array([-x0_hlength - x_offset, -y_hlength, 0.0]),
            np.array([x0_hlength - x_offset, -y_hlength, 0.0]),
        ]

        x_mid = 82.067
        y_mid = 113.417
        z_mid = 71.249
        trans_vec = np.array([x_mid, y_mid, z_mid])

        rot_dicts2 = [
            {"axis": "Y", "angle": np.radians(-55.0)},
            {"axis": "X", "angle": np.radians(90.0)},
        ]

        poly2 = shield_polygon(orig_verts, trans_vec, rot_dicts2)
        name = "Shield2"
        self.shield_names.append(name)
        self.shield_dict[name] = poly2

        trans_vec = np.array([-x_mid, y_mid, z_mid])

        rot_dicts3 = [
            {"axis": "Y", "angle": np.radians(-125.0)},
            {"axis": "X", "angle": np.radians(90.0)},
        ]

        poly3 = shield_polygon(orig_verts, trans_vec, rot_dicts3)
        name = "Shield3"
        self.shield_names.append(name)
        self.shield_dict[name] = poly3


class Sun_Shield_Interactions(object):
    def __init__(self):
        self.shield_struct = Sun_Shield_Structure()
        self.Npolys = len(self.shield_struct.shield_names)

    def get_poly(self, ident):
        """
        ident is either the name as a string or
        the id (the list index) as an int
        """

        if (
            isinstance(ident, int)
            or isinstance(ident, float)
            or isinstance(ident, np.integer)
        ):
            name = self.shield_struct.shield_names[int(ident)]
        else:
            name = ident
        return self.shield_struct.shield_dict[name]

    def does_intersect_poly(self, theta, phi, x, y, z, ident):
        poly = self.get_poly(ident)
        return poly.does_intersect(theta, phi, x, y, z)

    def which_poly_it_intersects(self, theta, phi, x, y, z):
        poly_idents = -1 * np.ones(len(x), dtype=np.int64)
        for ident in range(self.Npolys):
            no_ints_yet = poly_idents == -1
            does_int = np.zeros(len(x), dtype=bool)
            does_int[no_ints_yet] = self.does_intersect_poly(
                theta, phi, x[no_ints_yet], y[no_ints_yet], z[no_ints_yet], ident
            )
            if np.sum(does_int) < 1:
                continue
            poly_idents[does_int] = ident
            if np.sum(poly_idents == -1) == 0:
                break
        return poly_idents

    def get_name_from_id(self, ID):
        return self.shield_struct.shield_names[ID]

    def get_angs2norm(self, theta, phi):
        gamma_vec = np.array(
            [np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        angs2norm = np.zeros(self.Npolys)
        for ident in range(self.Npolys):
            poly = self.get_poly(ident)
            angs2norm[ident] = np.arccos(np.dot(gamma_vec, poly.norm_vec))
        return angs2norm
