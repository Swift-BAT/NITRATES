import numpy as np
import matplotlib.path as mpltPath
from numba import njit
import os

from ..response.shield_structure import (
    get_intersection_pnt,
    get_intersection_pnts,
    shield_polygon,
    Shield_Structure,
    get_norm_vec_from_wall_bnds,
)


@njit(cache=True)
def get_ray_xy_at_z(x0, y0, z0, theta, phi, z):
    Vx = np.sin(theta) * np.cos(phi)
    Vy = np.sin(theta) * np.sin(-phi)
    Vz = np.cos(theta)

    t = (z - z0) / Vz
    x = x0 + Vx * t
    y = y0 + Vy * t

    return x, y


@njit(cache=True)
def get_cylinder_ray_intersection_pnt(x0, y0, R, gam_theta, gam_phi, detx, dety, detz):
    Vx = np.sin(gam_theta) * np.cos(gam_phi)
    Vy = np.sin(gam_theta) * np.sin(-gam_phi)
    Vz = np.cos(gam_theta)
    #     print(Vx, Vy, Vz)

    A = Vx**2 + Vy**2
    B = 2 * Vx * (detx - x0) + 2 * Vy * (dety - y0)
    C = (
        x0**2
        + detx**2
        - 2 * x0 * detx
        + y0**2
        + dety**2
        - 2 * y0 * dety
        - R**2
    )

    #     print(A,B,C)
    #     print(B**2 - 4*A*C)

    t_0 = (np.sqrt(B**2 - 4 * A * C) - B) / (2 * A)
    t_1 = (-np.sqrt(B**2 - 4 * A * C) - B) / (2 * A)

    x_0 = detx + Vx * t_0
    y_0 = dety + Vy * t_0
    z_0 = detz + Vz * t_0

    x_1 = detx + Vx * t_1
    y_1 = dety + Vy * t_1
    z_1 = detz + Vz * t_1

    return x_0, y_0, z_0, x_1, y_1, z_1


@njit(cache=True)
def get_cylinder_ray_intersection_pnts(x0, y0, R, gam_theta, gam_phi, detx, dety, detz):
    x_0 = np.zeros_like(detx)
    y_0 = np.zeros_like(detx)
    z_0 = np.zeros_like(detx)
    x_1 = np.zeros_like(detx)
    y_1 = np.zeros_like(detx)
    z_1 = np.zeros_like(detx)

    for i in range(len(detx)):
        (
            x_0[i],
            y_0[i],
            z_0[i],
            x_1[i],
            y_1[i],
            z_1[i],
        ) = get_cylinder_ray_intersection_pnt(
            x0, y0, R, gam_theta, gam_phi, detx[i], dety[i], detz[i]
        )

    return x_0, y_0, z_0, x_1, y_1, z_1


@njit(cache=True)
def calc_disc_ray_int_dist(x0, y0, R, zmin, zmax, gam_theta, gam_phi, batx, baty, batz):
    x0, y0, z0, x1, y1, z1 = get_cylinder_ray_intersection_pnt(
        x0, y0, R, gam_theta, gam_phi, batx, baty, batz
    )

    if np.isnan(x0) and np.isnan(x1):
        # doesn't intersect outer circle of cylinder so dist=0
        return 0.0
    if z0 < zmin and z1 < zmin:
        # enters and exits outer circle below tube
        return 0.0
    if z0 > zmax and z1 > zmax:
        # enters and exits outer circle above tube
        return 0.0

    if z0 < zmin:
        z0 = zmin
        x0, y0 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z0)
    elif z0 > zmax:
        z0 = zmax
        x0, y0 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z0)
    if z1 < zmin:
        z1 = zmin
        x1, y1 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z1)
    elif z1 > zmax:
        z1 = zmax
        x1, y1 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z1)
    if gam_theta < np.pi / 2.0:
        if ((z0 - batz) < 0) or ((z1 - batz) < 0):
            return 0.0
    elif gam_theta > np.pi / 2.0:
        if ((z0 - batz) > 0) or ((z1 - batz) > 0):
            return 0.0
    dist = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
    #         print(x10, y10, z10)
    #         print(x11, y11, z11)
    return dist


@njit(cache=True)
def calc_disc_ray_int_dists(
    x0, y0, R, zmin, zmax, gam_theta, gam_phi, batx, baty, batz
):
    dists = np.zeros_like(batx)
    if (gam_theta > 0) and (gam_theta < np.pi):
        for i in range(len(batx)):
            dists[i] = calc_disc_ray_int_dist(
                x0, y0, R, zmin, zmax, gam_theta, gam_phi, batx[i], baty[i], batz[i]
            )
    return dists


@njit(cache=True)
def calc_cylinder_ray_int_dist(
    x0, y0, r0, r1, zmin, zmax, gam_theta, gam_phi, batx, baty, batz
):
    # intersection with outer circle of cylinder
    x10, y10, z10, x11, y11, z11 = get_cylinder_ray_intersection_pnt(
        x0, y0, r1, gam_theta, gam_phi, batx, baty, batz
    )

    #     if np.isnan(x10) and np.isnan(x11):
    if (not np.isfinite(x10)) and (not np.isfinite(x11)):
        # doesn't intersect outer circle of cylinder so dist=0
        return 0.0
    if z10 < zmin and z11 < zmin:
        # enters and exits outer circle below tube
        return 0.0
    if z10 > zmax and z11 > zmax:
        # enters and exits outer circle above tube
        return 0.0

    # intersection with inner circle of cylinder
    x00, y00, z00, x01, y01, z01 = get_cylinder_ray_intersection_pnt(
        x0, y0, r0, gam_theta, gam_phi, batx, baty, batz
    )

    if np.isnan(x00):
        # doesn't intersect inner circle
        #         print "doesn't intersect inner circle"
        if z10 < zmin:
            z10 = zmin
            x10, y10 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z10)
        elif z10 > zmax:
            z10 = zmax
            x10, y10 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z10)
        if z11 < zmin:
            z11 = zmin
            x11, y11 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z11)
        elif z11 > zmax:
            z11 = zmax
            x11, y11 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z11)
        if (gam_theta < np.pi / 2.0) and (((z10 - batz) < 0) or ((z11 - batz) < 0)):
            return 0.0
        elif (gam_theta > np.pi / 2.0) and (((z10 - batz) > 0) or ((z11 - batz) > 0)):
            return 0.0
        dist = np.sqrt((x10 - x11) ** 2 + (y10 - y11) ** 2 + (z10 - z11) ** 2)
        #         print(x10, y10, z10)
        #         print(x11, y11, z11)
        return dist

    # get dist0
    # the distance through the first instersection of the ring
    if z00 < zmin and z10 < zmin:
        dist0 = 0.0
    elif z00 > zmax and z10 > zmax:
        dist0 = 0.0
    else:
        if z00 < zmin:
            z00 = zmin
            x00, y00 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z00)
        if z10 < zmin:
            z10 = zmin
            x10, y10 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z10)
        if z00 > zmax:
            z00 = zmax
            x00, y00 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z00)
        if z10 > zmax:
            z10 = zmax
            x10, y10 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z10)
        if (gam_theta < np.pi / 2.0) and (((z00 - batz) < 0) or ((z10 - batz) < 0)):
            dist0 = 0.0
        elif (gam_theta > np.pi / 2.0) and (((z00 - batz) > 0) or ((z10 - batz) > 0)):
            dist0 = 0.0
        else:
            dist0 = np.sqrt((x00 - x10) ** 2 + (y00 - y10) ** 2 + (z00 - z10) ** 2)

    if z01 < zmin and z11 < zmin:
        dist1 = 0.0
    elif z01 > zmax and z11 > zmax:
        dist1 = 0.0
    else:
        if z01 < zmin:
            z01 = zmin
            x01, y01 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z01)
        if z11 < zmin:
            z11 = zmin
            x11, y11 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z11)
        if z01 > zmax:
            z01 = zmax
            x01, y01 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z01)
        if z11 > zmax:
            z11 = zmax
            x11, y11 = get_ray_xy_at_z(batx, baty, batz, gam_theta, gam_phi, z11)
        if (gam_theta < np.pi / 2.0) and (((z01 - batz) < 0) or ((z11 - batz) < 0)):
            dist1 = 0.0
        elif (gam_theta > np.pi / 2.0) and (((z01 - batz) > 0) or ((z11 - batz) > 0)):
            dist1 = 0.0
        else:
            dist1 = np.sqrt((x01 - x11) ** 2 + (y01 - y11) ** 2 + (z01 - z11) ** 2)

    #     dist0 = np.sqrt((x00-x10)**2 + (y00-y10)**2 + (z00-z10)**2)
    #     dist1 = np.sqrt((x01-x11)**2 + (y01-y11)**2 + (z01-z11)**2)
    #     print(dist0, dist1)
    dist = dist0 + dist1

    return dist


@njit(cache=True)
def calc_cylinder_ray_int_dists(
    x0, y0, r0, r1, zmin, zmax, gam_theta, gam_phi, batx, baty, batz
):
    dists = np.zeros_like(batx)
    if (gam_theta > 0) and (gam_theta < np.pi):
        for i in range(len(batx)):
            dists[i] = calc_cylinder_ray_int_dist(
                x0,
                y0,
                r0,
                r1,
                zmin,
                zmax,
                gam_theta,
                gam_phi,
                batx[i],
                baty[i],
                batz[i],
            )
    return dists


class Polygon2D(object):
    def __init__(self, orig_verts, trans_vec, rotations=None, plane="z"):
        """
        rotations is a list of rot_dicts
        rot_dict = {'axis':'X', 'angle':np.pi}
        """

        self.orig_verts = orig_verts
        self.plane = plane
        if plane == "z":
            self.poly_pnts = [(vert[0], vert[1]) for vert in self.orig_verts]
        elif plane == "y":
            self.poly_pnts = [(vert[0], vert[2]) for vert in self.orig_verts]
        elif plane == "x":
            self.poly_pnts = [(vert[1], vert[2]) for vert in self.orig_verts]
        else:
            print("bad plane")
        self.path = mpltPath.Path(self.poly_pnts)
        self.trans_vec = trans_vec
        if rotations is not None:
            self.Nrots = len(rotations)
            self.rot_mat = rot_list2rot_mat(rotations)
            self.inv_rot_mat = np.linalg.inv(self.rot_mat)

            self.verts = [
                np.matmul(self.rot_mat, vert) + self.trans_vec
                for vert in self.orig_verts
            ]
        else:
            self.Nrots = 0
            self.verts = [vert + self.trans_vec for vert in self.orig_verts]

    def calc_intersect_point(self, src_theta, src_phi, detx, dety, detz):
        if np.isscalar(detx):
            return get_intersection_pnt(
                self.verts, src_theta, src_phi, detx, dety, detz
            )
        else:
            return get_intersection_pnts(
                self.verts, src_theta, src_phi, detx, dety, detz
            )

    def does_intersect(
        self, src_theta, src_phi, detx, dety, detz, ret_pos=False, z_height=0.01
    ):
        x, y, z = self.calc_intersect_point(src_theta, src_phi, detx, dety, detz)
        if np.isscalar(detx):
            if src_theta < np.pi / 2.0:
                if (z - detz) < -z_height / 2.0:
                    return False
            else:
                if (z - detz) > z_height / 2.0:
                    return False
            pos = np.array([x, y, z])
            if self.Nrots > 0:
                pos = np.matmul(pos - self.trans_vec, self.rot_mat)
            if self.plane == "z":
                path_pnt = [pos[0], pos[1]]
            elif self.plane == "y":
                path_pnt = [pos[0], pos[2]]
            elif self.plane == "x":
                path_pnt = [pos[1], pos[2]]
            if ret_pos:
                return self.path.contains_point(path_pnt), x, y, z
            else:
                return self.path.contains_point(path_pnt)
        else:
            bl_goodz = np.ones(len(detx), dtype=bool)
            if src_theta < np.pi / 2.0:
                bl_goodz[((z - detz) < (-z_height / 2.0))] = False
            else:
                bl_goodz[((z - detz) > (z_height / 2.0))] = False
            xs = x - self.trans_vec[0]
            ys = y - self.trans_vec[1]
            zs = z - self.trans_vec[2]
            if self.Nrots > 0:
                xs, ys, zs = rot_pnts(xs, ys, zs, self.inv_rot_mat)
            if self.plane == "z":
                path_pnts = np.swapaxes(np.array([xs, ys]), 0, 1)
            elif self.plane == "y":
                path_pnts = np.swapaxes(np.array([xs, zs]), 0, 1)
            elif self.plane == "x":
                path_pnts = np.swapaxes(np.array([ys, zs]), 0, 1)
            if ret_pos:
                return self.path.contains_points(path_pnts) & bl_goodz, x, y, z
            else:
                return self.path.contains_points(path_pnts) & bl_goodz


class Box_Polygon(object):
    def __init__(self, x_half, y_half, z_half, trans_vec):
        """ """

        self.trans_vec = trans_vec
        self.orig_verts = np.array(
            [
                [x_half, y_half, z_half],
                [-x_half, y_half, z_half],
                [-x_half, -y_half, z_half],
                [x_half, -y_half, z_half],
                [x_half, -y_half, -z_half],
                [-x_half, -y_half, -z_half],
                [-x_half, y_half, -z_half],
                [x_half, y_half, -z_half],
            ]
        )

        self.verts = self.orig_verts + trans_vec

        self.poly2d_orig_vert_list = [
            [
                self.orig_verts[0],
                self.orig_verts[1],
                self.orig_verts[2],
                self.orig_verts[3],
            ],
            [
                self.orig_verts[0],
                self.orig_verts[1],
                self.orig_verts[6],
                self.orig_verts[7],
            ],
            [
                self.orig_verts[1],
                self.orig_verts[2],
                self.orig_verts[5],
                self.orig_verts[6],
            ],
            [
                self.orig_verts[2],
                self.orig_verts[3],
                self.orig_verts[4],
                self.orig_verts[5],
            ],
            [
                self.orig_verts[3],
                self.orig_verts[0],
                self.orig_verts[7],
                self.orig_verts[4],
            ],
            [
                self.orig_verts[4],
                self.orig_verts[5],
                self.orig_verts[6],
                self.orig_verts[7],
            ],
        ]
        self.face_planes = ["z", "y", "x", "y", "x", "z"]

        self.face_poly_list = []

        for vert_list, plane in zip(self.poly2d_orig_vert_list, self.face_planes):
            self.face_poly_list.append(Polygon2D(vert_list, trans_vec, plane=plane))

    def calc_intersect_point(self, src_theta, src_phi, detx, dety, detz):
        x0s = np.nan * np.ones_like(detx)
        y0s = np.nan * np.ones_like(detx)
        z0s = np.nan * np.ones_like(detx)

        x1s = np.nan * np.ones_like(detx)
        y1s = np.nan * np.ones_like(detx)
        z1s = np.nan * np.ones_like(detx)

        has_int0 = np.zeros_like(detx, dtype=bool)
        has_int1 = np.zeros_like(detx, dtype=bool)

        for poly_face in self.face_poly_list:
            does_int, x, y, z = poly_face.does_intersect(
                src_theta, src_phi, detx, dety, detz, ret_pos=True
            )
            if np.sum(does_int) > 0:
                bl0 = does_int & (~has_int0)
                bl1 = does_int & (~has_int1) & has_int0
                x0s[bl0] = x[bl0]
                y0s[bl0] = y[bl0]
                z0s[bl0] = z[bl0]
                has_int0[bl0] = True

                x1s[bl1] = x[bl1]
                y1s[bl1] = y[bl1]
                z1s[bl1] = z[bl1]
                has_int1[bl1] = True

        return x0s, y0s, z0s, x1s, y1s, z1s

    def calc_intersection_dist(self, src_theta, src_phi, detx, dety, detz):
        x0s, y0s, z0s, x1s, y1s, z1s = self.calc_intersect_point(
            src_theta, src_phi, detx, dety, detz
        )

        dists = np.sqrt((x0s - x1s) ** 2 + (y0s - y1s) ** 2 + (z0s - z1s) ** 2)
        dists[np.isnan(dists)] = 0.0
        return dists


class Cylinder_Polygon(object):
    def __init__(self, x0, y0, z0, r0, r1, half_height, axis="z"):
        """ """

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.z_max = z0 + half_height
        self.z_min = z0 - half_height
        self.half_height = half_height

        self.r0 = r0
        self.r1 = r1

    def calc_intersect_point(self, src_theta, src_phi, detx, dety, detz, R=None):
        if R is None:
            R = self.r1
        if np.isscalar(detx):
            return get_cylinder_ray_intersection_pnt(
                self.x0, self.y0, R, src_theta, src_phi, detx, dety, detz
            )
        else:
            return get_cylinder_ray_intersection_pnts(
                self.x0, self.y0, R, src_theta, src_phi, detx, dety, detz
            )

    def calc_intersection_dist(self, src_theta, src_phi, detx, dety, detz):
        if np.isscalar(detx):
            if self.r0 > 0:
                return calc_cylinder_ray_int_dist(
                    self.x0,
                    self.y0,
                    self.r0,
                    self.r1,
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )
            else:
                return calc_disc_ray_int_dist(
                    self.x0,
                    self.y0,
                    self.r1,
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )
        else:
            if self.r0 > 0:
                return calc_cylinder_ray_int_dists(
                    self.x0,
                    self.y0,
                    self.r0,
                    self.r1,
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )
            else:
                return calc_disc_ray_int_dists(
                    self.x0,
                    self.y0,
                    self.r1,
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )


class Cylinder_wHoles_Polygon(object):
    def __init__(self, x0, y0, z0, R, hole_cents, hole_rs, half_height, axis="z"):
        """
        Holes can't overlap
        """

        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        self.Nholes = len(hole_rs)
        self.hole_cents = hole_cents
        self.hole_rs = hole_rs

        self.z_max = z0 + half_height
        self.z_min = z0 - half_height
        self.half_height = half_height

        self.R = R

    #     def calc_intersect_point(self, src_theta, src_phi, detx, dety, detz, R=None):

    #         if R is None:
    #             R = self.r1
    #         if np.isscalar(detx):
    #             return get_cylinder_ray_intersection_pnt(self.x0, self.y0, R,\
    #                                                      src_theta, src_phi, detx, dety, detz)
    #         else:
    #             return get_cylinder_ray_intersection_pnts(self.x0, self.y0, R,\
    #                                                       src_theta, src_phi, detx, dety, detz)

    def calc_intersection_dist(self, src_theta, src_phi, detx, dety, detz):
        # calc dist with whole cylinder (with disc func)
        # if dist > 0
        # then calc dist of each hole and subtract that off

        if np.isscalar(detx):
            tot_dist = calc_disc_ray_int_dist(
                self.x0,
                self.y0,
                self.R,
                self.z_min,
                self.z_max,
                src_theta,
                src_phi,
                detx,
                dety,
                detz,
            )
            for i in range(self.Nholes):
                dist = calc_disc_ray_int_dist(
                    self.hole_cents[i][0],
                    self.hole_cents[i][1],
                    self.hole_rs[i],
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )
                tot_dist -= dist

            return tot_dist
        else:
            tot_dist = calc_disc_ray_int_dists(
                self.x0,
                self.y0,
                self.R,
                self.z_min,
                self.z_max,
                src_theta,
                src_phi,
                detx,
                dety,
                detz,
            )
            for i in range(self.Nholes):
                dist = calc_disc_ray_int_dists(
                    self.hole_cents[i][0],
                    self.hole_cents[i][1],
                    self.hole_rs[i],
                    self.z_min,
                    self.z_max,
                    src_theta,
                    src_phi,
                    detx,
                    dety,
                    detz,
                )
                tot_dist -= dist

            return tot_dist
