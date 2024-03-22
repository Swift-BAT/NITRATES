import numpy as np

from ..response.Materials import (
    CarbonFibre,
    SiC,
    AL,
    AG,
    ElecMix,
    ElecMixDense,
    AlHoney1,
    AlHoney2,
    AlHoney3,
    PB,
    TA,
    SN,
    CU,
)
from ..response.shield_structure import Shield_Interactions, Sun_Shield_Interactions
from ..response.Polygons import Polygon2D


class Swift_Structure(object):
    def __init__(self, Polygon, Material, Name=""):
        self.Polygon = Polygon
        self.Material = Material
        self.Name = Name

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus = self.Material.get_tot_rhomu(self.energy)
        self.comp_rho_mus = self.Material.get_comp_rhomu(self.energy)
        self.photoe_rho_mus = self.Material.get_photoe_rhomu(self.energy)
        if hasattr(self, "dists"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi

        self.dists = self.Polygon.calc_intersection_dist(
            self.theta, self.phi, self.batxs, self.batys, self.batzs
        )
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def get_dists(self, theta=None, phi=None):
        if (np.abs(theta - self.theta) > 1e-3) or (np.abs(phi - self.phi) > 1e-3):
            self.set_theta_phi(theta, phi)
        return self.dists

    def calc_tot_rhomu_dist(self):
        self.tot_rhomu_dists = self.dists[:, np.newaxis] * self.tot_rho_mus

    def get_trans(self, dist=None):
        if dist is None:
            dist = self.dists
        #         trans = np.exp(-dist*self.tot_rho_mus)
        trans = np.exp(-self.tot_rhomu_dists)
        return trans

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_Compound(object):
    def __init__(self, ParentPolygon, ChildPolygons, Materials, Name=""):
        self.Nchild = len(ChildPolygons)
        self.Parent_Polygon = ParentPolygon
        self.Child_Polygons = ChildPolygons
        self.material_list = Materials
        self.Name = Name

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus_list = []
        self.comp_rho_mus_list = []
        self.photoe_rho_mus_list = []

        for material in self.material_list:
            self.tot_rho_mus_list.append(material.get_tot_rhomu(self.energy))
            self.comp_rho_mus_list.append(material.get_comp_rhomu(self.energy))
            self.photoe_rho_mus_list.append(material.get_photoe_rhomu(self.energy))

        if hasattr(self, "parent_dist"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi

        self.calc_dists()
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def calc_dists(self):
        tot_dist = self.Parent_Polygon.calc_intersection_dist(
            self.theta, self.phi, self.batxs, self.batys, self.batzs
        )

        tot_child_dist = 0.0
        child_dists = []
        for child_poly in self.Child_Polygons:
            dist = child_poly.calc_intersection_dist(
                self.theta, self.phi, self.batxs, self.batys, self.batzs
            )
            tot_child_dist += dist
            child_dists.append(dist)

        self.parent_dist = tot_dist - tot_child_dist
        self.child_dists = child_dists

    #     def get_dists(self, theta=None, phi=None):

    #         if (np.abs(theta - self.theta) > 1e-3) or (np.abs(phi - self.phi) > 1e-3):
    #             self.set_theta_phi(theta, phi)
    #         return self.dists

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists)
        return self.trans

    def calc_tot_rhomu_dist(self):
        self.tot_rhomu_dists = self.parent_dist * self.tot_rho_mus_list[0]
        for i in range(self.Nchild):
            self.tot_rhomu_dists += self.child_dists[i] * self.tot_rho_mus_list[i + 1]

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_wEmbededPolys(object):
    def __init__(self, ParentPolygon, ChildPolygons, Materials, Name=""):
        self.Nchild = len(ChildPolygons)
        self.Parent_Polygon = ParentPolygon
        self.Child_Polygons = ChildPolygons
        self.material_list = Materials
        self.Name = Name

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus_list = []
        self.comp_rho_mus_list = []
        self.photoe_rho_mus_list = []

        for material in self.material_list:
            self.tot_rho_mus_list.append(material.get_tot_rhomu(self.energy))
            self.comp_rho_mus_list.append(material.get_comp_rhomu(self.energy))
            self.photoe_rho_mus_list.append(material.get_photoe_rhomu(self.energy))

        if hasattr(self, "parent_dist"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi

        self.calc_dists()
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def calc_dists(self):
        tot_dist = self.Parent_Polygon.calc_intersection_dist(
            self.theta, self.phi, self.batxs, self.batys, self.batzs
        )

        tot_child_dist = 0.0
        child_dists = []
        for child_poly in self.Child_Polygons:
            dist = child_poly.calc_intersection_dist(
                self.theta, self.phi, self.batxs, self.batys, self.batzs
            )
            tot_child_dist += dist
            child_dists.append(dist)

        self.parent_dist = tot_dist - tot_child_dist
        self.child_dists = child_dists

    #     def get_dists(self, theta=None, phi=None):

    #         if (np.abs(theta - self.theta) > 1e-3) or (np.abs(phi - self.phi) > 1e-3):
    #             self.set_theta_phi(theta, phi)
    #         return self.dists

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists)
        return self.trans

    def calc_tot_rhomu_dist(self):
        self.tot_rhomu_dists = (
            self.parent_dist[:, np.newaxis] * self.tot_rho_mus_list[0]
        )
        for i in range(self.Nchild):
            self.tot_rhomu_dists += (
                self.child_dists[i][:, np.newaxis] * self.tot_rho_mus_list[i + 1]
            )

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_Shield(object):
    def __init__(self):
        self.shield_obj = Shield_Interactions()

        self.ds_base = [
            0.00254 * np.array([3, 3, 2, 1]),
            0.00254 * np.array([8, 7, 6, 2]),
            0.00254 * np.array([5, 5, 4, 1]),
            0.00254 * np.array([3, 3, 2, 1]),
        ]

        self.material_list = [PB, TA, SN, CU]
        self.Nmaterials = len(self.material_list)
        self.Name = "Shield"
        self.polyIDs2ignore = []

    def add_polyID2ignore(self, ID):
        self.polyIDs2ignore.append(ID)

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus_list = []
        self.comp_rho_mus_list = []
        self.photoe_rho_mus_list = []

        for material in self.material_list:
            self.tot_rho_mus_list.append(material.get_tot_rhomu(self.energy))
            self.comp_rho_mus_list.append(material.get_comp_rhomu(self.energy))
            self.photoe_rho_mus_list.append(material.get_photoe_rhomu(self.energy))

        if hasattr(self, "dists"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi

        self.angs2norm = self.shield_obj.get_angs2norm(self.theta, self.phi)

        self.calc_dists()
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def calc_dists(self):
        self.poly_ids = self.shield_obj.which_poly_it_intersects(
            self.theta,
            self.phi,
            self.batxs,
            self.batys,
            self.batzs,
            polyIDs2ignore=self.polyIDs2ignore,
        )
        self.poly_ids2use = np.unique(self.poly_ids)

        self.dists = [np.zeros(self.ndets) for i in range(self.Nmaterials)]
        for polyid in self.poly_ids2use:
            poly_bl = self.poly_ids == polyid
            if polyid < 0:
                for i in range(self.Nmaterials):
                    self.dists[i][poly_bl] = 0.0
                continue
            poly = self.shield_obj.get_poly(polyid)
            layer = self.shield_obj.shield_layer[polyid]
            base_dists = self.ds_base[layer]
            cos_ang = np.abs(np.cos(self.angs2norm[polyid]))
            for i in range(self.Nmaterials):
                self.dists[i][poly_bl] = base_dists[i] / cos_ang

    def calc_tot_rhomu_dist(self):
        self.tot_rhomu_dists = np.zeros((self.ndets, self.Ne))
        for i in range(self.Nmaterials):
            self.tot_rhomu_dists += (
                self.dists[i][:, np.newaxis] * self.tot_rho_mus_list[i]
            )

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists)
        return self.trans

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_Sun_Shield(object):
    def __init__(self):
        self.shield_obj = Sun_Shield_Interactions()
        self.ds_base = 0.0145
        self.material = AG
        self.Name = "SunShield"

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus = self.material.get_tot_rhomu(self.energy)
        self.comp_rho_mus = self.material.get_comp_rhomu(self.energy)
        self.photoe_rho_mus = self.material.get_photoe_rhomu(self.energy)

        if hasattr(self, "dists"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi

        self.angs2norm = self.shield_obj.get_angs2norm(self.theta, self.phi)

        self.calc_dists()
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def calc_dists(self):
        self.poly_ids = self.shield_obj.which_poly_it_intersects(
            self.theta, self.phi, self.batxs, self.batys, self.batzs
        )
        self.poly_ids2use = np.unique(self.poly_ids)

        self.dists = np.zeros(self.ndets)
        for polyid in self.poly_ids2use:
            poly_bl = self.poly_ids == polyid
            if polyid < 0:
                self.dists[poly_bl] = 0.0
                continue
            poly = self.shield_obj.get_poly(polyid)
            cos_ang = np.abs(np.cos(self.angs2norm[polyid]))
            self.dists[poly_bl] = self.ds_base / cos_ang

    def calc_tot_rhomu_dist(self):
        self.tot_rhomu_dists = np.zeros((self.ndets, self.Ne))
        self.tot_rhomu_dists = self.dists[:, np.newaxis] * self.tot_rho_mus

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists)
        return self.trans

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_Mask(object):
    def __init__(self):
        self.ds_base = 0.1

        self.material = PB
        self.Nmaterials = 1
        self.Name = "Mask"

        self.norm_vec = np.array([0.0, 0.0, -1.0])

        self.verts = np.array(
            [
                (121.92, 60.95, 103.187),
                (121.92, -1.41, 103.187),
                (61.5, -60.95, 103.187),
                (-61.5, -60.95, 103.187),
                (-121.92, -1.41, 103.187),
                (-121.92, 60.95, 103.187),
            ]
        )
        trans_vec = np.zeros(3)

        self.mask_poly = Polygon2D(self.verts, trans_vec)

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)

        self.tot_rho_mus = self.material.get_tot_rhomu(self.energy)
        self.comp_rho_mus = self.material.get_comp_rhomu(self.energy)
        self.photoe_rho_mus = self.material.get_photoe_rhomu(self.energy)

        if hasattr(self, "dists"):
            self.calc_tot_rhomu_dist()

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.gamma_vec = -np.array(
            [np.sin(theta) * np.cos(-phi), np.sin(theta) * np.sin(-phi), np.cos(theta)]
        )

        self.ang2norm = np.arccos(np.dot(self.gamma_vec, self.norm_vec))
        self.d = self.ds_base / np.cos(self.ang2norm)

        self.calc_dists()
        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def calc_dists(self):
        does_int = self.mask_poly.does_intersect(
            self.theta, self.phi, self.batxs, self.batys, self.batzs
        )
        self.dists = does_int.astype(np.float64) * self.d

    def calc_tot_rhomu_dist(self):
        #         self.tot_rhomu_dists = np.zeros((self.ndets,self.Ne))
        self.tot_rhomu_dists = self.dists[:, np.newaxis] * self.tot_rho_mus

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists) + 0.5
        return self.trans

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists


class Swift_Structure_Manager(object):
    def __init__(self):
        self.struct_names = []
        self.struct_dict = {}
        self.Nstructs = 0

    def add_struct(self, struct):
        name = struct.Name
        if name in self.struct_names:
            print((name + "is already added"))
            return
        if hasattr(self, "energy"):
            struct.set_energy_arr(self.energy)
        if hasattr(self, "batxs"):
            struct.set_batxyzs(self.batxs, self.batys, self.batzs)
            if hasattr(self, "theta"):
                struct.set_theta_phi(self.theta, self.phi)

        self.struct_names.append(name)
        self.struct_dict[name] = struct
        self.Nstructs += 1

        if hasattr(self, "energy") and hasattr(self, "theta"):
            self.calc_tot_rhomu_dist()

    def set_energy_arr(self, energy):
        self.energy = energy
        self.Ne = len(energy)
        for name, struct in self.struct_dict.items():
            struct.set_energy_arr(self.energy)

    def set_batxyzs(self, batxs, batys, batzs):
        self.batxs = batxs
        self.batys = batys
        self.batzs = batzs
        self.ndets = len(batxs)
        for name, struct in self.struct_dict.items():
            struct.set_batxyzs(batxs, batys, batzs)

    def set_theta_phi(self, theta, phi):
        self.theta = theta
        self.phi = phi
        for name, struct in self.struct_dict.items():
            struct.set_theta_phi(theta, phi)

        if hasattr(self, "energy"):
            self.calc_tot_rhomu_dist()

    def get_trans(self):
        self.trans = np.exp(-self.tot_rhomu_dists)
        return self.trans

    def calc_tot_rhomu_dist(self):
        #         self.tot_rhomu_dists = self.parent_dist*self.tot_rho_mus_list[0]
        self.tot_rhomu_dists = np.zeros((self.ndets, self.Ne))
        for name, struct in self.struct_dict.items():
            self.tot_rhomu_dists += struct.get_tot_rhomu_dist()

    def get_tot_rhomu_dist(self):
        return self.tot_rhomu_dists
