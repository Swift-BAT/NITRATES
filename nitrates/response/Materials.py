import numpy as np
from scipy import interpolate
from astropy.table import Table
import os


class Element(object):
    def __init__(self, name, cross_section_dname=None):
        if cross_section_dname is None:
            from ..config import ELEMENT_CROSS_SECTION_DNAME

            self.dname = ELEMENT_CROSS_SECTION_DNAME
        else:
            self.dname = cross_section_dname

        self.tab_fname = os.path.join(self.dname, name + ".txt")
        self.tab = Table.read(self.tab_fname, format="ascii")

        self.energies = self.tab["Energy"] * 1e3

        self.tot_mu_intp = interpolate.interp1d(
            np.log10(self.energies), np.log10(self.tab["TotwCoh"])
        )
        self.photoe_mu_intp = interpolate.interp1d(
            np.log10(self.energies), np.log10(self.tab["PhotoelAbsorb"])
        )
        self.comp_mu_intp = interpolate.interp1d(
            np.log10(self.energies), np.log10(self.tab["IncoherScatter"])
        )

    def get_tot_rho(self, Energy):
        return 10.0 ** self.tot_mu_intp(np.log10(Energy))

    def get_photoe_rho(self, Energy):
        return 10.0 ** self.photoe_mu_intp(np.log10(Energy))

    def get_comp_rho(self, Energy):
        return 10.0 ** self.comp_mu_intp(np.log10(Energy))


class Material(object):
    def __init__(self, density, elements=None, mass_fracs=None):
        self.density = density  # g/cm3
        self.elements = []
        self.element_obj_dict = {}
        self.mass_frac_dict = {}
        if elements is not None:
            for element, mass_frac in zip(elements, mass_fracs):
                self.add_element(element, mass_frac)

    def add_element(self, element, mass_frac):
        self.elements.append(element)
        element_obj = Element(element)

        self.element_obj_dict[element] = element_obj
        self.mass_frac_dict[element] = mass_frac

    def get_tot_rhomu(self, Energy):
        tot_rhomu = np.zeros_like(Energy)
        for element in self.elements:
            tot_rhomu += (
                self.mass_frac_dict[element]
                * self.density
                * self.element_obj_dict[element].get_tot_rho(Energy)
            )
        return tot_rhomu

    def get_comp_rhomu(self, Energy):
        rhomu = np.zeros_like(Energy)
        for element in self.elements:
            rhomu += (
                self.mass_frac_dict[element]
                * self.density
                * self.element_obj_dict[element].get_comp_rho(Energy)
            )
        return rhomu

    def get_photoe_rhomu(self, Energy):
        rhomu = np.zeros_like(Energy)
        for element in self.elements:
            rhomu += (
                self.mass_frac_dict[element]
                * self.density
                * self.element_obj_dict[element].get_photoe_rho(Energy)
            )
        return rhomu


class MaterialNone(object):
    def __init__(self):
        self.rhomu = 0.0

    def get_tot_rhomu(self, Energy):
        return self.rhomu

    def get_comp_rhomu(self, Energy):
        return self.rhomu

    def get_photoe_rhomu(self, Energy):
        return self.rhomu


CF_dens = 1.65
CarbonFibre = Material(CF_dens)
CarbonFibre.add_element("c", 1.0)

Korex = Material(0.032)
Korex.add_element("c", 1.0)

Mylar = Material(1.39)
Mylar.add_element("h", 0.3636)
Mylar.add_element("c", 0.4545)
Mylar.add_element("o", 0.1818)

SiCar_dens = 2.2985
si_au = 28.0855
c_au = 12.01
si_mass_frac = si_au / (si_au + c_au)
c_mass_frac = 1.0 - si_mass_frac
SiC = Material(SiCar_dens)
SiC.add_element("si", si_mass_frac)
SiC.add_element("c", c_mass_frac)

al_dens = 2.7
AL = Material(al_dens)
AL.add_element("al", 1.0)

AG = Material(10.49)
AG.add_element("ag", 1.0)

PB = Material(11.34)
PB.add_element("pb", 1.0)

TA = Material(16.6)
TA.add_element("ta", 1.0)

SN = Material(7.29)
SN.add_element("sn", 1.0)

CU = Material(8.93)
CU.add_element("cu", 1.0)

TI = Material(4.507)
TI.add_element("ti", 1.0)


elec_dens = 1.0
ElecMix = Material(elec_dens)
ElecMix.add_element("c", 0.05)
ElecMix.add_element("cu", 0.1)
ElecMix.add_element("si", 0.05)
ElecMix.add_element("al", 0.8)

ElecMixDense = Material(4.0)
ElecMixDense.add_element("c", 0.05)
ElecMixDense.add_element("cu", 0.1)
ElecMixDense.add_element("si", 0.05)
ElecMixDense.add_element("al", 0.8)


AlHoney1 = Material(0.2)
AlHoney1.add_element("al", 1.0)

AlHoney2 = Material(0.091)
AlHoney2.add_element("al", 1.0)

AlHoney3 = Material(0.055)
AlHoney3.add_element("al", 1.0)


c_aus = 22 * 12.01
cu_aus = 3 * 63.546
o_aus = 5 * 16.0
n_aus = 2 * 14.01
tot_aus = c_aus + cu_aus + o_aus + n_aus

pcb2lite = Material(0.393)
pcb2lite.add_element("c", c_aus / tot_aus)
pcb2lite.add_element("cu", cu_aus / tot_aus)
pcb2lite.add_element("o", o_aus / tot_aus)
pcb2lite.add_element("n", n_aus / tot_aus)

pcb2 = Material(2.07)
pcb2.add_element("c", c_aus / tot_aus)
pcb2.add_element("cu", cu_aus / tot_aus)
pcb2.add_element("o", o_aus / tot_aus)
pcb2.add_element("n", n_aus / tot_aus)


c_aus = 37 * 12.01
h_aus = 24 * 1.01
o_aus = 6 * 16.0
n_aus = 2 * 14.01
tot_aus = c_aus + h_aus + o_aus + n_aus

PolythLite = Material(0.28)
PolythLite.add_element("c", c_aus / tot_aus)
PolythLite.add_element("h", h_aus / tot_aus)
PolythLite.add_element("o", o_aus / tot_aus)
PolythLite.add_element("n", n_aus / tot_aus)

Polyth = Material(1.27)
Polyth.add_element("c", c_aus / tot_aus)
Polyth.add_element("h", h_aus / tot_aus)
Polyth.add_element("o", o_aus / tot_aus)
Polyth.add_element("n", n_aus / tot_aus)


AlMix = Material(1.0)
AlMix.add_element("al", 1.0)

plastic = Material(1.5)
c_au = 12.01
h_au = 1.01
h_mass_frac = 2 * h_au / (2 * h_au + c_au)
c_mass_frac = 1.0 - h_mass_frac
plastic.add_element("c", c_mass_frac)
plastic.add_element("h", h_mass_frac)


CZT = Material(5.78)
cd_aus = 9 * 112.411
zn_aus = 1 * 65.38
te_aus = 10 * 127.6
tot_aus = cd_aus + zn_aus + te_aus
CZT.add_element("cd", cd_aus / tot_aus)
CZT.add_element("zn", zn_aus / tot_aus)
CZT.add_element("te", te_aus / tot_aus)
