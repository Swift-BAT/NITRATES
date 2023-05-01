import numpy as np

PB_RHO = 11.35
PB_LCFF = 199826.0
PB_LIND = -2.585738
PB_HCFF = 3018560.0
PB_HIND = -2.92218
PB_VCFF = 0.00144026
PB_VIND = 0.373161
PB_SMTH = 0.371126
PB_EDGEN = 0.35176069


def pb_mu_low(energy):
    pb_mu = PB_LCFF * (energy**PB_LIND)

    return pb_mu


def pb_mu_high(energy):
    mid = PB_HCFF * (energy**PB_HIND)
    high = PB_VCFF * (energy**PB_VIND)
    base = mid**PB_SMTH + high**PB_SMTH
    power = 1.0 / PB_SMTH
    pb_mu = base**power

    return pb_mu


def trans(pb_mu, imx, imy):
    cos_theta = 1.0 / np.sqrt(1.0 + imx**2 + imy**2)  # I think

    t_pb = np.exp(-0.1 * pb_mu * PB_RHO / cos_theta)

    t_pb_near = (cos_theta / (0.05 * pb_mu * PB_RHO)) * (
        1.0 - np.exp(-0.05 * pb_mu * PB_RHO / cos_theta)
    )

    t_pb_far = (cos_theta / (0.05 * pb_mu * PB_RHO)) * (
        np.exp(-0.05 * pb_mu * PB_RHO / cos_theta)
        - np.exp(-0.1 * pb_mu * PB_RHO / cos_theta)
    )

    nonedge_fraction = (1.0 - 0.2 * abs(imx)) * (1.0 - 0.2 * abs(imy))
    edge_fraction = 1.0 - nonedge_fraction
    edge_fraction = PB_EDGEN * edge_fraction

    t_pb_overall = (1.0 - edge_fraction) * (1.0 - t_pb) + edge_fraction * (
        t_pb_near - t_pb_far
    )

    return t_pb_overall


def get_pb_mu(energy):
    if np.isscalar(energy):
        energy = np.array([energy])
    pb_mu = pb_mu_low(energy)
    pb_mu[(energy > 88)] = pb_mu_high(energy[(energy > 88)])

    return pb_mu


def get_pb_absortion(energy, imx, imy):
    pb_mu = get_pb_mu(energy)

    pb_absorb = trans(pb_mu, imx, imy)

    return pb_absorb
