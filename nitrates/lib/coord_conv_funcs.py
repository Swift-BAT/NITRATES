import numpy as np


align_mat = np.array(
    [
        [-1.96305411e-04, 6.56927099e-06, 9.99999981e-01],
        [9.99994245e-01, -3.38686178e-03, 1.96326534e-04],
        [3.38686300e-03, 9.99994265e-01, -5.90437390e-06],
    ]
)


def maintain_quat(q, round_off_err=1e-6):
    norm = np.sqrt(np.sum(np.square(q)))
    if (q[3] < 0) or (abs(norm - 1.0) > round_off_err):
        # print "fixing quat"
        invnorm = 1.0 / norm
        err = abs(norm - 1.0)
        if q[3] < 0:
            invnorm = -invnorm
        q = q * invnorm
    return q


def convertRotMatrixToQuat(rot):
    diag_sum = np.zeros(4)
    q = np.zeros(4)

    diag_sum[0] = 1 + rot[0][0] - rot[1][1] - rot[2][2]
    diag_sum[1] = 1 - rot[0][0] + rot[1][1] - rot[2][2]
    diag_sum[2] = 1 - rot[0][0] - rot[1][1] + rot[2][2]
    diag_sum[3] = 1 + rot[0][0] + rot[1][1] + rot[2][2]

    maxi = np.argmax(diag_sum)

    q[maxi] = 0.5 * np.sqrt(diag_sum[maxi])
    recip = 1.0 / (4.0 * q[maxi])

    if maxi == 0:
        q[1] = recip * (rot[0, 1] + rot[1, 0])
        q[2] = recip * (rot[2, 0] + rot[0, 2])
        q[3] = recip * (rot[1, 2] - rot[2, 1])

    elif maxi == 1:
        q[0] = recip * (rot[0, 1] + rot[1, 0])
        q[2] = recip * (rot[1, 2] + rot[2, 1])
        q[3] = recip * (rot[2, 0] - rot[0, 2])

    elif maxi == 2:
        q[0] = recip * (rot[2, 0] + rot[0, 2])
        q[1] = recip * (rot[1, 2] + rot[2, 1])
        q[3] = recip * (rot[0, 1] - rot[1, 0])

    elif maxi == 3:
        q[0] = recip * (rot[1, 2] - rot[2, 1])
        q[1] = recip * (rot[2, 0] - rot[0, 2])
        q[2] = recip * (rot[0, 1] - rot[1, 0])

    q = maintain_quat(q)

    return q


def convertQuatToRotMatrix(q):
    rot = np.zeros((3, 3))
    q2 = np.square(q)

    rot[0, 0] = q2[0] - q2[1] - q2[2] + q2[3]
    rot[1, 1] = -q2[0] + q2[1] - q2[2] + q2[3]
    rot[2, 2] = -q2[0] - q2[1] + q2[2] + q2[3]

    rot[0, 1] = 2.0 * (q[0] * q[1] + q[2] * q[3])
    rot[1, 0] = 2.0 * (q[0] * q[1] - q[2] * q[3])

    rot[0, 2] = 2.0 * (q[0] * q[2] - q[1] * q[3])
    rot[2, 0] = 2.0 * (q[0] * q[2] + q[1] * q[3])

    rot[1, 2] = 2.0 * (q[1] * q[2] + q[0] * q[3])
    rot[2, 1] = 2.0 * (q[1] * q[2] - q[0] * q[3])

    return rot


def invertQuat(q):
    q1 = -1.0 * q
    q1[3] *= -1

    return q1


def productOfQuats(q1, q2):
    q = np.zeros_like(q1)

    q[0] = q2[3] * q1[0] + q2[2] * q1[1] - q2[1] * q1[2] + q2[0] * q1[3]
    q[1] = -q2[2] * q1[0] + q2[3] * q1[1] + q2[0] * q1[2] + q2[1] * q1[3]
    q[2] = q2[1] * q1[0] - q2[0] * q1[1] + q2[3] * q1[2] + q2[2] * q1[3]
    q[3] = -q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2] + q2[3] * q1[3]

    q = maintain_quat(q)

    return q


def applyRotMatrixToVector(rot, vec):
    newvec = np.zeros_like(vec)

    for i in range(len(newvec)):
        if vec.ndim > 1:
            newvec[i] = np.sum(vec * rot[i, :, np.newaxis], axis=0)
        else:
            newvec[i] = np.sum(vec * rot[i, :])

    return newvec


def radec2skyvec(ra, dec):
    z_sky = np.sin(np.radians(dec))
    x_sky = (np.cos(np.radians(dec))) * (np.cos(np.radians(ra)))
    y_sky = (np.cos(np.radians(dec))) * (np.sin(np.radians(ra)))
    skyunit = np.array([x_sky, y_sky, z_sky])
    return skyunit


def skyvec2radec(sky_vec):
    dec = np.rad2deg(np.arcsin(sky_vec[2]))
    ra = np.rad2deg(np.arctan2(sky_vec[1], sky_vec[0]))
    if np.isscalar(ra):
        if ra < 0:
            ra += 360.0
    else:
        if np.any(ra < 0):
            ra[(ra < 0)] += 360.0
    return ra, dec


def imxy2detvec(imx, imy):
    z = 1.0 / np.sqrt(imx**2.0 + imy**2.0 + 1.0)
    x = imx * z
    y = imy * z
    detunit = np.array([x, y, z])
    return detunit


def theta_phi2detvec(theta, phi):
    z = np.cos(np.radians(theta))
    x = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = np.sin(np.radians(theta)) * np.sin(np.radians(-phi))
    detunit = np.array([x, y, z])
    return detunit


def convertSkyToSensor(teldef_qinv, sky_vec, att_q):
    cor_q = productOfQuats(att_q, teldef_qinv)

    rot = convertQuatToRotMatrix(cor_q)

    det_vec = applyRotMatrixToVector(rot, sky_vec)

    return det_vec


def convertDetToSky(teldef_qinv, det_vec, att_q):
    cor_q = productOfQuats(att_q, teldef_qinv)

    cor_qinv = invertQuat(cor_q)

    rot = convertQuatToRotMatrix(cor_qinv)

    sky_vec = applyRotMatrixToVector(rot, det_vec)

    return sky_vec


def convert_radec2batxyz(ra, dec, att_q):
    """
    Converts ra, dec to the BAT x,y,z vector

    Parameters:
    ra: float in degrees
    dec: float in degrees
    att_q: the attitude quaternion (length 4 1D array)

    Returns:
    detector vec: length 3 float vector
    """
    teldef_q = convertRotMatrixToQuat(align_mat)
    teldef_qinv = invertQuat(teldef_q)
    att_q = maintain_quat(att_q)

    sky_vec = radec2skyvec(ra, dec)
    det_vec = convertSkyToSensor(teldef_qinv, sky_vec, att_q)

    return det_vec


def convert_radec2thetaphi(ra, dec, att_q):
    """
    Converts ra, dec to the BAT theta, phi

    Parameters:
    ra: float in degrees
    dec: float in degrees
    att_q: the attitude quaternion (length 4 1D array)

    Returns:
    theta, phi: float in degrees
    """
    teldef_q = convertRotMatrixToQuat(align_mat)
    teldef_qinv = invertQuat(teldef_q)
    att_q = maintain_quat(att_q)

    sky_vec = radec2skyvec(ra, dec)
    det_vec = convertSkyToSensor(teldef_qinv, sky_vec, att_q)

    theta = np.rad2deg(np.arccos(det_vec[2]))
    phi = np.rad2deg(np.arctan2(-det_vec[1], det_vec[0]))
    if np.isscalar(phi):
        if phi < 0.0:
            phi += 360.0
    else:
        bl = phi < 0.0
        if np.sum(bl) > 0:
            phi[bl] += 360.0

    return theta, phi


def convert_radec2imxy(ra, dec, att_q):
    """
    Converts ra, dec to imx, imy

    Parameters:
    ra: float in degrees
    dec: float in degrees
    att_q: the attitude quaternion (length 4 1D array)

    Returns:
    imx: float
    imy: float
    """
    teldef_q = convertRotMatrixToQuat(align_mat)
    teldef_qinv = invertQuat(teldef_q)
    att_q = maintain_quat(att_q)

    sky_vec = radec2skyvec(ra, dec)
    det_vec = convertSkyToSensor(teldef_qinv, sky_vec, att_q)

    imx = det_vec[0] / det_vec[2]
    imy = det_vec[1] / det_vec[2]
    return imx, imy


def convert_theta_phi2radec(theta, phi, att_q):
    """
    Converts theta, phi to ra, dec

    Parameters:
    theta: float in degrees
    phi: float in degrees
    att_q: the attitude quaternion (length 4 1D array)

    Returns:
    ra: float in degrees
    dec: float in degrees
    """

    teldef_q = convertRotMatrixToQuat(align_mat)
    teldef_qinv = invertQuat(teldef_q)
    att_q = maintain_quat(att_q)

    # det_vec = imxy2detvec(imx, imy)
    det_vec = theta_phi2detvec(theta, phi)

    sky_vec = convertDetToSky(teldef_qinv, det_vec, att_q)

    ra, dec = skyvec2radec(sky_vec)

    return ra, dec


def convert_imxy2radec(imx, imy, att_q):
    """
    Converts imx, imy to ra, dec

    Parameters:
    imx: float
    imy: float
    att_q: the attitude quaternion (length 4 1D array)

    Returns:
    ra: float in degrees
    dec: float in degrees
    """

    teldef_q = convertRotMatrixToQuat(align_mat)
    teldef_qinv = invertQuat(teldef_q)
    att_q = maintain_quat(att_q)

    det_vec = imxy2detvec(imx, imy)

    sky_vec = convertDetToSky(teldef_qinv, det_vec, att_q)

    ra, dec = skyvec2radec(sky_vec)

    return ra, dec


def theta_phi2imxy(theta, phi):
    imr = np.tan(np.radians(theta))
    imx = imr * np.cos(np.radians(phi))
    imy = imr * np.sin(np.radians(-phi))
    return imx, imy


def imxy2theta_phi(imx, imy):
    theta = np.rad2deg(np.arctan(np.sqrt(imx**2 + imy**2)))
    phi = np.rad2deg(np.arctan2(-imy, imx))
    if np.isscalar(phi):
        if phi < 0:
            phi += 360.0
    else:
        bl = phi < 0
        if np.sum(bl) > 0:
            phi[bl] += 360.0
    return theta, phi


def pnt2euler(ra, dec, roll, roll_sign=-1.0, roll_offset=0.0):
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    psi = np.radians(roll * roll_sign + 90.0 - roll_offset)

    return phi, theta, psi


def convertEulerToRotMatrix(phi, theta, psi):
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    stheta = np.sin(theta)
    ctheta = np.cos(theta)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)

    rot = np.zeros((3, 3))

    rot[0, 0] = cpsi * ctheta * cphi - spsi * sphi
    rot[0, 1] = cpsi * ctheta * sphi + spsi * cphi

    rot[1, 0] = -spsi * ctheta * cphi - cpsi * sphi
    rot[1, 1] = -spsi * ctheta * sphi + cpsi * cphi

    rot[0, 2] = -cpsi * stheta
    rot[1, 2] = spsi * stheta

    rot[2, 0] = stheta * cphi
    rot[2, 1] = stheta * sphi

    rot[2, 2] = ctheta

    return rot


def convertEulerToQuat(phi, theta, psi):
    rot = convertEulerToRotMatrix(phi, theta, psi)

    q = convertRotMatrixToQuat(rot)

    return q


def pnt2quat(ra, dec, roll):
    """
    Converts pointing direciton to a quaternion

    Parameters:
    ra: float in degrees
    dec: float in degrees
    roll: float in degrees

    Returns:
    q: the attitude quaternion (length 4 1D array)
    """

    teldef_q = convertRotMatrixToQuat(align_mat)

    phi, theta, psi = pnt2euler(ra, dec, roll)

    q_realigned = convertEulerToQuat(phi, theta, psi)

    q = productOfQuats(q_realigned, teldef_q)

    return q
