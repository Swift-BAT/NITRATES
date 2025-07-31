from astropy.time import Time
from astropy.io import fits


def get_t_offset(att):
    if isinstance(att, str):
        att = fits.open(att)

    head = att[0].Header

    utcf = head["UTCFINIT"]

    return utcf


def met2mjd(times, fn=None, utcf=None, mjdref=None):
    if fn is not None:
        if isinstance(fn, str):
            fn = fits.open(fn)

        try:
            head = fn[1].header
        except:
            head = fn[0].header
        utcf = head["UTCFINIT"]
        mjdref = head["MJDREFI"]
    elif utcf is None or mjdref is None:
        print("Need to enter either file or utcf and mjdref")
        return

    mjds = mjdref + (times + utcf) / 86400.0

    return mjds


def met2astropy(times, fn, utcf=None, mjdref=51910.0):
    mjds = met2mjd(times, fn, utcf=utcf, mjdref=mjdref)

    ats = Time(mjds, format="mjd")

    return ats


def met2utc_str(times, fn, utcf=None):
    ats = met2astropy(times, fn)

    return ats.iso


def apy_time2met(apy_time, fn, utcf=None, mjdref=51910.0):
    mjds = apy_time.mjd

    if utcf is None:
        try:
            fn_ = fits.open(fn)
        except:
            fn_ = fn

        head = fn_[1].header
        utcf = head["UTCFINIT"]
        mjdref = head["MJDREFI"]

    mets = (mjds - mjdref) * 86400.0 - utcf

    return mets


def utc2met(utc_str, fn, utcf=None, mjdref=51910.0):
    ats = Time(utc_str, format="isot")

    mjds = ats.mjd

    if utcf is None:
        try:
            fn_ = fits.open(fn)
        except:
            fn_ = fn

        head = fn_[1].header
        utcf = head["UTCFINIT"]
        mjdref = head["MJDREFI"]

    mets = (mjds - mjdref) * 86400.0 - utcf

    return mets
