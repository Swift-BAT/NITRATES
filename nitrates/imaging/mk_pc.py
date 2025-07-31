from ..lib.funcs2run_bat_tools import do_pc

if __name__ == "__main__":
    do_pc("detmask.fits", "attitude.fits", ".", ovrsmp=2, detapp=True)
