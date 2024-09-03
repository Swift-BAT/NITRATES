from swifttools.swift_too import Data, Clock, ObsQuery
from urllib.request import urlretrieve

def get_sao_file(trigtime):
    '''
    args
    trigtime: isot string

    Returns:
    fname: file name of sao file after being downloaded
    '''

    filename_suffix = "sao.fits.gz"

    obsid = None

    if trigtime[-1] != 'z':
        trigtime += 'z'

    afst = ObsQuery(begin=trigtime)
    if len(afst) == 1:
        obsid = afst[0].obsid
    print(obsid)

        
    d = Data()
    d.obsid = obsid
    d.bat = True
    d.match = f"*{filename_suffix}"
    d.uksdc = True

    if d.submit():
        if len(d.entries) == 1:
            urlretrieve(d[0].url, d[0].filename)
            fname = d[0].filename

    return fname
