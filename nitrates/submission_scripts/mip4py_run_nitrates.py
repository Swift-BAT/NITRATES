import nitrates as nt
from mpi4py import MPI
import numpy
import sys
import argparse

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evfname', type=str,\
            help="Event data file",
            default=None)
    parser.add_argument('--dmask', type=str,\
            help="Detmask fname",
            default=None)
    parser.add_argument('--dbfname', type=str,\
            help="Name to save the database to",\
            default=None)
    parser.add_argument('--job_id', type=int,\
            help="Job ID",\
            default=0)
    parser.add_argument('--Njobs', type=int,\
            help="Number of jobs",\
            default=1)
    parser.add_argument('--twind', type=float,\
            help="Number of seconds to go +/- from the trigtime",\
            default=20*1.024)
    parser.add_argument('--min_dt', type=float,\
            help="Min time from trigger to do",\
            default=None)
    parser.add_argument('--bkg_dur', type=float,\
            help="bkg duration",\
            default=60.0)
    parser.add_argument('--archive',\
            help="Adjust for longer event duration",\
            action='store_true')
    parser.add_argument('--bkg_nopost',\
            help="Don't use time after signal window for bkg",\
            action='store_true')
    parser.add_argument('--bkg_nopre',\
            help="Don't use time before signal window for bkg",\
            action='store_true')
    parser.add_argument('--min_tbin', type=float,\
            help="Smallest tbin size to use",\
            default=0.256)
    parser.add_argument('--max_tbin', type=float,\
            help="Largest tbin size to use",\
            default=0.256*(2**6))
    parser.add_argument('--snr_min', type=float,\
            help="Min snr cut for time seeds",\
            default=2.5)

    args = parser.parse_args()
    return args



main(args):

    #get MPI rand and its place in the processes
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    #set some default variables for all processes
    has_sky_map = False
    conn = None
    info_tab = None
    trigtime = -1
    attfname = ''
    evfname = ''
    
    #only have proc 0 do these since they are single core jobs
    if rank == 0:
        pass
    #Step 1. Do the full rates Light curve creation/SNR calc & cutoff
        nt.analysis_seeds.main(args)
    
    #Step 2. Get the skymap info
        
        try:
            sky_map_fnames = [fname for fname in os.listdir('.') if\
                            'cWB.fits.gz' in fname or 'bayestar' in fname\
                            or 'skymap' in fname]
            sky_map = hp.read_map(sky_map_fnames[0], field=(0,), nest=True)
            logging.info('Opened sky map')
            perc_map = probm2perc(sky_map)
            logging.info('Made perc map')
            has_sky_map = True
        except Exception as E:
            logging.warning("problem reading skymap")
            logging.error(E)
            logging.error(traceback.format_exc())

    
    #Step 3. Get info from the database
        try:
            logging.info('Connecting to DB')
            if args.dbfname is None:
                db_fname = guess_dbfname()
                if isinstance(db_fname, list):
                    db_fname = db_fname[0]
            else:
                db_fname = args.dbfname
            conn = get_conn(db_fname)
            info_tab = get_info_tab(conn)
            logging.info('Got info table')
            trigtime = info_tab['trigtimeMET'][0]
            files_tab = get_files_tab(conn)
            logging.info('Got files table')
            attfname = files_tab['attfname'][0]
            evfname = files_tab['evfname'][0]

        except Exception as E:
            logging.warning("problem getting files tab from DB")
            logging.error(E)
            logging.error(traceback.format_exc())
            attfname = 'attitude.fits'
            evfname = 'filter_evdata.fits'

    
    #Step 4. Read in the attitude file
    
    #Step 5. Get the GTI information
    
    #Step 6. Determine if there are any "good pixels", if not then the trigger probaly happened outside the FOV
    
    #Step 7. Do the background estimation (this is currently done with a single process)
    else:
        #other procs get the info from rank 0 proc
        pass
    
    #Step 8. Do the rates analysis
    #How to break this up and send up peices of the problem to each rank
    
    #Step 9. Aggregate all the rate analysis data and save to a CSV file
    
    #Step 10. Send email with info, single process should do this
    if rank == 0:
        pass
    
    #Step 11. Create the position seeds for the LLH analysis
    #How to break up the position seeds among all ranks
    
    #Step 12. Do the inFOV and outFOV calculations
    # A given rank has a tiny part of the position space to conduct its calculations over
    
    #Step 13. Aggregate results with RA/DEC info and save to a CSV file
    #Saving to CSV can be a single process task
    
    return 0
    
if __name__ == "__main__":

    args = cli()

    main(args)
