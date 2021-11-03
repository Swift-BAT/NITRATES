#!/bin/bash

# run as
# . run_stuff_grb2_open_queue_wfiles.sh triggertime name event_file_name detmask_file_name attitude_file_name
# or usually more convenient to use nohup and run in the background
# nohup run_stuff_grb2_open_queue_wfiles.sh triggertime name event_file_name detmask_file_name attitude_file_name > run_stuff.out 2>&1 &


# batml_path should be wherever the code is
batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

# workdir here is where the analysis directory for this search will be made
workdir='/gpfs/scratch/jjd330/bat_data/'
workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'

# pbs file used by do_manage to submit analysis jobs to cluster
pbsfname='/storage/work/j/jjd330/local/bat_data/BatML/submission_scripts/pyscript_template_rhel7.pbs'

drmdir='/storage/home/j/jjd330/work/local/bat_data/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

# HEASOFT stuff
HEADAS=/storage/work/jjd330/heasoft/heasoft-6.24/x86_64-pc-linux-gnu-libc2.12
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/storage/work/jjd330/caldb_files; export CALDB
source $CALDB/software/tools/caldbinit.sh

export HEADASNOQUERY=
export HEADASPROMPT=/dev/null

export PFILES="/tmp/$$.tmp/pfiles;$HEADAS/syspfiles"


# first arg is the trigger time in either isot or MET
trigtime=$1
# gwname is the name of the event/analysis
# it'll be the name of the directory inside workdir and used for job names
gwname=$2
# file name of the event file to use
evfname=$3
# file name of the detmask to use
dmask=$4
# file name of the attitude to use
attfname=$5
# file names can also be urls

twind=20.0
tmin=-20.0
Ntdbls=6

# $Njobs=

# new workdir is the old workdir/gwname
workdir=$workdir$gwname
if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

# if a 6th arg is given, that's used as the min time bin used
# if not it defaults to 0.256s
if [ "$#" -ne 5 ]; then
    mintbin=$6
else
    mintbin=0.256
fi


echo $trigtime
echo $workdir
echo $twind
echo $mintbin

cd $batml_path

curdir=$(pwd)

echo $curdir

# write PID to file if you want to kill it
echo $$ > $workdir'/run_stuff.pid'

python mkdb.py --trig_time $trigtime --work_dir $workdir --drm_dir $drmdir

cd $workdir
#cd $obsid

python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --search_twind $twind --min_dt $tmin --Ntdbls $Ntdbls --min_tbin $mintbin --evfname $evfname --dmask $dmask --att_fname $attfname --acs_fname $attfname
if [ -f "filter_evdata.fits" ]; then
    # python $batml_path'do_bkg_estimation.py' > bkg_estimation_out.log 2>&1
    # python $batml_path'do_bkg_estimation_wSA.py' --twind $twind > bkg_estimation_out.log 2>&1
    # python $sub_path'submit_jobs.py' --Njobs $Nratejobs --workdir $workdir --name $gwname --ssh --pbs_fname $ratespbs > submit_jobs.log 2>&1 &
    # python $sub_path'submit_jobs.py' --Njobs $Nratejobs --workdir $workdir --name $gwname --pbs_fname $ratespbs > submit_jobs.log 2>&1 &
    # python $batml_path'do_manage.py' --Nrate_jobs $Nratejobs --GWname $gwname > manager.out 2>&1 &
    python $batml_path'do_full_rates.py' --min_tbin $mintbin --archive > full_rates.out 2>&1 &
    python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh --archive --queue open --pbs_fname $pbsfname --pbs_rhel7_fname $pbsfname > manager.out 2>&1 &
fi

cd $curdir
