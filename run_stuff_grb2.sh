#!/bin/bash

batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

workdir='/gpfs/scratch/jjd330/bat_data/'
workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'
ratespbs='/storage/work/jjd330/local/bat_data/BatML/submission_scripts/pbs_rates_fp_realtime.pbs'

drmdir='/storage/home/j/jjd330/work/local/bat_data/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

HEADAS=/storage/work/jjd330/heasoft/heasoft-6.24/x86_64-pc-linux-gnu-libc2.12
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/storage/work/jjd330/caldb_files; export CALDB
source $CALDB/software/tools/caldbinit.sh

export HEADASNOQUERY=
export HEADASPROMPT=/dev/null

export PFILES="/tmp/$$.tmp/pfiles;$HEADAS/syspfiles"

# evfname=$1
# dmask=$2
# attfname=$3
# trigtime=$4
# gwname=$5

trigtime=$1
gwname=$2
# if [ "$#" -ne 2 ]; then
#     nimgs=$3
# else
#     nimgs=60
# fi

Nratejobs=16
twind=20.0
tmin=-20.0
Ntdbls=6

# $Njobs=

workdir=$workdir$gwname
if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

if [ "$#" -ne 2 ]; then
    mintbin=$3
else
    mintbin=0.256
fi


echo $trigtime
echo $workdir
echo $Nratejobs
echo $twind
echo $mintbin

cd $batml_path

curdir=$(pwd)

echo $curdir

echo $$ > $workdir'/run_stuff.pid'

python mkdb.py --trig_time $trigtime --work_dir $workdir --drm_dir $drmdir

cd $workdir
#cd $obsid

python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --search_twind $twind --min_dt $tmin --Ntdbls $Ntdbls --min_tbin $mintbin
if [ -f "filter_evdata.fits" ]; then
    # python $batml_path'do_bkg_estimation.py' > bkg_estimation_out.log 2>&1
    # python $batml_path'do_bkg_estimation_wSA.py' --twind $twind > bkg_estimation_out.log 2>&1
    # python $sub_path'submit_jobs.py' --Njobs $Nratejobs --workdir $workdir --name $gwname --ssh --pbs_fname $ratespbs > submit_jobs.log 2>&1 &
    # python $sub_path'submit_jobs.py' --Njobs $Nratejobs --workdir $workdir --name $gwname --pbs_fname $ratespbs > submit_jobs.log 2>&1 &
    # python $batml_path'do_manage.py' --Nrate_jobs $Nratejobs --GWname $gwname > manager.out 2>&1 &
    python $batml_path'do_full_rates.py' --min_tbin $mintbin > full_rates.out 2>&1 &
    python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh > manager.out 2>&1 &
fi

cd $curdir
