#!/bin/bash

batml_path='/storage/home/jjd330/work/local/NITRATES/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

workdir='/gpfs/scratch/jjd330/bat_data/'
#workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'

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

twind=20.0
tmin=-20.0
Ntdbls=6

queue='jak51_b_g_bc_default'
NinFOVjobs=260
NoutFOVjobs=40

# $Njobs=

workdir=$workdir$gwname
if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

if [ "$#" -ne 2 ]; then
    mintbin=$3
else
    mintbin=0.128
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
    python $batml_path'do_full_rates.py' --min_tbin $mintbin > full_rates.out 2>&1 &
    python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh --queue $queue --N_infov_jobs $NinFOVjobs --N_outfov_jobs $NoutFOVjobs > manager.out 2>&1 &
fi

cd $curdir
