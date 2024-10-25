#!/bin/bash

batml_path='/gpfs/group/jak51/default/nitrates_realtime/NITRATES/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

#workdir='/gpfs/scratch/jjd330/bat_data/'
#workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'
workdir='/gpfs/group/jak51/default/realtime_workdir/'
ratespbs='/gpfs/group/jak51/default/nitrates_realtime/NITRATES/submission_scripts/pbs_rates_fp_realtime.pbs'

drmdir='/gpfs/group/jak51/default/responses/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

#HEADAS=/storage/work/jjd330/heasoft/heasoft-6.24/x86_64-pc-linux-gnu-libc2.12
#export HEADAS
#. $HEADAS/headas-init.sh

# CALDB stuff
#CALDB=/storage/work/jjd330/caldb_files; export CALDB
#source $CALDB/software/tools/caldbinit.sh

HEADAS=/storage/work/jjd330/heasoft/heasoft-6.21/x86_64-unknown-linux-gnu-libc2.12
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

queue='jak51_b_g_bc_default'
NinFOVjobs=260
NoutFOVjobs=40

workdir=$workdir$gwname

if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

if [ "$#" -ne 5 ]; then
    mintbin=$6
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

python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --search_twind $twind --min_dt $tmin --Ntdbls $Ntdbls --min_tbin $mintbin --evfname $evfname --dmask $dmask --att_fname $attfname --acs_fname $attfname
if [ -f "filter_evdata.fits" ]; then

    python $batml_path'do_full_rates.py' --min_tbin $mintbin > full_rates.out 2>&1 &
    #python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh --queue $queue --N_infov_jobs $NinFOVjobs --N_outfov_jobs $NoutFOVjobs > manager.out 2>&1 &
    qsub /gpfs/group/jak51/default/nitrates_realtime/NITRATES/submission_scripts/launch_g3.pbs -v workdir=$workdir,gwname=$gwname -A jak51_b_g_bc_default -N launch -l feature=rhel7

fi

cd $curdir
