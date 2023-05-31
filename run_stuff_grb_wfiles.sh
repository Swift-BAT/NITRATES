#!/bin/bash

batml_path='/home/shared/nitrates_new/NITRATES/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

workdir='/home/shared/realtime_workdir/'
#workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'
# Need to figure out pbs path later
#ratespbs='/storage/work/jjd330/local/bat_data/BatML/submission_scripts/pbs_rates_fp_realtime.pbs'

drmdir='/home/shared/response/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

HEADAS=/home/gayathri/Softwares/heasoft-6.28/x86_64-pc-linux-gnu-libc2.31
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/home/gayathri/Softwares/CALDB; export CALDB
source $CALDB/software/tools/caldbinit.sh

export HEADASNOQUERY=
export HEADASPROMPT=/dev/null

export PFILES="/tmp/$$.tmp/pfiles;$HEADAS/syspfiles"

#workdir=$1
#evfname=$2
#dmask=$3
#attfname=$4
#trigtime=$5
#gwname=$6


# My convention - g3
trigtime=$1
gwname=$2
evfname=$3
dmask=$4
attfname=$5

#export gwname

twind=20.0
tmin=-20.0
Ntdbls=6

#queue='jak51_b_g_bc_default'
NinFOVjobs=120
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
#echo $Nratejobs
echo $twind
echo $mintbin

cd $batml_path

curdir=$(pwd)

echo $curdir

echo $$ > $workdir'/run_stuff.pid'

python mkdb.py --trig_time $trigtime --work_dir $workdir --drm_dir $drmdir

echo 'finished mkdb.py' 
cd $workdir
#cd $obsid

python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --search_twind $twind --min_dt $tmin --Ntdbls $Ntdbls --min_tbin $mintbin --evfname $evfname --dmask $dmask --att_fname $attfname --acs_fname $attfname
if [ -f "filter_evdata.fits" ]; then
    python $batml_path'do_full_rates.py' --min_tbin $mintbin > full_rates.out 2>&1 &
    python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh --N_infov_jobs $NinFOVjobs --N_outfov_jobs $NoutFOVjobs> manager.out 2>&1 &
   # condor_submit /home/shared/nitrates_new/NITRATES/submission_scripts/condor_submit_do_manage.sub
fi

cd $curdir
