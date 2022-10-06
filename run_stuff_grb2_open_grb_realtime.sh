#!/bin/bash

batml_path='/storage/home/gzr5209/work/BatML_code_work/NITRATES/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

#workdir='/storage/home/gzr5209/work/bat-data/'
#workdir='/storage/home/gzr5209/work/realtime_workdir/result/'
#workdir='/gpfs/group/jak51/default/gzr5209/realtime_results/'
#workdir='/storage/home/gzr5209/work/realtime_workdir_NITRATES/'
workdir='/storage/work/g/gzr5209/realtime_workdir_NITRATES/'

ratespbs='/storage/home/gzr5209/work/BatML_code_work/NITRATES/submission_scripts/pyscript_template_rhel7_g3_10gb.pbs'

drmdir='/storage/home/gzr5209/work/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH


HEADAS=/storage/home/gzr5209/work/Softwares/heasoft/heasoft-6.28/x86_64-pc-linux-gnu-libc2.17
export HEADAS
alias heainit='/storage/home/gzr5209/work/Softwares/heasoft/heasoft-6.28/x86_64-pc-linux-gnu-libc2.17/headas-init.sh'

. $HEADAS/headas-init.sh


export CALDB=/storage/home/gzr5209/work/Softwares/CALDB
source $CALDB/software/tools/caldbinit.sh

export HEADASNOQUERY=
export HEADASPROMPT=/dev/null

export PFILES="/tmp/$$.tmp/pfiles;$HEADAS/syspfiles"


#HEADAS=/storage/work/jjd330/heasoft/heasoft-6.24/x86_64-pc-linux-gnu-libc2.12
#export HEADAS
#. $HEADAS/headas-init.sh

# CALDB stuff
#CALDB=/storage/work/jjd330/caldb_files; export CALDB
#source $CALDB/software/tools/caldbinit.sh

#export HEADASNOQUERY=
#export HEADASPROMPT=/dev/null

#export PFILES="/tmp/$$.tmp/pfiles;$HEADAS/syspfiles"

# evfname=$1
# dmask=$2
# attfname=$3
# trigtime=$4
# gwname=$5

trigtime=$1
gwname=$2


#------------------------------------
#Adding these lines on Jan 21st, 2021

# first arg is the trigger time in either isot or MET
trigtime=$1
# gwname is the name of the event/analysis
# it'll be the name of the directory inside workdir and used for job names
gwname=$2
# file name of the event file to use
#evfname=$3 #com out Apr 19th 
# file name of the detmask to use
#dmask=$4 # com out Apr 19th
# file name of the attitude to use
#attfname=$5 # com out Apr 19th
# file names can also be urls
#-----------------------------------



# if [ "$#" -ne 2 ]; then
#     nimgs=$3
# else
#     nimgs=60
# fi

Nratejobs=16 # commenting out on Jan 24th 2022
twind=20.0
tmin=-20.0
Ntdbls=6

# $Njobs=

workdir=$workdir$gwname
if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

if [ "$#" -ne 5 ]; then
    mintbin=$3 # edited from 6 to 3 here Apr 19th
else
    mintbin=0.256
fi


#mintbin=0.256

echo $trigtime
echo $workdir
echo $Nratejobs
echo $twind
#echo 'mintbin=' $mintbin

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
 
    python $batml_path'do_manage2_open.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh --queue open --N_infov_jobs 96 --N_outfov_jobs 24 > manager.out 2>&1 &
    #python $batml_path'do_manage2.py' --GWname $gwname --rhel7 --do_bkg --do_rates --do_llh > manager.out 2>&1 &
fi

cd $curdir
