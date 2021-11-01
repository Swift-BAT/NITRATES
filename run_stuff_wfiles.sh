#!/bin/bash

batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

workdir='/gpfs/scratch/jjd330/bat_data/'
workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'

drmdir='/storage/home/j/jjd330/work/local/bat_data/drms/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

HEADAS=/storage/work/jjd330/heasoft/heasoft-6.24/x86_64-pc-linux-gnu-libc2.12
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/storage/work/jjd330/caldb_files; export CALDB
source $CALDB/software/tools/caldbinit.sh

evfname=$1
dmask=$2
attfname=$3
trigtime=$4
gwname=$5

workdir=$workdir$gwname
if [ ! -d "$workdir" ]; then
    echo "Making the directory"
    mkdir $workdir
fi

echo $evfname
echo $dmask
echo $attfname
echo $trigtime
echo $workdir

cd $batml_path

curdir=$(pwd)

echo $curdir

echo $$ > $workdir'/run_stuff.pid'

python mkdb.py --trig_time $trigtime --work_dir $workdir --drm_dir $drmdir

cd $workdir
#cd $obsid

python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --evfname $evfname --dmask $dmask --att_fname $attfname

python $sub_path'submit_jobs.py' --workdir $workdir --name $gwname > submit_jobs.log 2>&1 &
python $batml_path'do_bkg_estimation.py' > bkg_estimation_out.log 2>&1
python $batml_path'do_rates_mle.py' > rates_mle_out.log 2>&1
python $batml_path'do_blip_search.py' --Nimgs 60 > blip_search_out.log 2>&1 &
python $batml_path'assign_seeds2jobs.py' > assign_seeds_out.log 2>&1 &

cd $curdir
