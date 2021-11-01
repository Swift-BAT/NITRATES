#!/bin/bash

batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

workdir='/gpfs/scratch/jjd330/bat_data/archive_workdir/'
# workdir='/storage/home/j/jjd330/work/local/bat_data/realtime_workdir/'

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
idname=$5

if [ "$#" -ne 5 ]; then
    njobs=$6
else
    njobs=20
fi


workdir=$workdir$idname
if [ ! -d "$workdir" ]; then
  mkdir $workdir
fi

echo "evfname: "$evfname
echo "dmask: "$dmask
echo "attfname: "$attfname
echo "trigtime: "$trigtime
echo "workdir: "$workdir
echo "njobs: "$njobs

cd $batml_path

curdir=$(pwd)

echo $curdir

echo $$ > $workdir'/run_stuff.pid'

#python mkdb.py --trig_time $trigtime --work_dir $workdir --drm_dir $drmdir

cd $workdir
#cd $obsid

#python $batml_path'do_data_setup.py' --work_dir $workdir --trig_time $trigtime --evfname $evfname --dmask $dmask --att_fname $attfname

#python $sub_path'submit_jobs.py' --workdir $workdir --name $idname --Njobs $njobs --pbs_fname /storage/work/jjd330/local/bat_data/BatML/submission_scripts/pbs_archive.pbs > submit_jobs.log 2>&1 &
#python $batml_path'do_bkg_estimation.py' > bkg_estimation_out.log 2>&1
#python $batml_path'do_rates_mle.py' --nproc 2 > rates_mle_out.log 2>&1
python $batml_path'do_sig_sky_imgs.py' --Nproc 2 > img_sig_out.log 2>&1 &
#python $batml_path'do_blip_search.py' --Nimgs 120 --Nproc 2 > blip_search_out.log 2>&1 &
#python $batml_path'assign_seeds2jobs.py' --njobs $njobs > assign_seeds_out.log 2>&1 &

cd $curdir
