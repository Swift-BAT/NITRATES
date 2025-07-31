#!/bin/bash

module load python/2.7.14-anaconda5.0.1
source activate myenv
batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'
export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH
# batml_path='/storage/work/jjd330/local/bat_data/BatML/'
# ht_path=$batml_path'HeasoftTools/'
sub_path=$batml_path'submission_scripts/'

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


workdir=$1
evfname=$2
dmaskfname=$3
attfname=$4
twind=$5

echo $workdir
echo $evfname
echo $dmaskfname
echo $attfname

curdir=$(pwd)

echo $curdir

cd $batml_path

echo $$ > $workdir'/setup_analysis.pid'

python mkdb.py --work_dir $workdir

cd $workdir

python $batml_path'do_data_setup_archive.py' --work_dir $workdir --evfname $evfname --dmask $dmaskfname --att_fname $attfname --search_twind $twind

python $batml_path'do_bkg_estimation_wPSs_mp.py' --archive --twind $twind > bkg_out.log 2>&1

cd $curdir
