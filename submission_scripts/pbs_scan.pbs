#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=6:00:00
#PBS -l mem=10gb

echo "#-#-#Job started on `hostname` at `date` "
echo This job runs on the following processors:
echo `cat $PBS_NODEFILE`

module load python/2.7.14-anaconda5.0.1

source activate myenv

batml_path='/storage/work/jjd330/local/bat_data/BatML/'
ht_path=$batml_path'HeasoftTools/'

export PYTHONPATH=$batml_path:$PYTHONPATH
export PYTHONPATH=$ht_path:$PYTHONPATH

echo ${jobid}
echo ${workdir}

cd ${workdir}

python $batml_path'do_signal_llhs_scan.py' --job_id ${jobid} --posfname /gpfs/scratch/jjd330/bat_data/S190927an4scan.npz --dbfname /storage/work/jjd330/local/bat_data/realtime_workdir/S190927an/results.db --dt0 12.51 --rt_dir /gpfs/scratch/jjd330/bat_data/ray_traces_detapp_npy/

echo "#-#-#Job Ended at `date`"
