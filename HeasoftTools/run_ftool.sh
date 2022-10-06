#!/bin/bash
#HEADAS=/storage/work/jjd330/heasoft/heasoft-6.25/x86_64-pc-linux-gnu-libc2.12
#export HEADAS
#. $HEADAS/headas-init.sh

#CALDB stuff
#CALDB=/storage/work/jjd330/caldb_files; export CALDB
#source $CALDB/software/tools/caldbinit.sh

#HEADAS=/storage/work/jjd330/heasoft/heasoft-6.21/x86_64-unknown-linux-gnu-libc2.12
#export HEADAS
#. $HEADAS/headas-init.sh


HEADAS=/storage/home/gzr5209/work/Softwares/heasoft/heasoft-6.28/x86_64-pc-linux-gnu-libc2.17
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/storage/home/gzr5209/work/Softwares/CALDB; export CALDB
source $CALDB/software/tools/caldbinit.sh


$*
