#!/bin/bash


HEADAS=/home/gayathri/Softwares/heasoft-6.28/x86_64-pc-linux-gnu-libc2.31
export HEADAS
. $HEADAS/headas-init.sh

# CALDB stuff
CALDB=/home/gayathri/Softwares/CALDB; export CALDB
source $CALDB/software/tools/caldbinit.sh



$*
