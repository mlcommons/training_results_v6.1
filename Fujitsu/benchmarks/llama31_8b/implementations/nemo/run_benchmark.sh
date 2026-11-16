#! /bin/bash
#
#

# check if P2P is enabled
#bash ./check-p2p.sh
#p2p_status=$?
#if [ $p2p_status -gt 0 ] ; then
#    echo "Please set CDI's P2P capability"
#    exit 1
#fi

# setup slurm command path
SLURM_HOME=/opt/slurm-26-05-pmix3
export PATH=${SLURM_HOME}/bin:${PATH}

#OMPI_HOME=/opt/ompi-5.0.10
#export LD_LIBRARY_PATH=${OMPI_HOME}/lib:${LD_LIBRARY_PATH}

PMIX3_HOME=/opt/pmix-3.2.5/lib
export LD_LIBRARY_PATH=${PMIX3_HOME}/lib:${LD_LIBRARY_PATH}

# setup logdir
logname=log_$(date +%Y%m%d_%H%M%S)
logdir=./logs/${logname}

mkdir ${logdir}

export DATADIR="/mnt/data4/work/training" # set your </path/to/dataset>
#export MODEL="/mnt/data4/work/llama2-v60/model"  # set your </path/to/dataset>
export LOGDIR="$(realpath ${logdir})"  # set the place where the output logs will be saved
export CONT=/home/notsu/tr_6.1/sqsh/llama31_tr60_nv.sqsh
export NEXP=10
source configs/config_PGCDI_1x8x2xtp1pp1cp1_8b.sh

sbatch -N $DGXNNODES -t $WALLTIME -o outputs/slurm-%j.out run.sub  # you may be required to set --account and --partition here
