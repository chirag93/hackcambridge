#!/bin/bash

#SBATCH
#SBATCH -p gpu # or -p parallel
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on TNG Cluster
#---------------------------------------------------------------------
module unload git
source activate dnn

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
ml python/3.6.0
# ml cuda/8.0
# ml cuda/6.0
ml cuda

# grep for SLURM_EXPORT_ENV
echo ${tempdatadir}
echo ${outputdatadir}
echo ${CUDA_VISIBLE_DEVICES}

# create log directory 
logdir='_logs'
if [ -d "$logdir" ]; then  
	echo "log directory exists!"
else
	mkdir $logdir
fi
# set jobname
logfilename="submit_trainpy.log"

printf "Running training model"
python ./main.py ${outputdatadir} ${tempdatadir} ${CUDA_VISIBLE_DEVICES}

exit