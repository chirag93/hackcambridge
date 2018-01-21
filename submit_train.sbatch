#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -p gpu # or -p parallel

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on TNG Cluster
#---------------------------------------------------------------------
ml python/3.6.0
ml cuda/8.0

echo ${tempdatadir}
echo ${outputdatadir}

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
python ./main.py ${outputdatadir} ${tempdatadir}

exit