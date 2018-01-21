#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on TNG Cluster
#---------------------------------------------------------------------
ml python

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