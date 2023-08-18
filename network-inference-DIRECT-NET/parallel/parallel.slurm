#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --output=python_array_job_slurm_%A_%a.out
#SBATCH --array=1
#SBATCH --mail-user=sarah.f.maddox@vanderbilt.edu
#SBATCH --time=240:00:00


ml Intel/2017.4.196 IntelMPI Anaconda3/5.0.1
source activate boba_accre

config=/data/qlab/smgroves/parallel_bb/parallel/commands_test.txt

network=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
data=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
cluster=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
norm=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
threshold=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)

echo "This is array task ${SLURM_ARRAY_TASK_ID}, ${network} ${data} ${cluster} ${norm} ${threshold}"
python3 -u parallel_rule_builder.py '/data/qlab/smgroves/parallel_bb/parallel/booleabayes_results/' ${network} ${data} ${cluster} ${norm} ${threshold}