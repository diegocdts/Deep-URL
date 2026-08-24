#!/bin/bash

#SBATCH --partition=tc
#SBATCH -A mddlgp
#SBATCH -J DL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=100:00:00
#SBATCH --output=./slurms/Result-%x.%j.out
#SBATCH --error=./slurms/Result-%x.%j.err
#SBATCH --exclusive
#SBATCH --array=0-0%1

nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

DIR_SRC=${SLURM_SUBMIT_DIR}
DIR_DATA="/beegfs/gov7/jessica/data/Filtro_Hessiana"
DIR_CONT="/beegfs/gaia/tcs/ufrj_ml_u30s/containers"

srun singularity exec \
-B ${DIR_SRC}:/home/src \
-B ${DIR_DATA}:/home/data \
--nv ${DIR_CONT}/pytorch-ngc-digitalrockframework-14082024.sif \
python -u /home/src/workflow_test.py \
--config_idx $SLURM_ARRAY_TASK_ID