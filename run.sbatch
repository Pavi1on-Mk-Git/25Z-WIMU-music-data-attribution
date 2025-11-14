#!/bin/bash
#SBATCH --job-name=trak
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=0-04:00:00
#SBATCH --array=0-3  # <NUMBER OF CHECKPOINTS>
#SBATCH --output=logs/%j.out

MODEL_ID=$SLURM_ARRAY_TASK_ID

# environment setup
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# CUDA
module load CUDA/12.4.0
echo $CUDA_HOME
export CUDA_LAUNCH_BLOCKING=1

python featurize_and_score.py --model_id $MODEL_ID