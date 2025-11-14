#!/bin/bash
#SBATCH --job-name=tta
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=$ATHENA_GRANT
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=0-4:00:00
#SBATCH --output=logs/%j.out

if [ $# -lt 1 ]; then
  echo "Usage: $0 <script> [arg1] [arg2]..."
  exit 1
fi

pwd;hostname;date

set -e

mkdir -p logs/

script_name=$1
extension="${script_name##*.}"
# shift arguments to leave only the command line arguments for the script
shift 

source .env
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# CUDA
module load CUDA/12.4.0
echo $CUDA_HOME
export CUDA_LAUNCH_BLOCKING=1

if [ "$extension" == "py" ]; then
    python $script_name "$@"
elif [ "$extension" == "sh" ]; then
    bash $script_name "$@"
else
    echo "Unsupported file format: .$extension"
    exit 1
fi
