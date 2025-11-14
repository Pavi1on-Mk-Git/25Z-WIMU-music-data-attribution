#!/bin/bash
eval "$(conda shell.bash hook)"
source .env
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n $CONDA_ENV_NAME python=3.11
  conda activate $CONDA_ENV_NAME
  mkdir pip-build
  TMPDIR=pip-build pip --no-input --no-cache-dir install torch torchvision torchaudio
  module load CUDA/12.4.0
  echo $CUDA_HOME
  module load GCC/12.3.0
  nvcc --version
  gcc --version
  TMPDIR=pip-build pip --no-input --no-cache-dir pip install traker[fast]
  TMPDIR=pip-build pip --no-input --no-cache-dir matplotlib jupyter
  rm -rf pip-build
  conda env export | grep -v "^prefix: " > environment.yml
fi