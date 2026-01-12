#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 audiocraft_repo_dir" >&2
    echo "Error: Expected 1 argument, got $#." >&2
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Activate audiocraft's virtual environment before running the script." >&2
    exit 1
fi

if [ -z "$AUDIOCRAFT_DORA_DIR" ]; then
    echo "Error: Set AUDIOCRAFT_DORA_DIR environment variable before running the script." >&2
    exit 1
fi

AUDIOCRAFT_REPO_DIR="$1"
MAIN_REPO_DIR=$(pwd)
DEBUG_SET_FILE="$AUDIOCRAFT_REPO_DIR/egs/musiccaps/debug/data.jsonl"

if [ ! -f "$DEBUG_SET_FILE" ]; then
    echo "Error: First prepare MusicCaps debug split file in $DEBUG_SET_FILE" >&2
    exit 1
fi

DEBUG_SET_SIZE=$(wc -l < "$DEBUG_SET_FILE")

BATCH_SIZE=4
DEBUG_SET_SIZE_BATCHES=$(python3 -c "import math; print(math.ceil($DEBUG_SET_SIZE / $BATCH_SIZE))")

cd "$AUDIOCRAFT_REPO_DIR"

echo '# @package __global__

datasource:
  max_sample_rate: 48000
  max_channels: 2

  train: egs/musiccaps/debug
  valid: egs/musiccaps/debug
  evaluate: egs/musiccaps/debug
  generate: egs/musiccaps/debug
' > config/dset/audio/musiccaps_custom.yaml

dora run solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small \
    conditioner=text2music \
    dset=audio/musiccaps_custom \
    dataset.num_workers=2 \
    dataset.train.num_samples="$DEBUG_SET_SIZE" \
    dataset.valid.num_samples="$DEBUG_SET_SIZE" \
    dataset.batch_size="$BATCH_SIZE" \
    schedule.cosine.warmup=8 \
    optim.optimizer=adamw \
    optim.lr=2e-5 \
    optim.max_norm=1e-3 \
    optim.epochs=60 \
    optim.updates_per_epoch="$DEBUG_SET_SIZE_BATCHES" \
    optim.adam.weight_decay=0.01 \
    generate.every=null \
    dataset.generate.num_samples=0
