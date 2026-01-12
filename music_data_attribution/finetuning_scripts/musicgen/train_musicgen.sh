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
FULL_TRAIN_SET_FILE="$AUDIOCRAFT_REPO_DIR/egs/musiccaps/train/data.jsonl"
FULL_TEST_SET_FILE="$AUDIOCRAFT_REPO_DIR/egs/musiccaps/test/data.jsonl"

if [ ! -f "$FULL_TRAIN_SET_FILE" ] || [ ! -f "$FULL_TEST_SET_FILE" ]; then
    echo "Error: First prepare full MusicCaps train split file in $FULL_TRAIN_SET_FILE and test split file in $FULL_TEST_SET_FILE" >&2
    exit 1
fi

FULL_TRAIN_SET_SIZE=$(wc -l < "$FULL_TRAIN_SET_FILE")
FULL_TEST_SET_SIZE=$(wc -l < "$FULL_TEST_SET_FILE")
TRAIN_SET_FRACTION=0.5
TRAIN_SET_SIZE=$(python3 -c "print(round($TRAIN_SET_FRACTION * $FULL_TRAIN_SET_SIZE))")

BATCH_SIZE=4
TRAIN_SET_SIZE_BATCHES=$(python3 -c "import math; print(math.ceil($TRAIN_SET_SIZE / $BATCH_SIZE))")
FULL_TRAIN_SET_SIZE_BATCHES=$(python3 -c "import math; print(math.ceil($FULL_TRAIN_SET_SIZE / $BATCH_SIZE))")

cd "$AUDIOCRAFT_REPO_DIR"

echo '# @package __global__

datasource:
  max_sample_rate: 48000
  max_channels: 2

  train: egs/musiccaps/train
  valid: egs/musiccaps/test
  evaluate: egs/musiccaps/test
  generate: egs/musiccaps/test
' > config/dset/audio/musiccaps_custom.yaml

for seed in {1..10}; do
    TRAIN_SET_DIR="egs/musiccaps/train_$seed"
    mkdir -p "$TRAIN_SET_DIR"

    python3 "$MAIN_REPO_DIR"/music_data_attribution/finetuning_scripts/musicgen/sample_train_set.py \
        "$FULL_TRAIN_SET_FILE" \
        --output "$TRAIN_SET_DIR/data.jsonl" \
        --fraction "$TRAIN_SET_FRACTION" \
        --seed "$seed"

    dora run solver=musicgen/musicgen_base_32khz \
        model/lm/model_scale=small \
        continue_from=//pretrained/facebook/musicgen-small \
        conditioner=text2music \
        dset=audio/musiccaps_custom \
        dataset.num_workers=2 \
        dataset.train.num_samples="$TRAIN_SET_SIZE" \
        dataset.valid.num_samples="$FULL_TEST_SET_SIZE" \
        dataset.batch_size="$BATCH_SIZE" \
        schedule.cosine.warmup=8 \
        optim.optimizer=dadam \
        optim.lr=1 \
        optim.max_norm=1e-3 \
        optim.epochs=15 \
        optim.updates_per_epoch="$TRAIN_SET_SIZE_BATCHES" \
        optim.adam.weight_decay=0.01 \
        checkpoint.save_every=1 \
        checkpoint.keep_last=10 \
        checkpoint.keep_every_states="[model, xp.cfg]" \
        datasource.train="$TRAIN_SET_DIR" \
        generate.every=null \
        dataset.generate.num_samples=0
done

dora run solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small \
    conditioner=text2music \
    dset=audio/musiccaps_custom \
    dataset.num_workers=2 \
    dataset.train.num_samples="$FULL_TRAIN_SET_SIZE" \
    dataset.valid.num_samples="$FULL_TEST_SET_SIZE" \
    dataset.batch_size="$BATCH_SIZE" \
    schedule.cosine.warmup=8 \
    optim.optimizer=dadam \
    optim.lr=1 \
    optim.max_norm=1e-3 \
    optim.epochs=15 \
    optim.updates_per_epoch="$FULL_TRAIN_SET_SIZE_BATCHES" \
    optim.adam.weight_decay=0.01 \
    checkpoint.save_every=1 \
    checkpoint.keep_last=10 \
    checkpoint.keep_every_states="[model, xp.cfg]"
    generate.every=null \
    dataset.generate.num_samples=0
