#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 dora_experiments_dir trak_dir" >&2
    echo "Error: Expected 2 arguments, got $#." >&2
    exit 1
fi

DORA_EXPERIMENTS_DIR="$1"
TRAK_DIR="$2"

for dir in "$DORA_EXPERIMENTS_DIR"/*; do
    if [ ! -d "$dir" ] || [ ! -f "$dir"/checkpoint_15.th ]; then
        continue
    fi
    TRAIN_RUN_ID=$(cat "$dir"/outputs/hyperparams.json | python -m json.tool | grep -o 'train_[0-9]\+' | sed 's/train_//')

    for checkpoint_id in {6..15}; do
        CHECKPOINT_PATH="$dir/checkpoint_$checkpoint_id.th"
        pdm run ./music_data_attribution/trak_scripts/featurize_musicgen.py \
            "$dir/checkpoint_$checkpoint_id.th" \
            --train-run-id $TRAIN_RUN_ID \
            --checkpoint-id $checkpoint_id \
            --music-data-path data/processed/musiccaps/music_data_train \
            --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
            --trak-dir "$TRAK_DIR" \
            --batch-size 4
    done
done
