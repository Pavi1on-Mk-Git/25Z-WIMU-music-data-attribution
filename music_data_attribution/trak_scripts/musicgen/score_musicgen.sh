#!/bin/bash
set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 audiocraft_dir dora_experiments_dir trak_dir" >&2
    echo "Error: Expected 3 arguments, got $#." >&2
    exit 1
fi

AUDIOCRAFT_REPO_DIR="$1"
DORA_EXPERIMENTS_DIR="$2"
TRAK_DIR="$3"

FULL_TRAIN_SET_FILE="$AUDIOCRAFT_REPO_DIR/egs/musiccaps/train/data.jsonl"
FULL_TRAIN_SET_SIZE=$(wc -l < "$FULL_TRAIN_SET_FILE")

for dir in "$DORA_EXPERIMENTS_DIR"/*; do
    if [ ! -d "$dir" ] || [ ! -f "$dir"/checkpoint_15.th ]; then
        continue
    fi
    TRAIN_RUN_ID=$(cat "$dir"/outputs/hyperparams.json | python -m json.tool | grep -o 'train_[0-9]\+' | sed 's/train_//')

    for checkpoint_id in {6..15}; do
        pdm run ./music_data_attribution/trak_scripts/musicgen/score_musicgen.py \
            "$dir/checkpoint_$checkpoint_id.th" \
            --train-run-id $TRAIN_RUN_ID \
            --checkpoint-id $checkpoint_id \
            --music-data-path results/generated/musicgen \
            --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
            --train-set-size "$FULL_TRAIN_SET_SIZE" \
            --trak-dir "$TRAK_DIR" \
            --batch-size 2
    done
done
