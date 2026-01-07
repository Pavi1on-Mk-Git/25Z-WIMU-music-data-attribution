#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 dora_experiments_dir" >&2
    echo "Error: Expected 1 argument, got $#." >&2
    exit 1
fi

DORA_EXPERIMENTS_DIR="$1"

# Find the checkpoint finetuned on the entire training set
for dir in "$DORA_EXPERIMENTS_DIR"/*; do
    if [ ! -d "$dir" ] || [ ! -f "$dir"/checkpoint.th ]; then
        continue
    fi

    cat "$dir"/outputs/hyperparams.json | python -m json.tool | grep 'egs/musiccaps/train"' > /dev/null && break
done

CHECKPOINT_PATH="$(realpath $dir)/checkpoint.th"

pdm run ./music_data_attribution/trak_scripts/musicgen/generate_samples.py \
    "$CHECKPOINT_PATH" \
    --music-data-path data/processed/musiccaps/music_data_test \
    --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
    --output-dir results/generated/musicgen \
    --batch-size 4
