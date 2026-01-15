#!/bin/bash
set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 dora_experiments_dir trak_dir model_output use_cfg" >&2
    echo "Error: Expected 4 arguments, got $#." >&2
    exit 1
fi

DORA_EXPERIMENTS_DIR="$1"
TRAK_DIR="$2"
MODEL_OUTPUT="$3"
USE_CFG="$4"

DEBUG_SET_SIZE=$(find data/processed/musiccaps/music_data_debug/* -prune | wc -l)

# Find the test checkpoint
for dir in "$DORA_EXPERIMENTS_DIR"/*; do
    if [ ! -d "$dir" ] || [ ! -f "$dir"/checkpoint.th ]; then
        continue
    fi

    cat "$dir"/outputs/hyperparams.json | python -m json.tool | grep '"optim.epochs": 60' > /dev/null && break
done

CHECKPOINT_PATH="$(realpath "$dir")/checkpoint.th"

# Featurize
pdm run ./music_data_attribution/trak_scripts/musicgen/featurize_musicgen.py \
    "$CHECKPOINT_PATH" \
    --train-run-id 1 \
    --checkpoint-id 6 \
    --music-data-path data/processed/musiccaps/music_data_debug \
    --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
    --trak-dir "$TRAK_DIR" \
    --batch-size 4 \
    --model-output "$MODEL_OUTPUT" \
    --use-cfg "$USE_CFG"

# Generate samples if not generated yet
if [ ! -d "results/generated/musicgen_debug" ]; then
    pdm run ./music_data_attribution/trak_scripts/musicgen/generate_samples.py \
        "$CHECKPOINT_PATH" \
        --music-data-path data/processed/musiccaps/music_data_debug \
        --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
        --output-dir results/generated/musicgen_debug \
        --batch-size 16
fi

# Score
pdm run ./music_data_attribution/trak_scripts/musicgen/score_musicgen.py \
    "$CHECKPOINT_PATH" \
    --train-run-id 1 \
    --checkpoint-id 6 \
    --generated-path results/generated/musicgen_debug \
    --music-data-path data/processed/musiccaps/music_data_debug \
    --descriptions-path data/raw/musiccaps/musiccaps-public.csv \
    --train-set-size "$DEBUG_SET_SIZE" \
    --trak-dir "$TRAK_DIR" \
    --batch-size 2 \
    --experiment-name musicgen_test \
    --model-output "$MODEL_OUTPUT" \
    --use-cfg "$USE_CFG"

# Finalize scores
pdm run ./music_data_attribution/trak_scripts/musicgen/finalize_scores.py \
    --models-count 1 \
    --train-set-size "$DEBUG_SET_SIZE" \
    --trak-dir "$TRAK_DIR" \
    --experiment-name musicgen_test \
    --model-output "$MODEL_OUTPUT" \
    --use-cfg "$USE_CFG"
