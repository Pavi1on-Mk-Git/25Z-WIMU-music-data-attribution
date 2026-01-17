#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 checkpoint_dir trak_dir" >&2
    echo "Error: Expected 2 arguments, got $#." >&2
    exit 1
fi

CHECKPOINT_DIR=$1
TRAK_DIR=$2

for dir in "$CHECKPOINT_DIR"/*; do
    if [ ! -d "$dir" ] || [ "$dir" = "lightning_logs" ]; then
        continue
    fi

    TRAIN_RUN_ID=$(basename "$dir")

    for checkpoint in "$dir"/*.ckpt; do

        echo "Featurizing seed: $TRAIN_RUN_ID, checkpoint: $checkpoint"

        file=$(basename "$checkpoint")
        epoch_suffix="${file#epoch=}"
        CHECKPOINT_ID="${epoch_suffix%%-*}"

        pdm run ./music_data_attribution/trak_scripts/sao/featurize_sao.py \
            --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_config.json \
            --batch-size 2 \
            --model-ckpt-path "$checkpoint" \
            --train-run-id "$TRAIN_RUN_ID" \
            --checkpoint-id "$CHECKPOINT_ID" \
            --proj-dim 4096 \
            --num-timesteps 6 \
            --trak-dir "$TRAK_DIR"

    done
done
