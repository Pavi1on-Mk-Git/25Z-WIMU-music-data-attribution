#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 checkpoint_dir" >&2
    echo "Error: Expected 1 argument, got $#." >&2
    exit 1
fi

CHECKPOINT_DIR=$1

for checkpoint in "$CHECKPOINT_DIR"/*.ckpt; do
    pdm run ./music_data_attribution/trak_scripts/sao/generate_samples.py \
        --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
        --model-ckpt-path "$checkpoint" \
        --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_config.json \
        --diffusion-steps 50 \
        --cfg-scale 7.0 \
        --output-dir "results/generated/sao"
done
