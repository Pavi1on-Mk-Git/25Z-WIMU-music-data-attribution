#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 trak_dir" >&2
    echo "Error: Expected 1 argument, got $#." >&2
    exit 1
fi

TRAK_DIR="$1"

CHECKPOINT_PATH="results/checkpoints/epoch=299-step=2400.ckpt"

# Featurize
pdm run ./music_data_attribution/trak_scripts/sao/featurize_sao.py \
    --dataset-config music_data_attribution/finetuning_scripts/sao/debug_dataset_config.json \
    --batch-size 2 \
    --model-ckpt-path "$CHECKPOINT_PATH" \
    --train-run-id 0 \
    --checkpoint-id 5 \
    --proj-dim 4096 \
    --num-timesteps 6 \
    --trak-dir "$TRAK_DIR"

# Generate samples if not generated yet
if [ ! -d "results/generated/sao_debug" ]; then
    pdm run ./music_data_attribution/trak_scripts/sao/generate_samples.py \
        --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
        --model-ckpt-path "$CHECKPOINT_PATH" \
        --dataset-config music_data_attribution/finetuning_scripts/sao/debug_dataset_config.json \
        --diffusion-steps 50 \
        --cfg-scale 7.0 \
        --output-dir results/generated/sao_debug
fi

# Score
pdm run ./music_data_attribution/trak_scripts/sao/score_sao.py \
    --dataset-config music_data_attribution/trak_scripts/sao/generated_debug_config.json \
    --train-dataset-config music_data_attribution/finetuning_scripts/sao/debug_dataset_config.json \
    --batch-size 2 \
    --model-ckpt-path "$CHECKPOINT_PATH" \
    --train-run-id 0 \
    --checkpoint-id 5 \
    --proj-dim 4096 \
    --num-timesteps 6 \
    --trak-dir "$TRAK_DIR"

# Finalize scores
pdm run ./music_data_attribution/trak_scripts/sao/finalize_scores.py \
    --train-dataset-config music_data_attribution/finetuning_scripts/sao/debug_dataset_config.json \
    --num-timesteps 6 \
    --models-count 1 \
    --proj-dim 4096 \
    --trak-dir "$TRAK_DIR"
