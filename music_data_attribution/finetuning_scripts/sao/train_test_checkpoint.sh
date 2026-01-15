#!/bin/bash
set -e

CHECKPOINT_DIR="results/checkpoints"
SAO_FINETUNING_DIR="music_data_attribution/finetuning_scripts/sao"

pdm run python3 ./stable-audio-tools/train.py \
    --config-file ./stable-audio-tools/defaults.ini \
    --dataset-config "$SAO_FINETUNING_DIR"/debug_dataset_config.json \
    --model-config "$SAO_FINETUNING_DIR"/base_model_config.json \
    --name stable_audio_open_small_finetune \
    --save-dir "$CHECKPOINT_DIR" \
    --batch-size 8 \
    --precision 16-mixed \
    --pretrained-ckpt-path results/checkpoints/base_model.ckpt \
    --max-epochs 300 \
    --checkpoint-every 300 \
    --seed 42 \
    --random-subset-percentage 1.0

for checkpoint in "$CHECKPOINT_DIR"/epoch=*-*.ckpt; do
    pdm run python3 ./stable-audio-tools/unwrap_model.py \
        --model-config "$SAO_FINETUNING_DIR"/base_model_config.json \
        --ckpt-path "$checkpoint" \
        --name "${checkpoint%.*}"
done

