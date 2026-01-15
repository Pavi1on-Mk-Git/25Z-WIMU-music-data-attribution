#!/bin/bash
set -e

CHECKPOINT_DIR="/data/jproboszcz/sao/checkpoints"
SAO_FINETUNING_DIR="music_data_attribution/finetuning_scripts/sao"

for seed_id in {0..9}; do
    pdm run python3 ./stable-audio-tools/train.py \
        --config-file ./stable-audio-tools/defaults.ini \
        --dataset-config "$SAO_FINETUNING_DIR"/dataset_config.json \
        --model-config "$SAO_FINETUNING_DIR"/base_model_config.json \
        --name stable_audio_open_small_finetune \
        --save-dir "$CHECKPOINT_DIR/$seed_id" \
        --batch-size 8 \
        --precision 16-mixed \
        --pretrained-ckpt-path results/checkpoints/base_model.ckpt \
        --max-epochs 15 \
        --seed "$seed_id" \
        --random-subset-percentage 0.5

    find "$CHECKPOINT_DIR/$seed_id"/epoch=*-*.ckpt -maxdepth 1 | sort -t= -k2,2n | head -n -10 | xargs -r rm --

    for checkpoint in "$CHECKPOINT_DIR/$seed_id"/*.ckpt; do
        pdm run python3 ./stable-audio-tools/unwrap_model.py \
            --model-config "$SAO_FINETUNING_DIR"/base_model_config.json \
            --ckpt-path "$checkpoint" \
            --name "${checkpoint%.*}"
    done

done