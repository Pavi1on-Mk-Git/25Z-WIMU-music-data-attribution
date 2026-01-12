#!/bin/bash
set -e

export FLASH_ATTN_DISABLE=1

MODEL_CKPT_PATH="checkpoints/sao_small/9/stable_audio_open_small_finetune/xfqjs690/checkpoints/epoch=299-step=1200.ckpt"

seed=$(echo "$MODEL_CKPT_PATH" | sed -n 's|.*sao_small/\([0-9]\+\)/.*|\1|p')
epoch=$(echo "$MODEL_CKPT_PATH" | sed -n 's|.*epoch=\([0-9]\+\)-.*|\1|p')

pdm run ./music_data_attribution/trak_scripts/score_sao.py \
    --dataset-config music_data_attribution/trak_scripts/dataset_config.json \
    --batch-size 2 \
    --model-ckpt-path  $MODEL_CKPT_PATH \
    --train-run-id $seed \
    --checkpoint-id $epoch \
    --proj-dim 4096 \
    --num-timesteps 6 \
    --trak-dir ./results/trak_sao
