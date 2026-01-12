#!/bin/bash
set -e

MODEL_DIR="checkpoints/sao_small/9/stable_audio_open_small_finetune/xfqjs690/"
CHECKPOINT_NAME="epoch=299-step=1200"


pdm run ./test_sao.py \
    --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
    --model-ckpt-path  "${MODEL_DIR}/checkpoints/${CHECKPOINT_NAME}.ckpt" \
    --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_config.json \
    --diffusion-steps 50 \
    --cfg-scale 7.0 \
    --output-dir $MODEL_DIR/generations