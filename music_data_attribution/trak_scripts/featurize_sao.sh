#!/bin/bash
set -e

export FLASH_ATTN_DISABLE=1

pdm run ./music_data_attribution/trak_scripts/featurize_sao.py \
    --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_configs/datset_config.json \
    --batch-size 2 \
    --random-subset-percentage 0.1 \
    --model-ckpt-path checkpoints/sao/0/epoch=9-step=4570.ckpt \
    --train-run-id 0 \
    --checkpoint-id 9 \
    --proj-dim 4096 \
    --num-noise-samples 6 \
    --trak-dir ./results/trak_sao
