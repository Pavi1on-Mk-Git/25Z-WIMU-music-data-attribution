#!/bin/bash
set -e

pdm run ./music_data_attribution/trak_scripts/featurize_sao.py \
    --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_configs/datset_config.json \
    --batch-size 1 \
    --num-workers 4 \
    --random-subset-percentage 0.1
