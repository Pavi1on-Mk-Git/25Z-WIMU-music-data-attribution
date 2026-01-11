#!/bin/bash
set -e

STABLE_AUDIO_TOOLS_DIR=./stable-audio-tools
CHECKPOINTS_DIR=./checkpoints/sao_small

seed=9

pdm run python3 $STABLE_AUDIO_TOOLS_DIR/train.py \
    --config-file $STABLE_AUDIO_TOOLS_DIR/defaults.ini \
    --dataset-config music_data_attribution/finetuning_scripts/sao/dataset_config.json \
    --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
    --name stable_audio_open_small_finetune \
    --pretrained-ckpt-path $CHECKPOINTS_DIR/base_model.ckpt \
    --save-dir $CHECKPOINTS_DIR/$seed \
    --batch-size 8 \
    --accum-batches 2 \
    --num-gpus 1 \
    --precision 16-mixed \
    --max-epochs 10 \
    --seed $seed \
    --random-subset-percentage 1.0

ls $CHECKPOINTS_DIR/$seed/epoch=*-*.ckpt | sort -t= -k2,2n | head -n -10 | xargs -r rm --

for checkpoint in $CHECKPOINTS_DIR/$seed/*.ckpt; do
    pdm run python3 $STABLE_AUDIO_TOOLS_DIR/unwrap_model.py \
        --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
        --ckpt-path "$checkpoint" \
        --name "${checkpoint%.*}"
done