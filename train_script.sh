#!/bin/bash

TRAIN_SET_FRACTION=0.5
TRAIN_SET_SIZE=$(python3 -c "print(round($TRAIN_SET_FRACTION * 3659))")

for seed in {1..10}; do
    mkdir -p egs/musiccaps/train_"$seed"
    python3 sample_train_set.py egs/musiccaps/train/data.jsonl egs/musiccaps/train_"$seed"/data.jsonl "$TRAIN_SET_FRACTION" --seed "$seed"

    AUDIOCRAFT_DORA_DIR=/data/jproboszcz/musicgen \
    dora run solver=musicgen/musicgen_base_32khz \
        model/lm/model_scale=small \
        continue_from=//pretrained/facebook/musicgen-small \
        conditioner=text2music \
        dset=audio/musiccaps_custom \
        dataset.num_workers=2 \
        dataset.valid.num_samples=915 \
        dataset.batch_size=2 \
        schedule.cosine.warmup=8 \
        optim.optimizer=adamw \
        optim.lr=1e-4 \
        optim.epochs=15 \
        optim.updates_per_epoch="$TRAIN_SET_SIZE" \
        optim.adam.weight_decay=0.01 \
        checkpoint.save_every=1 \
        checkpoint.keep_last=10 \
        checkpoint.keep_every_states="[model]" \
        datasource.train=egs/musiccaps/train_"$seed"
done
