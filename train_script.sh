TRAIN_SET_FRACTION=0.5

for seed in {1..10}; do
    python3 sample_train_set.py egs/musiccaps/train/data_full.jsonl egs/musiccaps/train/data.jsonl "$TRAIN_SET_FRACTION" --seed "$seed"

    dora run solver=musicgen/musicgen_base_32khz \
        model/lm/model_scale=small \
        continue_from=//pretrained/facebook/musicgen-small \
        conditioner=text2music \
        dset=audio/musiccaps_custom \
        dataset.num_workers=2 \
        dataset.valid.num_samples=915 \
        dataset.batch_size=8 \
        schedule.cosine.warmup=8 \
        optim.optimizer=adamw \
        optim.lr=1e-4 \
        optim.epochs=10 \
        optim.updates_per_epoch=3659 \
        optim.adam.weight_decay=0.01 \
        checkpoint.save_every=1 \
        checkpoint.keep_last=10 \
        checkpoint.keep_every_states="[model]"
done
