TRAIN_SET_FRACTION=0.8

for seed in {1..10}; do
    python3 sample_train_set.py egs/musiccaps/train/data.jsonl egs/musiccaps/train/data.jsonl 0.8 --seed "$seed"

    CUDA_VISIBLE_DEVICES=1 dora run solver=musicgen/musicgen_base_32khz \
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
        optim.adam.weight_decay=0.01
done
