STABLE_AUDIO_TOOLS_DIR=./stable-audio-tools
CHECKPOINTS_DIR=checkpoints/sao_small/9/stable_audio_open_small_finetune/xfqjs690/checkpoints

ls $CHECKPOINTS_DIR/epoch=*-*.ckpt | sort -t= -k2,2n | head -n -10 | xargs -r rm --

for checkpoint in $CHECKPOINTS_DIR/*.ckpt; do
    pdm run python3 $STABLE_AUDIO_TOOLS_DIR/unwrap_model.py \
        --model-config music_data_attribution/finetuning_scripts/sao/base_model_config.json \
        --ckpt-path "$checkpoint" \
        --name "${checkpoint%.*}"
done