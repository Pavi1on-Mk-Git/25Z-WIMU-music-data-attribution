AUDIOCRAFT_REPO_DIR := "/home/jproboszcz/audiocraft"
AUDIOCRAFT_DORA_DIR := "/data/jproboszcz/musicgen"

split_musiccaps:
    pdm run ./music_data_attribution/finetuning_scripts/split_musiccaps.py \
        data/raw/musiccaps/music_data \
        --output-train-dir data/processed/musiccaps/music_data_train \
        --output-test-dir data/processed/musiccaps/music_data_test

prepare_musiccaps_for_musicgen:
    pdm run ./music_data_attribution/finetuning_scripts/musicgen/prepare_musicgen_data_json.py \
        "{{`pwd`}}/data/processed/musiccaps/music_data_train" \
        data/raw/musiccaps/musiccaps-public.csv \
        --output-dir {{AUDIOCRAFT_REPO_DIR}}/egs/musiccaps/train
    pdm run ./music_data_attribution/finetuning_scripts/musicgen/prepare_musicgen_data_json.py \
        "{{`pwd`}}/data/processed/musiccaps/music_data_test" \
        data/raw/musiccaps/musiccaps-public.csv \
        --output-dir {{AUDIOCRAFT_REPO_DIR}}/egs/musiccaps/test

finetune_musicgen:
    AUDIOCRAFT_DORA_DIR="{{AUDIOCRAFT_DORA_DIR}}" \
    bash ./music_data_attribution/finetuning_scripts/musicgen/train_musicgen.sh {{AUDIOCRAFT_REPO_DIR}}

featurize_musicgen:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash ./music_data_attribution/trak_scripts/featurize_musicgen.sh "{{AUDIOCRAFT_DORA_DIR}}/xps"

format:
    pdm run python3 -m ruff format .

lint:
    pdm run python3 -m ruff check .
