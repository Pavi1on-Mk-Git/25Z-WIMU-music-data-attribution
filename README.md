# 25Z-WIMU-music-data-attribution - Instructions

## Install dependencies:
Install needed system packages (this may not be an exhaustive list):
```bash
sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
# or use conda:
conda create -n ffmpeg python=3.11 ffmpeg -c conda-forge
conda activate ffmpeg
```

Install PDM, instructions: https://pdm-project.org/en/latest/, then pull submodules and install Python dependencies:
```bash
git submodule init
git submodule update
pdm install --dev --no-isolation
```

Install Just, instructions: https://github.com/casey/just.

## Dataset preparation

Download MusicCaps CSV file into data/raw/musiccaps/musiccaps-public.csv. Download the audio files, called `{index}.wav` where index is their row in the CSV file, into data/raw/musiccaps/music_data.

Then, split the dataset into train and test:
```bash
just split_musiccaps
```

## Finetuning models

For MusicGen:
- clone the audiocraft repository https://github.com/facebookresearch/audiocraft
- adjust AUDIOCRAFT_REPO_DIR and AUDIOCRAFT_DORA_DIR variables in justfile
- install audiocraft's dependencies based on the requirements.txt in the cloned repository:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
- with audiocraft's venv active, `cd` to this repository and run:
```bash
just prepare_musiccaps_for_musicgen
just finetune_musicgen
```
The training artifacts will be located in AUDIOCRAFT_DORA_DIR.

Training the single overfit test model is done analogously, except the training command is:
```bash
# no just prepare_musiccaps, the single task does everything
just train_test_musicgen
```

For SAO:
- download the initial model checkpoint:
```bash
pdm run music_data_attribution/finetuning_scripts/sao/download.py
```
- run the training script:
```bash
music_data_attribution/finetuning_scripts/sao/train_sao.sh
```

Training the single overfit test model is done by running the training command:
```bash
just train_test_sao
```

## Running TRAK

For MusicGen, in the current directory, run:

```bash
just featurize_musicgen
just generate_musicgen
just score_musicgen
```

The resulting scores will be in `results/trak_musicgen/scores/musicgen_trak.mmap`. They can be read and visualized via `display_audio_trak.ipynb` notebook.
