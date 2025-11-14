# 25Z-WIMU-music-data-attribution

## Install dependencies:
```bash
sudo apt install libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
pdm install
```

## Running TRAK

Compute TRAK features:
```bash
pdm run run_trak.py
```

Visualize attribution results using visualize.ipynb notebook.

## Running MusicGen and Stable Audio Open

Run the respective scripts:
```bash
pdm run run_musicgen.py
```

and

```bash
pdm run run_sao.py
```

Both produce a file called output.wav, containing the generated audio.
