# 25Z-WIMU-music-data-attribution

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
