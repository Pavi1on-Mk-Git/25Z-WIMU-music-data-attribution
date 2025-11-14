# 25Z-WIMU-music-data-attribution

## Usage

Compute TRAK features for training data, scores for target examples for each checkpoint in parallel using SLURM:
```bash
source .env
sbatch --account="$ATHENA_GRANT" run.sbatch
```

Gather final scores:
```bash
bash slurm_scripts/start_bash.sh
python gather.py
```

To visualize attribution results use visualize.ipynb notebook.