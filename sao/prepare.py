from huggingface_hub import hf_hub_download
from pathlib import Path

SAO_DIR = Path("./sao")

MODEL_NAME = "stabilityai/stable-audio-open-small"

CHECKPOINT_NAME = "model.ckpt"

MODEL = SAO_DIR / Path(CHECKPOINT_NAME)

if not MODEL.exists():
    hf_hub_download(
        repo_id=MODEL_NAME,
        filename=CHECKPOINT_NAME,
        local_dir=SAO_DIR,
    )
