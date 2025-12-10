import polars
from typing import Any

DATASET = polars.read_csv("../musiccaps/musiccaps-public.csv").rows(named=True)


def get_custom_metadata(info: dict[str, Any], _):
    song_id = int(info["relpath"].removesuffix(".wav"))
    prompt = DATASET[song_id]["caption"]
    return {"prompt": prompt}
