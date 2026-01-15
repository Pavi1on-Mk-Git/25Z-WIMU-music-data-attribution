from typing import Any

import pandas

DATASET = pandas.read_csv("../musiccaps/musiccaps-public.csv").to_dict(orient="records")


def get_custom_metadata(info: dict[str, Any], _):
    song_id = int(info["relpath"].removesuffix(".wav"))
    prompt = DATASET[song_id]["caption"]
    return {"prompt": prompt}
