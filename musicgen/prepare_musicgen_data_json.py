import json
import csv
import os
import os.path
import librosa
import tqdm
import ast
from itertools import islice


SUBSET = "train"
# SUBSET = "test"


MUSIC_DATA_PATH = f"music_data_{SUBSET}/"
DESCRIPTIONS_PATH = "musiccaps-public.csv"
MUSIC_DATA_FULL_PATH = f"/home/jproboszcz/musiccaps/music_data_{SUBSET}/"
# AUDIOCRAFT_DATASET_PATH = f"/home/jproboszcz/audiocraft/egs/musiccaps/{SUBSET}"
AUDIOCRAFT_DATASET_PATH = f"musiccaps_{SUBSET}"
DATA_FILE_PATH = os.path.join(AUDIOCRAFT_DATASET_PATH, "data.jsonl")


def parse_csv_descriptions(path: str) -> list[tuple[list[str], str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        # use ast.literal_eval to handle single quotes used sometimes (not valid JSON)
        return [(ast.literal_eval(row[4]), row[5]) for row in islice(reader, 1, None)]


if __name__ == "__main__":
    descriptions = parse_csv_descriptions(DESCRIPTIONS_PATH)
    os.makedirs(AUDIOCRAFT_DATASET_PATH, exist_ok=True)

    with open(DATA_FILE_PATH, "w") as output_file:
        for file_name in tqdm.tqdm(os.listdir(MUSIC_DATA_PATH)):
            file_path = os.path.join(MUSIC_DATA_PATH, file_name)
            y, sr = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)

            index = int(file_name[:-4])
            keywords, description = descriptions[index]

            file_data = {
                "path": os.path.join(MUSIC_DATA_FULL_PATH, file_name),
                "duration": duration,
                "sample_rate": sr,
                "description": description,
                "keywords": keywords,
            }

            output_file.write(json.dumps(file_data) + "\n")
