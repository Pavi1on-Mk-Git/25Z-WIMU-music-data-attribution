from itertools import islice

import json
import csv
import os
import os.path
import librosa
import tqdm
import ast
import argparse


def parse_csv_descriptions(path: str) -> list[tuple[list[str], str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        # use ast.literal_eval to handle single quotes used sometimes (not valid JSON)
        return [(ast.literal_eval(row[4]), row[5]) for row in islice(reader, 1, None)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "music_data_path",
        type=str,
        help="Path to the MusicCaps wav files of the chosen split.",
    )
    parser.add_argument("descriptions_path", type=str, help="Path to the MusicCaps captions csv file.")
    parser.add_argument("--output-dir", type=str, help="Output dataset directory in audiocraft repo.")
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output data json file name.",
        default="data.jsonl",
    )
    args = parser.parse_args()

    descriptions = parse_csv_descriptions(args.descriptions_path)
    os.makedirs(args.output_dir, exist_ok=True)

    data_file_path = os.path.join(args.output_dir, args.output_file)
    with open(data_file_path, "w") as output_file:
        for file_name in tqdm.tqdm(os.listdir(args.music_data_path)):
            file_path = os.path.join(args.music_data_path, file_name)
            y, sr = librosa.load(file_path, sr=None, mono=False)
            duration = librosa.get_duration(y=y, sr=sr)

            index = int(file_name[:-4])
            keywords, description = descriptions[index]

            file_data = {
                "path": os.path.join(args.music_data_path, file_name),
                "duration": duration,
                "sample_rate": sr,
                "description": description,
            }

            output_file.write(json.dumps(file_data) + "\n")
