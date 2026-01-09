# This script was run in a separate environment with only one package installed: datasets[audio]
# This is because of incompatibility of datasets with the version of torch in the main env.

from datasets import Dataset, DatasetDict, Audio, Value, Features
import pandas as pd
import os


def prepare_split(audio_dir: str, csv_data: pd.DataFrame) -> Dataset:
    filenames = [filename for filename in os.listdir(audio_dir) if filename[-4:] == ".wav"]
    filenames = sorted(filenames, key=lambda name: int(name[:-4]))
    indices = [int(filename[:-4]) for filename in filenames]
    filenames = [os.path.join(audio_dir, filename) for filename in filenames]

    csv_data = csv_data.iloc[indices].reset_index()
    ytid = csv_data["ytid"].tolist()
    start_s = [int(value) for value in csv_data["start_s"].tolist()]
    end_s = [int(value) for value in csv_data["end_s"].tolist()]
    audioset_positive_labels = csv_data["audioset_positive_labels"].tolist()
    aspect_list = csv_data["aspect_list"].tolist()
    caption = csv_data["caption"].tolist()
    author_id = [int(value) for value in csv_data["author_id"].tolist()]
    is_balanced_subset = [bool(value) for value in csv_data["is_balanced_subset"].tolist()]
    is_audioset_eval = [bool(value) for value in csv_data["is_audioset_eval"].tolist()]

    data = {
        "audio": filenames,
        "ytid": ytid,
        "start_s": start_s,
        "end_s": end_s,
        "audioset_positive_labels": audioset_positive_labels,
        "aspect_list": aspect_list,
        "caption": caption,
        "author_id": author_id,
        "is_balanced_subset": is_balanced_subset,
        "is_audioset_eval": is_audioset_eval,
    }
    features = Features(
        {
            "audio": Audio(sampling_rate=None),
            "ytid": Value("string"),
            "start_s": Value("int32"),
            "end_s": Value("int32"),
            "audioset_positive_labels": Value("string"),
            "aspect_list": Value("string"),
            "caption": Value("string"),
            "author_id": Value("int32"),
            "is_balanced_subset": Value("bool"),
            "is_audioset_eval": Value("bool"),
        }
    )

    return Dataset.from_dict(data, features=features)


if __name__ == "__main__":
    train_audio_dir = "data/processed/musiccaps/music_data_train"
    test_audio_dir = "data/processed/musiccaps/music_data_test"
    labels_csv_path = "data/raw/musiccaps/musiccaps-public.csv"

    csv_data = pd.read_csv(labels_csv_path)
    train_dataset = prepare_split(train_audio_dir, csv_data)
    test_dataset = prepare_split(test_audio_dir, csv_data)

    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    dataset_dict.push_to_hub("jproboszcz/split-musiccaps")
