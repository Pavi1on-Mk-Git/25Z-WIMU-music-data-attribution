import os
import os.path
import librosa
import tqdm
import shutil
import argparse


def get_number_of_channels(y) -> int:
    if y.ndim == 1:
        return 1
    else:
        return y.shape[0]


def copy_subset(source: str, destination: str, file_names: list[str]):
    os.makedirs(destination, exist_ok=True)

    for file_name in file_names:
        shutil.copy(os.path.join(source, file_name), destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("music_data_path", type=str, help="Path to the MusicCaps wav files.")
    parser.add_argument("--output-train-dir", type=str, help="Path to the output directory for the train split.")
    parser.add_argument("--output-test-dir", type=str, help="Path to the output directory for the test split.")
    args = parser.parse_args()

    file_names = sorted(os.listdir(args.music_data_path))

    train_file_names = []
    test_file_names = []

    valid_file_index = 0
    for file_name in tqdm.tqdm(file_names):
        file_path = os.path.join(args.music_data_path, file_name)
        y, sr = librosa.load(file_path, sr=None, mono=False)
        duration = librosa.get_duration(y=y, sr=sr)
        channels = get_number_of_channels(y)

        if channels > 2 or duration != 10.0:
            continue

        if valid_file_index % 5 == 0:
            test_file_names.append(file_name)
        else:
            train_file_names.append(file_name)

        valid_file_index += 1

    copy_subset(args.music_data_path, args.output_train_dir, train_file_names)
    copy_subset(args.music_data_path, args.output_test_dir, test_file_names)
