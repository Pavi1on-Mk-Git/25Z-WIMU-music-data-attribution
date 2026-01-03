from torch.utils.data import Dataset, DataLoader
from audiocraft.data.audio import audio_read
from audiocraft.data.audio_utils import convert_audio
from torch import Tensor

import os
import os.path
import pandas as pd
import torch


class MusicCapsDataset(Dataset):
    def __init__(
        self, audio_dir: os.PathLike, labels_csv_path: os.PathLike, sample_rate: int = 32000, channels: int = 1
    ):
        self.sample_rate = sample_rate
        self.channels = channels

        self.descriptions = pd.read_csv(labels_csv_path)["caption"].tolist()

        filenames = [filename for filename in os.listdir(audio_dir) if filename[-4:] == ".wav"]
        filenames = sorted(filenames, key=lambda name: int(name[:-4]))
        filenames = [os.path.join(audio_dir, filename) for filename in filenames]
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        filename = self.filenames[idx]
        caption_idx = int(filename[:-4].split("/")[-1])
        caption = self.descriptions[caption_idx]

        return self._read_audio(filename), caption

    def _read_audio(self, filename: os.PathLike) -> Tensor:
        audio, sample_rate = audio_read(filename)
        return convert_audio(audio, sample_rate, self.sample_rate, self.channels)


def get_musiccaps_dataloader(dataset: Dataset, num_workers: int, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        # all audios should have the same length, so no need for padding collated audios
        collate_fn=lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch]),
    )


if __name__ == "__main__":
    dataset = MusicCapsDataset(
        audio_dir="data/musiccaps/music_data_train",
        labels_csv_path="data/musiccaps/musiccaps-public.csv",
    )
    dataloader = get_musiccaps_dataloader(dataset, 1, 4)

    from tqdm import tqdm

    for audio, descriptions in tqdm(dataloader):
        print(audio.shape)
