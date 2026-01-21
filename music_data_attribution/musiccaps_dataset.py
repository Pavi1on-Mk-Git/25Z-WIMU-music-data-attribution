import os
import os.path
from typing import Iterable

import pandas as pd
import torch
from audiocraft.data.audio import audio_read
from audiocraft.data.audio_utils import convert_audio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MusicCapsDataset(Dataset):
    def __init__(
        self,
        audio_dir: os.PathLike,
        labels_csv_path: os.PathLike,
        sample_rate: int = 32000,
        channels: int = 1,
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

    def __getitem__(self, idx: int) -> tuple[int, int, Tensor, str]:
        filename = self.filenames[idx]
        caption_idx = int(filename[:-4].split("/")[-1])
        caption = self.descriptions[caption_idx]

        return idx, caption_idx, self._read_audio(filename), caption

    def _read_audio(self, filename: os.PathLike) -> Tensor:
        audio, sample_rate = audio_read(filename)
        return convert_audio(audio, sample_rate, self.sample_rate, self.channels)

    @staticmethod
    def collate(batch: Iterable[tuple[int, Tensor, str]]) -> tuple[Tensor, Tensor, Tensor, list[str]]:
        indices = []
        caption_indices = []
        audios = []
        descriptions = []

        for index, caption_index, audio, description in batch:
            indices.append(index)
            caption_indices.append(caption_index)
            audios.append(audio)
            descriptions.append(description)

        # all audios should have the same length, so no need for padding collated audios
        return torch.tensor(indices), torch.tensor(caption_indices), torch.stack(audios), descriptions


if __name__ == "__main__":
    dataset = MusicCapsDataset(
        audio_dir="data/musiccaps/music_data_train",
        labels_csv_path="data/musiccaps/musiccaps-public.csv",
    )
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)

    from tqdm import tqdm

    for indices, caption_indices, audio, descriptions in tqdm(dataloader):
        print(audio.shape)
