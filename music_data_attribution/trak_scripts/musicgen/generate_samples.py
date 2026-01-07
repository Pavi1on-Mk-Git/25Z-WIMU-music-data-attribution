from audiocraft.utils.checkpoint import load_checkpoint
from audiocraft.data.audio import audio_write
from music_data_attribution.musiccaps_dataset import MusicCapsDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from audiocraft.models.musicgen import MusicGen

import torch
import logging
import numpy as np
import sys
import argparse
import os

SEED = 201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to the MusicGen checkpoint to generate from.")
    parser.add_argument("--music-data-path", type=str, help="Path to the MusicCaps wav files of the test split.")
    parser.add_argument("--descriptions-path", type=str, help="Path to the MusicCaps captions csv file.")
    parser.add_argument("--output-dir", type=str, help="Directory to output the generated samples to.")
    parser.add_argument("--batch-size", type=int, help="Batch size for generation.", default=2)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = MusicCapsDataset(args.music_data_path, args.descriptions_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate)

    musicgen = MusicGen.get_pretrained("facebook/musicgen-small")
    musicgen.compression_model.eval()
    musicgen.lm.eval()

    checkpoint = load_checkpoint(args.checkpoint)
    musicgen.lm.load_state_dict(checkpoint["model"])

    musicgen.set_generation_params(duration=10, temperature=1)

    for indices, _, descriptions in tqdm(dataloader, desc="Generating samples..."):
        audios = musicgen.generate(descriptions)
        for index, audio in zip(indices, audios):
            output_file_path = os.path.join(args.output_dir, f"{index}.wav")
            audio_write(output_file_path, audio.cpu(), musicgen.sample_rate, strategy="loudness")
