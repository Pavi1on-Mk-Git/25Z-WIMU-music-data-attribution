import argparse
import logging
import os
import sys

import numpy as np
import torch
from audiocraft.data.audio import audio_write
from audiocraft.models.musicgen import MusicGen
from torch.utils.data import DataLoader
from tqdm import tqdm

from music_data_attribution.musiccaps_dataset import MusicCapsDataset

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

    checkpoint = torch.load(args.checkpoint, "cpu", weights_only=False)
    musicgen.lm.load_state_dict(checkpoint["model"])

    musicgen.set_generation_params(duration=10, temperature=0)

    for indices, _, descriptions in tqdm(dataloader, desc="Generating samples..."):
        audios, all_tokens = musicgen.generate(descriptions, return_tokens=True)
        for index, audio, tokens in zip(indices, audios, all_tokens):
            audio_output_file_path = os.path.join(args.output_dir, f"{index}")  # audiocraft appends .wav
            audio_write(
                audio_output_file_path,
                audio.cpu(),
                musicgen.sample_rate,
                strategy="loudness",
            )
            tokens_output_file_path = os.path.join(args.output_dir, f"tokens_{index}.pt")
            torch.save(tokens, tokens_output_file_path)
