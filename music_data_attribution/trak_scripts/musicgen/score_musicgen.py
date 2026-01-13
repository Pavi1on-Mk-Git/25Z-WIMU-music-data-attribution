import argparse
import logging
import sys

import numpy as np
import torch
from audiocraft.models.musicgen import MusicGen
from torch.utils.data import DataLoader
from tqdm import tqdm
from trak.gradient_computers import IterativeGradientComputer

from music_data_attribution.modelout_functions.musicgen import MusicGenModelOutput
from music_data_attribution.musiccaps_dataset import MusicCapsDataset
from trak import TRAKer

SEED = 201


def prepare_generated_dataset(args: argparse.Namespace) -> MusicCapsDataset:
    valid_dataset = MusicCapsDataset(
        audio_dir=args.music_data_path,
        labels_csv_path=args.descriptions_path,
        # sample rate and channels as expected by MusicGen
        sample_rate=32000,
        channels=1,
    )
    generated_dataset = MusicCapsDataset(
        audio_dir=args.generated_path,
        labels_csv_path=args.descriptions_path,
        # sample rate and channels as expected by MusicGen
        sample_rate=32000,
        channels=1,
    )
    generated_dataset.descriptions = valid_dataset.descriptions

    return generated_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to the MusicGen checkpoint to generate from.")
    parser.add_argument("--train-run-id", type=int, help="ID of the training run that produced the checkpoint.")
    parser.add_argument("--checkpoint-id", type=int, help="Number of epochs after which the checkpoint was saved.")
    parser.add_argument("--generated-path", type=str, help="Path to the generated wav files to score.")
    parser.add_argument(
        "--music-data-path", type=str, help="Path to the original test set wav files (for retrieving descriptions)."
    )
    parser.add_argument("--descriptions-path", type=str, help="Path to the MusicCaps captions csv file.")
    parser.add_argument("--train-set-size", type=int, help="Size of the training set used for featurizing.")
    parser.add_argument("--trak-dir", type=str, help="Directory for TRAK intermediate results.")
    parser.add_argument("--batch-size", type=int, help="Batch size for gradient calculations.", default=2)
    parser.add_argument("--experiment-name", type=str, help="TRAK experiment name.")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger("score_musicgen")

    dataset = prepare_generated_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate,
    )

    logger.debug("created dataloader")

    m = MusicGen.get_pretrained("facebook/musicgen-small")
    m.compression_model.eval()
    m.lm.eval()

    logger.debug("loaded pretrained musicgen")

    task = MusicGenModelOutput(m)

    traker = TRAKer(
        model=m.lm,
        task=task,
        save_dir=args.trak_dir,
        load_from_save_dir=True,
        train_set_size=args.train_set_size,
        device="cuda",
        gradient_computer=IterativeGradientComputer,
        proj_dim=4096,
        proj_max_batch_size=8,
        use_half_precision=False,
    )

    logger.debug("created traker")

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model_id = (args.train_run_id - 1) * 10 + (args.checkpoint_id - 6)

    traker.start_scoring_checkpoint(
        exp_name=args.experiment_name,
        checkpoint=checkpoint["model"],
        model_id=model_id,
        num_targets=len(dataloader.dataset),
    )

    logger.debug("started scoring checkpoint")

    for indices, audios, descriptions in tqdm(dataloader, desc="Scoring..."):
        traker.score(batch=(audios.cuda(), descriptions), inds=indices)

    logger.info("finished")
