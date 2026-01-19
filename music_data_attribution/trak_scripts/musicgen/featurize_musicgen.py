import argparse
import logging
import sys

import numpy as np
import torch
from audiocraft.models.musicgen import MusicGen
from torch.utils.data import DataLoader
from tqdm import tqdm
from trak.gradient_computers import IterativeGradientComputer

from music_data_attribution.modelout_functions.musicgen import get_model_output_function
from music_data_attribution.musiccaps_dataset import MusicCapsDataset
from trak import TRAKer

SEED = 201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to the MusicGen checkpoint to featurize.")
    parser.add_argument("--train-run-id", type=int, help="ID of the training run that produced the checkpoint.")
    parser.add_argument("--checkpoint-id", type=int, help="Number of epochs after which the checkpoint was saved.")
    parser.add_argument("--music-data-path", type=str, help="Path to the MusicCaps wav files of the train split.")
    parser.add_argument("--descriptions-path", type=str, help="Path to the MusicCaps captions csv file.")
    parser.add_argument("--trak-dir", type=str, help="Directory for TRAK intermediate results.")
    parser.add_argument("--batch-size", type=int, help="Batch size for gradient calculations.", default=2)
    parser.add_argument(
        "--model-output", choices=["loss", "single", "summed"], help="Model output function version to use."
    )
    parser.add_argument("--use-cfg", choices=["true", "false"], help="Whether to use CFG for logit calculation.")
    parser.add_argument("--proj-dim", type=int, help="TRAK projection dimension.")
    args = parser.parse_args()

    model_id = (args.train_run_id - 1) * 10 + (args.checkpoint_id - 6)

    np.random.seed(SEED + model_id)
    torch.manual_seed(SEED + model_id)
    torch.backends.cudnn.deterministic = True
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger("featurize_musicgen")

    dataset = MusicCapsDataset(
        audio_dir=args.music_data_path,
        labels_csv_path=args.descriptions_path,
        # sample rate and channels as expected by MusicGen
        sample_rate=32000,
        channels=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate,
    )

    logger.debug("created dataloader")

    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.compression_model.eval()
    model.lm.eval()

    logger.debug("loaded pretrained musicgen")

    task = get_model_output_function(args.model_output, model, args.use_cfg == "true")

    traker = TRAKer(
        model=model.lm,
        task=task,
        save_dir=args.trak_dir,
        load_from_save_dir=True,
        train_set_size=len(dataset),
        device="cuda",
        gradient_computer=IterativeGradientComputer,
        proj_dim=args.proj_dim,
        proj_max_batch_size=8,
        use_half_precision=False,
    )

    logger.debug("created traker")

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    traker.load_checkpoint(checkpoint["model"], model_id=model_id)

    logger.debug("loaded checkpoint with traker")

    for indices, audios, descriptions in tqdm(dataloader, desc="Computing TRAK embeddings..."):
        traker.featurize(batch=(audios.cuda(), descriptions), inds=indices)
    traker.finalize_features(model_ids=[model_id])

    logger.info("finished")
