import argparse
import json
import logging

import torch
from pytorch_lightning import seed_everything
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from tqdm import tqdm

from music_data_attribution.modelout_functions.sao import SAOSmallModelOutput
from trak import TRAKer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Path to the dataset config JSON file.",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for gradient calculations.",
        default=2,
    )
    parser.add_argument(
        "--random-subset-percentage",
        type=float,
        help="Percentage of the dataset to use as a random subset.",
        default=1.0,
    )
    parser.add_argument(
        "--model-ckpt-path",
        type=str,
        help="Path to the MusicGen checkpoint to featurize.",
        default=None,
    )
    parser.add_argument(
        "--train-run-id",
        type=int,
        help="ID of the training run that produced the checkpoint.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-id",
        type=int,
        help="Number of epochs after which the checkpoint was saved.",
        default=None,
    )
    parser.add_argument(
        "--trak-dir",
        type=str,
        help="Directory for TRAK intermediate results.",
        default="trak_results",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        help="Projection dimension for TRAK.",
        default=4096,
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Number of timesteps to average gradients over.",
        default=1,
    )

    return parser.parse_args()


def main(args):
    seed = 123
    seed_everything(seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small", checkpoint_name="base_model")
    model = model.to(device)
    model.eval()

    logging.info("Loaded pretrained model 'stabilityai/stable-audio-open-small' from checkpoint 'base_model'.")

    # prepare dataset and dataloader
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    dataloader, dataset_size = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
        random_subset_percentage=args.random_subset_percentage,
        return_dataset_size=True,
    )
    logging.info("Dataloader created.")

    # create TRAK
    task = SAOSmallModelOutput(num_timesteps=args.num_timesteps)

    traker = TRAKer(
        model=model,
        task=task,
        save_dir=args.trak_dir,
        load_from_save_dir=True,
        proj_max_batch_size=8,
        train_set_size=dataset_size,
        device=device,
        proj_dim=args.proj_dim,
    )
    logging.info("Created TRAKer.")

    # load checkpoint
    if args.model_ckpt_path is not None:
        checkpoint = load_ckpt_state_dict(args.model_ckpt_path)
        model_id = int(f"{args.train_run_id}{args.checkpoint_id}")
        traker.load_checkpoint(checkpoint, model_id=model_id)
        logging.info(f"Loaded checkpoint from {args.model_ckpt_path} under model ID {model_id} with TRAKer.")

    for batch in tqdm(dataloader, desc="Computing TRAK embeddings..."):
        reals, metadata = batch
        with torch.no_grad():
            conditioning = model.conditioner(metadata, device)

        traker.featurize(batch=(reals.to(device), conditioning), num_samples=reals.shape[0])
        torch.cuda.empty_cache()

    traker.finalize_features(model_ids=[model_id])
    logging.info("Finished.")


if __name__ == "__main__":
    main(parse_args())
