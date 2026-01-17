import argparse
import json
import logging

import torch
from pytorch_lightning import seed_everything
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import create_dataloader_from_config

from music_data_attribution.modelout_functions.sao import SAOSmallModelOutput
from trak import TRAKer

SEED = 201

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dataset-config",
        type=str,
        help="Path to the train dataset config JSON file.",
        required=True,
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Number of timesteps to average gradients over.",
        default=1,
    )
    parser.add_argument("--models-count", type=int, help="Count of different model checkpoints used.")
    parser.add_argument("--trak-dir", type=str, help="Directory for TRAK intermediate results.")
    parser.add_argument("--train-set-size", type=int, help="Size of the training set used for featurizing.")
    args = parser.parse_args()

    seed_everything(SEED, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small", checkpoint_name="base_model")
    model = model.to(device)
    model.eval()

    logging.info("Loaded pretrained model 'stabilityai/stable-audio-open-small' from checkpoint 'base_model'.")

    # prepare train dataset and dataloader
    with open(args.train_dataset_config) as f:
        train_dataset_config = json.load(f)

    _, train_dataset_size = create_dataloader_from_config(
        train_dataset_config,
        batch_size=args.batch_size,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
        return_dataset_size=True,
    )
    logging.info("Train dataloader created.")

    # create TRAK
    task = SAOSmallModelOutput(num_timesteps=args.num_timesteps)

    traker = TRAKer(
        model=model,
        task=task,
        save_dir=args.trak_dir,
        load_from_save_dir=True,
        proj_max_batch_size=8,
        train_set_size=train_dataset_size,
        device=device,
        proj_dim=args.proj_dim,
        lambda_reg=1e-6,
    )
    logging.info("Created TRAKer.")

    traker.finalize_scores(exp_name="sao_small_finetune", model_ids=list(range(args.models_count)))

    logging.info("Finished")
