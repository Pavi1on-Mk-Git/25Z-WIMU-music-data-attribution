import argparse
import json

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.inference.sampling import (
    get_alphas_sigmas,
    truncated_logistic_normal_rescaled,
    sample_timesteps_logsnr,
)


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
        "--num-workers",
        type=int,
        help="Number of workers for the dataloader.",
        default=4,
    )
    parser.add_argument(
        "--random-subset-percentage",
        type=float,
        help="Percentage of the dataset to use as a random subset.",
        default=1.0,
    )
    parser.add_argument(
        "--checkpoint",
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

    return parser.parse_args()


def main(args):
    seed = 123
    seed_everything(seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # with open(args.model_config) as f:
    #     model_config = json.load(f)
    # model = create_model_from_config(model_config)
    # if args.model_ckpt_path:
    # copy_state_dict(model, load_ckpt_state_dict(args.model_ckpt_path))

    model, model_config = get_pretrained_model(
        "stabilityai/stable-audio-open-small", checkpoint_name="base_model"
    )
    model = model.to(device)
    model.eval()

    # load checkpoint
    # if args.checkpoint is not None:
    #     model_ckpt_path = args.checkpoint
    #     checkpoint = load_ckpt_state_dict(model_ckpt_path)
    #     model.load_state_dict(checkpoint)
    #     model_id = (args.train_run_id - 1) * 10 + (args.checkpoint_id - 6)
    #     print(
    #         f"Loaded model checkpoint from {model_ckpt_path} with model ID {model_id}"
    #     )

    print(f"Model config:\n{model_config}")

    # prepare dataset and dataloader
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    print(f"Dataset config:\n{dataset_config}")

    dataloader = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
        random_subset_percentage=args.random_subset_percentage,
    )

    batch = next(iter(dataloader))
    print(batch)

    def compute_loss_on_batch(model, batch, repeat_batch=1):
        reals, metadata = batch

        if repeat_batch > 1:
            reals = reals.repeat(repeat_batch, 1, 1)
            metadata = metadata * repeat_batch

        reals = reals.to(device)
        print(f"Audio shape: {reals.shape}")

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        conditioning = model.conditioner(metadata, device)

        if model.pretransform is not None:
            model.pretransform.to(device)

            with torch.amp.autocast("cuda") and torch.set_grad_enabled(
                model.pretransform.enable_grad
            ):
                model.pretransform.train(model.pretransform.enable_grad)

                diffusion_input = model.pretransform.encode(diffusion_input)

        if model_config["training"]["timestep_sampler"] == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = torch.rand(reals.shape[0], device=device)
        elif model_config["training"]["timestep_sampler"] == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=device))
        elif model_config["training"]["timestep_sampler"] == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(device)
            # Flip the distribution
            t = 1 - t
        elif model_config["training"]["timestep_sampler"] == "log_snr":
            t = sample_timesteps_logsnr(reals.shape[0]).to(device)
        else:
            raise ValueError(
                f"Invalid timestep_sampler: {model_config['training']['timestep_sampler']}"
            )

        if model.dist_shift is not None:
            # Shift the distribution
            t = model.dist_shift.time_shift(t, reals.shape[2])

        print(f"Timestep shape: {t.shape}")

        # Calculate the noise schedule parameters for those timesteps
        if model.diffusion_objective in ["v"]:
            alphas, sigmas = get_alphas_sigmas(t)
        elif model.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            alphas, sigmas = 1 - t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if model.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif model.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            targets = noise - diffusion_input

        extra_args = {}

        with torch.amp.autocast("cuda"):
            output = model(
                noised_inputs, t, cond=conditioning, cfg_dropout_prob=0, **extra_args
            )
            print("Computed model output.")
            print(f"Output shape: {output.shape}")
            print(f"Targets shape: {targets.shape}")

        return F.mse_loss(output, targets)

    loss = compute_loss_on_batch(model, batch, repeat_batch=32)
    print(f"Computed loss on batch: {loss.item()}")


if __name__ == "__main__":
    main(parse_args())
