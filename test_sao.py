import os
import json
import argparse

from tqdm import tqdm
from loguru import logger
import torch
import torchaudio
from einops import rearrange
from dotenv import load_dotenv
from pytorch_lightning import seed_everything

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models.factory import create_model_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-ckpt-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    return parser.parse_args()

def main(args):
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.model_config) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(args.model_ckpt_path))
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded pretrained model from {args.model_ckpt_path}.")

    # prepare dataset and dataloader
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    dataloader = create_dataloader_from_config(
        dataset_config,
        batch_size=1,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
    )
    logger.info("Test dataloader created.")

    logger.info(f"Using {args.diffusion_steps} diffusion steps with CFG scale {args.cfg_scale}")
    for batch in tqdm(dataloader, desc="Generating..."):
        _, metadata = batch
        conditioning = [{k: v for k, v in metadata[0].items() if k in ['prompt', 'seconds_total']}]
    
        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=args.diffusion_steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            sample_size=model_config["sample_size"],
            sampler_type="pingpong",
            device=device,
        )

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        
        output_path = os.path.join(args.output_dir, metadata[0]['relpath'])
        torchaudio.save(output_path, output, model_config["sample_rate"])

        with open(output_path + ".txt", 'w') as f:
            f.write(conditioning[0]['prompt'])

    logger.info("Done.")

if __name__ == "__main__":
    main(parse_args())