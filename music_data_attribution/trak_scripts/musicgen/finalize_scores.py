import argparse
import logging
import sys

import numpy as np
import torch
from audiocraft.models.musicgen import MusicGen
from trak.gradient_computers import IterativeGradientComputer

from music_data_attribution.modelout_functions.musicgen import get_model_output_function
from trak import TRAKer

SEED = 201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-count", type=int, help="Count of different model checkpoints used.")
    parser.add_argument("--train-set-size", type=int, help="Size of the training set used for featurizing.")
    parser.add_argument("--trak-dir", type=str, help="Directory for TRAK intermediate results.")
    parser.add_argument("--experiment-name", type=str, help="TRAK experiment name.")
    parser.add_argument("--model-output", choices=["loss", "binary"], help="Model output function version to use.")
    parser.add_argument("--use-cfg", type=bool, help="Whether to use CFG for logit calculation.")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger("finalize_scores")

    model = MusicGen.get_pretrained("facebook/musicgen-small")
    model.compression_model.eval()
    model.lm.eval()

    logger.debug("loaded pretrained musicgen")

    task = get_model_output_function(args.model_output, model, args.use_cfg)

    traker = TRAKer(
        model=model.lm,
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

    traker.finalize_scores(exp_name=args.experiment_name, model_ids=list(range(args.models_count)))

    logger.info("finished")
