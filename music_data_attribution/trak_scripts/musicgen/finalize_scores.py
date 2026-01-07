from music_data_attribution.modelout_functions.musicgen import MusicGenModelOutput
from trak import TRAKer
from trak.gradient_computers import IterativeGradientComputer
from audiocraft.models.musicgen import MusicGen
from music_data_attribution.trak_scripts.musicgen.score_musicgen import EXPERIMENT_NAME

import torch
import logging
import numpy as np
import sys
import argparse

SEED = 201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-count", type=int, help="Count of different model checkpoints used.")
    parser.add_argument("--train-set-size", type=int, help="Size of the training set used for featurizing.")
    parser.add_argument("--trak-dir", type=str, help="Directory for TRAK intermediate results.")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    logger = logging.getLogger("finalize_scores")

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

    traker.finalize_scores(exp_name=EXPERIMENT_NAME, model_ids=list(range(args.models_count)))

    logger.info("finished")
