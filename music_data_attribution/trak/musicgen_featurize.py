from audiocraft.utils.checkpoint import load_checkpoint
from music_data_attribution.musiccaps_dataset import MusicCapsDataset
from music_data_attribution.modelout_functions.musicgen_trak import MusicGenModelOutput
from trak import TRAKer
from tqdm import tqdm
from torch.utils.data import DataLoader
from trak.gradient_computers import IterativeGradientComputer
from audiocraft.models.musicgen import MusicGen

import torch


if __name__ == "__main__":
    torch.manual_seed(201)

    dataset = MusicCapsDataset(
        audio_dir="/home/jproboszcz/musiccaps/music_data_train",
        labels_csv_path="/home/jproboszcz/musiccaps/musiccaps-public.csv",
        sample_rate=32000,
        channels=1,
    )
    dataloader = dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)

    print("created dataloader")

    m = MusicGen.get_pretrained("facebook/musicgen-small")
    m.compression_model.eval()
    m.lm.eval()
    # m.lm.load_state_dict(checkpoint["model"])

    print("loaded pretrained musicgen")

    task = MusicGenModelOutput(m)

    traker = TRAKer(
        model=m.lm,
        task=task,
        save_dir="./trak_debug",
        load_from_save_dir=True,
        train_set_size=len(dataset),
        device="cuda",
        gradient_computer=IterativeGradientComputer,
    )

    print("created traker")

    checkpoint = load_checkpoint("/data/jproboszcz/musicgen/xps/21a44b8e/checkpoint_15.th")
    traker.load_checkpoint(checkpoint["model"], model_id=0)

    print("loaded checkpoint with traker")

    for audios, descriptions in tqdm(dataloader, desc="Computing TRAK embeddings..."):
        traker.featurize(batch=(audios.cuda(), descriptions), num_samples=dataloader.batch_size)
    traker.finalize_features(model_ids=[0])
