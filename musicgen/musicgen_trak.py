from trak.modelout_functions import AbstractModelOutput
from trak.gradient_computers import IterativeGradientComputer
from torch import Tensor
from torch.nn import Module
from audiocraft.modules.conditioners import (
    ConditioningAttributes,
    ClassifierFreeGuidanceDropout,
)
from audiocraft.models.musicgen import MusicGen
from audiocraft.models.lm import LMModel
from typing import Iterable

import torch


class MusicGenModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, temperature: float = 1.0):
        self.musicgen = musicgen
        self.temperature = temperature

    def get_output(
        self,
        lm_model: Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        """
        Calculates model output function analogously to the text classification case, as:

        log(probability[correct] / (1 - probability[correct]))

        equivalent to

        logit[correct] - log(sum(exp(logit[i] if i != correct)))

        The "correct" tokens are the ones actually generated.
        """
        tokens = self._tokenize(audios)

        logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        B, K, T, card = logits.shape
        assert mask.shape == (B, K, T)

        tokens = tokens.unsqueeze(-1)
        logits_correct = torch.gather(logits, -1, tokens).squeeze(-1)
        logits_correct[~mask] = 0

        # this is passed to torch.logsumexp, so -inf after exp becomes 0
        logits_incorrect = logits.clone()
        logits_incorrect = logits_incorrect.scatter(
            -1, tokens, torch.full_like(logits_correct, -torch.inf).unsqueeze(-1)
        )
        logits_incorrect[~mask] = -torch.inf

        margins = logits_correct - torch.logsumexp(logits_incorrect, dim=-1)
        output = margins.sum(dim=-1).sum(dim=-1)
        print(margins.shape)
        print(output.shape)
        return output

    def get_out_to_loss_grad(
        self,
        lm_model: LMModel,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        tokens = self._tokenize(audios)

        logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        B, K, T, card = logits.shape

        ps = torch.softmax(logits / self.temperature, dim=-1)  # B, K, T, card
        ps[~mask] = 0
        ps = torch.gather(ps, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ps = ps.reshape(B, K * T).sum(dim=-1)

        return (1 - ps).clone().detach()

    def _tokenize(self, audios: Tensor) -> Tensor:
        tokens, _ = self.musicgen.compression_model.encode(audios)
        return tokens

    def _compute_cfg_logits(self, lm_model: LMModel, tokens: Tensor, descriptions: list[str]) -> tuple[Tensor, Tensor]:
        conditions = [ConditioningAttributes(text={"description": description}) for description in descriptions]
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        all_conditions = conditions + null_conditions

        tokenized_attributes = self.musicgen.lm.condition_provider.tokenize(all_conditions)
        condition_tensors = self.musicgen.lm.condition_provider(tokenized_attributes)

        with self.musicgen.autocast:
            out = lm_model.compute_predictions(torch.cat([tokens, tokens], dim=0), [], condition_tensors)
            all_logits, mask = out.logits, out.mask

        batch_size = all_logits.shape[0] // 2
        logits_cond, logits_uncond = torch.split(all_logits, batch_size, dim=0)
        logits = logits_uncond + (logits_cond - logits_uncond) * self.musicgen.lm.cfg_coef

        mask = mask[:batch_size, :, :]

        return logits, mask


if __name__ == "__main__":
    from audiocraft.utils.checkpoint import load_checkpoint
    from music_data_attribution.musiccaps_dataset import (
        MusicCapsDataset,
        get_musiccaps_dataloader,
    )
    from trak import TRAKer
    from tqdm import tqdm

    torch.manual_seed(201)

    dataset = MusicCapsDataset(
        audio_dir="/home/jproboszcz/musiccaps/music_data_train",
        labels_csv_path="/home/jproboszcz/musiccaps/musiccaps-public.csv",
        sample_rate=32000,
        channels=1,
    )
    dataloader = get_musiccaps_dataloader(dataset, 1, 2)

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
