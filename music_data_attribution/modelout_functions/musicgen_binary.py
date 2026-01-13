from typing import Iterable

import torch
from audiocraft.models.lm import LMModel
from audiocraft.models.musicgen import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
from torch import Tensor
from torch.nn import Module
from trak.modelout_functions import AbstractModelOutput


class MusicGenBinaryModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, temperature: float = 1.0):
        self.musicgen = musicgen
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(-1)

        self.compression_model = musicgen.compression_model
        self.model = musicgen.lm
        self.autocast = musicgen.autocast
        self.device = musicgen.device

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
        B, K, T = tokens.shape

        logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        logits = logits[:, 0, 0, :]
        label = tokens[:, 0, 0]

        logits_correct = logits[:, label.unsqueeze(0)]

        cloned_logits = logits.clone()
        cloned_logits[:, label.unsqueeze(0)] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins.sum(dim=-1)

    def get_out_to_loss_grad(
        self,
        lm_model: LMModel,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        tokens = self._tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        logits = logits[:, 0, 0, :]
        label = tokens[:, 0, 0]
        assert logits.shape == (B, self.musicgen.lm.card)
        assert label.shape == (B,)

        ps = self.softmax(logits / self.temperature)[torch.arange(logits.size(0)), label]
        assert ps.shape == (B,)
        return (1 - ps).clone().detach().unsqueeze(-1)

    def _tokenize(self, audios: Tensor) -> Tensor:
        tokens, _ = self.musicgen.compression_model.encode(audios)
        return tokens

    def _compute_cfg_logits(self, lm_model: LMModel, tokens: Tensor, descriptions: list[str]) -> tuple[Tensor, Tensor]:
        conditions = [ConditioningAttributes(text={"description": description}) for description in descriptions]
        # null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        all_conditions = conditions  # + null_conditions

        tokenized_attributes = self.musicgen.lm.condition_provider.tokenize(all_conditions)
        condition_tensors = self.musicgen.lm.condition_provider(tokenized_attributes)

        # print(f"my {condition_tensors=}")

        with self.musicgen.autocast:
            out = lm_model.compute_predictions(tokens, [], condition_tensors)
            all_logits, mask = out.logits, out.mask

        batch_size = tokens.shape[0]
        # logits_cond, logits_uncond = torch.split(all_logits, batch_size, dim=0)
        # logits = logits_uncond + (logits_cond - logits_uncond) * self.musicgen.lm.cfg_coef
        logits = all_logits

        mask = mask[:batch_size, :, :]

        return logits, mask
