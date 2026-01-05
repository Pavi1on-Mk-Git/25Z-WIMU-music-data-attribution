from trak.modelout_functions import AbstractModelOutput
from torch import Tensor
from torch.nn import Module
from audiocraft.modules.conditioners import ConditioningAttributes, ClassifierFreeGuidanceDropout
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
        B, K, T = tokens.shape

        logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        margins = self._get_margins_from_logits(tokens, logits, mask)
        assert margins.shape == (B, K * T)

        return torch.sum(margins, dim=-1)

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

        ps = torch.softmax(logits / self.temperature, dim=-1)
        ps[~mask] = 0
        ps = torch.gather(ps, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ps = ps.reshape(B, K * T)

        base_grads = 1 - ps
        margins = self._get_margins_from_logits(tokens, logits, mask)
        assert margins.shape == (B, K * T)

        return torch.sum(base_grads * margins, dim=-1) / torch.sum(margins, dim=-1)

    def _get_margins_from_logits(tokens: Tensor, logits: Tensor, mask: Tensor) -> Tensor:
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
        return margins.reshape(tokens.shape[0], -1)

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
