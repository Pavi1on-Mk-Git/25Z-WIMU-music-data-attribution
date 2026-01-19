from typing import Iterable, Literal

import torch
from audiocraft.models.lm import LMModel
from audiocraft.models.musicgen import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout, ConditioningAttributes
from torch import Tensor
from torch.nn import Module
from trak.modelout_functions import AbstractModelOutput


class MusicGenWrapper:
    def __init__(self, musicgen: MusicGen, use_cfg: bool, cfg_coef: float = 3.0):
        self.musicgen = musicgen
        self.use_cfg = use_cfg
        self.cfg_coef = cfg_coef

    def tokenize(self, audios: Tensor) -> Tensor:
        tokens, _ = self.musicgen.compression_model.encode(audios)
        return tokens

    def compute_logits(self, lm_model: LMModel, tokens: Tensor, descriptions: list[str]) -> tuple[Tensor, Tensor]:
        batch_size = tokens.shape[0]
        conditions = [ConditioningAttributes(text={"description": description}) for description in descriptions]

        if self.use_cfg:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            tokens = torch.cat([tokens, tokens], dim=0)

        tokenized_attributes = self.musicgen.lm.condition_provider.tokenize(conditions)

        with self.musicgen.autocast:
            condition_tensors = self.musicgen.lm.condition_provider(tokenized_attributes)

            out = lm_model.compute_predictions(tokens, [], condition_tensors)
            logits, mask = out.logits, out.mask

        if self.use_cfg:
            logits_cond, logits_uncond = torch.split(logits, batch_size, dim=0)
            logits = logits_uncond + (logits_cond - logits_uncond) * self.cfg_coef
            mask = mask[:batch_size, :, :]

        return logits, mask


class MusicGenLossModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, use_cfg: bool):
        self.musicgen = MusicGenWrapper(musicgen, use_cfg)

    def get_output(
        self,
        lm_model: Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        """
        Calculates model output function as the cross entropy loss, same as in MusicGen model training.
        The correct tokens are the ones actually generated.
        """
        tokens = self.musicgen.tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self.musicgen.compute_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        tokens = tokens.reshape(B, K * T)
        logits = logits.reshape(B, K * T, -1)
        mask = mask.reshape(B, K * T)

        masked_tokens = tokens[mask].view(tokens.size(0), -1)
        masked_logits = logits[mask].view(logits.size(0), -1, logits.size(2))

        masked_logits = masked_logits.transpose(1, 2)

        loss_per_token = torch.nn.functional.cross_entropy(masked_logits, masked_tokens, reduction="none")
        loss_per_sample = loss_per_token.sum(dim=-1, keepdim=True)

        return loss_per_sample

    def get_out_to_loss_grad(
        self,
        lm_model: LMModel,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        return torch.ones(audios.shape[0], 1, dtype=torch.float32, device=audios.device)


class MusicGenBinaryModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, use_cfg: bool, temperature: float = 1.0):
        self.musicgen = MusicGenWrapper(musicgen, use_cfg)
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(-1)

    def get_output(
        self,
        lm_model: Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        """
        Considers only the first logit for the output calculation,
        so that the calculation can be exactly analogous to the text classification case.
        That is, the model output function is:

        log(probability[correct] / (1 - probability[correct]))

        equivalent to

        logit[correct] - log(sum(exp(logit[i] if i != correct)))

        The "correct" token is the one actually generated.
        """
        tokens = self.musicgen.tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self.musicgen.compute_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        logits = logits[:, 0, 0, :]
        label = tokens[:, 0, 0]

        logits_correct = logits[torch.arange(B), label]

        cloned_logits = logits.clone()
        cloned_logits[torch.arange(B), label] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return margins

    def get_out_to_loss_grad(
        self,
        lm_model: LMModel,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        tokens = self.musicgen.tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self.musicgen.compute_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        logits = logits[:, 0, 0, :]
        label = tokens[:, 0, 0]
        assert logits.shape == (B, self.musicgen.musicgen.lm.card)
        assert label.shape == (B,)

        ps = self.softmax(logits / self.temperature)[torch.arange(logits.size(0)), label]
        assert ps.shape == (B,)
        return (1 - ps).clone().detach().unsqueeze(-1)


class MusicGenSummedModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, use_cfg: bool, temperature: float = 1.0):
        self.musicgen = MusicGenWrapper(musicgen, use_cfg)
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(-1)

    def get_output(
        self,
        lm_model: Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        """
        Calculated model output function analogously to the text classification case,
        summing the results over codebooks and tokens.
        """
        tokens = self.musicgen.tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self.musicgen.compute_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        logits_correct = torch.gather(logits, -1, tokens.unsqueeze(-1)).squeeze(-1)
        logits_correct[~mask] = 0
        assert logits_correct.shape == (B, K, T)

        logits_incorrect = torch.scatter(
            logits, -1, tokens.unsqueeze(-1), torch.full_like(logits_correct, -torch.inf).unsqueeze(-1)
        )
        logits_incorrect = torch.masked_fill(logits_incorrect, ~mask.unsqueeze(-1), -torch.inf)
        assert logits_incorrect.shape == (B, K, T, self.musicgen.musicgen.lm.card)

        margins = logits_correct - logits_incorrect.logsumexp(dim=-1)
        margins[torch.isposinf(margins)] = 0  # handle masked tokens
        return margins.reshape(tokens.shape[0], -1).sum(dim=-1)

    def get_out_to_loss_grad(
        self,
        lm_model: LMModel,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audios: Tensor,
        descriptions: list[str],
    ) -> Tensor:
        tokens = self.musicgen.tokenize(audios)
        B, K, T = tokens.shape

        logits, mask = self.musicgen.compute_logits(lm_model, tokens, descriptions)
        assert logits.shape == (B, K, T, self.musicgen.musicgen.lm.card)
        assert mask.shape == (B, K, T)

        ps = self.softmax(logits / self.temperature)
        ps_correct = torch.gather(ps, -1, tokens.unsqueeze(-1)).squeeze(-1)
        ps_correct[~mask] = 1
        assert ps_correct.shape == (B, K, T)

        return (1 - ps_correct.reshape(B, K * T)).sum(dim=-1, keepdim=True)


MODEL_OUTPUT_FUNCTIONS = {
    "loss": MusicGenLossModelOutput,
    "binary": MusicGenBinaryModelOutput,
    "summed": MusicGenSummedModelOutput,
}


def get_model_output_function(
    version: Literal["loss", "binary", "summed"], model: MusicGen, use_cfg: bool
) -> MusicGenLossModelOutput | MusicGenBinaryModelOutput | MusicGenSummedModelOutput:
    return MODEL_OUTPUT_FUNCTIONS[version](model, use_cfg)
