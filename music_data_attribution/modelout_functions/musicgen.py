import typing as tp
from typing import Iterable

import torch
from audiocraft.models.lm import LMModel
from audiocraft.models.musicgen import MusicGen
from audiocraft.modules.conditioners import (
    # ClassifierFreeGuidanceDropout,
    ConditioningAttributes,
    SegmentWithAttributes,
)
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from trak.modelout_functions import AbstractModelOutput


class MusicGenModelOutput(AbstractModelOutput):
    def __init__(self, musicgen: MusicGen, temperature: float = 1.0):
        self.musicgen = musicgen
        self.temperature = temperature

        self.compression_model = musicgen.compression_model
        self.model = musicgen.lm
        self.autocast = musicgen.autocast
        self.device = musicgen.device

    def _get_audio_tokens(self, audio: torch.Tensor):
        with torch.no_grad():
            audio_tokens, scale = self.compression_model.encode(audio)
            assert scale is None, "Scaled compression model not supported with LM."
            return audio_tokens

    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[str]]
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        audio, infos = batch
        audio = audio.to(self.device)
        audio_tokens = None
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})",
        )

        # prepare attributes
        attributes = [ConditioningAttributes(text={"description": description}) for description in infos]
        tokenized = self.model.condition_provider.tokenize(attributes)

        if audio_tokens is None:
            audio_tokens = self._get_audio_tokens(audio)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)

        return condition_tensors, audio_tokens, padding_mask

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def run_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]]) -> float:
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(batch)

        print(f"{condition_tensors=}")
        print(f"{audio_tokens=}")
        print(f"{padding_mask=}")

        with self.autocast:
            style_mask = None
            if hasattr(self.model.condition_provider.conditioners, "self_wav"):
                if hasattr(self.model.condition_provider.conditioners.self_wav, "mask"):
                    print("HAS STYLE MASK")
                    style_mask = self.model.condition_provider.conditioners.self_wav.mask

            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            if style_mask is not None:
                mask = padding_mask & model_output.mask & style_mask
            else:
                mask = padding_mask & model_output.mask

            print(f"{logits=}")
            print(f"{audio_tokens=}")
            print(f"{mask=}")

            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce

        return loss

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

        # print(f"my {logits=}")
        # print(f"my {tokens=}")
        # print(f"my {mask=}")

        tokens = tokens.reshape(B, K * T)
        logits = logits.reshape(B, K * T, -1)
        mask = mask.reshape(B, K * T)

        masked_tokens = tokens[mask].view(tokens.size(0), -1)
        masked_logits = logits[mask].view(logits.size(0), -1, logits.size(2))

        # margins = self._get_margins_from_logits(tokens, logits, mask)
        # assert margins.shape == (B, K * T)

        # output = torch.sum(margins, dim=-1)
        # assert not output.isnan().any()
        # assert not output.isinf().any()

        masked_logits = masked_logits.transpose(1, 2)

        loss_per_token = torch.nn.functional.cross_entropy(masked_logits, masked_tokens, reduction="none")
        loss_per_sample = loss_per_token.sum(dim=-1, keepdim=True)

        # print(f"{loss_per_sample=}")

        # from_run_step = self.run_step((audios, descriptions))

        # print(f"{from_run_step=}")

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

        # tokens = self._tokenize(audios)
        # B, K, T = tokens.shape

        # logits, mask = self._compute_cfg_logits(lm_model, tokens, descriptions)
        # assert logits.shape == (B, K, T, self.musicgen.lm.card)
        # assert mask.shape == (B, K, T)

        # ps = torch.softmax(logits / self.temperature, dim=-1)
        # ps = torch.gather(ps, -1, tokens.unsqueeze(-1)).squeeze(-1)
        # ps = torch.masked_fill(ps, ~mask, 0)
        # ps = ps.reshape(B, K * T)

        # base_grads = 1 - ps
        # margins = self._get_margins_from_logits(tokens, logits, mask)
        # assert margins.shape == (B, K * T)

        # out_to_loss_grad = (torch.sum(base_grads * margins, dim=-1) / torch.sum(margins, dim=-1)).unsqueeze(-1)
        # assert not out_to_loss_grad.isnan().any()
        # assert not out_to_loss_grad.isinf().any()

        # return out_to_loss_grad

    def _get_margins_from_logits(self, tokens: Tensor, logits: Tensor, mask: Tensor) -> Tensor:
        tokens = tokens.unsqueeze(-1)
        logits_correct = torch.gather(logits, -1, tokens).squeeze(-1)
        logits_correct[~mask] = 0

        # this is passed to torch.logsumexp, so -inf after exp becomes 0
        logits_incorrect = logits.clone()
        logits_incorrect = torch.scatter(
            logits_incorrect, -1, tokens, torch.full_like(logits_correct, -torch.inf).unsqueeze(-1)
        )
        logits_incorrect = torch.masked_fill(logits_incorrect, ~mask.unsqueeze(-1), -torch.inf)

        margins = logits_correct - torch.logsumexp(logits_incorrect, dim=-1)
        margins = torch.masked_fill(margins, ~mask, 0)

        return margins.reshape(tokens.shape[0], -1)

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
