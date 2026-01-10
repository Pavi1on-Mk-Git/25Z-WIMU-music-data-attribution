import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable

from trak.modelout_functions import AbstractModelOutput


class SAOSmallModelOutput(AbstractModelOutput):
    def __init__(self, sao: nn.Module):
        self.sao = sao
        self._are_we_featurizing = False

    def get_output(self, *args, **kwargs) -> Tensor:
        if self._are_we_featurizing:
            return self._get_output_featurizing(*args, **kwargs)
        else:
            return self._get_output_scoring(*args, **kwargs)

    def _get_output_featurizing(
        self,
        model: nn.Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        audio: Tensor,
        metadata: dict,
    ) -> Tensor:
        raise NotImplementedError(
            "Featurizing not implemented for SAOSmallModelOutput."
        )

    def _get_output_scoring(
        self,
        model: nn.Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        index: Tensor,
        audio: Tensor,
        prompt: list[str],
    ) -> Tensor:
        raise NotImplementedError("Scoring not implemented for SAOSmall.")

    def get_out_to_loss_grad(
        self,
        model: nn.Module,
        weights: Iterable[Tensor] | None,
        buffers: Iterable[Tensor] | None,
        indices: Tensor,
        audios: Tensor,
        prompts: list[str],
    ) -> Tensor:
        return torch.ones(audios.shape[0]).to(audios.device).unsqueeze(-1)
