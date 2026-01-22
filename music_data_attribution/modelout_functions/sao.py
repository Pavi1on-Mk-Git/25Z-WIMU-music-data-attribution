import torch
import torch.nn.functional as F
from stable_audio_tools.inference.sampling import truncated_logistic_normal_rescaled
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from torch import Tensor
from trak.modelout_functions import AbstractModelOutput


class SAOSmallModelOutput(AbstractModelOutput):
    def __init__(self, num_timesteps: int = 1):
        super().__init__()
        self.num_timesteps = num_timesteps

    def get_output(
        self,
        model: ConditionedDiffusionModelWrapper,
        weights: dict[str, Tensor],
        buffers: dict[str, Tensor],
        audio: Tensor,
        conditioning: dict,
    ) -> Tensor:
        device = audio.device

        def unsqueeze_all(dim=0, *args):
            def _unsqueeze(obj, dim):
                if isinstance(obj, Tensor):
                    return obj.unsqueeze(dim)
                elif isinstance(obj, list):
                    return [_unsqueeze(o, dim) for o in obj]
                elif isinstance(obj, tuple):
                    return tuple(_unsqueeze(o, dim) for o in obj)
                elif isinstance(obj, dict):
                    return {k: _unsqueeze(v, dim) for k, v in obj.items()}
                else:
                    return obj

            for arg in args:
                yield _unsqueeze(arg, dim)

        audio, conditioning = unsqueeze_all(0, audio, conditioning)

        if self.num_timesteps > 1:

            def repeat_all(num_repeats, *args):
                def _repeat(obj, num_repeats):
                    if isinstance(obj, Tensor):
                        return obj.repeat(num_repeats, *[1 for _ in range(obj.ndim - 1)])
                    elif isinstance(obj, list):
                        return [_repeat(o, num_repeats) for o in obj]
                    elif isinstance(obj, tuple):
                        return tuple(_repeat(o, num_repeats) for o in obj)
                    elif isinstance(obj, dict):
                        return {k: _repeat(v, num_repeats) for k, v in obj.items()}
                    else:
                        return obj

                for arg in args:
                    yield _repeat(arg, num_repeats)

            audio, conditioning = repeat_all(self.num_timesteps, audio, conditioning)

        if audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]

        diffusion_input = audio

        if model.pretransform is not None:
            with torch.amp.autocast(device.type) and torch.set_grad_enabled(model.pretransform.enable_grad):
                model.pretransform.train(model.pretransform.enable_grad)
                diffusion_input = model.pretransform.encode(diffusion_input)

        # Draw from logistic truncated normal distribution
        t = truncated_logistic_normal_rescaled(audio.shape[0]).to(device)
        # Flip the distribution
        t = 1 - t

        if model.dist_shift is not None:
            # Shift the distribution
            t = model.dist_shift.time_shift(t, audio.shape[2])

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = 1 - t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        targets = noise - diffusion_input

        with torch.amp.autocast(device.type):
            output = torch.func.functional_call(
                model,
                (weights, buffers),
                args=(noised_inputs, t),
                kwargs={"cond": conditioning, "cfg_dropout_prob": 0, "use_checkpointing": False},
            )

        return F.mse_loss(output, targets)

    def get_out_to_loss_grad(self, model, weights, buffers, batch) -> Tensor:
        audios, _ = batch
        return torch.ones(audios.shape[0]).to(audios.device).unsqueeze(-1)
