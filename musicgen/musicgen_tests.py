from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import (
    ConditioningAttributes,
    ClassifierFreeGuidanceDropout,
)
from audiocraft.utils.checkpoint import load_checkpoint

# from audiocraft.data.audio import audio_write

import torch

torch.manual_seed(201)

m = MusicGen.get_pretrained("facebook/musicgen-small")
checkpoint = load_checkpoint("checkpoint_1.th")
m.lm.load_state_dict(checkpoint["model"])

m.set_generation_params(duration=0.5, temperature=0, two_step_cfg=False)
audio, tokens = m.generate(["powerful metal song"], return_tokens=True)
# audio_write("output", audio[0].cpu(), m.sample_rate, strategy="loudness")
print(f"{audio.shape=}")
print(f"{tokens.shape=}")

# attributes = [ConditioningAttributes(text={"description": "powerful metal song"})]
# null_attributes = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
# all_conditions = attributes + null_attributes

# m.compression_model

# tokenized = m.lm.condition_provider.tokenize(all_conditions)
# condition_tensors = m.lm.condition_provider(tokenized)

# with m.autocast:
#     out = m.lm.compute_predictions(torch.cat([tokens, tokens], dim=0), [], condition_tensors).logits

# logits_cond, logits_uncond = torch.split(out, 1, dim=0)
# logits = logits_uncond + (logits_cond - logits_uncond) * m.lm.cfg_coef
