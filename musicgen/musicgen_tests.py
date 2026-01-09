from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import (
    ConditioningAttributes,
    ClassifierFreeGuidanceDropout,
)
from audiocraft.utils.checkpoint import load_checkpoint
from audiocraft.data.audio import audio_write

import torch
import sys
import pandas as pd

torch.manual_seed(201)

checkpoint = sys.argv[1]
indexes = [int(idx) for idx in sys.argv[2].split(",")]
prefix = "" if len(sys.argv) < 4 else sys.argv[3]

m = MusicGen.get_pretrained("facebook/musicgen-small")
checkpoint = torch.load(checkpoint, weights_only=False)
print(list(checkpoint))
m.lm.load_state_dict(checkpoint["model"])

descriptions = pd.read_csv("data/raw/musiccaps/musiccaps-public.csv")["caption"][indexes]

m.set_generation_params(duration=10, temperature=1)
audios = m.generate(descriptions)

for idx, audio in zip(indexes, audios):
    audio_write(f"{prefix}{idx}", audio.cpu(), m.sample_rate, strategy="loudness")

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
