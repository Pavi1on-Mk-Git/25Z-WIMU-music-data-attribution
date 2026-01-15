import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import copy_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

copy_state_dict(model, torch.load("./sao/base_model.ckpt", weights_only=True)["state_dict"])

model = model.to(device)
model.eval()

# Set up text and timing conditioning
conditioning = [
    {
        "prompt": "The song is a techno dance song with a groovy bass line, strong drumming rhythm and a keyboard accompaniment. The song is so groovy and serves as a dance track for the dancing children. The audio quality is very poor with high gains and hissing noise.",
        "seconds_total": 10,
    }
]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=50,
    cfg_scale=6.0,
    conditioning=conditioning,
    sample_size=sample_size,
    sampler_type="pingpong",
    device=device,
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)
