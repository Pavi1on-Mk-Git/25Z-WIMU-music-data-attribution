from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

m = MusicGen.get_pretrained("small")
m.set_generation_params(duration=8)
w = m.generate(["a powerful metal song"])
audio_write("output", w[0].cpu(), m.sample_rate, strategy="loudness")
