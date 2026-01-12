#!/usr/bin/env python3
"""
Generate music using a finetuned MusicGen model from HuggingFace.
Usage: python generate_musicgen.py "your prompt here"
"""

import sys
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile


def generate_music(prompt: str, duration: float = 10.0, output_path: str = "output.wav"):
    """Generate music from a text prompt using finetuned MusicGen model."""

    print(f"Loading model from HuggingFace...")
    # Load the finetuned model and processor
    model = MusicgenForConditionalGeneration.from_pretrained("danhtran2mind/MusicGen-Small-MusicCaps-finetuning")
    processor = AutoProcessor.from_pretrained("danhtran2mind/MusicGen-Small-MusicCaps-finetuning")

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Generating music with prompt: '{prompt}'")
    print(f"Duration: {duration}s")

    # Process the prompt
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Calculate max_new_tokens based on duration
    # MusicGen typically generates at 50 Hz (tokens per second)
    sample_rate = model.config.audio_encoder.sampling_rate
    audio_tokens_per_second = model.config.audio_encoder.frame_rate
    max_new_tokens = int(duration * audio_tokens_per_second)

    print(f"Max new tokens: {max_new_tokens}")

    # Generate audio
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    # Save to file
    # audio_values shape is [batch, channels, samples]
    audio = audio_values[0, 0].cpu().numpy()  # Take first batch, first channel

    print(f"Saving to {output_path}...")
    wavfile.write(output_path, rate=sample_rate, data=audio)

    print(f"Done! Generated audio saved to {output_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio shape: {audio.shape}")


if __name__ == "__main__":
    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "An aggressive metal track with distorted guitars, fast double-kick drums, and harsh screamed vocals. The song maintains a high-energy, intense mood with chugging riffs, brief breakdowns, and a powerful, headbang-driven feel."

    # Optional: customize duration and output path
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output.wav"

    generate_music(prompt, duration, output_path)
