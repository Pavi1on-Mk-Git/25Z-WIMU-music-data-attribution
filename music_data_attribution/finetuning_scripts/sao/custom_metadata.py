import pandas as pd

DESCRIPTIONS_PATH = "data/raw/musiccaps/musiccaps-public.csv"
DATASET = pd.read_csv(DESCRIPTIONS_PATH).to_dict(orient="records")


def get_custom_metadata(info, audio):
    song_id = int(info["relpath"].removesuffix(".wav"))
    prompt = DATASET[song_id]["caption"]
    return {"prompt": prompt}


if __name__ == "__main__":
    # Example usage
    sample_info = {"relpath": "1.wav"}
    sample_audio = None  # Placeholder for audio data
    metadata = get_custom_metadata(sample_info, sample_audio)
    print(metadata)
