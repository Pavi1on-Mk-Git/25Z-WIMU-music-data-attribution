from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    hf_hub_download(
        "stabilityai/stable-audio-open-small",
        filename="base_model.ckpt",
        repo_type="model",
        local_dir="results/checkpoints",
    )
