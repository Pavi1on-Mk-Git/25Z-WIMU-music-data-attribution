from huggingface_hub import hf_hub_download


def download_pretrained_model(name: str, output_dir: str, checkpoint_name: str = "model"):
    model_config_path = hf_hub_download(
        name,
        filename=f"{checkpoint_name}_config.json",
        repo_type="model",
        local_dir=output_dir,
    )

    # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
    try:
        model_ckpt_path = hf_hub_download(
            name,
            filename=f"{checkpoint_name}.safetensors",
            repo_type="model",
            local_dir=output_dir,
        )
    except Exception as e:
        model_ckpt_path = hf_hub_download(
            name,
            filename=f"{checkpoint_name}.ckpt",
            repo_type="model",
            local_dir=output_dir,
        )


if __name__ == "__main__":
    download_pretrained_model(
        "stabilityai/stable-audio-open-small",
        output_dir="./checkpoints/sao_small",
        checkpoint_name="base_model",
    )
