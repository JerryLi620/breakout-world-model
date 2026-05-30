"""
modal_train_transformer.py – Run Stage-2 GPT training on a Modal cloud GPU.

Prereqs on the Volume (from the tokenizer stage):
    /atari_data_128.pt    (dataset)
    /tokenizer_128.pt     (trained, frozen tokenizer)

Train:
    modal run modal_train_transformer.py
    modal run modal_train_transformer.py --epochs 40 --d-model 512 --n-layers 8

Download:
    modal volume get --force worldmodel-data /transformer.pt ./transformer.pt
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow", "imageio")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "rollouts", ".git", "*.tex", "*.log"],
    )
)

app = modal.App("worldmodel-transformer", image=image)
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 60 * 8)
def train_remote(
    epochs: int = 30,
    batch_size: int = 16,
    context_frames: int = 8,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    lr: float = 3e-4,
    action_dim: int = 4,
    tokenize_chunk: int = 512,
    debug_every: int = 100,
    data_file: str = "/data/atari_data_128.pt",
    tokenizer: str = "/data/tokenizer_128.pt",
    checkpoint: str = "/data/transformer.pt",
):
    import sys
    import types
    sys.path.insert(0, "/root/project")

    import torch
    import train_transformer

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    args = types.SimpleNamespace(
        data_file=data_file,
        tokenizer=tokenizer,
        action_dim=action_dim,
        context_frames=context_frames,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=0.1,
        batch_size=batch_size,
        tokenize_chunk=tokenize_chunk,
        epochs=epochs,
        lr=lr,
        weight_decay=0.01,
        seed=42,
        device="cuda",
        checkpoint=checkpoint,
        debug_every=debug_every,
    )

    train_transformer.run(args)
    volume.commit()
    print("\nDone. Fetch with:")
    print(f"  modal volume get --force worldmodel-data {checkpoint} ./{checkpoint.split('/')[-1]}")


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 16,
    context_frames: int = 8,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    data_file: str = "/data/atari_data_128.pt",
    tokenizer: str = "/data/tokenizer_128.pt",
    checkpoint: str = "/data/transformer.pt",
):
    train_remote.remote(
        epochs=epochs, batch_size=batch_size, context_frames=context_frames,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        data_file=data_file, tokenizer=tokenizer, checkpoint=checkpoint,
    )
