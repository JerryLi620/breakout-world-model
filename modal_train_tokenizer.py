"""
modal_train_tokenizer.py – Run VQ-VAE tokenizer training on a Modal cloud GPU.

One-time setup (local terminal):
    pip install modal
    modal setup                                   # authenticate
    python3 export_dataset.py --num_sequences 2000 --seq_len 20 --out atari_data.pt
    modal volume create worldmodel-data
    modal volume put worldmodel-data atari_data.pt /atari_data.pt

Train:
    modal run modal_train_tokenizer.py

    # override settings:
    modal run modal_train_tokenizer.py --epochs 40 --object-weight 50 --gpu A100

Download results when done:
    modal volume get worldmodel-data /tokenizer.pt ./tokenizer.pt
    modal volume get worldmodel-data /tokenizer_recon.gif ./tokenizer_recon.gif

Notes:
    - Data + outputs live on the Modal Volume "worldmodel-data" (mounted at /data).
    - No ALE in the cloud image — training loads atari_data.pt directly.
    - GPU billed per-second only while the function runs.
"""

import modal

# ---------------------------------------------------------------------------
# Image: torch + the few deps our code needs. No ale-py (we use the saved file).
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow", "imageio")
    # ship the project source into the image
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "rollouts", ".git", "*.tex", "*.log"],
    )
)

app = modal.App("worldmodel-tokenizer", image=image)

# Persistent volume for the dataset file and training outputs.
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 60 * 6)
def train_remote(
    epochs: int = 30,
    batch_size: int = 64,
    frame_batch: int = 256,      # 128x128 frames use ~4x the memory of 64x64
    num_codes: int = 512,
    embedding_dim: int = 256,
    hidden: int = 128,
    lr: float = 3e-4,
    object_weight: float = 30.0,
    commitment_cost: float = 0.25,
    debug_every: int = 100,
    data_file: str = "/data/atari_data_128.pt",
    checkpoint: str = "/data/tokenizer_128.pt",
    recon_gif: str = "/data/tokenizer_recon_128.gif",
):
    import sys
    import types
    sys.path.insert(0, "/root/project")

    import torch
    import train_tokenizer

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Build an args namespace equivalent to the CLI parser, pointing at the
    # Volume for input data and outputs so results persist after the run.
    args = types.SimpleNamespace(
        dataset="atari",
        num_sequences=2000,
        seq_len=20,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
        hidden=hidden,
        commitment_cost=commitment_cost,
        batch_size=batch_size,
        frame_batch=frame_batch,
        epochs=epochs,
        lr=lr,
        seed=42,
        device="cuda",
        checkpoint=checkpoint,
        recon_gif=recon_gif,
        data_file=data_file,
        object_weight=object_weight,
        object_threshold=0.05,
        debug_every=debug_every,
    )

    train_tokenizer.run(args)

    # Persist Volume writes so `modal volume get` sees them.
    volume.commit()
    print("\nDone. Fetch results with:")
    print(f"  modal volume get worldmodel-data {checkpoint} ./{checkpoint.split('/')[-1]}")
    print(f"  modal volume get worldmodel-data {recon_gif} ./{recon_gif.split('/')[-1]}")


@app.local_entrypoint()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    frame_batch: int = 256,
    object_weight: float = 30.0,
    num_codes: int = 512,
    embedding_dim: int = 256,
    data_file: str = "/data/atari_data_128.pt",
    checkpoint: str = "/data/tokenizer_128.pt",
    recon_gif: str = "/data/tokenizer_recon_128.gif",
):
    train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        frame_batch=frame_batch,
        object_weight=object_weight,
        num_codes=num_codes,
        embedding_dim=embedding_dim,
        data_file=data_file,
        checkpoint=checkpoint,
        recon_gif=recon_gif,
    )
