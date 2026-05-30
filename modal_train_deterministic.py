"""
modal_train_deterministic.py – Train the deterministic FiLM U-Net on a Modal GPU.

Used here to produce a TRUE-FREEZE checkpoint (strong brightness-conservation +
false-motion penalties) for the coherence plot's freeze curve.

train.py builds its dataset live from the Arcade Learning Environment, so the
image needs gymnasium[atari] + ale-py + the Breakout ROM.

Run:
    modal run modal_train_deterministic.py
    modal run modal_train_deterministic.py --bright-weight 80 --false-motion-weight 300 --epochs 40

Download:
    modal volume get --force worldmodel-data /ckpt_freeze.pt ./ckpt_freeze.pt
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("swig")
    .pip_install(
        "torch", "numpy", "pillow", "imageio",
        "gymnasium[atari]", "ale-py", "autorom[accept-rom-license]",
    )
    .run_commands("AutoROM --accept-license || true")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "rollouts", ".git", "*.tex", "*.log"],
    )
)

app = modal.App("worldmodel-det-train", image=image)
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 60 * 2)
def train_remote(
    bright_weight: float = 80.0,
    false_motion_weight: float = 300.0,
    epochs: int = 40,
    num_sequences: int = 2000,
    checkpoint: str = "/data/ckpt_freeze.pt",
):
    import sys
    sys.path.insert(0, "/root/project")

    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # train.py reads argv via get_args(); construct the CLI invocation here.
    sys.argv = [
        "train.py",
        "--dataset", "atari",
        "--num_sequences", str(num_sequences),
        "--seq_len", "20",
        "--num_frames", "4",
        "--latent_dim", "256",
        "--hidden_dim", "512",
        "--decoder_type", "residual",
        "--residual_scale", "0.2",
        "--action_dim", "4",
        "--object_weight", "50",
        "--change_weight", "100",
        "--delta_weight", "100",
        "--false_motion_weight", str(false_motion_weight),
        "--bright_weight", str(bright_weight),
        "--rollout_weight", "1.0",
        "--rollout_horizon", "6",
        "--no_detach_rollout",
        "--epochs", str(epochs),
        "--lr", "1e-4",
        "--batch_size", "32",
        "--seed", "42",
        "--checkpoint", checkpoint,
        "--device", "cuda",
    ]

    import train
    train.train()
    volume.commit()
    print("\nDone. Fetch with:")
    print(f"  modal volume get --force worldmodel-data {checkpoint} ./{checkpoint.split('/')[-1]}")


@app.local_entrypoint()
def main(
    bright_weight: float = 80.0,
    false_motion_weight: float = 300.0,
    epochs: int = 40,
    num_sequences: int = 2000,
):
    train_remote.remote(
        bright_weight=bright_weight,
        false_motion_weight=false_motion_weight,
        epochs=epochs,
        num_sequences=num_sequences,
    )
