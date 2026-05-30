"""
modal_coherence_plot.py – Generate the coherence plot (Figure 3) on a Modal GPU.

The token-model rollout is autoregressive (256 tokens/frame, many frames, many
sequences), so it is far too slow on a laptop CPU. This runs it on an A100.

Prereqs on the Volume:
    /atari_data.pt        (64x64 dataset, for deterministic models)
    /atari_data_128.pt    (128x128 dataset, for token model)
    /checkpoint.pt        (deterministic, freeze regime)
    /ckpt_static.pt       (deterministic, blow-up regime)
    /tokenizer_128.pt     (frozen tokenizer)
    /transformer.pt       (trained GPT dynamics)

Run:
    modal run modal_coherence_plot.py

Download:
    modal volume get --force worldmodel-data /coherence_plot_vertical.png \
        ./report/figures/coherence_plot_vertical.png
    modal volume get --force worldmodel-data /coherence_plot_vertical.pdf \
        ./report/figures/coherence_plot_vertical.pdf
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow", "imageio", "matplotlib")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "rollouts", ".git", "*.tex", "*.log"],
    )
)

app = modal.App("worldmodel-coherence", image=image)
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 60)
def plot_remote(n_seqs: int = 30):
    import sys
    sys.path.insert(0, "/root/project")

    import torch
    import make_coherence_plot

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Write outputs to the volume so we can download them.
    make_coherence_plot.make_plot(
        "/data/coherence_plot_vertical",
        device="cuda",
        data_dir="/data",
        n_seqs=n_seqs,
    )
    volume.commit()
    print("\nDone. Fetch with:")
    print("  modal volume get --force worldmodel-data "
          "/coherence_plot_vertical.png ./report/figures/coherence_plot_vertical.png")
    print("  modal volume get --force worldmodel-data "
          "/coherence_plot_vertical.pdf ./report/figures/coherence_plot_vertical.pdf")


@app.local_entrypoint()
def main(n_seqs: int = 30):
    plot_remote.remote(n_seqs=n_seqs)
