"""
utils.py – Helper functions for the latent world model.
"""

import os
import torch
import numpy as np
from typing import List


def save_checkpoint(model_dict: dict, path: str = "checkpoint.pt") -> None:
    """Save a dictionary of model state dicts to disk."""
    torch.save(model_dict, path)
    print(f"[✓] Checkpoint saved → {path}")


def load_checkpoint(path: str = "checkpoint.pt", device: str = "cpu") -> dict:
    """Load a checkpoint dictionary from disk."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    print(f"[✓] Checkpoint loaded ← {path}")
    return ckpt


def frames_to_gif(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    """
    Save a list of numpy frames (H, W) or (H, W, C) as an animated GIF.
    Uses imageio if available, otherwise falls back to matplotlib.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    try:
        import imageio.v2 as imageio
        # Convert to uint8
        uint8_frames = []
        for f in frames:
            f = np.clip(f, 0.0, 1.0)
            uint8_frames.append((f * 255).astype(np.uint8))
        imageio.mimsave(path, uint8_frames, fps=fps, loop=0)
    except ImportError:
        # Fallback: save as a filmstrip PNG via matplotlib
        import matplotlib.pyplot as plt
        n = len(frames)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        if n == 1:
            axes = [axes]
        for ax, f in zip(axes, frames):
            ax.imshow(f, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        fig.tight_layout(pad=0.2)
        png_path = path.replace(".gif", ".png")
        fig.savefig(png_path, dpi=100)
        plt.close(fig)
        print(f"  [imageio not found — saved filmstrip to {png_path}]")
        return
    print(f"[✓] GIF saved → {path}")


def tensor_to_numpy_frames(tensor: torch.Tensor) -> List[np.ndarray]:
    """
    Convert a (T, 1, H, W) or (T, C, H, W) tensor to a list of numpy arrays
    suitable for GIF creation.
    """
    frames = []
    for t in range(tensor.shape[0]):
        img = tensor[t].detach().cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]  # (H, W)
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        frames.append(img)
    return frames


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
