"""
export_dataset.py – Generate the Atari dataset once and save it to a file.

Run this LOCALLY (where ALE is installed). The output file can then be uploaded
to a Modal Volume so cloud GPU training loads tensors directly — no ALE in the
cloud, no per-run regeneration, and a fixed reproducible dataset for the report.

Usage:
    python3 export_dataset.py --num_sequences 2000 --seq_len 20 --out atari_data.pt

Then upload to Modal:
    modal volume create worldmodel-data          # once
    modal volume put worldmodel-data atari_data.pt /atari_data.pt
"""

import argparse
import torch

from dataset import AtariDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sequences", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="atari_data.pt")
    args = p.parse_args()

    ds = AtariDataset(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        img_size=args.img_size,
        seed=args.seed,
    )

    print(f"\nStacking {len(ds)} sequences into tensors...")
    frames = torch.stack([ds[i]["frames"] for i in range(len(ds))], dim=0)   # (N,T,1,64,64)
    actions = torch.stack([ds[i]["actions"] for i in range(len(ds))], dim=0)  # (N,T)

    torch.save({"frames": frames, "actions": actions}, args.out)

    size_mb = (frames.numel() * 4 + actions.numel() * 8) / 1e6
    print(f"Saved {args.out}")
    print(f"  frames : {tuple(frames.shape)}  dtype={frames.dtype}")
    print(f"  actions: {tuple(actions.shape)}  dtype={actions.dtype}")
    print(f"  approx size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
