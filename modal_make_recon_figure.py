"""
modal_make_recon_figure.py – Generate a clean VQ-VAE reconstruction figure
for the paper (report/figures/fs_recon.png).

The current fs_recon.png has two problems:
  1. The ball is missing – the frames chosen didn't show it.
  2. The step labels in the top row were incorrect.

This script fixes both by:
  - Scanning the dataset for frames where the ball is clearly visible
    (bright pixel cluster in the field of play, away from the paddle row).
  - Picking N_COLS diverse frames across different sequences and timesteps.
  - Rendering a clean 2-row figure: top = ground truth, bottom = reconstruction.
  - Labels: "GT" / "Recon" on the y-axis; no per-frame numbers (avoids the
    label-correctness issue entirely).

Run:
    modal run modal_make_recon_figure.py

Download:
    modal volume get --force worldmodel-data /fs_recon.png \
        ./report/figures/fs_recon.png
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow", "matplotlib")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "rollouts", ".git", "*.tex", "*.log"],
    )
)

app = modal.App("worldmodel-recon-figure", image=image)
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 20)
def make_recon_figure(
    tokenizer_ckpt: str = "/data/tokenizer_128_v2.pt",
    data_file: str = "/data/atari_data_128_v2.pt",
    out_path: str = "/data/fs_recon.png",
    n_cols: int = 6,          # number of example frames shown
    dpi: int = 200,
):
    import sys
    sys.path.insert(0, "/root/project")

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenizer import VQVAE
    from dataset import SavedSequenceDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load tokenizer ─────────────────────────────────────────────────────
    blob = torch.load(tokenizer_ckpt, map_location=device)
    a = blob["args"]
    vq = VQVAE(
        in_channels=1,
        hidden=a["hidden"],
        embedding_dim=a["embedding_dim"],
        num_codes=a["num_codes"],
        commitment_cost=a.get("commitment_cost", 0.25),
    ).to(device)
    vq.load_state_dict(blob["model"])
    vq.eval()
    print(f"Loaded tokenizer from {tokenizer_ckpt}  (epoch {blob['epoch']}, "
          f"recon={blob['recon']:.5f})")

    # ── Load dataset ────────────────────────────────────────────────────────
    ds = SavedSequenceDataset(data_file)
    all_frames = ds.frames  # (N_seq, T, 1, H, W)  values in [0,1]
    N_seq, T, C, H, W = all_frames.shape
    print(f"Dataset: {N_seq} sequences × {T} frames, {H}×{W}")

    # ── Find frames where the ball is clearly visible ───────────────────────
    # Heuristic: the ball is a small bright object (>0.7) that lives in the
    # "field of play" – roughly rows 8..H-24 (above the paddle, below the score).
    # We look for at least 1 pixel that bright in that band.
    # Additionally require that the ball isn't just the paddle (which sits at
    # the bottom ~10 rows) – check the middle 60% of the image height.
    field_top = H // 8          # skip score bar at very top
    field_bot = int(H * 0.78)   # stop before paddle row
    ball_thresh = 0.65

    candidates = []  # list of (seq_idx, t, score)
    for seq_i in range(N_seq):
        for t in range(T):
            f = all_frames[seq_i, t, 0]        # (H, W)  numpy float32
            field = f[field_top:field_bot, :]
            bright_count = int((field > ball_thresh).sum())
            # Want a small bright region (ball ~2-8 px²) – reject frames where
            # too many pixels are bright (just the score row artifact).
            if 1 <= bright_count <= 40:
                # Score = brightness × pixel count for ranking
                score = float(field[field > ball_thresh].sum())
                candidates.append((seq_i, t, score, bright_count))

    print(f"Found {len(candidates)} candidate frames with visible ball.")

    # ── Diverse selection: spread across different sequences ────────────────
    # Sort by sequence index, then stride to pick n_cols evenly spaced.
    candidates.sort(key=lambda x: (x[0], x[1]))

    # Pick frames from different sequences (diversity)
    if len(candidates) >= n_cols:
        step = max(1, len(candidates) // (n_cols * 4))
        strided = candidates[::step]
        # De-duplicate by sequence – prefer one frame per sequence
        seen_seq = set()
        selected = []
        for c in strided:
            if c[0] not in seen_seq:
                selected.append(c)
                seen_seq.add(c[0])
            if len(selected) == n_cols:
                break
        # If still short, fill from remaining candidates
        if len(selected) < n_cols:
            for c in candidates:
                if len(selected) == n_cols:
                    break
                if c[0] not in {s[0] for s in selected}:
                    selected.append(c)
        # Last resort: just take first n_cols
        if len(selected) < n_cols:
            selected = candidates[:n_cols]
    else:
        selected = candidates[:n_cols]
        print(f"WARNING: only {len(selected)} ball-visible frames found; "
              f"filling with best available.")

    print(f"Selected {len(selected)} frames:")
    for seq_i, t, score, bc in selected:
        print(f"  seq={seq_i:4d}  t={t:2d}  bright_px={bc:2d}  score={score:.3f}")

    # ── Run through tokenizer ───────────────────────────────────────────────
    gt_frames = []
    rc_frames = []
    with torch.no_grad():
        for seq_i, t, _, _ in selected:
            x = all_frames[seq_i, t].to(device).unsqueeze(0)  # (1,1,H,W)
            x_recon, _, _, _ = vq(x)
            gt_frames.append(x[0, 0].cpu().numpy())           # (H, W)
            rc_frames.append(x_recon[0, 0].cpu().numpy())

    # ── Quick quality check ─────────────────────────────────────────────────
    for i, (g, r) in enumerate(zip(gt_frames, rc_frames)):
        field_g = g[field_top:field_bot]
        field_r = r[field_top:field_bot]
        ball_gt  = int((field_g > ball_thresh).sum())
        ball_rec = int((field_r > ball_thresh).sum())
        print(f"  frame {i}: GT bright_px={ball_gt}  Recon bright_px={ball_rec}")

    # ── Build figure ────────────────────────────────────────────────────────
    # Two rows (GT / Recon), n_cols columns.
    # No per-column numbers – y-axis labels only.
    n = len(selected)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.4),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})

    row_labels = ["Ground\ntruth", "Recon"]
    for row in range(2):
        frames_row = gt_frames if row == 0 else rc_frames
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(frames_row[col], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Left-most column: row label
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, labelpad=4,
                              rotation=0, ha="right", va="center")

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved figure → {out_path}")

    volume.commit()
    print("\nDone. Download with:")
    print(f"  modal volume get --force worldmodel-data {out_path} "
          f"./report/figures/fs_recon.png")


@app.local_entrypoint()
def main(
    tokenizer_ckpt: str = "/data/tokenizer_128_v2.pt",
    data_file: str = "/data/atari_data_128_v2.pt",
    out_path: str = "/data/fs_recon.png",
    n_cols: int = 6,
):
    make_recon_figure.remote(
        tokenizer_ckpt=tokenizer_ckpt,
        data_file=data_file,
        out_path=out_path,
        n_cols=n_cols,
    )
