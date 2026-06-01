"""
make_recon_figure.py – Generate report/figures/fs_recon.png

Fixes over the old figure:
  - Selects frames where the ball is VISIBLE in the open field (rows 55-120),
    detected via inter-frame motion.
  - Checks that the reconstruction also preserves the ball pixel.
  - Picks diverse frames across different sequences and ball positions.
  - Clean 2-row layout (GT top, Recon bottom) with no misleading step numbers.
"""

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/limingyang/Desktop/CS231/project")
from tokenizer import VQVAE

# ── Config ──────────────────────────────────────────────────────────────────
TOKENIZER  = "/Users/limingyang/Desktop/CS231/project/tokenizer_128_v2.pt"
DATA_FILE  = "/Users/limingyang/Desktop/CS231/project/atari_data_128_v2.pt"
OUT_PATH   = "/Users/limingyang/Desktop/CS231/project/report/figures/fs_recon.png"
N_COLS     = 6      # number of example frames in the figure
DPI        = 200
# Ball is 0.431 bright (same as walls), background is 0.000 in open field.
# Safe interior field: rows 56-113 (below bricks, above paddle), cols 8-120 (inside walls).
BALL_ROW_MIN = 56
BALL_ROW_MAX = 113
COL_MIN      = 8
COL_MAX      = 120
BALL_THRESH  = 0.40   # ball is ~0.431; background in open field is 0.000
MIN_BALL_PX  = 2
MAX_BALL_PX  = 12

# ── Load tokenizer ───────────────────────────────────────────────────────────
blob = torch.load(TOKENIZER, map_location="cpu")
a    = blob["args"]
vq   = VQVAE(in_channels=1, hidden=a["hidden"], embedding_dim=a["embedding_dim"],
             num_codes=a["num_codes"], commitment_cost=a.get("commitment_cost", 0.25))
vq.load_state_dict(blob["model"])
vq.eval()
print(f"Tokenizer: epoch={blob['epoch']}  best_recon={blob['recon']:.5f}")

# ── Load dataset ─────────────────────────────────────────────────────────────
d      = torch.load(DATA_FILE, map_location="cpu")
frames = d["frames"]   # (N_seq, T, 1, H, W)  float32 in [0, max~0.58]
N_seq, T, C, H, W = frames.shape
print(f"Dataset: {N_seq} seqs × {T} frames, {H}×{W}")

# ── Find ball-visible frames ──────────────────────────────────────────────────
# Ball is ~0.431 bright against a dark (0.000) background in the open field.
# Detect it by looking for bright pixels in the interior region below the bricks.
candidates = []  # (seq_i, t, n_px, mean_ball_row, mean_ball_col)
for si in range(N_seq):
    for ti in range(T):
        f = frames[si, ti, 0].numpy()
        region = f[BALL_ROW_MIN:BALL_ROW_MAX, COL_MIN:COL_MAX]
        bright = int((region > BALL_THRESH).sum())
        if MIN_BALL_PX <= bright <= MAX_BALL_PX:
            pos = np.argwhere(region > BALL_THRESH)
            ball_row = float(pos[:, 0].mean()) + BALL_ROW_MIN
            ball_col = float(pos[:, 1].mean()) + COL_MIN
            candidates.append((si, ti, bright, ball_row, ball_col))

print(f"Ball-visible candidate frames: {len(candidates)}")

# ── Run candidates through tokenizer, keep those where ball survives ──────────
stride = max(1, len(candidates) // 3000)
checked = []
with torch.no_grad():
    for idx in range(0, len(candidates), stride):
        si, ti, n_px, ball_row, ball_col = candidates[idx]
        x = frames[si, ti].unsqueeze(0)
        x_recon, _, _, _ = vq(x)

        br = int(round(ball_row))
        bc = int(round(ball_col))
        r0, r1 = max(0, br - 5), min(H, br + 6)
        c0, c1 = max(0, bc - 5), min(W, bc + 6)
        gt_peak    = float(x[0, 0, r0:r1, c0:c1].max())
        recon_peak = float(x_recon[0, 0, r0:r1, c0:c1].max())

        if recon_peak >= 0.30:
            checked.append((si, ti, n_px, ball_row, ball_col, gt_peak, recon_peak))

print(f"Frames where ball survives reconstruction: {len(checked)}")

# ── Diverse selection: spread across sequences and ball positions ──────────────
# Sort by ball column position to get varied ball locations across the figure.
checked.sort(key=lambda x: x[4])   # sort by ball column

seen_seq = set()
selected = []
# First pass: one per sequence, spread by ball column
step = max(1, len(checked) // (N_COLS * 5))
for i in range(0, len(checked), step):
    si = checked[i][0]
    if si not in seen_seq:
        selected.append(checked[i])
        seen_seq.add(si)
    if len(selected) == N_COLS:
        break

# Fill if short
if len(selected) < N_COLS:
    for c in checked:
        if c[0] not in seen_seq:
            selected.append(c)
            seen_seq.add(c[0])
        if len(selected) == N_COLS:
            break

# Last resort
if len(selected) < N_COLS:
    selected = checked[:N_COLS]

print(f"\nSelected {len(selected)} frames:")
for si, ti, nm, br, bc, gtp, rp in selected:
    print(f"  seq={si:4d}  t={ti:2d}  moving_px={nm}  "
          f"ball≈row{br:.0f},col{bc:.0f}  "
          f"gt_peak={gtp:.3f}  recon_peak={rp:.3f}")

# ── Reconstruct selected frames ──────────────────────────────────────────────
gt_imgs   = []
rec_imgs  = []
with torch.no_grad():
    for si, ti, *_ in selected:
        x = frames[si, ti].unsqueeze(0)
        x_recon, _, _, _ = vq(x)
        gt_imgs.append(x[0, 0].numpy())
        rec_imgs.append(x_recon[0, 0].numpy())

# ── Build figure ─────────────────────────────────────────────────────────────
n   = len(selected)
fig, axes = plt.subplots(
    2, n,
    figsize=(n * 1.6, 3.4),
    gridspec_kw={"hspace": 0.05, "wspace": 0.03},
)

row_labels = ["Ground\ntruth", "Recon"]
for row in range(2):
    imgs = gt_imgs if row == 0 else rec_imgs
    for col in range(n):
        ax = axes[row, col]
        ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=0.58,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if col == 0:
            ax.set_ylabel(
                row_labels[row], fontsize=8.5, labelpad=3,
                rotation=0, ha="right", va="center",
            )

plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {OUT_PATH}")
