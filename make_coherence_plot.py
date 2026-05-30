"""
make_coherence_plot.py – Regenerate Figure 3 with token-model rollout curves added.

Outputs:
    report/figures/coherence_plot.png   (overwrites existing figure)
    report/figures/coherence_plot.pdf   (overwrites existing figure)

Deterministic model (64x64):
    checkpoint.pt     -> bright_weight=8.0  (freeze regime)
    ckpt_static.pt    -> bright_weight=0.0  (blow-up regime)

Token model (128x128):
    tokenizer_128.pt + transformer.pt

Metrics per rollout step t:
    max_pixel(t)  = max pixel value of the predicted frame at step t
    motion(t)     = mean-abs per-pixel delta between consecutive predicted frames
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Deterministic rollout
# ---------------------------------------------------------------------------

def load_det_model(ckpt_path, device):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from models import WorldModel
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    model = WorldModel(
        in_channels=a["num_frames"],
        out_channels=1,
        action_dim=a["action_dim"],
        latent_dim=a["latent_dim"],
        hidden_dim=a["hidden_dim"],
        decoder_type=a["decoder_type"],
        residual_scale=a["residual_scale"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, a


@torch.no_grad()
def det_rollout_metrics(ckpt_path, data_path, n_seqs=20, rollout_len=20, device="cpu"):
    model, a = load_det_model(ckpt_path, device)
    nf = a["num_frames"]

    blob = torch.load(data_path, map_location=device)
    frames_all = blob["frames"]   # (N, T, 1, 64, 64)
    actions_all = blob["actions"] # (N, T)
    N = min(n_seqs, frames_all.shape[0])

    max_pixels = np.zeros((N, rollout_len))
    motions = np.zeros((N, rollout_len))

    for i in range(N):
        seq = frames_all[i]      # (T, 1, 64, 64)
        acts = actions_all[i]    # (T,)
        T = seq.shape[0]

        # prime with first nf real frames
        stack = seq[:nf].unsqueeze(0).squeeze(2)  # (1, nf, 64, 64)
        h = model.init_hidden(1, device, seq.dtype)
        prev = seq[nf - 1]  # (1, 64, 64)

        for t in range(rollout_len):
            src_t = min(nf - 1 + t, T - 2)
            a_oh = F.one_hot(acts[src_t:src_t+1].long(), num_classes=a["action_dim"]).float()
            x_hat, _, _, h, _ = model(stack, action=a_oh, h=h, deterministic=True, return_aux=True)
            x_hat = x_hat.squeeze(0)  # (1, 64, 64)

            max_pixels[i, t] = x_hat.max().item()
            motions[i, t] = (x_hat - prev).abs().mean().item()

            prev = x_hat
            stack = torch.cat([stack[:, 1:], x_hat.unsqueeze(0)], dim=1)

    return max_pixels.mean(0), motions.mean(0)


# ---------------------------------------------------------------------------
# Token model rollout
# ---------------------------------------------------------------------------

def load_token_models(tok_path, trans_path, device):
    from tokenizer import VQVAE
    from transformer import TokenWorldModel

    tb = torch.load(tok_path, map_location=device)
    ta = tb["args"]
    vq = VQVAE(in_channels=1, hidden=ta["hidden"], embedding_dim=ta["embedding_dim"],
               num_codes=ta["num_codes"], commitment_cost=ta.get("commitment_cost", 0.25)).to(device)
    vq.load_state_dict(tb["model"])
    vq.eval()

    xb = torch.load(trans_path, map_location=device)
    c = xb["config"]
    gpt = TokenWorldModel(
        num_codes=c["num_codes"], action_dim=c["action_dim"],
        tokens_per_frame=c["tokens_per_frame"], context_frames=c["context_frames"],
        d_model=c["d_model"], n_layers=c["n_layers"], n_heads=c["n_heads"], dropout=0.0,
    ).to(device)
    gpt.load_state_dict(xb["model"])
    gpt.eval()
    return vq, gpt, c


@torch.no_grad()
def token_rollout_metrics(tok_path, trans_path, data_path, n_seqs=20,
                          prime=4, temperature=1.0, top_k=50, device="cpu"):
    from dataset import SavedSequenceDataset
    vq, gpt, cfg = load_token_models(tok_path, trans_path, device)
    P = cfg["tokens_per_frame"]
    grid = int(P ** 0.5)

    ds = SavedSequenceDataset(data_path)
    N = min(n_seqs, len(ds))

    rollout_len = ds.frames.shape[1] - prime  # frames to generate
    max_pixels = np.zeros((N, rollout_len))
    motions = np.zeros((N, rollout_len))

    for i in range(N):
        torch.manual_seed(42 + i)
        seq = ds.frames[i].to(device)    # (T, 1, H, W)
        acts = ds.actions[i].long().to(device)

        prime_idx = vq.encode_to_indices(seq[:prime])        # (prime, grid, grid)
        prime_tokens = prime_idx.reshape(1, prime, P)
        actions = acts.unsqueeze(0)

        gen = gpt.generate(prime_tokens, actions, n_new_frames=rollout_len,
                           temperature=temperature, top_k=top_k)  # (1, rollout_len, P)
        gen_idx = gen.reshape(rollout_len, grid, grid)
        gen_frames = vq.decode_from_indices(gen_idx)         # (rollout_len, 1, H, W)

        prev = seq[prime - 1]
        for t in range(rollout_len):
            f = gen_frames[t]
            max_pixels[i, t] = f.max().item()
            motions[i, t] = (f - prev).abs().mean().item()
            prev = f

    return max_pixels.mean(0), motions.mean(0)


# ---------------------------------------------------------------------------
# Ground-truth curves
# ---------------------------------------------------------------------------

@torch.no_grad()
def gt_curves(data_path, prime=4, n_seqs=20, device="cpu"):
    blob = torch.load(data_path, map_location=device)
    frames_all = blob["frames"]   # (N, T, 1, H, W)
    N = min(n_seqs, frames_all.shape[0])
    T = frames_all.shape[1]
    rollout_len = T - prime

    max_pixels = np.zeros((N, rollout_len))
    motions = np.zeros((N, rollout_len))
    for i in range(N):
        seq = frames_all[i]
        prev = seq[prime - 1]
        for t in range(rollout_len):
            f = seq[prime + t]
            max_pixels[i, t] = f.max().item()
            motions[i, t] = (f - prev).abs().mean().item()
            prev = f
    return max_pixels.mean(0), motions.mean(0)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(out_prefix, device="cpu", data_dir=None, n_seqs=30):
    import os
    base = data_dir if data_dir is not None else os.path.dirname(__file__)
    data64  = os.path.join(base, "atari_data.pt")
    data128 = os.path.join(base, "atari_data_128.pt")
    ckpt_freeze  = os.path.join(base, "ckpt_freeze.pt")
    ckpt_blowup  = os.path.join(base, "ckpt_static.pt")
    tok_path     = os.path.join(base, "tokenizer_128.pt")
    trans_path   = os.path.join(base, "transformer.pt")

    N = n_seqs
    prime = 4

    print("Running deterministic blow-up rollout...")
    bu_px, bu_mo = det_rollout_metrics(ckpt_blowup, data64, n_seqs=N, device=device)
    print("Running deterministic freeze rollout...")
    fr_px, fr_mo = det_rollout_metrics(ckpt_freeze, data64, n_seqs=N, device=device)
    print("Running token rollout...")
    tok_px, tok_mo = token_rollout_metrics(tok_path, trans_path, data128,
                                           n_seqs=N, prime=prime, device=device)
    print("Computing ground-truth curves...")
    gt_px, gt_mo = gt_curves(data128, prime=prime, n_seqs=N, device=device)

    steps_det = np.arange(1, len(bu_px) + 1)
    steps_tok = np.arange(1, len(tok_px) + 1)
    steps_gt  = np.arange(1, len(gt_px) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=False)

    # --- Top panel: max pixel ---
    ax = axes[0]
    ax.plot(steps_det, bu_px, color="#d62728", lw=1.5, label="Det. blow-up")
    ax.plot(steps_det, fr_px, color="#ff7f0e", lw=1.5, label="Det. freeze")
    ax.plot(steps_tok, tok_px, color="#1f77b4", lw=1.5, label="Token rollout")
    ax.plot(steps_gt,  gt_px,  color="gray",    lw=1.2, ls="--", label="Ground truth")
    ax.set_ylabel("Max pixel value", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Rollout stability: max pixel value over time", fontsize=9)
    ax.tick_params(labelsize=8)

    # --- Bottom panel: per-step motion ---
    ax = axes[1]
    ax.plot(steps_det, bu_mo * 1e3, color="#d62728", lw=1.5, label="Det. blow-up")
    ax.plot(steps_det, fr_mo * 1e3, color="#ff7f0e", lw=1.5, label="Det. freeze")
    ax.plot(steps_tok, tok_mo * 1e3, color="#1f77b4", lw=1.5, label="Token rollout")
    ax.plot(steps_gt,  gt_mo * 1e3,  color="gray",    lw=1.2, ls="--", label="Ground truth")
    ax.set_xlabel("Rollout step", fontsize=9)
    ax.set_ylabel("Per-pixel motion (×10⁻³)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Rollout stability: per-step motion", fontsize=9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="report/figures/coherence_plot_vertical")
    p.add_argument("--data_dir", default=None,
                   help="Directory holding .pt files (defaults to script dir)")
    p.add_argument("--n_seqs", type=int, default=30)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    else:
        dev = args.device

    print(f"Using device: {dev}")
    make_plot(args.out, device=dev, data_dir=args.data_dir, n_seqs=args.n_seqs)
