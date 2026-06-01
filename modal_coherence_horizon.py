"""
modal_coherence_horizon.py – Per-frame MAE error curves comparing rollouts to GT.

Generates report/figures/coherence_horizon.png showing how quickly each model
diverges from ground truth, measured by mean-absolute-error per pixel at each
rollout step.  This directly quantifies the "coherence horizon" mentioned in
the paper's future-work section.

Three curves:
  - Token model rollout    (blue, with ±1σ shaded band)
  - Copy-last-frame        (gray dashed — stationary baseline)
  - Ground-truth self-error (gray dotted — lower-bound noise floor)

Resolution: everything computed at 128×128 (token model's native resolution).
The deterministic model runs at 64×64 so is not included here (its rollout
fails within ~6 steps anyway, which is already shown in the coherence plot).

Run:
    modal run modal_coherence_horizon.py

Download:
    modal volume get --force worldmodel-data /coherence_horizon.png \
        ./report/figures/coherence_horizon.png
    modal volume get --force worldmodel-data /coherence_horizon.pdf \
        ./report/figures/coherence_horizon.pdf
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "matplotlib")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["_trash_cleanup", "__pycache__", "*.gif", "*.pt", "*.png",
                "*.pdf", "rollouts", ".git", "*.tex", "*.log"],
    )
)

app    = modal.App("worldmodel-coherence-horizon", image=image)
volume = modal.Volume.from_name("worldmodel-data", create_if_missing=True)


# ── Remote function ────────────────────────────────────────────────────────────

@app.function(gpu="A100", volumes={"/data": volume}, timeout=60 * 30)
def compute_and_plot(
    tok_ckpt:   str = "/data/tokenizer_128_v2.pt",
    trans_ckpt: str = "/data/transformer_v2.pt",
    data_file:  str = "/data/atari_data_128_v2.pt",
    out_prefix: str = "/data/coherence_horizon",
    n_seqs:     int = 30,
    prime:      int = 4,
    temperature: float = 1.0,
    top_k:      int = 50,
):
    import sys
    sys.path.insert(0, "/root/project")

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenizer import VQVAE
    from transformer import TokenWorldModel
    from dataset import SavedSequenceDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load models ────────────────────────────────────────────────────────
    tb = torch.load(tok_ckpt, map_location=device)
    ta = tb["args"]
    vq = VQVAE(in_channels=1, hidden=ta["hidden"], embedding_dim=ta["embedding_dim"],
               num_codes=ta["num_codes"], commitment_cost=ta.get("commitment_cost", 0.25)).to(device)
    vq.load_state_dict(tb["model"]); vq.eval()
    print(f"Tokenizer: epoch={tb['epoch']} recon={tb['recon']:.5f}")

    xb = torch.load(trans_ckpt, map_location=device)
    c  = xb["config"]
    gpt = TokenWorldModel(
        num_codes=c["num_codes"], action_dim=c["action_dim"],
        tokens_per_frame=c["tokens_per_frame"], context_frames=c["context_frames"],
        d_model=c["d_model"], n_layers=c["n_layers"], n_heads=c["n_heads"], dropout=0.0,
    ).to(device)
    gpt.load_state_dict(xb["model"]); gpt.eval()
    print(f"Transformer: config={c}")

    # ── Load dataset ───────────────────────────────────────────────────────
    ds = SavedSequenceDataset(data_file)
    N  = min(n_seqs, len(ds))
    T  = ds.frames.shape[1]
    rollout_len = T - prime
    P    = c["tokens_per_frame"]
    grid = int(P ** 0.5)
    print(f"Dataset: {len(ds)} seqs × {T} frames | rolling out {rollout_len} steps")

    # ── Collect per-step MAE for each model/baseline ───────────────────────
    # mae[seq, step] = mean |pred_t - gt_t|  (pixel-level, frame at step t)
    tok_mae  = np.zeros((N, rollout_len))   # token model
    copy_mae = np.zeros((N, rollout_len))   # copy-last-frame
    gt_mae   = np.zeros((N, rollout_len))   # gt vs. a different-seed gt (noise floor)
    #   gt noise floor: compare gt_frame[t] to gt_frame[t-1] shifted by 1 seq
    #   (shows irreducible stochasticity / inter-sequence variance)

    for i in range(N):
        torch.manual_seed(42 + i)
        seq  = ds.frames[i].to(device)     # (T, 1, H, W)
        acts = ds.actions[i].long().to(device)

        # -- Token rollout --------------------------------------------------
        prime_idx    = vq.encode_to_indices(seq[:prime])           # (prime, grid, grid)
        prime_tokens = prime_idx.reshape(1, prime, P)
        actions_in   = acts.unsqueeze(0)

        gen = gpt.generate(
            prime_tokens, actions_in,
            n_new_frames=rollout_len,
            temperature=temperature, top_k=top_k,
        )                                                           # (1, rollout_len, P)
        gen_idx    = gen.reshape(rollout_len, grid, grid)
        gen_frames = vq.decode_from_indices(gen_idx)               # (rollout_len, 1, H, W)

        for t in range(rollout_len):
            gt_t   = seq[prime + t]                                # (1, H, W)
            pred_t = gen_frames[t]                                 # (1, H, W)
            tok_mae[i, t]  = (pred_t - gt_t).abs().mean().item()
            copy_mae[i, t] = (seq[prime - 1] - gt_t).abs().mean().item()

        # -- GT noise floor: compare each GT frame to the same frame from seq i+1
        j = (i + 1) % N
        seq_j = ds.frames[j].to(device)
        for t in range(rollout_len):
            gt_mae[i, t] = (seq[prime + t] - seq_j[prime + t]).abs().mean().item()

        if (i + 1) % 5 == 0:
            print(f"  Done {i+1}/{N} sequences")

    # ── Summarise ──────────────────────────────────────────────────────────
    steps = np.arange(1, rollout_len + 1)

    tok_mean,  tok_std  = tok_mae.mean(0),  tok_mae.std(0)
    copy_mean, copy_std = copy_mae.mean(0), copy_mae.std(0)
    gt_mean,   gt_std   = gt_mae.mean(0),   gt_mae.std(0)

    print(f"\nToken model MAE at step 1:  {tok_mean[0]:.5f}")
    print(f"Token model MAE at step {rollout_len}: {tok_mean[-1]:.5f}")
    print(f"Copy-last MAE at step 1:   {copy_mean[0]:.5f}")
    print(f"Copy-last MAE at step {rollout_len}: {copy_mean[-1]:.5f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.8))

    # Token model
    ax.plot(steps, tok_mean * 1e3, color="#1f77b4", lw=1.8,
            label="Token rollout")
    ax.fill_between(steps,
                    (tok_mean - tok_std) * 1e3,
                    (tok_mean + tok_std) * 1e3,
                    color="#1f77b4", alpha=0.18)

    # Copy-last-frame
    ax.plot(steps, copy_mean * 1e3, color="#7f7f7f", lw=1.4,
            ls="--", label="Copy-last-frame")
    ax.fill_between(steps,
                    (copy_mean - copy_std) * 1e3,
                    (copy_mean + copy_std) * 1e3,
                    color="#7f7f7f", alpha=0.12)

    # GT inter-sequence noise floor
    ax.plot(steps, gt_mean * 1e3, color="#bcbcbc", lw=1.1,
            ls=":", label="GT inter-seq. variance")

    ax.set_xlabel("Rollout step", fontsize=10)
    ax.set_ylabel("Mean absolute error (×10⁻³)", fontsize=10)
    ax.set_title(
        f"Per-step MAE vs. ground truth  (n={N} sequences, prime={prime} frames)",
        fontsize=9,
    )
    ax.legend(fontsize=8.5)
    ax.tick_params(labelsize=9)
    ax.set_xlim(1, rollout_len)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = f"{out_prefix}.{ext}"
        plt.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()

    volume.commit()
    print("\nDone. Download with:")
    print(f"  modal volume get --force worldmodel-data {out_prefix}.png "
          f"./report/figures/coherence_horizon.png")
    print(f"  modal volume get --force worldmodel-data {out_prefix}.pdf "
          f"./report/figures/coherence_horizon.pdf")


# ── Local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    tok_ckpt:    str = "/data/tokenizer_128_v2.pt",
    trans_ckpt:  str = "/data/transformer_v2.pt",
    data_file:   str = "/data/atari_data_128_v2.pt",
    out_prefix:  str = "/data/coherence_horizon",
    n_seqs:      int = 30,
    prime:       int = 4,
    temperature: float = 1.0,
    top_k:       int = 50,
):
    compute_and_plot.remote(
        tok_ckpt=tok_ckpt,
        trans_ckpt=trans_ckpt,
        data_file=data_file,
        out_prefix=out_prefix,
        n_seqs=n_seqs,
        prime=prime,
        temperature=temperature,
        top_k=top_k,
    )
