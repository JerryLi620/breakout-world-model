"""
train_tokenizer.py – Stage 1 training for the discrete-token world model.

Trains the VQ-VAE tokenizer on individual Atari Breakout frames. The dynamics
transformer (Stage 2) is built only AFTER this reconstructs frames crisply.

Success criteria:
    1. Reconstruction shows a visible ball AND paddle (not just bricks).
    2. perplexity stays well above a handful of codes (no codebook collapse).

Recommended run:
    python3 train_tokenizer.py --num_sequences 2000 --seq_len 20 \
        --num_codes 512 --embedding_dim 256 --batch_size 64 --epochs 30 \
        --object_weight 30

Outputs:
    tokenizer.pt
    tokenizer_recon.gif   (top row GT vs reconstructed frames, interleaved)
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokenizer import VQVAE
from utils import set_seed, frames_to_gif, tensor_to_numpy_frames


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="atari", choices=["atari", "ball"])
    p.add_argument("--num_sequences", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--num_codes", type=int, default=512)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--commitment_cost", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Data-loader batch (sequences). Frames per optimizer step "
                        "is capped by --frame_batch, so this can stay large.")
    p.add_argument("--frame_batch", type=int, default=128,
                   help="Max frames pushed through the model per step (memory cap). "
                        "Lower this if you still hit MPS/GPU OOM.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--checkpoint", type=str, default="tokenizer.pt")
    p.add_argument("--recon_gif", type=str, default="tokenizer_recon.gif")
    p.add_argument("--data_file", type=str, default="",
                   help="If set, load sequences from this .pt file (export_dataset.py) "
                        "instead of generating via ALE. Used for cloud/Modal training.")
    # The ball is tiny + bright; weight bright pixels so it isn't ignored.
    p.add_argument("--object_weight", type=float, default=30.0)
    p.add_argument("--object_threshold", type=float, default=0.05)
    p.add_argument("--debug_every", type=int, default=100)
    return p.parse_args()


def get_device(req):
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(args):
    if args.data_file:
        from dataset import SavedSequenceDataset
        print(f"  Loading dataset from file: {args.data_file}")
        return SavedSequenceDataset(args.data_file)
    if args.dataset == "atari":
        from dataset import AtariDataset
        return AtariDataset(num_sequences=args.num_sequences, seq_len=args.seq_len, seed=args.seed)
    from dataset import BouncingBallDataset
    return BouncingBallDataset(num_sequences=args.num_sequences, seq_len=args.seq_len, seed=args.seed)


def recon_loss_fn(x_recon, x, object_weight, object_threshold):
    """Object-weighted reconstruction so the tiny bright ball/paddle survive."""
    per_pixel = F.smooth_l1_loss(x_recon, x, beta=0.01, reduction="none")
    object_mask = (x > object_threshold).float()
    weight = 1.0 + object_weight * object_mask
    return (weight * per_pixel).mean()


def train():
    run(get_args())


def run(args):
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"\n{'='*70}")
    print("  VQ-VAE Tokenizer – Stage 1 Training")
    print(f"{'='*70}")
    print(f"  Device         : {device}")
    print(f"  Codebook (K)   : {args.num_codes}")
    print(f"  Embedding (D)  : {args.embedding_dim}")
    print(f"  Object weight  : {args.object_weight}")
    print(f"  Batch size     : {args.batch_size}  Epochs: {args.epochs}  LR: {args.lr}")
    print(f"{'='*70}\n")

    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    print(f"  {len(dataset)} sequences loaded.\n")

    model = VQVAE(
        in_channels=1, hidden=args.hidden, embedding_dim=args.embedding_dim,
        num_codes=args.num_codes, commitment_cost=args.commitment_cost,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    global_step = 0
    fixed_seq = None  # for reconstruction GIF

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_recon = ep_commit = ep_ppl = 0.0
        nb = 0
        t0 = time.time()

        for batch in loader:
            frames = batch["frames"].to(device)        # (B, T, 1, 64, 64)
            if fixed_seq is None:
                fixed_seq = frames[0].detach().clone()  # (T, 1, 64, 64)

            B, T, C, H, W = frames.shape
            x_all = frames.view(B * T, C, H, W)         # all frames in this batch

            # The tokenizer trains on individual frames, so chunk the flattened
            # frames into memory-safe mini-batches (each its own optimizer step).
            # This bounds peak memory by --frame_batch regardless of the
            # data-loader batch_size * seq_len, which is what caused MPS OOM.
            perm = torch.randperm(x_all.shape[0], device=x_all.device)
            x_all = x_all[perm]

            for start in range(0, x_all.shape[0], args.frame_batch):
                x = x_all[start:start + args.frame_batch]

                x_recon, commit_loss, perplexity, _ = model(x)
                recon = recon_loss_fn(x_recon, x, args.object_weight, args.object_threshold)
                loss = recon + commit_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                global_step += 1
                ep_recon += recon.item()
                ep_commit += commit_loss.item()
                ep_ppl += perplexity.item()
                nb += 1

                if global_step % args.debug_every == 0:
                    print(f"  [step {global_step}] recon={recon.item():.5f} "
                          f"commit={commit_loss.item():.5f} "
                          f"perplexity={perplexity.item():.1f}/{args.num_codes}")

        avg_recon = ep_recon / nb
        avg_ppl = ep_ppl / nb
        print(f"  Epoch {epoch:3d}/{args.epochs} | recon={avg_recon:.6f} "
              f"commit={ep_commit/nb:.6f} perplexity={avg_ppl:.1f} | {time.time()-t0:.1f}s")

        if avg_recon < best:
            best = avg_recon
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "recon": avg_recon}, args.checkpoint)

    print(f"\nBest recon: {best:.6f}  ->  {args.checkpoint}")
    save_recon_gif(model, fixed_seq, device, args.recon_gif)


@torch.no_grad()
def save_recon_gif(model, seq, device, out_path="tokenizer_recon.gif"):
    """Interleave GT and reconstruction frame-by-frame so you can eyeball quality."""
    model.eval()
    seq = seq.to(device)                    # (T, 1, 64, 64)
    x_recon, _, _, _ = model(seq)
    interleaved = []
    for i in range(seq.shape[0]):
        interleaved.append(seq[i])          # GT
        interleaved.append(x_recon[i])      # recon
    out = torch.stack(interleaved, dim=0)
    frames_to_gif(tensor_to_numpy_frames(out), out_path)
    print(f"Saved {out_path} (alternating GT, recon, GT, recon, ...)")


if __name__ == "__main__":
    train()
