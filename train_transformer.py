"""
train_transformer.py – Stage 2 training for the discrete-token world model.

Loads the FROZEN VQ-VAE tokenizer, pre-tokenizes the whole dataset once (fast,
the tokenizer never updates), then trains the GPT TokenWorldModel with
next-token cross-entropy on action-interleaved token windows.

Run locally (small) or via modal_train_transformer.py (recommended).

    python3 train_transformer.py --data_file atari_data_128.pt \
        --tokenizer tokenizer_128.pt --context_frames 8 \
        --d_model 512 --n_layers 8 --n_heads 8 \
        --batch_size 16 --epochs 30

Outputs:
    transformer.pt
"""

import argparse
import time

import torch
import torch.nn.functional as F

from dataset import SavedSequenceDataset
from tokenizer import VQVAE
from transformer import TokenWorldModel
from utils import set_seed


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="tokenizer_128.pt")
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--context_frames", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tokenize_chunk", type=int, default=256,
                   help="Frames per tokenizer forward pass during pre-tokenization (memory cap).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--checkpoint", type=str, default="transformer.pt")
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


def load_tokenizer(path, device):
    blob = torch.load(path, map_location=device)
    a = blob["args"]
    vqvae = VQVAE(
        in_channels=1, hidden=a["hidden"], embedding_dim=a["embedding_dim"],
        num_codes=a["num_codes"], commitment_cost=a.get("commitment_cost", 0.25),
    ).to(device)
    vqvae.load_state_dict(blob["model"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return vqvae, a["num_codes"]


@torch.no_grad()
def tokenize_dataset(vqvae, frames, device, chunk):
    """frames: (N, T, 1, H, W) -> tokens (N, T, P) long, on CPU."""
    N, T, C, H, W = frames.shape
    flat = frames.view(N * T, C, H, W)
    toks = []
    for s in range(0, flat.shape[0], chunk):
        idx = vqvae.encode_to_indices(flat[s:s + chunk].to(device))   # (b, h, w)
        toks.append(idx.reshape(idx.shape[0], -1).cpu())
    tokens = torch.cat(toks, dim=0).view(N, T, -1)                    # (N, T, P)
    return tokens.long()


def run(args):
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"\n{'='*70}")
    print("  Token World Model (GPT) – Stage 2 Training")
    print(f"{'='*70}")
    print(f"  Device         : {device}")
    print(f"  Tokenizer      : {args.tokenizer}")
    print(f"  Context frames : {args.context_frames}")
    print(f"  d_model/layers : {args.d_model} / {args.n_layers}  heads {args.n_heads}")
    print(f"  Batch size     : {args.batch_size}  Epochs: {args.epochs}  LR: {args.lr}")
    print(f"{'='*70}\n")

    vqvae, num_codes = load_tokenizer(args.tokenizer, device)
    ds = SavedSequenceDataset(args.data_file)
    frames = ds.frames                                  # (N, T, 1, H, W)
    actions = ds.actions.long()                         # (N, T)
    N, T = frames.shape[0], frames.shape[1]

    if T < args.context_frames:
        raise ValueError(f"seq_len {T} < context_frames {args.context_frames}")

    print(f"  Pre-tokenizing {N} sequences ({N*T} frames)...")
    tokens = tokenize_dataset(vqvae, frames, device, args.tokenize_chunk)  # (N, T, P)
    P = tokens.shape[-1]
    print(f"  tokens: {tuple(tokens.shape)}  ({P} tokens/frame)\n")

    # keep token data on device if it fits (small), else CPU
    tokens = tokens.to(device)
    actions = actions.to(device)

    model = TokenWorldModel(
        num_codes=num_codes, action_dim=args.action_dim, tokens_per_frame=P,
        context_frames=args.context_frames, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
    ).to(device)
    print(f"GPT parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    L = args.context_frames
    max_start = T - L                                   # inclusive range [0, max_start]
    steps_per_epoch = max(1, N // args.batch_size)
    best = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        t0 = time.time()

        for _ in range(steps_per_epoch):
            seq_idx = torch.randint(0, N, (args.batch_size,), device=device)
            starts = torch.randint(0, max_start + 1, (args.batch_size,), device=device)
            # gather windows (B, L, P) and (B, L)
            offs = torch.arange(L, device=device)
            time_idx = starts[:, None] + offs[None, :]          # (B, L)
            bt = seq_idx[:, None].expand(-1, L)                 # (B, L)
            ft = tokens[bt, time_idx]                           # (B, L, P)
            act = actions[bt, time_idx]                         # (B, L)

            _, loss = model(ft, act)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            ep_loss += loss.item()
            if global_step % args.debug_every == 0:
                print(f"  [step {global_step}] loss={loss.item():.4f}")

        avg = ep_loss / steps_per_epoch
        print(f"  Epoch {epoch:3d}/{args.epochs} | loss={avg:.4f} | {time.time()-t0:.1f}s")

        if avg < best:
            best = avg
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "num_codes": num_codes, "action_dim": args.action_dim,
                    "tokens_per_frame": P, "context_frames": args.context_frames,
                    "d_model": args.d_model, "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                },
                "tokenizer": args.tokenizer,
                "epoch": epoch, "loss": avg,
            }, args.checkpoint)

    print(f"\nBest loss: {best:.4f}  ->  {args.checkpoint}")


def train():
    run(get_args())


if __name__ == "__main__":
    train()
