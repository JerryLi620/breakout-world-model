"""
sample_rollout.py – Generate rollouts from the trained token world model.

Primes the GPT with the first --prime frames of a real sequence, then
autoregressively generates the rest, conditioned on the real action sequence.
Decodes generated tokens back to pixels with the frozen tokenizer.

Because sampling is stochastic (temperature / top-k), running with --n_samples > 1
produces DIFFERENT futures from the SAME prime — the stochastic simulation goal.

Usage:
    python3 sample_rollout.py --data_file atari_data_128.pt \
        --tokenizer tokenizer_128.pt --transformer transformer.pt \
        --prime 4 --n_samples 3 --temperature 1.0 --top_k 50 --gif_seq 0

Outputs:
    rollout_gt.gif
    rollout_sample0.gif, rollout_sample1.gif, ...
"""

import argparse
import torch

from dataset import SavedSequenceDataset
from tokenizer import VQVAE
from transformer import TokenWorldModel
from utils import frames_to_gif, tensor_to_numpy_frames


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="tokenizer_128.pt")
    p.add_argument("--transformer", type=str, default="transformer.pt")
    p.add_argument("--gif_seq", type=int, default=0)
    p.add_argument("--prime", type=int, default=4, help="# real frames to prime with")
    p.add_argument("--n_samples", type=int, default=3, help="stochastic rollouts from same prime")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="auto")
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
    vq = VQVAE(in_channels=1, hidden=a["hidden"], embedding_dim=a["embedding_dim"],
               num_codes=a["num_codes"], commitment_cost=a.get("commitment_cost", 0.25)).to(device)
    vq.load_state_dict(blob["model"])
    vq.eval()
    return vq


def load_transformer(path, device):
    blob = torch.load(path, map_location=device)
    c = blob["config"]
    m = TokenWorldModel(
        num_codes=c["num_codes"], action_dim=c["action_dim"],
        tokens_per_frame=c["tokens_per_frame"], context_frames=c["context_frames"],
        d_model=c["d_model"], n_layers=c["n_layers"], n_heads=c["n_heads"], dropout=0.0,
    ).to(device)
    m.load_state_dict(blob["model"])
    m.eval()
    return m, c


@torch.no_grad()
def main():
    args = get_args()
    device = get_device(args.device)

    vq = load_tokenizer(args.tokenizer, device)
    model, cfg = load_transformer(args.transformer, device)
    P = cfg["tokens_per_frame"]
    grid = int(P ** 0.5)   # 16 for 256 tokens

    ds = SavedSequenceDataset(args.data_file)
    seq = ds.frames[args.gif_seq].to(device)        # (T, 1, H, W)
    acts = ds.actions[args.gif_seq].long().to(device)  # (T,)
    T = seq.shape[0]

    n_new = T - args.prime
    if n_new < 1:
        raise ValueError(f"prime {args.prime} >= seq_len {T}")

    # ground truth
    frames_to_gif(tensor_to_numpy_frames(seq), "rollout_gt.gif")

    # tokenize the prime frames
    prime_idx = vq.encode_to_indices(seq[:args.prime])          # (prime, grid, grid)
    prime_tokens = prime_idx.reshape(1, args.prime, P)          # (1, prime, P)
    actions = acts.unsqueeze(0)                                 # (1, T)

    print(f"seq {args.gif_seq}: prime={args.prime} generate={n_new} "
          f"(temp={args.temperature}, top_k={args.top_k})")

    for s in range(args.n_samples):
        torch.manual_seed(1000 + s)   # different seed -> different future
        gen = model.generate(prime_tokens, actions, n_new_frames=n_new,
                             temperature=args.temperature, top_k=args.top_k)  # (1, n_new, P)
        gen_idx = gen.reshape(n_new, grid, grid)
        gen_frames = vq.decode_from_indices(gen_idx)            # (n_new, 1, H, W)

        full = torch.cat([seq[:args.prime], gen_frames], dim=0)  # prime(real)+generated
        out = f"rollout_sample{s}.gif"
        frames_to_gif(tensor_to_numpy_frames(full), out)
        # how much does it move? quick collapse check
        d = (gen_frames[1:] - gen_frames[:-1]).abs().flatten(1).sum(1)
        print(f"  sample {s}: per-frame motion {[round(float(x),1) for x in d[:8]]}...  -> {out}")

    print("\nSaved rollout_gt.gif + rollout_sample*.gif")
    print("Compare samples: same prime, different futures = stochastic simulation working.")


if __name__ == "__main__":
    main()
