"""
diagnose.py – Inspect a trained world model.

Two checks:

    1. Action sensitivity:
       Load one sequence, take the first 4 frames as the input stack.
       Predict the next frame under each of the K possible actions.
       Report the max-abs difference between predictions for different actions.
       If this is ~0, the model is ignoring its action input.

    2. Action-usage along the saved overfit sequence:
       For each teacher-forced prediction, print the action that was fed in and
       the predicted-vs-true motion sum.
"""

import argparse
import torch
import torch.nn.functional as F

from dataset import AtariDataset, BouncingBallDataset
from models import WorldModel


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--dataset", type=str, default="atari", choices=["atari", "ball"])
    p.add_argument("--num_sequences", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=4)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--decoder_type", type=str, default="residual")
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--save_gif", action="store_true",
                   help="Save teacher-forced GT + prediction GIFs for sequence --gif_seq.")
    p.add_argument("--rollout_gif", action="store_true",
                   help="Save a free autoregressive rollout GIF (model feeds itself).")
    p.add_argument("--gif_seq", type=int, default=0, help="Which sequence index to visualize.")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cpu")

    if args.dataset == "atari":
        ds = AtariDataset(num_sequences=args.num_sequences, seq_len=args.seq_len, seed=42)
    else:
        ds = BouncingBallDataset(num_sequences=args.num_sequences, seq_len=args.seq_len, seed=42)

    seq = ds[0]["frames"].to(device)        # (T, 1, 64, 64)
    actions = ds[0]["actions"].to(device)   # (T,)

    model = WorldModel(
        in_channels=args.num_frames,
        out_channels=1,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        decoder_type=args.decoder_type,
        residual_scale=args.residual_scale,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n=== Loaded {args.checkpoint} ===")
    print(f"   action_dim = {args.action_dim}")

    # ------------------------------------------------------------------
    # 1. Action sensitivity test
    # ------------------------------------------------------------------
    print("\n=== Action sensitivity (same input, vary action) ===")
    x_stack = seq[: args.num_frames].unsqueeze(0).squeeze(2)  # (1, num_frames, 64, 64)
    x_last = seq[args.num_frames - 1]

    with torch.no_grad():
        preds = []
        for a_idx in range(args.action_dim):
            a = F.one_hot(torch.tensor([a_idx]), num_classes=args.action_dim).float()
            x_hat, _, _, _, _ = model(x_stack, action=a, h=None, deterministic=True, return_aux=True)
            preds.append(x_hat.squeeze(0))   # (1, 64, 64)

        for i in range(args.action_dim):
            for j in range(i + 1, args.action_dim):
                d = (preds[i] - preds[j]).abs()
                print(
                    f"  |pred(a={i}) - pred(a={j})|  sum={d.sum().item():.3f}  "
                    f"max={d.max().item():.4f}"
                )

        # Same vs x_last so we can compare to "real motion"
        for i, p in enumerate(preds):
            m = (p - x_last).abs().sum().item()
            print(f"  pred(a={i}) motion vs x_last: {m:.3f}")

    # Reference: how much does ground truth move per step?
    real_motion = (seq[args.num_frames] - seq[args.num_frames - 1]).abs().sum().item()
    print(f"  GT motion seq[{args.num_frames}] - seq[{args.num_frames-1}]: {real_motion:.3f}")

    # ------------------------------------------------------------------
    # 2. Per-step teacher-forced trace with action printed
    # ------------------------------------------------------------------
    print("\n=== Teacher-forced trace ===")
    print("  i  action   pred_motion   true_motion   pred_max")

    with torch.no_grad():
        for i in range(args.num_frames - 1, args.seq_len - 1):
            x_stack = seq[i - args.num_frames + 1 : i + 1].unsqueeze(0).squeeze(2)
            x_last = seq[i]
            x_target = seq[i + 1]
            a_idx = actions[i].item()
            a = F.one_hot(actions[i : i + 1].long(), num_classes=args.action_dim).float()

            x_hat, _, _, _, _ = model(x_stack, action=a, h=None, deterministic=True, return_aux=True)
            x_hat = x_hat.squeeze(0)

            pred_motion = (x_hat - x_last).abs().sum().item()
            true_motion = (x_target - x_last).abs().sum().item()

            print(
                f"  {i:>2}  a={a_idx:<2}   "
                f"{pred_motion:>10.3f}   {true_motion:>10.3f}   "
                f"{x_hat.max().item():.3f}"
            )

    # ------------------------------------------------------------------
    # 3. Optional: save teacher-forced GIFs for visual inspection
    # ------------------------------------------------------------------
    if args.save_gif:
        save_teacher_forced_gif(model, ds, args, device)

    if args.rollout_gif:
        save_rollout_gif(model, ds, args, device)


def save_teacher_forced_gif(model, ds, args, device):
    """
    Teacher-forced visualization on one real sequence: every prediction uses the
    true previous frames as input (h=None), so it reflects pure one-step quality.
    """
    from utils import tensor_to_numpy_frames, frames_to_gif

    seq = ds[args.gif_seq]["frames"].to(device)     # (T, 1, 64, 64)
    actions = ds[args.gif_seq]["actions"].to(device)
    T_len = seq.shape[0]

    frames_to_gif(tensor_to_numpy_frames(seq), "diagnose_gt.gif")

    pred_frames = [seq[i] for i in range(args.num_frames)]  # real context
    with torch.no_grad():
        for i in range(args.num_frames - 1, T_len - 1):
            x_stack = seq[i - args.num_frames + 1 : i + 1].unsqueeze(0).squeeze(2)
            a = F.one_hot(actions[i : i + 1].long(), num_classes=args.action_dim).float()
            x_hat, _, _, _, _ = model(x_stack, action=a, h=None, deterministic=True, return_aux=True)
            pred_frames.append(x_hat.squeeze(0))

    pred_tensor = torch.stack(pred_frames, dim=0)
    frames_to_gif(tensor_to_numpy_frames(pred_tensor), "diagnose_teacher_pred.gif")

    print("\nSaved:")
    print("  diagnose_gt.gif")
    print("  diagnose_teacher_pred.gif")


def save_rollout_gif(model, ds, args, device):
    """
    Free autoregressive rollout: start from the first num_frames REAL frames,
    then feed the model its own predictions back as input. Uses the recurrent
    hidden state (h carried forward) and the real action sequence.

    This is the honest test of whether the model can simulate forward. The
    per-step trace shows pred_motion (model's own step-to-step change) next to
    true_motion — when pred_motion decays toward 0 while true_motion stays
    positive, that's the frame where rollout collapsed.
    """
    from utils import tensor_to_numpy_frames, frames_to_gif

    seq = ds[args.gif_seq]["frames"].to(device)     # (T, 1, 64, 64)
    actions = ds[args.gif_seq]["actions"].to(device)
    T_len = seq.shape[0]

    frames_to_gif(tensor_to_numpy_frames(seq), "diagnose_gt.gif")

    rollout_frames = [seq[i] for i in range(args.num_frames)]   # real context
    current_stack = seq[: args.num_frames].unsqueeze(0).squeeze(2)  # (1, nf, 64, 64)
    h = model.init_hidden(1, device, seq.dtype)

    print("\n=== Free rollout trace ===")
    print("  i  action   pred_motion   true_motion   pred_max   (pred_motion->0 = collapse)")

    with torch.no_grad():
        for i in range(args.num_frames - 1, T_len - 1):
            prev = current_stack[:, -1:].clone()
            a = F.one_hot(actions[i : i + 1].long(), num_classes=args.action_dim).float()

            x_hat, _, _, h, _ = model(
                current_stack, action=a, h=h, deterministic=True, return_aux=True,
            )

            pred_motion = (x_hat - prev).abs().sum().item()
            true_motion = (seq[i + 1] - seq[i]).abs().sum().item()
            print(
                f"  {i:>2}  a={actions[i].item():<2}   "
                f"{pred_motion:>10.3f}   {true_motion:>10.3f}   "
                f"{x_hat.max().item():.3f}"
            )

            rollout_frames.append(x_hat.squeeze(0))
            current_stack = torch.cat([current_stack[:, 1:], x_hat], dim=1)

    roll_tensor = torch.stack(rollout_frames, dim=0)
    frames_to_gif(tensor_to_numpy_frames(roll_tensor), "diagnose_rollout.gif")

    print("\nSaved:")
    print("  diagnose_gt.gif")
    print("  diagnose_rollout.gif")


if __name__ == "__main__":
    main()
