"""
train.py – Training loop for the Atari Breakout latent world model.

Debugging-first design:

    1. First make teacher-forced one-step prediction clean.
       Default decoder is "residual", default loss is the simple 3-term loss,
       default rollout is OFF.

    2. Only after teacher-forced overfit looks clean should you enable
       --rollout_weight > 0 and --rollout_horizon > 0.

Recommended debug command (one-step overfit, gap=1):

    python train.py \
      --dataset atari --num_frames 4 \
      --latent_dim 256 --hidden_dim 512 \
      --batch_size 16 --lr 1e-4 \
      --overfit_batch \
      --decoder_type residual \
      --simple_one_step \
      --prediction_gap 1 \
      --change_weight 500 --object_weight 50 \
      --delta_weight 100 --false_motion_weight 2 \
      --change_threshold 0.005 \
      --rollout_weight 0 --rollout_horizon 0 \
      --residual_scale 0.25 \
      --debug_stats

Outputs in overfit mode:
    overfit_gt.gif
    overfit_teacher_forced_pred.gif
    overfit_pred.gif      (only meaningful once rollout works)
"""

import argparse
import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BouncingBallDataset
from models import WorldModel
from utils import save_checkpoint, set_seed


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent world model")

    parser.add_argument("--dataset", type=str, default="ball", choices=["ball", "atari"])
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--num_frames", type=int, default=4)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--noise_std", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--overfit_batch", action="store_true")
    parser.add_argument("--overfit_steps", type=int, default=2000)

    parser.add_argument("--action_dim", type=int, default=4)

    # Decoder
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="residual",
        choices=["residual", "erase_draw"],
    )
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--edit_scale", type=float, default=1.0)

    # Teacher-forced uses a fresh hidden state every step by default. The
    # 4-frame stack already provides plenty of temporal context.
    parser.add_argument(
        "--use_recurrent_teacher",
        type=_str2bool,
        default=False,
        help="If False, h=None is used for every teacher-forced step.",
    )

    # Prediction gap: predict frame t+num_frames-1+gap given [t..t+num_frames-1].
    parser.add_argument("--prediction_gap", type=int, default=1)

    # Loss mode
    parser.add_argument(
        "--simple_one_step",
        action="store_true",
        default=True,
        help="Use simple 3-term loss. Default ON for debugging.",
    )
    parser.add_argument(
        "--no_simple_one_step",
        dest="simple_one_step",
        action="store_false",
        help="Disable simple mode and use the full structured loss.",
    )

    # Image / motion loss weights
    parser.add_argument("--object_weight", type=float, default=50.0)
    parser.add_argument("--change_weight", type=float, default=500.0)
    parser.add_argument("--delta_weight", type=float, default=100.0)
    parser.add_argument("--false_motion_weight", type=float, default=2.0)
    parser.add_argument("--bright_weight", type=float, default=0.0,
                        help="Brightness-conservation loss weight; fights rollout white-out.")
    parser.add_argument("--object_threshold", type=float, default=0.05)
    parser.add_argument("--change_threshold", type=float, default=0.005)

    # Heavy structured-loss weights (only used when --no_simple_one_step).
    parser.add_argument("--false_disappear_weight", type=float, default=0.0)
    parser.add_argument("--erase_aux_weight", type=float, default=0.0)
    parser.add_argument("--draw_aux_weight", type=float, default=0.0)
    parser.add_argument("--edit_l1_weight", type=float, default=0.0)

    # Rollout (off by default until teacher-forced is clean)
    parser.add_argument("--rollout_weight", type=float, default=0.0)
    parser.add_argument("--rollout_horizon", type=int, default=0)
    parser.add_argument("--rollout_warmup_steps", type=int, default=500)
    parser.add_argument("--rollout_ramp_steps", type=int, default=1000)
    parser.add_argument(
        "--no_detach_rollout",
        action="store_true",
        help="Backprop through the rollout chain (off by default).",
    )

    # Debugging
    parser.add_argument("--debug_stats", action="store_true")
    parser.add_argument("--debug_every", type=int, default=100)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset / device
# ---------------------------------------------------------------------------

def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(args):
    if args.dataset == "atari":
        from dataset import AtariDataset
        return AtariDataset(
            num_sequences=args.num_sequences,
            seq_len=args.seq_len,
            seed=args.seed,
        )
    return BouncingBallDataset(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        seed=args.seed,
    )


def get_action_dim(args: argparse.Namespace) -> int:
    if args.dataset == "atari":
        return args.action_dim
    return 0


def make_action_onehot(actions: torch.Tensor, action_dim: int) -> Optional[torch.Tensor]:
    if action_dim <= 0:
        return None
    return F.one_hot(actions.long(), num_classes=action_dim).float()


def effective_rollout_weight(
    global_step: int,
    target_weight: float,
    warmup_steps: int,
    ramp_steps: int,
) -> float:
    if target_weight <= 0.0:
        return 0.0
    if global_step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return target_weight
    progress = (global_step - warmup_steps) / float(ramp_steps)
    progress = max(0.0, min(1.0, progress))
    return target_weight * progress


def predict_next_frame(
    model: WorldModel,
    x_stack: torch.Tensor,
    act_onehot: Optional[torch.Tensor],
    h: Optional[torch.Tensor],
    edit_scale: float,
):
    return model(
        x_stack,
        action=act_onehot,
        h=h,
        deterministic=True,
        edit_scale=edit_scale,
        return_aux=True,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def simple_one_step_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_frame: torch.Tensor,
    object_weight: float,
    change_weight: float,
    delta_weight: float,
    false_motion_weight: float,
    object_threshold: float,
    change_threshold: float,
    bright_weight: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Clean loss for one-step + rollout training:

        1. weighted image loss
        2. changed-pixel delta loss
        3. false-motion loss (weak)
        4. brightness-conservation loss (fights autoregressive brightness drift)
    """
    if pred.shape != target.shape or prev_frame.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape}, target {target.shape}, prev {prev_frame.shape}"
        )

    object_mask = (target > object_threshold).float()
    change_mask = ((target - prev_frame).abs() > change_threshold).float()

    per_pixel = F.smooth_l1_loss(pred, target, beta=0.01, reduction="none")
    weight = 1.0 + object_weight * object_mask + change_weight * change_mask
    image_loss = (weight * per_pixel).mean()

    pred_delta = pred - prev_frame
    target_delta = target - prev_frame
    delta_error = F.smooth_l1_loss(pred_delta, target_delta, beta=0.01, reduction="none")

    changed_pixels = change_mask.sum().clamp_min(1.0)
    changed_delta_loss = (delta_error * change_mask).sum() / changed_pixels

    false_motion_loss = (delta_error * (1.0 - change_mask)).mean()

    # Brightness conservation: per-image mean-intensity must match the target.
    # Penalizes the small net-positive residual that compounds into white-out
    # during autoregressive rollout.
    pred_bright = pred.mean(dim=[1, 2, 3])
    target_bright = target.mean(dim=[1, 2, 3])
    bright_loss = (pred_bright - target_bright).abs().mean()

    total = (
        image_loss
        + delta_weight * changed_delta_loss
        + false_motion_weight * false_motion_loss
        + bright_weight * bright_loss
    )

    stats = {
        "image_loss": image_loss.detach(),
        "changed_delta_loss": changed_delta_loss.detach(),
        "false_motion_loss": false_motion_loss.detach(),
        "bright_loss": bright_loss.detach(),
        "change_pixels": change_mask.sum().detach(),
    }
    return total, stats


def structured_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    prev_frame: torch.Tensor,
    aux: dict,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, dict]:
    """
    Heavier loss with false-disappearance and erase/draw aux terms.
    Only used when --no_simple_one_step is passed AND decoder_type == "erase_draw".
    """
    total, stats = simple_one_step_loss(
        pred=pred,
        target=target,
        prev_frame=prev_frame,
        object_weight=args.object_weight,
        change_weight=args.change_weight,
        delta_weight=args.delta_weight,
        false_motion_weight=args.false_motion_weight,
        object_threshold=args.object_threshold,
        change_threshold=args.change_threshold,
        bright_weight=args.bright_weight,
    )

    change_mask = ((target - prev_frame).abs() > args.change_threshold).float()
    true_erase_mask = (
        (prev_frame > args.object_threshold)
        & ((prev_frame - target) > args.change_threshold)
    ).float()
    true_draw_mask = (
        (target > args.object_threshold)
        & ((target - prev_frame) > args.change_threshold)
    ).float()
    should_remain_mask = (
        (prev_frame > args.object_threshold) & (true_erase_mask < 0.5)
    ).float()

    if args.false_disappear_weight > 0.0:
        false_disappear_amount = F.relu(prev_frame - pred - args.change_threshold)
        remain_pixels = should_remain_mask.sum().clamp_min(1.0)
        false_disappear_loss = (
            false_disappear_amount * should_remain_mask
        ).sum() / remain_pixels
        total = total + args.false_disappear_weight * false_disappear_loss
        stats["false_disappear_loss"] = false_disappear_loss.detach()

    if "erase_logits" in aux and (
        args.erase_aux_weight > 0.0 or args.draw_aux_weight > 0.0
    ):
        erase_aux = F.binary_cross_entropy_with_logits(aux["erase_logits"], true_erase_mask)
        draw_aux = F.binary_cross_entropy_with_logits(aux["draw_mask_logits"], true_draw_mask)
        total = total + args.erase_aux_weight * erase_aux + args.draw_aux_weight * draw_aux
        stats["erase_aux_loss"] = erase_aux.detach()
        stats["draw_aux_loss"] = draw_aux.detach()

    if "erase_mask" in aux and args.edit_l1_weight > 0.0:
        edit_l1 = aux["erase_mask"].mean() + aux["draw_mask"].mean()
        total = total + args.edit_l1_weight * edit_l1
        stats["edit_l1"] = edit_l1.detach()

    return total, stats


def compute_step_loss(pred, target, prev_frame, aux, args):
    if args.simple_one_step:
        return simple_one_step_loss(
            pred=pred,
            target=target,
            prev_frame=prev_frame,
            object_weight=args.object_weight,
            change_weight=args.change_weight,
            delta_weight=args.delta_weight,
            false_motion_weight=args.false_motion_weight,
            object_threshold=args.object_threshold,
            change_threshold=args.change_threshold,
            bright_weight=args.bright_weight,
        )
    return structured_loss(pred, target, prev_frame, aux, args)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    args = get_args()
    set_seed(args.seed)

    device = get_device(args.device)
    action_dim = get_action_dim(args)

    print(f"\n{'=' * 84}")
    print("  Latent World Model – Training")
    print(f"{'=' * 84}")
    print(f"  Device                  : {device}")
    print(f"  Dataset                 : {args.dataset}")
    print(f"  Seq length              : {args.seq_len}")
    print(f"  Num frames/state        : {args.num_frames}")
    print(f"  Prediction gap          : {args.prediction_gap}")
    print(f"  Action dim              : {action_dim}")
    print(f"  Latent dim              : {args.latent_dim}")
    print(f"  Hidden dim              : {args.hidden_dim}")
    print(f"  Decoder type            : {args.decoder_type}")
    print(f"  Residual scale          : {args.residual_scale}")
    print(f"  Recurrent teacher       : {args.use_recurrent_teacher}")
    print(f"  Simple one-step loss    : {args.simple_one_step}")
    print(f"  Batch size              : {args.batch_size}")
    print(f"  Epochs                  : {args.epochs}")
    print(f"  LR                      : {args.lr}")
    print(f"  Object weight           : {args.object_weight}")
    print(f"  Change weight           : {args.change_weight}")
    print(f"  Delta weight            : {args.delta_weight}")
    print(f"  False motion weight     : {args.false_motion_weight}")
    print(f"  Brightness weight       : {args.bright_weight}")
    print(f"  Rollout weight target   : {args.rollout_weight}")
    print(f"  Rollout horizon         : {args.rollout_horizon}")
    print(f"{'=' * 84}\n")

    print("Loading dataset...")
    dataset = build_dataset(args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    print(f"  {len(dataset)} sequences loaded.\n")

    fixed_batch_dict = None
    if args.overfit_batch:
        fixed_batch_dict = next(iter(loader))
        loader = [fixed_batch_dict] * args.overfit_steps
        args.epochs = 1
        print(f"!!! OVERFIT MODE: training on one fixed batch for {args.overfit_steps} steps !!!\n")

    model = WorldModel(
        in_channels=args.num_frames,
        out_channels=1,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        noise_std=args.noise_std,
        decoder_type=args.decoder_type,
        residual_scale=args.residual_scale,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_loss = float("inf")
    global_step = 0

    # Number of one-step predictions in a sequence of length T.
    #   inputs:  [t .. t+num_frames-1]
    #   target:  t+num_frames-1+gap
    # so valid t ranges from 0 to T - num_frames - gap (inclusive).
    def compute_n_steps(T: int) -> int:
        return T - args.num_frames - args.prediction_gap + 1

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_total = 0.0
        epoch_one = 0.0
        epoch_roll = 0.0
        num_batches = 0
        t0 = time.time()

        debug_values = {}

        for batch_dict in loader:
            global_step += 1

            batch = batch_dict["frames"].to(device)

            if "actions" in batch_dict:
                actions = batch_dict["actions"].to(device)
            else:
                actions = torch.zeros(
                    batch.shape[0], batch.shape[1], device=device, dtype=torch.long,
                )

            if batch.dim() != 5:
                raise ValueError(f"Expected frames (B,T,1,H,W), got {batch.shape}")

            B, T, C, H, W = batch.shape
            if C != 1:
                raise ValueError(f"Expected grayscale C=1, got C={C}")

            n_steps = compute_n_steps(T)
            if n_steps < 1:
                raise ValueError(
                    f"T={T} too short for num_frames={args.num_frames} "
                    f"+ prediction_gap={args.prediction_gap}"
                )

            one_step_loss = torch.tensor(0.0, device=device)
            rollout_loss = torch.tensor(0.0, device=device)

            # --------------------------------------------------------------
            # 1. Teacher-forced one-step training
            # --------------------------------------------------------------
            h_state = None
            if args.use_recurrent_teacher:
                h_state = model.init_hidden(B, device, batch.dtype)

            for t in range(n_steps):
                x_stack = batch[:, t : t + args.num_frames].squeeze(2)
                x_last = batch[:, t + args.num_frames - 1]
                target_idx = t + args.num_frames - 1 + args.prediction_gap
                x_target = batch[:, target_idx]

                act_t = actions[:, t + args.num_frames - 1]
                act_onehot = make_action_onehot(act_t, action_dim)

                if not args.use_recurrent_teacher:
                    h_in = None
                else:
                    h_in = h_state

                x_hat, _, _, h_next, aux = predict_next_frame(
                    model=model,
                    x_stack=x_stack,
                    act_onehot=act_onehot,
                    h=h_in,
                    edit_scale=args.edit_scale,
                )

                if args.use_recurrent_teacher:
                    h_state = h_next

                step_loss, stats = compute_step_loss(x_hat, x_target, x_last, aux, args)
                one_step_loss = one_step_loss + step_loss

                if t == 0 and args.debug_stats:
                    with torch.no_grad():
                        true_motion = (x_target - x_last).abs()
                        pred_motion = (x_hat - x_last).abs()

                        debug_values = {
                            "pred_max": x_hat.max().item(),
                            "target_max": x_target.max().item(),
                            "pred_motion_sum": pred_motion.sum().item(),
                            "true_motion_sum": true_motion.sum().item(),
                            "change_pixels": stats["change_pixels"].item(),
                            "image_loss": stats["image_loss"].item(),
                            "changed_delta_loss": stats["changed_delta_loss"].item(),
                            "false_motion_loss": stats["false_motion_loss"].item(),
                            "bright_loss": stats.get("bright_loss", torch.tensor(0.0)).item(),
                        }
                        if "delta" in aux:
                            debug_values["delta_abs_max"] = aux["delta"].abs().max().item()
                        if "erase_mask" in aux:
                            debug_values["erase_mean"] = aux["erase_mask"].mean().item()
                            debug_values["draw_mean"] = aux["draw_mask"].mean().item()

            one_step_loss = one_step_loss / n_steps

            # --------------------------------------------------------------
            # 2. Optional autoregressive rollout (off by default)
            # --------------------------------------------------------------
            eff_roll_w = effective_rollout_weight(
                global_step=global_step,
                target_weight=args.rollout_weight,
                warmup_steps=args.rollout_warmup_steps,
                ramp_steps=args.rollout_ramp_steps,
            )
            horizon = min(args.rollout_horizon, n_steps)

            if horizon > 0 and eff_roll_w > 0.0:
                current_stack = batch[:, : args.num_frames].squeeze(2)
                h_roll = model.init_hidden(B, device, batch.dtype)

                for k in range(horizon):
                    x_last = current_stack[:, -1:].contiguous()
                    target_idx = args.num_frames + k + (args.prediction_gap - 1)
                    if target_idx >= T:
                        break
                    x_target = batch[:, target_idx]

                    act_t = actions[:, args.num_frames + k - 1]
                    act_onehot = make_action_onehot(act_t, action_dim)

                    x_hat, _, _, h_roll, aux = predict_next_frame(
                        model=model,
                        x_stack=current_stack,
                        act_onehot=act_onehot,
                        h=h_roll,
                        edit_scale=args.edit_scale,
                    )

                    step_roll_loss, _ = compute_step_loss(x_hat, x_target, x_last, aux, args)
                    rollout_loss = rollout_loss + step_roll_loss

                    next_frame_for_stack = x_hat
                    if not args.no_detach_rollout:
                        next_frame_for_stack = next_frame_for_stack.detach()
                        h_roll = h_roll.detach()

                    current_stack = torch.cat(
                        [current_stack[:, 1:], next_frame_for_stack], dim=1
                    )

                rollout_loss = rollout_loss / max(horizon, 1)

            total_loss = one_step_loss + eff_roll_w * rollout_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_one += one_step_loss.item()
            epoch_roll += rollout_loss.item() if isinstance(rollout_loss, torch.Tensor) else 0.0
            num_batches += 1

            if args.debug_stats and global_step % args.debug_every == 0:
                extra = ""
                if "delta_abs_max" in debug_values:
                    extra += f" delta_abs_max={debug_values['delta_abs_max']:.4f}"
                if "erase_mean" in debug_values:
                    extra += (
                        f" erase_mean={debug_values['erase_mean']:.5f}"
                        f" draw_mean={debug_values['draw_mean']:.5f}"
                    )
                print(
                    f"[step {global_step}] "
                    f"loss={total_loss.item():.5f} "
                    f"one={one_step_loss.item():.5f} "
                    f"img={debug_values.get('image_loss', 0):.5f} "
                    f"cdelta={debug_values.get('changed_delta_loss', 0):.5f} "
                    f"fmot={debug_values.get('false_motion_loss', 0):.5f} "
                    f"bright={debug_values.get('bright_loss', 0):.5f} | "
                    f"pred_max={debug_values.get('pred_max', 0):.3f} "
                    f"tgt_max={debug_values.get('target_max', 0):.3f} | "
                    f"pred_motion={debug_values.get('pred_motion_sum', 0):.2f} "
                    f"true_motion={debug_values.get('true_motion_sum', 0):.2f} "
                    f"chg_px={debug_values.get('change_pixels', 0):.0f}"
                    f"{extra}"
                )

        avg_total = epoch_total / max(num_batches, 1)
        avg_one = epoch_one / max(num_batches, 1)
        avg_roll = epoch_roll / max(num_batches, 1)
        elapsed = time.time() - t0

        bar_len = 30
        filled = int(bar_len * epoch / args.epochs)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"  Epoch {epoch:3d}/{args.epochs} │ {bar} │ "
            f"total: {avg_total:.6f}  one: {avg_one:.6f}  roll: {avg_roll:.6f} │ "
            f"{elapsed:.1f}s"
        )

        if avg_total < best_loss:
            best_loss = avg_total
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": avg_total,
                    "args": vars(args),
                },
                path=args.checkpoint,
            )

    print(f"\n{'=' * 84}")
    print(f"  Training complete. Best loss: {best_loss:.6f}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'=' * 84}\n")

    if args.overfit_batch and fixed_batch_dict is not None:
        save_overfit_gifs(
            model=model,
            fixed_batch_dict=fixed_batch_dict,
            args=args,
            action_dim=action_dim,
            device=device,
        )


# ---------------------------------------------------------------------------
# GIF saving
# ---------------------------------------------------------------------------

def save_overfit_gifs(
    model: WorldModel,
    fixed_batch_dict: dict,
    args: argparse.Namespace,
    action_dim: int,
    device: torch.device,
):
    print("Saving overfit GIFs...")
    model.eval()

    from utils import tensor_to_numpy_frames, frames_to_gif

    with torch.no_grad():
        seq = fixed_batch_dict["frames"][0].to(device)  # (T, 1, H, W)

        if "actions" in fixed_batch_dict:
            act = fixed_batch_dict["actions"][0].to(device)
        else:
            act = torch.zeros(seq.shape[0], device=device, dtype=torch.long)

        T_len = seq.shape[0]

        # ----- Ground truth -----
        frames_to_gif(tensor_to_numpy_frames(seq), "overfit_gt.gif")

        # ----- Teacher-forced (input stack always real frames) -----
        teacher_frames = [seq[i] for i in range(args.num_frames)]
        h_teacher = None
        if args.use_recurrent_teacher:
            h_teacher = model.init_hidden(1, device, seq.dtype)

        # We can produce predictions for indices num_frames-1+gap .. T_len-1
        # using input stacks [i-num_frames+1 .. i] where i = pred_idx - gap.
        gap = args.prediction_gap
        first_pred_idx = args.num_frames - 1 + gap

        for pred_idx in range(first_pred_idx, T_len):
            i_last = pred_idx - gap  # last input frame index
            x_stack = seq[i_last - args.num_frames + 1 : i_last + 1].unsqueeze(0).squeeze(2)

            a = act[i_last : i_last + 1]
            a_onehot = make_action_onehot(a, action_dim)

            h_in = h_teacher if args.use_recurrent_teacher else None

            x_hat, _, _, h_next, _ = predict_next_frame(
                model=model,
                x_stack=x_stack,
                act_onehot=a_onehot,
                h=h_in,
                edit_scale=args.edit_scale,
            )

            if args.use_recurrent_teacher:
                h_teacher = h_next

            # Pad missing slots (when gap>1) with the last real input frame, so
            # the GIF stays aligned to GT length without lying about predictions.
            while len(teacher_frames) < pred_idx:
                teacher_frames.append(seq[len(teacher_frames)])
            teacher_frames.append(x_hat.squeeze(0))

        teacher_tensor = torch.stack(teacher_frames, dim=0)
        frames_to_gif(
            tensor_to_numpy_frames(teacher_tensor),
            "overfit_teacher_forced_pred.gif",
        )

        # ----- Free rollout (only meaningful once teacher-forced is clean) -----
        rollout_frames = [seq[i] for i in range(args.num_frames)]
        current_stack = seq[: args.num_frames].unsqueeze(0).squeeze(2)
        h_roll = model.init_hidden(1, device, seq.dtype)

        for t in range(args.num_frames, T_len):
            a = act[t - 1 : t]
            a_onehot = make_action_onehot(a, action_dim)

            x_hat, _, _, h_roll, _ = predict_next_frame(
                model=model,
                x_stack=current_stack,
                act_onehot=a_onehot,
                h=h_roll,
                edit_scale=args.edit_scale,
            )

            rollout_frames.append(x_hat.squeeze(0))
            current_stack = torch.cat([current_stack[:, 1:], x_hat], dim=1)

        rollout_tensor = torch.stack(rollout_frames, dim=0)
        frames_to_gif(tensor_to_numpy_frames(rollout_tensor), "overfit_pred.gif")

    print("Saved:")
    print("  overfit_gt.gif")
    print("  overfit_teacher_forced_pred.gif")
    print("  overfit_pred.gif")


if __name__ == "__main__":
    train()
