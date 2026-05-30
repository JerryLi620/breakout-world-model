"""
dataset.py – Video sequence datasets for the latent world model.

Provides two options:
  1. BouncingBallDataset  – synthetic, zero-dependency (default)
  2. AtariDataset          – Gymnasium Atari (requires gymnasium[atari])
"""

import torch
import numpy as np
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Option A (default): Synthetic bouncing ball
# ---------------------------------------------------------------------------

class BouncingBallDataset(Dataset):
    """
    Generates sequences of a white ball bouncing inside a 64×64 frame.
    Each item is a tensor of shape (T, 1, 64, 64) with values in [0, 1].
    """

    def __init__(
        self,
        num_sequences: int = 1000,
        seq_len: int = 20,
        img_size: int = 64,
        ball_radius: int = 8,
        seed: int = 42,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.img_size = img_size
        self.ball_radius = ball_radius

        rng = np.random.RandomState(seed)
        self.sequences = []

        for _ in range(num_sequences):
            frames = self._generate_trajectory(rng)
            self.sequences.append(frames)

    def _generate_trajectory(self, rng: np.random.RandomState) -> np.ndarray:
        """Create one bouncing-ball trajectory → (T, 1, H, W) float32."""
        s = self.img_size
        r = self.ball_radius

        # Random initial position and velocity
        x = rng.uniform(r, s - r)
        y = rng.uniform(r, s - r)
        vx = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])
        vy = rng.uniform(1.5, 3.0) * rng.choice([-1, 1])

        frames = np.zeros((self.seq_len, 1, s, s), dtype=np.float32)

        # Pre-compute a soft circle mask for anti-aliased look
        yy, xx = np.mgrid[:s, :s]

        for t in range(self.seq_len):
            # Draw ball with soft edges
            dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            ball = np.clip(1.0 - (dist - r + 1.0), 0.0, 1.0)
            frames[t, 0] = ball

            # Update position
            x += vx
            y += vy

            # Bounce off walls
            if x - r < 0:
                x = r
                vx = abs(vx)
            elif x + r > s:
                x = s - r
                vx = -abs(vx)
            if y - r < 0:
                y = r
                vy = abs(vy)
            elif y + r > s:
                y = s - r
                vy = -abs(vy)

        return frames

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        frames = torch.from_numpy(self.sequences[idx])  # (T, 1, H, W)
        actions = torch.zeros(self.seq_len, dtype=torch.long)
        return {"frames": frames, "actions": actions}


# ---------------------------------------------------------------------------
# Option B: Atari random trajectories
# ---------------------------------------------------------------------------

class AtariDataset(Dataset):
    """
    Collects random-action trajectories from an Atari environment.
    Requires: conda environment with gymnasium, ale-py, Pillow

    Each item is a tensor of shape (T, 1, 64, 64) with values in [0, 1].
    """

    def __init__(
        self,
        env_name: str = "ALE/Breakout-v5",
        num_sequences: int = 200,
        seq_len: int = 20,
        img_size: int = 64,
        frame_skip: int = 2,
        seed: int = 42,
        fire_on_life_loss: bool = True,
    ):
        super().__init__()
        import gymnasium as gym
        import ale_py
        from PIL import Image

        # Register ALE environments (required for newer ale-py versions)
        gym.register_envs(ale_py)

        self.sequences = []

        env = gym.make(env_name, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)

        def to_frame(o):
            img = Image.fromarray(o).convert("L").resize((img_size, img_size), Image.NEAREST)
            return np.array(img, dtype=np.float32) / 255.0

        print(f"  Collecting {num_sequences} Atari trajectories "
              f"(seq_len={seq_len}, fire_on_life_loss={fire_on_life_loss})...")
        collected = 0
        respawn_seqs = 0
        while collected < num_sequences:
            # Collect one continuous trajectory of length seq_len
            trajectory = []
            actions_seq = []
            obs, info = env.reset()

            # Force FIRE action to spawn the ball
            obs, _, _, _, info = env.step(1)
            lives = info.get("lives", 0)

            trajectory.append(to_frame(obs))
            actions_seq.append(1)  # the FIRE that launched the ball

            done = False
            force_fire = False
            had_respawn = False
            while len(trajectory) < seq_len and not done:
                # FIRE to relaunch right after a life is lost; otherwise random.
                # This is what makes lose -> FIRE -> respawn cycles appear in the
                # data, so the world model can learn the respawn mechanic.
                if force_fire:
                    action = 1
                    force_fire = False
                else:
                    action = env.action_space.sample()

                for _ in range(frame_skip):
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    if done:
                        break

                new_lives = info.get("lives", lives)
                if fire_on_life_loss and new_lives < lives:
                    force_fire = True       # relaunch on the next step
                    had_respawn = True
                lives = new_lives

                trajectory.append(to_frame(obs))
                actions_seq.append(action)

            # Only keep full-length sequences with actual movement
            if len(trajectory) >= seq_len:
                seq = np.array(trajectory[:seq_len])  # (T, H, W)
                acts = np.array(actions_seq[:seq_len], dtype=np.int64)

                # Verify that the sequence isn't just a static screen
                diffs = np.abs(seq[1:] - seq[:-1])
                if np.max(diffs) < 0.05:
                    continue  # Skip dead sequences

                seq = seq[:, None, :, :]               # (T, 1, H, W)
                self.sequences.append((seq, acts))
                collected += 1
                if had_respawn:
                    respawn_seqs += 1
                if collected % 50 == 0 or collected == num_sequences:
                    print(f"    {collected}/{num_sequences} collected "
                          f"({respawn_seqs} contain a respawn)")

        env.close()
        print(f"  Done — {len(self.sequences)} Atari sequences ready "
              f"({respawn_seqs} with a ball respawn).")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, acts = self.sequences[idx]
        return {"frames": torch.from_numpy(seq), "actions": torch.from_numpy(acts)}


class SavedSequenceDataset(Dataset):
    """
    Loads sequences from a file produced by export_dataset.py — no ALE needed.

    File format (torch.save dict):
        "frames":  FloatTensor (N, T, 1, 64, 64)  in [0, 1]
        "actions": LongTensor  (N, T)

    Same item interface as AtariDataset: {"frames": (T,1,64,64), "actions": (T,)}.
    Use this on cloud GPUs (Modal) so you don't install ALE or regenerate data.
    """

    def __init__(self, path: str):
        super().__init__()
        blob = torch.load(path, map_location="cpu")
        self.frames = blob["frames"]
        self.actions = blob["actions"]
        if self.frames.dim() != 5:
            raise ValueError(f"Expected frames (N,T,1,H,W), got {tuple(self.frames.shape)}")
        if self.frames.shape[0] != self.actions.shape[0]:
            raise ValueError("frames and actions disagree on N")

    def __len__(self):
        return self.frames.shape[0]

    def __getitem__(self, idx):
        return {"frames": self.frames[idx], "actions": self.actions[idx]}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Testing BouncingBallDataset...")
    ds = BouncingBallDataset(num_sequences=5, seq_len=10)
    sample = ds[0]["frames"]
    print(f"  Ball shape : {sample.shape}")

    print("Testing AtariDataset sanity check...")
    ds_atari = AtariDataset(num_sequences=2, seq_len=10)
    sample_atari = ds_atari[0]["frames"]
    
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(sample_atari[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f"t={i}")
    fig.tight_layout()
    fig.savefig("dataset_sanity_check.png")
    print("  Done ✓ Saved dataset_sanity_check.png")
