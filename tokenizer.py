"""
tokenizer.py – VQ-VAE tokenizer for 64x64 grayscale Atari frames (Stage 1 of the
discrete-token world model).

Each frame is encoded to an 8x8 grid of discrete tokens (64 tokens/frame), drawn
from a learned codebook of K entries. A decoder reconstructs the frame from the
quantized grid.

Why discrete tokens:
    The deterministic residual model hit the exposure-bias wall in rollout
    (confident-but-unstable vs safe-but-frozen). Tokens are *picks*, not
    regressions, so reconstruction does not blur or decay, and the Stage-2
    transformer can model next-frame tokens with cross-entropy + sampling ->
    sharp, stochastic rollouts.

Codebook updates use EMA (van den Oord et al.), the robust default against
codebook collapse. Monitor `perplexity`: high (toward K) = healthy usage;
low (a handful of codes) = collapse.

This file is standalone — it does NOT import or modify models.py / train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """64x64x1 -> (B, embedding_dim, 8, 8). No skip connections (the bottleneck
    IS the token grid)."""

    def __init__(self, in_channels=1, hidden=128, embedding_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, stride=2, padding=1),   # 64 -> 32
            nn.GroupNorm(8, hidden), nn.SiLU(),
            _conv_block(hidden, hidden),

            nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1),    # 32 -> 16
            nn.GroupNorm(8, hidden * 2), nn.SiLU(),
            _conv_block(hidden * 2, hidden * 2),

            nn.Conv2d(hidden * 2, hidden * 2, 4, stride=2, padding=1),  # 16 -> 8
            nn.GroupNorm(8, hidden * 2), nn.SiLU(),
            _conv_block(hidden * 2, hidden * 2),

            nn.Conv2d(hidden * 2, embedding_dim, 1),                  # project to D
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """(B, embedding_dim, 8, 8) -> 64x64x1."""

    def __init__(self, out_channels=1, hidden=128, embedding_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden * 2, 1),
            nn.GroupNorm(8, hidden * 2), nn.SiLU(),
            _conv_block(hidden * 2, hidden * 2),

            nn.ConvTranspose2d(hidden * 2, hidden * 2, 4, stride=2, padding=1),  # 8 -> 16
            nn.GroupNorm(8, hidden * 2), nn.SiLU(),
            _conv_block(hidden * 2, hidden * 2),

            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1),      # 16 -> 32
            nn.GroupNorm(8, hidden), nn.SiLU(),
            _conv_block(hidden, hidden),

            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1),          # 32 -> 64
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )

    def forward(self, z_q):
        return torch.sigmoid(self.net(z_q))   # frames are in [0, 1]


# ---------------------------------------------------------------------------
# EMA Vector Quantizer
# ---------------------------------------------------------------------------

class VectorQuantizerEMA(nn.Module):
    """
    EMA-updated codebook quantization.

    Args:
        num_codes:      K (codebook size)
        embedding_dim:  D
        commitment_cost: weight on the commitment loss (keeps encoder outputs
                         close to the codebook)
        decay:          EMA decay for codebook updates
    """

    def __init__(self, num_codes=512, embedding_dim=256, commitment_cost=0.25,
                 decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embed = torch.randn(num_codes, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", embed.clone())

    def forward(self, z):
        # z: (B, D, H, W) -> (B, H, W, D) -> (N, D)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        flat = z.view(-1, self.embedding_dim)

        # distances to codebook entries
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = dist.argmin(1)                       # (N,)
        encodings = F.one_hot(indices, self.num_codes).type(flat.dtype)  # (N, K)
        z_q = (encodings @ self.embedding).view(z_shape)  # (B, H, W, D)

        # EMA codebook update (training only)
        if self.training:
            with torch.no_grad():
                cluster = encodings.sum(0)             # (K,)
                self.cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
                dw = encodings.t() @ flat              # (K, D)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_norm = (
                    (self.cluster_size + self.epsilon)
                    / (n + self.num_codes * self.epsilon) * n
                )
                self.embedding.copy_(self.ema_w / cluster_norm.unsqueeze(1))

        # commitment loss (encoder is pushed toward the chosen codes)
        commit_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)

        # straight-through estimator
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()     # back to (B, D, H, W)

        # perplexity: codebook usage health (toward K = good)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        indices = indices.view(z_shape[0], z_shape[1], z_shape[2])  # (B, H, W)
        return z_q, commit_loss, perplexity, indices


# ---------------------------------------------------------------------------
# VQ-VAE
# ---------------------------------------------------------------------------

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden=128, embedding_dim=256,
                 num_codes=512, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, embedding_dim)
        self.quantizer = VectorQuantizerEMA(num_codes, embedding_dim, commitment_cost, decay)
        self.decoder = Decoder(in_channels, hidden, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_codes = num_codes

    def forward(self, x):
        z = self.encoder(x)
        z_q, commit_loss, perplexity, indices = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, commit_loss, perplexity, indices

    @torch.no_grad()
    def encode_to_indices(self, x):
        """x: (B, 1, 64, 64) -> indices (B, 8, 8)."""
        z = self.encoder(x)
        _, _, _, indices = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        """indices: (B, 8, 8) -> frames (B, 1, 64, 64)."""
        B, H, W = indices.shape
        flat = indices.view(-1)
        z_q = self.quantizer.embedding[flat].view(B, H, W, self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)


# ---------------------------------------------------------------------------
# Sanity check (tiny, fast — safe to run alongside other work)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = VQVAE(num_codes=512, embedding_dim=256)
    x = torch.rand(2, 1, 64, 64)

    x_recon, commit_loss, perplexity, indices = model(x)
    print(f"input        : {tuple(x.shape)}")
    print(f"recon        : {tuple(x_recon.shape)}  range [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"indices      : {tuple(indices.shape)}  (8x8 = 64 tokens/frame)")
    print(f"index range  : [{indices.min().item()}, {indices.max().item()}]  of {model.num_codes}")
    print(f"commit_loss  : {commit_loss.item():.4f}")
    print(f"perplexity   : {perplexity.item():.2f}  (toward {model.num_codes} = healthy)")

    # round-trip encode/decode
    idx = model.encode_to_indices(x)
    rt = model.decode_from_indices(idx)
    print(f"round-trip   : indices {tuple(idx.shape)} -> frame {tuple(rt.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params       : {n_params:,}")
    print("Sanity check passed.")
