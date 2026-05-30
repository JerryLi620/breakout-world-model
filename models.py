"""
models.py – FiLM-action U-Net world model.

Architecture:

    stacked frames (B, num_frames, 64, 64)
        |
        v
    [Encoder] -> spatial skips {f1, f2, f3, f4} + flat latent z_t
        |
        v
    [Recurrent Dynamics]
        z_next, h_next = GRU([z_t, action_embed], h)
        |
        v
    [Action-FiLM U-Net Decoder]
        decoder upsamples z_next, concatenates encoder skips, and after
        every merge block applies FiLM (per-channel scale + bias) computed
        from the action embedding.

Why FiLM:
    Concatenating a 4-dim one-hot action with a 256-dim latent and then
    LayerNorming makes the action gradient tiny (~1.5% of input variance).
    The model learns to ignore it. FiLM forces every decoder block to be
    modulated by the action, so the action's effect on the output cannot
    be optimized away.

action_dim=0 cleanly disables both the action embedding and the FiLM modules,
so the same code trains on Atari (with actions) and Moving MNIST / bouncing
ball (without actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape (B, C, H, W), got {x.shape}")
    B, C, H, W = x.shape
    y = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
    y = y.view(1, 1, H, 1).expand(B, 1, H, W)
    xc = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
    xc = xc.view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, y, xc], dim=1)


def _conv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(),
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 4, latent_dim: int = 256):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            _conv_block(32, 32),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            _conv_block(64, 64),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            _conv_block(128, 128),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.SiLU(),
            _conv_block(256, 256),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor):
        x = add_coord_channels(x)
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        z = self.fc(f4)
        return z, {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


# ---------------------------------------------------------------------------
# Action embedding
# ---------------------------------------------------------------------------

class ActionEmbedding(nn.Module):
    """Embed a (B, action_dim) one-hot action into a wide vector (B, embed_dim)."""

    def __init__(self, action_dim: int, embed_dim: int = 128):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("ActionEmbedding requires action_dim > 0.")
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(action)


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """
    Per-channel scale (gamma) + bias (beta) modulation conditioned on an
    embedding e:

        gamma, beta = Linear(e) -> 2C
        out = (1 + gamma) * feat + beta

    The (1 + gamma) form is the standard FiLM choice — initializing the linear
    to zero means the layer starts as identity, so it can only help, never hurt.
    """

    def __init__(self, embed_dim: int, num_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * num_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.num_channels = num_channels

    def forward(self, feat: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W), e: (B, embed_dim)
        gamma_beta = self.proj(e)                               # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)               # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * feat + beta


# ---------------------------------------------------------------------------
# Recurrent Dynamics
# ---------------------------------------------------------------------------

class RecurrentDynamicsModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        action_embed_dim: int = 0,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.noise_std = noise_std

        input_dim = latent_dim + action_embed_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.to_z = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def init_hidden(self, batch_size, device, dtype=torch.float32):
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def forward(self, z, action_embed=None, h=None, deterministic=False):
        if z.dim() != 2:
            raise ValueError(f"Expected z (B, latent_dim), got {z.shape}")
        B = z.size(0)
        if h is None:
            h = self.init_hidden(B, z.device, z.dtype)

        if self.action_embed_dim > 0:
            if action_embed is None:
                raise ValueError("action_embed required when action_embed_dim > 0")
            x = torch.cat([z, action_embed], dim=-1)
        else:
            x = z

        x = self.input_norm(x)
        h_next = self.gru(x, h)
        z_next = self.to_z(h_next)
        if self.noise_std > 0.0 and not deterministic:
            z_next = z_next + torch.randn_like(z_next) * self.noise_std
        return z_next, h_next


# ---------------------------------------------------------------------------
# FiLM-conditioned U-Net Decoder Body
# ---------------------------------------------------------------------------

class _UNetBody(nn.Module):
    def __init__(self, latent_dim: int = 256, action_embed_dim: int = 0):
        super().__init__()
        self.action_embed_dim = action_embed_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4),
            nn.SiLU(),
        )

        self.merge4 = _conv_block(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.merge3 = nn.Sequential(_conv_block(128 + 128, 128), _conv_block(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.merge2 = nn.Sequential(_conv_block(64 + 64, 64), _conv_block(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.merge1 = nn.Sequential(_conv_block(32 + 32, 32), _conv_block(32, 32))
        self.up0 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.final = nn.Sequential(_conv_block(32, 32), _conv_block(32, 32))

        if action_embed_dim > 0:
            self.film4 = FiLM(action_embed_dim, 256)
            self.film3 = FiLM(action_embed_dim, 128)
            self.film2 = FiLM(action_embed_dim, 64)
            self.film1 = FiLM(action_embed_dim, 32)
            self.film0 = FiLM(action_embed_dim, 32)
        else:
            self.film4 = self.film3 = self.film2 = self.film1 = self.film0 = None

    def _modulate(self, feat, film, e):
        if film is None or e is None:
            return feat
        return film(feat, e)

    def forward(self, z: torch.Tensor, skips: dict, action_embed=None) -> torch.Tensor:
        B = z.size(0)
        d = self.fc(z).view(B, 256, 4, 4)
        d = self.merge4(torch.cat([d, skips["f4"]], dim=1))
        d = self._modulate(d, self.film4, action_embed)

        d = self.up3(d)
        d = self.merge3(torch.cat([d, skips["f3"]], dim=1))
        d = self._modulate(d, self.film3, action_embed)

        d = self.up2(d)
        d = self.merge2(torch.cat([d, skips["f2"]], dim=1))
        d = self._modulate(d, self.film2, action_embed)

        d = self.up1(d)
        d = self.merge1(torch.cat([d, skips["f1"]], dim=1))
        d = self._modulate(d, self.film1, action_embed)

        d = self.up0(d)
        d = self.final(d)
        d = self._modulate(d, self.film0, action_embed)
        return d


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, feat):
        return {"delta": torch.tanh(self.delta_head(feat))}


class EraseDrawHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.erase_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.draw_mask_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.draw_value_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.erase_head.bias, -2.0)
        nn.init.constant_(self.draw_mask_head.bias, -2.0)
        nn.init.constant_(self.draw_value_head.bias, 0.0)

    def forward(self, feat):
        return {
            "erase_logits": self.erase_head(feat),
            "draw_mask_logits": self.draw_mask_head(feat),
            "draw_value_logits": self.draw_value_head(feat),
        }


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        action_dim: int = 0,
        action_embed_dim: int = 128,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        noise_std: float = 0.0,
        decoder_type: str = "residual",
        residual_scale: float = 0.3,
    ):
        super().__init__()

        if out_channels != 1:
            raise ValueError("Grayscale only (out_channels=1).")
        if decoder_type not in ("residual", "erase_draw"):
            raise ValueError(f"Bad decoder_type {decoder_type}")

        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)

        if action_dim > 0:
            self.action_embed = ActionEmbedding(action_dim, action_embed_dim)
            self._action_embed_dim = action_embed_dim
        else:
            self.action_embed = None
            self._action_embed_dim = 0

        self.dynamics = RecurrentDynamicsModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_embed_dim=self._action_embed_dim,
            noise_std=noise_std,
        )

        self.decoder_body = _UNetBody(
            latent_dim=latent_dim,
            action_embed_dim=self._action_embed_dim,
        )

        if decoder_type == "residual":
            self.head = ResidualHead()
        else:
            self.head = EraseDrawHead()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.decoder_type = decoder_type
        self.residual_scale = residual_scale

    def init_hidden(self, batch_size, device, dtype=torch.float32):
        return self.dynamics.init_hidden(batch_size, device, dtype)

    def _embed_action(self, action):
        if self.action_embed is None:
            return None
        if action is None:
            raise ValueError("Action required (action_dim > 0).")
        return self.action_embed(action.to(next(self.action_embed.parameters()).dtype))

    def forward(
        self,
        x_t,
        action=None,
        h=None,
        deterministic=False,
        edit_scale: float = 1.0,
        return_aux: bool = False,
    ):
        z_t, skips = self.encoder(x_t)
        a_e = self._embed_action(action) if self.action_dim > 0 else None

        z_next, h_next = self.dynamics(
            z=z_t, action_embed=a_e, h=h, deterministic=deterministic,
        )

        feat = self.decoder_body(z_next, skips, action_embed=a_e)
        decoder_out = self.head(feat)
        x_last = x_t[:, -1:].contiguous()

        if self.decoder_type == "residual":
            delta = decoder_out["delta"]
            x_hat = torch.clamp(x_last + self.residual_scale * delta, 0.0, 1.0)
            extras = {"delta": delta}
        else:
            erase = (torch.sigmoid(decoder_out["erase_logits"]) * edit_scale).clamp(0, 1)
            draw_m = (torch.sigmoid(decoder_out["draw_mask_logits"]) * edit_scale).clamp(0, 1)
            draw_v = torch.sigmoid(decoder_out["draw_value_logits"])
            x_hat = torch.clamp(x_last * (1 - erase) + draw_m * draw_v, 0, 1)
            extras = {"erase_mask": erase, "draw_mask": draw_m, "draw_value": draw_v}

        if return_aux:
            return x_hat, z_t, z_next, h_next, {**decoder_out, **extras}
        return x_hat, z_t, z_next, h_next


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B = 4
    num_frames = 4

    for action_dim in (0, 4):
        for decoder_type in ("residual", "erase_draw"):
            print(f"\n--- action_dim={action_dim} decoder={decoder_type} ---")
            model = WorldModel(
                in_channels=num_frames,
                out_channels=1,
                action_dim=action_dim,
                latent_dim=256,
                hidden_dim=512,
                decoder_type=decoder_type,
                residual_scale=0.3,
            )

            x = torch.rand(B, num_frames, 64, 64)
            a = None
            if action_dim > 0:
                a_idx = torch.randint(0, action_dim, size=(B,))
                a = F.one_hot(a_idx, num_classes=action_dim).float()

            x_hat, z, z_next, h_next, aux = model(
                x, action=a, h=None, deterministic=True, return_aux=True,
            )
            print(f"  x_hat: {tuple(x_hat.shape)} range [{x_hat.min():.3f}, {x_hat.max():.3f}]")
            print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

            # Action sensitivity check (random init): with FiLM init to zero,
            # at init every action should produce identical output. After
            # training, this should diverge.
            if action_dim > 0:
                a0 = F.one_hot(torch.zeros(B, dtype=torch.long), num_classes=action_dim).float()
                a1 = F.one_hot(torch.ones(B, dtype=torch.long), num_classes=action_dim).float()
                p0, *_ = model(x, action=a0, h=None, deterministic=True)
                p1, *_ = model(x, action=a1, h=None, deterministic=True)
                # At init the action embedding is small but nonzero. The diff
                # comes through the GRU input concat; FiLM contribution is 0.
                print(f"  init action diff sum: {(p0 - p1).abs().sum().item():.6f}")

    print("\nSanity check passed.")
