"""
transformer.py – Stage 2 of the discrete-token world model.

A GPT-style causal Transformer over the token sequence produced by the frozen
VQ-VAE tokenizer. Each frame is P = (H/8 * W/8) tokens (256 for 128x128 input).
A frame-step is laid out as:

    [ frame tokens (P) , action token (1) ]

and a window of L frames is the flattened interleaving:

    [ f0_0..f0_{P-1}, a0,  f1_0..f1_{P-1}, a1,  ...,  f_{L-1}.., a_{L-1} ]

The action token a_t sits immediately before frame f_{t+1}, so the causal model
predicts the next frame's tokens conditioned on all past frames AND the action
taken — exactly the action-conditioned next-state distribution we want.

Training: next-token cross-entropy, with loss computed only on frame-token
targets (actions are control inputs, given — we don't predict them).

Rollout: autoregressively sample frame tokens (temperature / top-k) -> different
samples give different futures -> stochastic, action-conditioned simulation.
Decode tokens back to pixels with the frozen tokenizer.

Vocabulary: frame tokens 0..num_codes-1, action token = num_codes + action_id.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GPT blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # KV cache: prepend previously computed keys/values (incremental decode)
        if cache is not None and "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        if cache is not None:
            cache["k"] = k
            cache["v"] = v

        # Causal mask only needed for multi-token (prefill) passes. A single
        # query token (decode step) attends to all cached keys — no mask.
        causal = T > 1
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=512, n_layers=8,
                 n_heads=8, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, caches=None, pos_offset=0):
        B, T = idx.shape
        if pos_offset + T > self.block_size:
            raise ValueError(f"position {pos_offset + T} > block_size {self.block_size}")
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for i, blk in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = blk(x, cache=cache)
        x = self.ln_f(x)
        return self.head(x)

    def empty_caches(self):
        return [dict() for _ in self.blocks]


# ---------------------------------------------------------------------------
# Token World Model wrapper
# ---------------------------------------------------------------------------

class TokenWorldModel(nn.Module):
    def __init__(self, num_codes, action_dim, tokens_per_frame, context_frames=8,
                 d_model=512, n_layers=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.action_dim = action_dim
        self.tokens_per_frame = tokens_per_frame      # P
        self.context_frames = context_frames          # L
        self.step = tokens_per_frame + 1               # P+1 (frame tokens + action)
        self.vocab_size = num_codes + action_dim
        block_size = context_frames * self.step
        self.gpt = GPT(self.vocab_size, block_size, d_model, n_layers, n_heads, dropout)

    # ----- sequence layout -----
    def build_sequence(self, frame_tokens, actions):
        """
        frame_tokens: (B, L, P) ints in [0, num_codes)
        actions:      (B, L)    ints in [0, action_dim)
        returns seq:  (B, L*(P+1))
        """
        B, L, P = frame_tokens.shape
        act_tok = (actions + self.num_codes).unsqueeze(-1)        # (B, L, 1)
        per_step = torch.cat([frame_tokens, act_tok], dim=-1)     # (B, L, P+1)
        return per_step.view(B, L * self.step)

    def forward(self, frame_tokens, actions):
        """Teacher-forced training. Returns (logits, loss)."""
        seq = self.build_sequence(frame_tokens, actions)
        inp = seq[:, :-1]
        tgt = seq[:, 1:]
        logits = self.gpt(inp)                                    # (B, T-1, vocab)
        # only supervise frame-token targets (ignore action targets)
        loss_mask = tgt < self.num_codes
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tgt.reshape(-1),
            reduction="none",
        )
        loss = (loss * loss_mask.reshape(-1).float()).sum() / loss_mask.sum().clamp_min(1)
        return logits, loss

    def _sample(self, logits, temperature, top_k):
        logits = logits.clone()
        logits[:, self.num_codes:] = float("-inf")   # frame tokens only
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)            # (B,)

    @torch.no_grad()
    def generate(self, prime_tokens, actions, n_new_frames, temperature=1.0, top_k=None):
        """
        KV-cached autoregressive rollout with a sliding window.

        For each new frame we (re)prefill the last (context_frames-1) frames as
        context — one forward pass — then decode that frame's P tokens using the
        KV cache (each token = one cheap single-token pass). This is ~10-30x
        faster than recomputing the full context per token, and supports rollouts
        longer than the training context (the window slides).

        prime_tokens: (B, k, P) tokens of the first k real frames
        actions:      (B, >= k + n_new_frames - 1) action ids
        returns:      (B, n_new_frames, P) generated frame tokens
        """
        B, k, P = prime_tokens.shape
        ctx_frames = self.context_frames - 1          # past frames to condition on
        all_frames = [prime_tokens[:, i] for i in range(k)]   # list of (B, P)

        generated = []
        for j in range(n_new_frames):
            f = k + j                                  # absolute index of frame to make
            start = max(0, f - ctx_frames)
            ctx_ft = torch.stack(all_frames[start:f], dim=1)   # (B, n_ctx, P)
            ctx_act = actions[:, start:f]                       # (B, n_ctx) = a_start..a_{f-1}
            prompt = self.build_sequence(ctx_ft, ctx_act)       # (B, n_ctx*(P+1))

            caches = self.gpt.empty_caches()
            logits = self.gpt(prompt, caches=caches, pos_offset=0)
            pos = prompt.shape[1]
            last = logits[:, -1, :]                             # predicts frame f token 0

            frame_tok = []
            for p in range(P):
                tok = self._sample(last, temperature, top_k)    # (B,)
                frame_tok.append(tok)
                if p < P - 1:
                    logits = self.gpt(tok[:, None], caches=caches, pos_offset=pos)
                    pos += 1
                    last = logits[:, -1, :]

            all_frames.append(torch.stack(frame_tok, dim=1))    # (B, P)
            generated.append(all_frames[-1])

        return torch.stack(generated, dim=1)                    # (B, n_new, P)


# ---------------------------------------------------------------------------
# Sanity check (tiny)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, P, K, A = 2, 4, 16, 512, 4   # tiny P=16 for speed
    model = TokenWorldModel(num_codes=K, action_dim=A, tokens_per_frame=P,
                            context_frames=L, d_model=128, n_layers=2, n_heads=4)

    ft = torch.randint(0, K, (B, L, P))
    acts = torch.randint(0, A, (B, L))
    logits, loss = model(ft, acts)
    print(f"seq len      : {L*(P+1)}  block_size: {model.gpt.block_size}")
    print(f"logits       : {tuple(logits.shape)}")
    print(f"loss         : {loss.item():.4f}  (~ln(vocab)={math.log(K+A):.3f} at init)")

    prime = torch.randint(0, K, (B, 2, P))
    a_full = torch.randint(0, A, (B, 4))
    gen = model.generate(prime, a_full, n_new_frames=2, temperature=1.0, top_k=50)
    print(f"generate     : prime {tuple(prime.shape)} -> {tuple(gen.shape)}")
    print(f"params       : {sum(p.numel() for p in model.parameters()):,}")
    print("Sanity check passed.")
