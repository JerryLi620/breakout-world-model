# Milestone 3 — Speaker Script (~8 min)

Target: ~8 minutes. Slides 3–4 (the wall vs the breakthrough) are the core — give them the most time. Times are cumulative budgets.

---

## Slide 1 — Title  (~15s)
"Hi — this is the results milestone for our action-conditioned world model for Atari Breakout. I'll show what we built, how it's evaluated, where the first approach broke, and how the second approach fixed it."

---

## Slide 2 — Recap & evaluation  (~50s)
"Quick recap. The goal is a learned simulator: give it a few start frames and an action sequence, and it rolls out a coherent Breakout clip.

We compare two models. The first is a **deterministic FiLM U-Net** that regresses the next frame with an L1 loss. The second is a **token model** — a VQ-VAE tokenizer plus a GPT that predicts frame tokens by classification and sampling.

We evaluate on five things: one-step prediction error; the **rollout coherence horizon** — how many frames until it falls apart; brightness stability over the rollout; action sensitivity; and whether the model produces genuinely different futures. Baselines are copy-last-frame and ablations of the deterministic model."

---

## Slide 3 — Result 1: the deterministic wall  (~1 min 45s)
"First result. The deterministic model learns **one-step prediction well** — teacher-forced predictions are clean, and adding FiLM conditioning made the action actually matter: action sensitivity went from exactly zero to 0.2.

But **rollout is where it breaks**, and this is the key finding. When the model feeds on its own predictions, errors compound — and the *type* of failure depends entirely on one regularization knob.

With weak brightness regularization, you get **blow-up** — the top filmstrip. The max pixel value climbs from 0.58 to 1.0 by about frame six, and the whole screen whites out.

With strong brightness regularization, you get the opposite — **freeze**, the bottom filmstrip. The ball dies, motion drops from about 12 to 1, and the frame is essentially frozen.

The important point: **there's no setting in between that's stable.** You trade one failure for the other. This is the classic exposure-bias wall, and it's a fundamental limitation of deterministic pixel regression — not something we could tune away. That's what motivated switching architectures."

---

## Slide 4 — Result 2: the token model works  (~1 min 45s)
"Second result — the token model. Stage one is a VQ-VAE that turns each 128×128 frame into a 16×16 grid of 256 discrete tokens, from a codebook of 512. It reconstructs frames at a loss of 0.0034 — the ball and paddle come through clearly.

Stage two is a GPT over those tokens, 27 million parameters, trained to a loss of 0.0084. We added a KV cache so generating a rollout takes seconds instead of minutes.

And here's the payoff — the filmstrips. The top row is ground truth; the bottom is the model's generated rollout, primed on four real frames. It stays **coherent across the whole clip** — the ball bounces, the paddle tracks, the bricks persist. **No blow-up, no freeze.** It broke through the wall that stopped the deterministic model.

Why does this work? Because predicting discrete tokens is a *classification* problem with sampling, not regression. It doesn't average over futures the way L1 does, and discrete codes can't gradually decay — so the rollout stays sharp and stable."

---

## Slide 5 — Comparison & analysis  (~1 min 15s)
"Putting it side by side. Copy-last-frame is static. The deterministic model does one-step well but collapses in rollout after about five or six frames, drifting to white-out or freezing, and it's purely deterministic. The token model is clean one-step, coherent across the full rollout, brightness-stable, and stochastic.

The analysis: the deterministic model fails because L1 regression averages over possible futures and has no mechanism to stop intensity drift in closed loop — so you get blur, then blow-up or freeze. The token model works because classification-plus-sampling avoids that averaging, and discrete codes can't decay.

And I want to stress — **this matched our hypothesis going in.** We predicted L1 would hit a wall in rollout, and it did. We predicted discrete tokens would break it, and they did. So the results align with what we expected, which is reassuring."

---

## Slide 6 — Stochasticity  (~45s)
"One more result, and it's the project's headline goal. These are three rollouts from the **identical** four priming frames — only the random seed differs. Each column is the same timestep across the three samples.

If you scan a column top to bottom, the ball is in **different places** — the three futures diverge. That means the model didn't memorize one continuation; it learned a **distribution** over futures and samples from it. That's the stochastic, action-conditioned simulation we set out to build."

---

## Slide 7 — Limitations & next steps  (~1 min)
"Limitations, honestly. First, the model doesn't respawn a new ball after one is lost — but we diagnosed this as a **data-coverage** issue, not an architecture flaw: our 20-frame random-policy clips almost never contained a lose-then-respawn cycle, so the model never saw it. Notably, it doesn't hallucinate a respawn either — it faithfully reproduces only what it observed, which is the right behavior.

Second, the codebook is underused — perplexity around 5 out of 512 — which is fine for low-entropy Breakout but would limit transfer.

For next steps: we've already **fixed the respawn at the data level** — a FIRE-on-life-loss collection policy plus longer clips, which now puts a respawn in nine out of ten sequences; we're retraining on that. Beyond that: quantifying the coherence horizon against ground truth, a temperature ablation, and color.

The takeaway: we **demonstrated** the deterministic rollout wall, then **broke it** with a discrete-token world model producing coherent, stochastic, action-conditioned Breakout. Thank you — happy to take questions."

---

## Anticipated Q&A (prep, not slides)
- **Q: Why is the codebook so underused?** "Breakout is low-entropy — mostly black, plus a few patch types. The VQ-VAE only needs a handful of codes, so perplexity is low but reconstruction is still excellent. It'd matter more on a visually richer game."
- **Q: How do you measure 'coherence horizon'?** "Right now qualitatively from the rollout, plus per-frame motion staying GT-like instead of collapsing to zero. Next step is a quantitative per-frame-error curve vs ground truth."
- **Q: Isn't comparing a 27M token model to the U-Net unfair?** "It's not about parameter count — the deterministic model fails for a structural reason (L1 averaging), and scaling it wouldn't fix the exposure-bias wall. The point is the architecture class, not size."
- **Q: Why discrete tokens over a stochastic continuous latent (Dreamer)?** "Tokens give sharp frames and easy temperature sampling, and the cross-entropy loss sidesteps the blur problem directly. Dreamer-style RSSM is a valid alternative we considered."
- **Q: Does it generalize to unseen sequences?** "Rollouts here prime on held-out clips' start frames, so it's not memorizing a single trajectory — but rigorous generalization metrics are future work."
