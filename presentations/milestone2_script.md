# Milestone 2 — Speaker Script (~6 min)

Target: ~6 minutes. ~45–50s per slide. Times are cumulative budgets, not hard limits.

---

## Slide 1 — Title  (~15s)
"Hi everyone. I'm building an action-conditioned **world model** for Atari Breakout — a model that learns to simulate the game itself. This is my Milestone 2: the technical approach and plan."

---

## Slide 2 — Goal  (~45s)
"The idea of a world model is simple: given the past few frames and an action, predict the next frame. If you can do that well, you can chain predictions together and *roll out* an entire imagined clip of gameplay.

My target is exactly that — feed in a few starting frames plus a sequence of actions, and have the model generate a playable-looking continuation: the paddle moves with the action, the ball bounces, bricks break.

The thing that makes this hard is scale: the ball is only one or two pixels at 64×64. During rollout the model feeds on its own output, so tiny errors compound — a faint ball quickly fades to nothing. Keeping that ball alive is the central challenge."

---

## Slide 3 — Task & Data  (~45s)
"Concretely: I use Atari Breakout through the Arcade Learning Environment. Observations are 64×64 grayscale. My state is a stack of the last four frames — that's what gives the model velocity information, so it knows which way the ball is moving. Actions are four discrete choices, one-hot encoded, collected by random play.

I use the data three ways. For **training**, predict frame t+1 from the previous four frames and the action. For **teacher-forced** evaluation, every input is a real frame — this isolates pure one-step quality. For **rollout** evaluation, the model feeds on its own predictions — this is the honest test of long-horizon stability."

---

## Slide 4 — Method  (~55s)
"Here's the architecture. A CNN **encoder** turns the four-frame stack into a latent z_t. A **GRU dynamics** module takes that latent plus the embedded action and predicts the next latent z_{t+1} — this is the part that models *where things move*. A **U-Net decoder** turns the predicted latent back into an image.

Two things to highlight on the diagram. First, the dashed **skip connections** from encoder to decoder — they carry pixel-level detail directly, bypassing the latent bottleneck. Second, the **action** in purple: it feeds both the GRU *and*, through FiLM, every layer of the decoder.

Finally, the output isn't a fresh image — the head predicts a small bounded **residual**, a change added to the previous frame. The next frame is the previous frame plus a scaled tanh delta, clamped to valid range."

---

## Slide 5 — Design intuition  (~55s)
"Every one of those choices came from diagnosing a concrete failure — this is the core of my approach.

First, **skip connections**. Without them, a 256-dimensional global latent has to encode the exact location of a one-pixel ball, and it just can't — you get a blurry, fading ball. Skips fix that by piping spatial detail straight through.

Second, **FiLM action conditioning**. My first version just concatenated the four-dim action onto the 256-dim latent. The action was under two percent of the signal, so the model learned to *completely ignore it* — I measured action sensitivity at exactly zero. FiLM makes the action modulate every decoder feature map, so it can't be optimized away.

Third, the **bounded residual decoder**. When I let the model output free pixels, it collapsed to predicting the average 'brick-wall' frame. Predicting a small change instead keeps it anchored to reality."

---

## Slide 6 — Why we expect this to work  (~45s)
"Why should this work at all? The key insight: one-step Breakout is *nearly deterministic*. Given the true four frames and the action, the next frame is basically fixed. So a simple L1 loss can fit it sharply — the famous 'L1 makes things blurry' problem only shows up under genuine uncertainty, which happens in long rollout, not in one-step prediction.

My loss is motion-aware: a weighted image term, plus a delta term that focuses on the pixels that actually changed, plus a false-motion term that penalizes editing pixels that should stay still. And because each module has one clear job — what's on screen, where it moves, small edits — failures are diagnosable, which is exactly how I found those three bugs."

---

## Slide 7 — Baselines  (~40s)
"My baselines are designed as ablations — each one removes a single design choice. Copy-last-frame is the trivial lower bound. A bottleneck model with no skips tests the value of skip connections. A concat-action model tests FiLM. And the residual model without rollout training shows the gap between one-step and rollout.

Because each baseline isolates one component, the comparison directly measures what each piece buys me — that's what I'll report."

---

## Slide 8 — Experimental setup & evaluation  (~50s)
"Setup: AdamW at 1e-4. I train teacher-forced first to get clean one-step prediction, then add an autoregressive rollout loss with a curriculum — warm up, then ramp it in. Heavy runs go on GPU via MPS or Colab.

For metrics, the headline is **rollout coherence horizon** — how many frames the model can generate before the motion collapses, shown in this plot: true motion stays flat, but a weak model's predicted motion decays to zero. I also report one-step error, action sensitivity — the difference in output across actions — and a physics check by tracking the ball's centroid trajectory. I built diagnostic tooling for all of these."

---

## Slide 9 — Current progress  (~45s)
"Where I am now. The data pipeline and the FiLM U-Net are implemented. Most importantly, **one-step teacher-forced prediction is clean** — I can clearly see both the ball and the paddle moving, with correct motion and no noise. And action sensitivity went from exactly zero to clearly nonzero once I added FiLM, so the action conditioning provably works. I've also built a diagnostic suite — sensitivity tests, motion traces, and GIF visualizations.

Next is rollout training and measuring that coherence horizon, then running the ablations. As a stretch goal, if the deterministic model can't hold long rollouts, I'll try a discrete-token approach in the IRIS style, which is designed for exactly that."

---

## Slide 10 — Summary  (~30s)
"To summarize: I'm building an action-conditioned world model for Breakout. The method is a FiLM-action U-Net — skips for detail, FiLM for action, bounded residual for stability — and each choice fixes a failure I actually measured. One-step works; rollout training and ablations are next; and I'll evaluate primarily on the rollout coherence horizon, the honest measure of a world model. Thank you — happy to take questions."

---

## Anticipated Q&A (prep, not slides)
- **Q: Why not just use Dreamer / a stochastic latent?** "Deterministic one-step is nearly sufficient and far easier to debug; stochastic latents are my fallback if rollout multi-modality becomes the bottleneck."
- **Q: Why grayscale / 64×64?** "Compute budget. The dynamics are the interesting part, not color; 64×64 keeps training feasible on a laptop/Colab."
- **Q: How will you know rollout 'works'?** "The coherence-horizon curve plus the ball-centroid physics check — quantitative, not just eyeballing the GIF."
- **Q: Biggest risk?** "Long-horizon collapse. If the deterministic model caps out at ~20–30 frames, the IRIS-style token model is the planned mitigation."
