# Breaking the Rollout Wall: A Discrete-Token World Model for Atari Breakout

An action-conditioned world model for Atari Breakout. Given a few starting
frames and a sequence of actions, the model rolls out a coherent, game-like
continuation. Built and analyzed for the CS231n final project.

The repo contains two models trained side-by-side, plus the full diagnostic
and reporting pipeline:

1. **Deterministic FiLM U-Net** — a CNN encoder, GRU dynamics with FiLM
   action conditioning, and a U-Net decoder with a bounded residual head.
   Learns clean one-step predictions but **cannot roll out stably**: errors
   compound into either brightness blow-up or motion freeze, with no
   intermediate setting that's stable. This is the exposure-bias wall of
   pixel-regression world models.
2. **Discrete-token world model** — a VQ-VAE tokenizer that maps each
   128×128 frame to a 16×16 grid of 256 discrete codes, and a causal GPT
   that models next-frame tokens given past tokens and actions. Generates
   **coherent, stochastic, action-conditioned rollouts** — the deterministic
   wall is broken.

Result GIFs live under [`results/`](results/) and the final write-up under
[`report/`](report/).

## Repo layout

```
.
├── README.md
├── .gitignore
│
├── dataset.py                    # AtariDataset (ALE), SavedSequenceDataset (file-backed)
├── models.py                     # Deterministic FiLM U-Net
├── tokenizer.py                  # VQ-VAE with EMA codebook
├── transformer.py                # GPT TokenWorldModel (KV-cached generation)
├── utils.py                      # checkpoint + GIF helpers
│
├── train.py                      # Train the deterministic model
├── train_tokenizer.py            # Stage 1: train the VQ-VAE
├── train_transformer.py          # Stage 2: train the GPT (uses frozen tokenizer)
│
├── export_dataset.py             # Bake ALE trajectories into a .pt file (for cloud training)
├── modal_train_tokenizer.py      # Modal A100 wrapper for the tokenizer
├── modal_train_transformer.py    # Modal A100 wrapper for the transformer
│
├── diagnose.py                   # Action sensitivity + rollout traces + GIFs
├── sample_rollout.py             # Autoregressive token rollout → GIFs (stochastic samples)
│
├── results/
│   ├── baseline/                 # Deterministic teacher-forced (clean one-step)
│   ├── failures/                 # Deterministic rollout failure modes (blow-up, freeze, static)
│   ├── tokenizer/                # VQ-VAE GT-vs-reconstruction
│   └── rollouts/                 # Token model rollouts (GT + 3 stochastic samples)
│
├── presentations/
│   ├── milestone2.tex + script   # Technical approach
│   ├── milestone3.tex + script   # Results & analysis
│   └── slide_frames/             # Extracted GIF frames used in milestone3 deck
│
└── report/
    ├── report.tex                # CVPR-style final report
    ├── references.bib
    ├── figures/                  # All paper figures (incl. measured coherence plot)
    └── cvpr.sty, ieee.bst, ...   # Template files (CVPR 2017 kit)
```

Trained checkpoints (`*.pt`) and exported dataset files are **not committed**
(they exceed GitHub's 100 MB limit and are easily regenerated). Reproduce them
with the workflow below.

## Setup

Python 3.11+, PyTorch 2.x, plus:

```bash
pip install torch numpy pillow imageio gymnasium ale-py matplotlib
# Optional, for cloud training:
pip install modal
```

For Atari data collection, ALE will auto-fetch the Breakout ROM the first
time you run a training/export script.

## Quickstart

### Local (small smoke test)

```bash
# 1. Sanity-check the architectures
python3 models.py
python3 tokenizer.py
python3 transformer.py

# 2. Overfit the deterministic model on one batch (~minutes on CPU/MPS)
python3 train.py --dataset atari --overfit_batch --overfit_steps 800 \
  --decoder_type residual --simple_one_step --debug_stats
```

### Full pipeline (recommended on GPU / Modal)

```bash
# 1. Export the dataset once (uses ALE; produces ~6.5 GB file at 128x128, seq_len 50)
python3 export_dataset.py --img_size 128 --num_sequences 2000 --seq_len 50 \
  --out atari_data_128_v2.pt

# 2. Train the VQ-VAE tokenizer (stage 1)
python3 train_tokenizer.py --data_file atari_data_128_v2.pt \
  --num_codes 512 --embedding_dim 256 --batch_size 64 --frame_batch 256 \
  --epochs 30 --object_weight 30

# 3. Train the GPT on tokenized frames (stage 2; uses frozen tokenizer)
python3 train_transformer.py --data_file atari_data_128_v2.pt \
  --tokenizer tokenizer.pt --context_frames 8 \
  --d_model 512 --n_layers 8 --n_heads 8 --batch_size 16 --epochs 30

# 4. Generate stochastic rollouts
python3 sample_rollout.py --data_file atari_data_128_v2.pt \
  --tokenizer tokenizer.pt --transformer transformer.pt \
  --prime 4 --n_samples 3 --temperature 1.0 --top_k 50 --gif_seq 0
```

### Cloud (Modal A100)

If you have Modal credits, the cloud workflow is faster and matches what
trained the checkpoints behind the figures:

```bash
modal volume create worldmodel-data
modal volume put worldmodel-data atari_data_128_v2.pt /atari_data_128_v2.pt

modal run modal_train_tokenizer.py \
  --data-file /data/atari_data_128_v2.pt \
  --checkpoint /data/tokenizer_128_v2.pt \
  --recon-gif /data/tokenizer_recon_128_v2.gif

modal run modal_train_transformer.py \
  --data-file /data/atari_data_128_v2.pt \
  --tokenizer /data/tokenizer_128_v2.pt \
  --checkpoint /data/transformer_v2.pt

# Pull results back down
modal volume get --force worldmodel-data /tokenizer_128_v2.pt    ./tokenizer.pt
modal volume get --force worldmodel-data /transformer_v2.pt      ./transformer.pt
modal volume get --force worldmodel-data /tokenizer_recon_128_v2.gif ./tokenizer_recon.gif
```

## Diagnostics

`diagnose.py` is the workhorse — action sensitivity, teacher-forced motion
trace, free-rollout trace, and optional GIFs:

```bash
# Token-model rollout traces (motion / brightness per step + GIFs)
python3 diagnose.py --dataset atari --num_frames 4 --action_dim 4 \
  --decoder_type residual --residual_scale 0.2 --rollout_gif --gif_seq 0
```

`sample_rollout.py` generates the stochastic-rollout GIFs that became the
project's headline result.

## Key findings

- **Deterministic L1 regression hits a structural wall** in rollout. The
  failure type (blow-up vs.\ freeze) is controlled by a single regularization
  weight, with no stable operating point between them.
- **Discrete tokens + autoregressive sampling break the wall.** The token
  model produces coherent rollouts and, because generation is stochastic,
  yields different plausible futures from the same prime — the
  action-conditioned simulator we set out to build.
- **A "missing behavior" turned out to be a data-coverage issue, not an
  architecture flaw.** A FIRE-on-life-loss collection policy puts a full
  respawn cycle in 9 of 10 trajectories (vs.\ essentially 0 before).

See [`report/report.tex`](report/report.tex) for the full write-up,
[`presentations/milestone3.tex`](presentations/milestone3.tex) for the slide
deck, and [`results/`](results/) for the visual evidence.

## Acknowledgments

CVPR 2017 author kit (template files in `report/`). VQ-VAE follows
van den Oord et al.\ (2017); FiLM follows Perez et al.\ (2018); the token
world-model design is inspired by IRIS (Micheli et al., 2023).
