# Parameter golf at laptop scale: ablations, a leave-one-out stack and two new layer options

Which training and architecture techniques lower the validation bits per byte (bpb) of a small GPT whose compressed weights must fit in 6.5 MB, when the training budget is fixed at 41M tokens? And when the helpful ones are combined, do their gains add up?

The model is nanolm's `L5_golf_mini` (6 layers × 384, 6.6M parameters, U-Net skips, Muon + Adam, int8 + lzma export) trained for 2,500 steps of 16,384 tokens (40,960,000 tokens) on FineWeb with nanolm's 1,024-token SentencePiece vocabulary. Every configuration in the results tables is trained with 5 seeds. The work runs on [nanolm](https://github.com/lkk688/DeepDataMiningLearning/tree/8418fc1bae85657190f11749483bf2e729b9702c/DeepDataMiningLearning/llm) at commit `8418fc1` with one patch (`patches/`) that adds two config options and their tests:

- `token_conv: K`: a causal depthwise convolution over the last K positions, applied to each block's normalized attention input, zero-initialized so it starts as the identity. It is the short convolution of Gated DeltaNet and Mamba layers, placed where Canon-A layers put it.
- `attn_window: W` with `window_layers`: sliding-window attention, so that a query in the listed layers sees only the last W tokens.

## Leaderboard row

| configuration | bpb, seeds 1337, 42, 1 | bpb, all 5 seeds | artifact (largest of the seeds) | config |
|---|---:|---:|---:|---|
| stack + the rest of L6, int6 + lzma | **1.5526 ± 0.0015** | 1.5536 ± 0.0018 | 5,695,318 bytes | `configs/sweeps/stack.yaml`, arm `stack_plus_l6` |

The bpb is that of the artifact: the weights are quantized, compressed, restored and evaluated on the first 2,097,152 validation tokens with a sliding window of stride 64. The artifact counts the compressed weights and nanolm's model and quantization code (23,730 bytes). It is the arm chosen by the leaderboard rule: the lowest mean dev bpb among the five stack-family arms that fit 6.5 MB on every seed (dev bpb: stack_plus_l6 1.6053, stack_adam_lr0.02 1.6128, stack_plus_mlp3x 1.6134, stack 1.6216, stack_matrix_lr0.01 1.6254).

## Setup

- **Data.** Training reads the first FineWeb training shard from its start (40,965,000 tokens). Validation numbers use the first 2,097,152 tokens of the validation split. Techniques and learning rates are chosen on a separate **dev** set: 2,097,152 training-shard tokens starting at token 50,000,000, which no run reads (`scripts/dev_eval.py`). Nothing is chosen on validation.
- **Metrics.** `val_bpb`: validation, non-overlapping windows of 512 tokens (nanolm's default). `sliding_bpb`: stride 64. `quant_val_bpb`: the restored artifact at stride 64. All runs train in fp32 (TF32 matrix multiplies).
- **Statistics.** The seeds are 1337, 42, 1, 2 and 3 for every arm. A difference is paired over seeds (t-test, 4 degrees of freedom). It counts as **resolved** when it is at least twice the noise floor of the two arms compared (the mean of their two within-arm standard deviations). H marks a Holm-rejected test at 0.05; the families are the ten single techniques and the four leave-one-out arms. Other p values are uncorrected.
- **Selection rule.** A technique joins the stack if its dev change is below −2× the noise floor and is negative on all 5 seeds.
- **Machines.** Three rented RTX 4090 machines (`results/machines.json`, `results/environment_*.txt`): **a** ran the ablation; **b** the stack, its leave-one-out arms, its MLP 3× and L6 arms, the baseline rerun, the machine check and 12 of the 15 learning-rate arms; **c** the stack's two learning-rate arms and the other three learning-rate arms. Machine a's Python and torch versions are in each run's `meta.json` (it has no environment file). Every run's `meta.json` records its host. Several runs shared each GPU, so recorded wall-clock times are not comparable; `results/bench.json` gives the training speed of each ablation and stack configuration alone (`nanolm.bench`, with depth recurrence on at every step).

## Results

### Baseline and the tutorial's L6 stack

| | val bpb | sliding (64) | artifact round trip | artifact | train FLOPs |
|---|---:|---:|---:|---:|---:|
| `L5_golf_mini` (int8), 5 seeds | 1.6371 ± 0.0025 | 1.5912 ± 0.0032 | 1.5913 ± 0.0032 | 6.11 MB | 2.11 P |
| nanolm's tutorial | 1.633 (reference run); 1.6338 ± 0.0018 (2 seeds) | | | 6.11 MB | 2.1 P |
| `L6_golf_mini_stack` (int6), 5 seeds | 1.6015 ± 0.0011 | 1.5532 ± 0.0014 | 1.5544 ± 0.0013 | 5.60 MB | 3.09 P |
| nanolm's tutorial | 1.601 | 1.554 (stride 128) | 1.555 (stride 128) | 5.59 MB | 3.1 P |

On seeds 1337, 42 and 1 (the leaderboard row's 3-seed column) the baseline is 1.6358 ± 0.0023. FLOPs: nanolm's per-token estimate (`bench.flops_per_token`: 6 × non-embedding parameters + 12 × layers × width × sequence length) times the training tokens, plus the re-run layers' share from 35% of training (`analysis/summarize.py`).

### Each technique alone (L5 + one change, 5 seeds)

Δ is the change from the baseline (negative is better). Dev Δ / floor: the selection test. Last columns: artifact round trip at stride 64 (int8), the largest artifact over the seeds (✗: over 6.5 MB), training time per step on its own (ms; `nanolm.bench`, depth recurrence on at every step) and training FLOPs (P = 10¹⁵).

| technique | val bpb | val Δ | seeds lower | dev Δ / floor | selected | sliding | round trip | MB | ms/step | FLOPs |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| baseline | 1.6371 ± 0.0025 | | | | | 1.5912 | 1.5913 | 6.11 | 47 | 2.11 |
| XSA, last 3 layers | 1.6284 ± 0.0016 | −0.0087 (p = 0.0009) H | 5/5 | −0.0087 / 0.0019 | yes | 1.5821 | 1.5822 | 6.12 | 49 | 2.11 |
| value residual | 1.6306 ± 0.0016 | −0.0066 (p = 0.0010) H | 5/5 | −0.0067 / 0.0019 | yes | 1.5829 | 1.5830 | 6.12 | 48 | 2.11 |
| **token_conv 4** (new) | 1.6306 ± 0.0013 | −0.0065 (p = 0.0007) H | 5/5 | −0.0068 / 0.0018 | yes | 1.5839 | 1.5839 | 6.13 | 50 | 2.11 |
| MLP 3× | 1.6318 ± 0.0029 | −0.0053 (p = 0.08) | 4/5 | −0.0052 / 0.0026 |  | 1.5864 | 1.5865 | 7.62 ✗ | 52 | 2.54 |
| LeakyReLU(0.5)² | 1.6329 ± 0.0043 | −0.0042 (p = 0.01) | 5/5 | −0.0041 / 0.0033 |  | 1.5879 | 1.5880 | 6.13 | 47 | 2.11 |
| depth recurrence (layers 2-3, from 35%) | 1.6333 ± 0.0041 | −0.0038 (p = 0.02) | 5/5 | −0.0041 / 0.0033 |  | 1.5869 | 1.5870 | 6.11 | 60 | 2.56 |
| **attn_window 128, layers 0-2** (new) | 1.6363 ± 0.0023 | −0.0008 (p = 0.38) | 3/5 | −0.0012 / 0.0023 |  | 1.5903 | 1.5904 | 6.12 | 50 | 2.11 |
| QK-gain 4 | 1.6418 ± 0.0044 | +0.0046 (p = 0.13) | 1/5 | +0.0046 / 0.0034 |  | 1.5968 | 1.5968 | 6.11 | 47 | 2.11 |
| SmearGate | 1.6475 ± 0.0047 | +0.0103 (p = 0.001) H | 0/5 | +0.0103 / 0.0036 |  | 1.6028 | 1.6029 | 6.09 | 47 | 2.11 |
| partial RoPE (16 dims) + LN-scale | 1.6478 ± 0.0042 | +0.0107 (p = 0.02) | 0/5 | +0.0103 / 0.0032 |  | 1.6012 | 1.6014 | 6.09 | 47 | 2.11 |

- Three techniques pass the selection rule: XSA, value residual and token_conv. Their val changes are Holm-significant too.
- token_conv, the new option, gives −0.0065, as much as value residual (−0.0066). SmearGate, the closest existing option (a one-step look-back on the embedding), makes the model worse (+0.0103, Holm-significant).
- attn_window does not change bpb measurably at this sequence length (512): −0.0008 (p = 0.38).
- MLP 3× does not fit 6.5 MB at int8. LeakyReLU² and depth recurrence lower dev bpb on all 5 seeds, but by less than twice the noise floor (0.0033 and 0.0033).
- Seed noise: the median within-arm std of val bpb over the 12 arms is 0.0027.

### Learning rate

`L5_golf_mini` inherits its learning rates (Muon matrix_lr 0.04, Adam lr 0.04) from nanolm's 8×H100 golf baseline (`04_golf_baseline.yaml`, 524,288 tokens per step). Dev bpb, mean over 5 seeds, with the change from 1× (†: resolved). Bold: the best point.

| arm, rate | 0.125× | 0.25× | 0.5× | 1× | 2× |
|---|---:|---:|---:|---:|---:|
| baseline, Muon matrix_lr |  | 1.6348 (−0.0079†) | **1.6341 (−0.0086†)** | 1.6427 | 1.6571 (+0.0144†) |
| token_conv, Muon matrix_lr |  | 1.6297 (−0.0062†) | **1.6275 (−0.0084†)** | 1.6359 | 1.6416 (+0.0057†) |
| attn_window, Muon matrix_lr |  | 1.6362 (−0.0053) | **1.6336 (−0.0079†)** | 1.6415 | 1.6514 (+0.0099†) |
| baseline, Adam lr | 1.6417 (−0.0010) | **1.6373 (−0.0054)** | 1.6386 (−0.0041) | 1.6427 |  |
| token_conv, Adam lr |  | 1.6312 (−0.0048†) | **1.6305 (−0.0055†)** | 1.6359 | 1.6477 (+0.0118†) |

Halving Muon's matrix_lr lowers dev bpb by 0.0086, 0.0084, 0.0079 (baseline, token_conv, attn_window), as much as the best single technique (XSA, dev −0.0087). So matrix_lr 0.02 is a stack member, ablated like a technique. Adam's lr is a smaller effect on the baseline, below the resolution rule, so the stack keeps 0.04 and `stack_adam_lr0.02` tests the other choice.

### The stack and leave-one-out

The stack is L5 + XSA + value residual + token_conv + matrix_lr 0.02, trained on machine b together with a rerun of the baseline. The rerun matches the ablation's baseline: Δ −0.0001 val bpb (p = 0.87). The stack's val bpb is 1.6165 ± 0.0012, a change of −0.0205 from the baseline (p = 0.0002, all 5 seeds).

| removed from the stack | val bpb | cost of removing it | seeds worse | dev cost | floor | the technique alone |
|---|---:|---:|---:|---:|---:|---:|
| value residual | 1.6218 ± 0.0057 | +0.0053 (p = 0.08) | 5/5 | +0.0057 | 0.0035 | −0.0066 |
| XSA, last 3 layers | 1.6201 ± 0.0019 | +0.0036 (p = 0.003) H | 5/5 | +0.0038 | 0.0016 | −0.0087 |
| token_conv 4 | 1.6160 ± 0.0006 | −0.0005 (p = 0.49) | 2/5 | −0.0001 | 0.0009 | −0.0065 |
| Muon matrix_lr 0.02 | 1.6210 ± 0.0018 | +0.0046 (p = 0.005) H | 5/5 | +0.0044 | 0.0015 | −0.0088 |

- **The gains do not add up.** The four single changes sum to −0.0305 per seed, the stack gives −0.0205, and the interaction is +0.0100 (95% CI +0.0033 to +0.0167, p = 0.01).
- **token_conv is redundant in the stack.** Alone it gives −0.0065; removing it from the stack changes nothing measurable. Its kernels are trained, not dead: their RMS per layer ranges from 0.764 to 2.379 in the stack and from 0.682 to 2.755 alone at the same matrix_lr, from a zero start (`results/token_conv_weights.json`). XSA and the learning rate keep resolved, Holm-significant costs. Removing value residual costs bpb on all 5 seeds, but one seed's large cost leaves it unresolved.
- **The tutorial's L6 stack is close to additive.** With every part measured alone (the tutorial's laptop ablation did not test XSA alone), the five parts of L6 sum to −0.0286, L6 gives −0.0357, and the interaction is −0.0070 (p = 0.11, not resolved).

| stack-family arm | val bpb | vs stack | dev bpb | round trip | MB | machine |
|---|---:|---:|---:|---:|---:|:---:|
| stack (int8) | 1.6165 ± 0.0012 | | 1.6216 ± 0.0007 | 1.5682 | 6.16 | b |
| stack, matrix_lr 0.01 | 1.6199 ± 0.0039 | +0.0034 (p = 0.11), not resolved | 1.6254 ± 0.0041 | 1.5720 | 6.16 | c |
| stack + Adam lr 0.02 | 1.6084 ± 0.0015 | −0.0080 (p = 0.0002) | 1.6128 ± 0.0013 | 1.5589 | 6.20 | c |
| stack + MLP 3× (int6) | 1.6085 ± 0.0017 | −0.0080 (p = 0.002) | 1.6134 ± 0.0016 | 1.5614 | 5.67 | b |
| stack + the rest of L6 (int6) | 1.6004 ± 0.0014 | −0.0161 (p < 0.0001) | 1.6053 ± 0.0017 | 1.5536 | 5.70 | b |

The two int6 arms trade bits per weight for a bigger model and both fit. Adding the rest of L6 (LeakyReLU², MLP 3×, depth recurrence, int6) gives the lowest bpb, but it differs from nanolm's own L6 stack only by token_conv and matrix_lr 0.02, and against L6 (ablation, machine a; paired by seed) these change val bpb by −0.0011 (p = 0.28, not resolved).

### When the gains appear

With the sweeps' `train.eval_every: 100`, nanolm evaluates each run on the first 262,144 validation tokens every 100 steps (`figures/dynamics.png`). Except for depth recurrence, which switches on at 35% of training, every gain in the figure is largest at step 500, the first point shown, and narrows at every later evaluation shown: the stack's change from the baseline is −0.145 at step 500 and −0.0196 at the end; value residual and token_conv end with 6.3% and 6.0% of their step-500 lead. At this budget most of what the techniques buy is a faster start.

### Bytes: the leaderboard model

`python -m nanolm.compress` on each seed's checkpoint (`runs/sweeps/stack/stack_plus_l6_s*/compress.json`): compressed weights in MB (10⁶ bytes, without the 23,730 bytes of code) and the round-trip cost in bpb against the unquantized model, both at no overlap; means over 5 seeds. "fits": weights + code at most 6.5 MB on every seed (lzma).

| bits | zlib MB | lzma MB | bits/param (lzma) | cost, zlib | cost, lzma | fits |
|---|---:|---:|---:|---:|---:|:---:|
| int8 | 7.690 | 7.706 | 7.36 | +0.0001 | +0.0001 ± 0.0000 | no |
| int6 | 5.849 | 5.663 | 5.41 | +0.0013 | +0.0013 ± 0.0001 | yes |
| int5 | 5.155 | 4.724 | 4.51 | +0.0047 | +0.0047 ± 0.0002 | yes |
| int4 | 4.087 | 3.816 | 3.64 | +0.0193 | +0.0193 ± 0.0016 | yes |

Bpb of the unquantized model by evaluation stride: no overlap 1.6004, stride 256 1.5594, stride 128 1.5541, stride 64 1.5523. zlib is slightly smaller than lzma at int8 and larger below it. Quantization costs little down to 6 bits (baseline +0.0001 at int8; stack +0.0001 at int8; stack + MLP 3× +0.0012 at int6) and much more at 5 and 4.

## The new options

`patches/nanolm-token-conv-attn-window.patch` changes `models/gpt.py`, `config.py` and `tests/test_nanolm.py`, and nothing else. `scripts/setup.sh` applies it to the pinned commit. Both options are added to the test file's `GOLF_FEATURES`, so nanolm's existing causality test (changing a future token must not change earlier outputs) covers them. Three new tests check that attn_window hides tokens outside the window, that token_conv starts as the identity, and that token_conv's output at a position depends on exactly that position and the K−1 before it, with its kernel trained by Adam rather than Muon. `results/tests.txt`: 34 passed, 3 skipped in 3.33s (skipped: one test needs the golf tokenizer, which was not downloaded on the test machine, and two need transformers or peft, which these runs do not use).

## Layout

| path | contents |
|---|---|
| `configs/sweeps/` | `ablation.yaml` (each technique alone), `lr.yaml`, `stack.yaml` (stack, leave-one-out, stack-family arms), `machine_check.yaml` (two baseline seeds on a second machine) |
| `runs/sweeps/<sweep>/<arm>_s<seed>/` | nanolm's record of every run (`config.yaml`, `meta.json`, `train_log.jsonl`, `eval_log.jsonl`, `metrics.json`, `stdout.log`), `dev.json`; for the stack sweep's baseline, stack, stack_plus_mlp3x and stack_plus_l6 runs also `compress.json` |
| `runs/sweeps/<sweep>/summary.csv`, `summary.md` | `python -m nanolm.analyze` on the ablation, lr and stack sweeps, with `curves.png` and `bars.png` |
| `results/` | `summary.json` (every result above except the token_conv kernel RMS, in `token_conv_weights.json`, and the test count, in `tests.txt`), `bench.json` (speed of each ablation and stack configuration alone), `machines.json`, `environment_b.txt`, `environment_c.txt`, `tests.txt`, `token_conv_weights.json` |
| `analysis/` | `summarize.py`, `figures.py` |
| `figures/` | `ablation.png`, `lr.png`, `stack.png`, `dynamics.png` |
| `scripts/` | `setup.sh`, `run_all.sh`, `dev_eval.py`, `bench_all.py`, `token_conv_weights.py`, `env.sh` |
| `patches/` | the nanolm patch |

## Reproducing

Requires Python 3.11 and an NVIDIA GPU (all runs: RTX 4090, 24 GB; the largest peak memory in `results/bench.json` is 2.87 GB, stack_plus_l6).

```bash
bash scripts/setup.sh          # nanolm at 8418fc1 into external/, apply the patch, install requirements
bash scripts/run_all.sh        # prepare tests ablation lr dev stack dev compress conv_weights bench analyze environment
python analysis/summarize.py   # results/summary.json (no GPU)
python analysis/figures.py     # figures/*.png (no GPU)
```

- Any subset of stages can be named (`bash scripts/run_all.sh stack dev`). Finished runs are skipped, so the script resumes. `SLOTS` (default 4) runs share the GPU.
- `prepare` downloads the first FineWeb shard and the validation split in nanolm's golf format (`python -m nanolm.prepare --golf-shards 1`).
- GPU training is not bitwise reproducible: the same seed on machine b differs from machine a by +0.00003 and −0.00155 val bpb (seeds 1337 and 42; `bash scripts/run_all.sh prepare machine_check dev` on a second machine). Reruns therefore match the means and spreads above, not every digit.
- Without a GPU, `summarize.py` and `figures.py` rebuild `results/summary.json` and `figures/*.png` from the committed `runs/`, `results/bench.json` and `results/machines.json`; the `analyze` stage (`python -m nanolm.analyze`) rebuilds the ablation, lr and stack sweeps' summaries and plots.
- Checkpoints are not committed.

## Limitations

- **Scale.** One model size and token budget (6.6M to 8.4M parameters, 41M tokens) and one sequence length (512). attn_window is meant for longer contexts than these.
- **Fixed settings.** New options at one setting each (token_conv 4, attn_window 128 in layers 0 to 2); learning rates on a factor-of-two grid.
- **Statistics.** 5 seeds per arm; paired t-tests with Holm correction only within the two families named above. Arms run on different machines are paired by seed across machines. The baseline rerun on machine b differs from machine a's by −0.0001 bpb (95% CI −0.0021 to +0.0019); machine c has no calibration run.
