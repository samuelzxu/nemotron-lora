# Nemotron LoRA — NVIDIA Nemotron Model Reasoning Challenge

LoRA fine-tuning pipeline, Bedrock reasoning-trace distillation, and Kaggle submission
automation for the **NVIDIA Nemotron Model Reasoning Challenge** (base model:
`Nemotron-3-Nano-30B-A3B`, a Mamba/MoE hybrid). The judge runs greedy (temp 0, top_p 1,
`max_tokens` 7680, `max_model_len` 8192) and extracts the answer from `\boxed{...}`.

## Results so far (public LB)

| Run | Data | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| dgxchen parity (reference) | train split | — | 0.84 | — |
| grid `a64/lr1e-4` (best recipe) | train split | 0.82 | **0.85** | 0.84 |
| eq_guess distill (cap=2) | train split + 71 | 0.84 | **0.85** | 0.84 |
| eq_guess distill (cap=4) | train split + 134 | 0.83 | 0.84 | 0.84 |
| eq_guess ×3 oversample | train split + 402 | 0.81 | 0.85 | _pending_ |
| full dataset + eq_guess ×3 | **full (+ held-out)** | _running_ | — | — |

**Best recipe:** `alpha 64, lr 1e-4, rank 32, micro-batch 1, grad_accum 32, cosine, warmup_ratio 0.03,
2 epochs`. Hyperparameters top out at ~0.85; data axis (eq_guess, oversampling) holds 0.85 but
hasn't beaten it. See `progress.txt` for the running log.

## Prerequisites (not in this repo — gitignored, obtain separately)

The big binaries and environments are excluded from git. To run, you need:

- **`foundation/model/`** — base `Nemotron-3-Nano-30B-A3B` weights (`trust_remote_code`, custom
  `modeling_nemotron_h.py`). ~59 GB.
- **`foundation/.venv/`** — Python 3.12 venv built from offline wheels. Key pins:
  `torch 2.10.0+cu128, transformers 4.57.6, peft 0.18.1, trl 0.24.0, vllm 0.18.0,
  unsloth 2026.3.17, mamba_ssm 2.3.1, triton 3.6.0`. (`requirements.lock` is tracked.)
  GPU used: RTX PRO 6000 Blackwell 96 GB (sm_120).
- **`foundation/datasets/dgxchen_nemotron-cot-tong/problem_ids_matched.csv`** — the winner's CoT
  dataset (`id, prompt, answer, type, generated_cot`), 7,830 rows / 6,171 problems.
- **`bedrock/.venv/`** — separate venv with `boto3` for the AWS Bedrock capture driver.
- **`artifacts/<run>/`** — generated LoRA adapters (created by training).

### Credentials (never printed/committed)

- **Kaggle:** `~/.claudeinit` exports `KAGGLE_API_TOKEN` (required by Kaggle CLI ≥ 2.2.0).
  `source ~/.claudeinit` before any `kaggle` command. Do **not** `cat` it.
- **AWS Bedrock:** default profile in `~/.aws/credentials`, region `us-east-1`.

## Repo layout

```
train/        SFT training (unsloth), grid runner, eq_guess corpus builder
eval/         official-metric grader, held-out scorer, train/held-out id splits
bedrock/      Bedrock capture driver (Converse + Claude extended-thinking) + capture results
puzzle_team/  deterministic solvers + capture-quality audits (bit_manip, eq_guess, cryptarithm)
submit/       crash-proof Kaggle auto-submit watcher (code-submission path)
kaggle_submission/   no-accel submission kernel (globs adapter -> submission.zip)
progress.txt  cross-iteration project log
```

## 1. Training

The proven dgxchen unsloth recipe. `import unsloth` first; experts are LoRA-adapted by default
(the `up_proj`/`down_proj` suffixes match all 128 routed experts).

```bash
foundation/.venv/bin/python train/train_sft.py \
  --run my_run --epochs 3 --alpha 64 --lr 1e-4 \
  --grad-accum 32 --scheduler cosine --warmup-ratio 0.03
```

Key flags:
- `--full-data` — train on the **full** COT dataset (do **not** exclude the held-out split).
  Use for submission models (the held-out eval is saturated; rank by LB).
- `--extra-data <jsonl>` — append extra `{id,prompt,answer,type,generated_cot}` records
  (e.g. the eq_guess corpus).
- `--extra-data-mult N` — repeat each extra-data record N times (oversampling).
- `--freeze-experts` — flag stub for the experts-frozen regularizer A/B (exclude_modules wiring
  is **not** implemented yet).

Per-epoch adapters are saved to `artifacts/<run>/adapter_ep{1,2,3}/` with a `.done` sentinel
(this is what the submit watcher polls).

**Hyperparameter grid** (4 configs × 2-epoch, checkpoint-optimized):

```bash
bash train/grid_runner.sh        # runs sequentially; logs to logs/grid/
```

## 2. Evaluation

The grader mirrors the official metric exactly: `\boxed{}` extraction, **binary strings compared
string-exact** (no numeric tolerance), non-binary numeric within `rel_tol=1e-2`, else
case-insensitive string.

```bash
foundation/.venv/bin/python eval/grader.py                       # self-tests
foundation/.venv/bin/python eval/score_heldout.py \
  --adapter artifacts/my_run/adapter_ep2 --run my_run            # greedy vLLM held-out score
```

> Note: the held-out score is saturated/over-fit relative to the LB — use the LB to rank models.

## 3. Bedrock reasoning-trace capture

Resumable concurrent Converse-API driver (skips completed `(id, gen)` pairs). Captures correct
reasoning traces for distillation.

```bash
source ~/.claudeinit   # not needed for AWS, but harmless; AWS uses ~/.aws/credentials
bedrock/.venv/bin/python bedrock/driver.py \
  --model-id deepseek.v3.2 --label eq_deepseek \
  --out-dir bedrock/eqguess_capture/eq_deepseek \
  --sample bedrock/eqguess_sample.jsonl \
  --max-tokens 8192 --n-gens 24 --temperature 0.8 --concurrency 8
```

Model-id notes:
- DeepSeek-R1 and other on-demand-unsupported models need the inference profile, e.g.
  `us.deepseek.r1-v1:0`.
- **Claude extended thinking** (reasoning captured in `reasoningContent`): pass
  `--thinking '{"reasoning_config":{"type":"enabled","budget_tokens":3072}}' --temperature 1.0`
  and a `--system` prompt that keeps the visible answer short. The driver auto-drops `topP` and
  forces `temperature=1` when `reasoning_config` is present.

## 4. eq_guess distillation corpus

Builds `train/eqguess_distill.jsonl` from the Bedrock captures + the deterministic solver, with
exact-string + Nemotron-token-budget filtering and per-problem capping. Tune the constants at the
top of the builder:

```python
MAX_PER_PROBLEM = 4     # captured traces kept per problem
DEDUP = False           # False = raw capped repetition; True = distinct approaches only
INCLUDE_SOLVER = True   # also include deterministic solver templates
```

```bash
foundation/.venv/bin/python train/build_eqguess_corpus.py
```

Related analysis: `train/investigate_eqguess_diversity.py`,
`puzzle_team/results/eqguess_capture_quality.md`.

## 5. Submission (Kaggle code competition)

This is a **code competition** — plain file uploads are rejected. Submission goes through the
no-accel kernel (`samuelzxu/nemotron-submission`), which mounts the adapter as a Kaggle model and
zips it into `submission.zip`.

**Per-adapter flow:** create model instance → point `kaggle_submission/kernel-metadata.json`
`model_sources` at it → `kaggle kernels push` (capture version N) → wait for the kernel run
(~4–5 min; the `kernels status` endpoint is unreliable, so retry the submit) →
`kaggle competitions submit -k samuelzxu/nemotron-submission -v N -f submission.zip -m "..."`.

**Automated:** the watcher does all of the above for `artifacts/<RUNS>/adapter_ep{1,2,3}` as they
appear. Set `RUNS`/`EPOCHS` at the top of the file, then:

```bash
source ~/.claudeinit
nohup foundation/.venv/bin/python submit/submit_watcher.py >> logs/submit_watcher.log 2>&1 &
```

It's idempotent (skips anything already on the board), crash-proof per adapter, and uses the
`KAGGLE_API_TOKEN` code-submission path. State in `submit/state.json`.

## 6. Puzzle-cracking team (deterministic solvers)

Maps the deterministic "roof" per hard category (what's solvable in pure Python → what's worth
distilling). A solver is `solve(prompt) -> str | None`.

```bash
foundation/.venv/bin/python puzzle_team/harness.py equation_numeric_guess \
  puzzle_team/solvers/equation_guess_rule.py [--status rule_unknown]
```

Findings: `puzzle_team/results/` (leaderboard, audits). Cryptarithm is information-theoretically
intractable; bit_manipulation is saturated by the existing data; eq_guess has a small typographic
signal (`-`→subtract) worth ~6 pt over baseline.

## Key gotchas

- **Always `source ~/.claudeinit`** before `kaggle` (CLI ≥ 2.2.0 dropped `kaggle.json`, needs
  `KAGGLE_API_TOKEN`). Never `cat` the credential files.
- **micro-batch must be 1** — batch 2 cost −0.10 LB.
- **rank ≤ 32** (submission constraint, `max_lora_rank=32`).
- Create a Kaggle model instance and **wait for it to finish processing** before pushing the
  kernel, or the kernel mounts nothing and errors.
- Watch disk — adapters are ~3.5 GB each; clean `artifacts/`, `kaggle_upload/`, `/tmp/*.zip`.
