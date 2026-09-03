# FedICL-MQA

Implementation of the native-MCQ experiment contract in
[`docs/superpowers/specs/2026-09-01-fedicl-mqa-final-prompt.md`](docs/superpowers/specs/2026-09-01-fedicl-mqa-final-prompt.md).

The code implements:

- native four-way MedQA and MedMCQA only;
- arms B0, B1, L0, L1, F0, F1, F2 and C0;
- one closure-constrained local dense retriever with exactly five exemplars;
- Local LoRA, weighted full-participation FedAvg-LoRA and Centralized LoRA;
- generated-answer-to-option matching as the primary evaluator;
- conditional likelihood as the mandatory secondary evaluator;
- paired item/seed bootstrap and Holm correction for the six primary contrasts;
- atomic, hash-verified and resumable checkpoints.

Security/DP components in the paper plan are not implemented here and therefore must not be
reported as validated privacy guarantees.

## Default hardware profile

The default [`configs/a5000.yaml`](configs/a5000.yaml) targets MedQA on one NVIDIA RTX A5000 with 24 GB
VRAM and uses `Qwen/Qwen3-0.6B`. This is the official text-only Qwen3 checkpoint closest to the
requested 0.5B scale. Qwen3 thinking mode is disabled by the runtime so generation stays short and
deterministic.

Use [`configs/a5000-medmcqa.yaml`](configs/a5000-medmcqa.yaml) for the separate MedMCQA experiment.
The two datasets intentionally use different output roots so each one has an independent sealed
configuration and results are never micro-pooled.

Performance defaults:

- BF16 weights and activations;
- PyTorch SDPA attention;
- fused AdamW when CUDA supports it;
- 2,048-token total context budget with fail-closed overflow checks;
- micro-batch 8 × gradient accumulation 2 = effective batch 16;
- dynamic padding to a multiple of eight;
- pinned-memory, non-blocking transfers and four data workers;
- generation batch 16 and likelihood candidate batch 8;
- TF32 enabled for remaining FP32 matrix operations.

Gradient checkpointing is disabled because the 0.6B model fits comfortably and recomputation would
reduce throughput. If a longer prompt causes OOM, first change micro-batch `8 → 4` and accumulation
`2 → 4` to preserve effective batch 16. Only then enable gradient checkpointing. Freeze the chosen
profile before comparing arms; do not tune hardware settings separately per arm.

Likelihood scoring keeps full-vocabulary logits in BF16 and upcasts only completion-token rows to
FP32 for the log-softmax. This avoids the largest avoidable evaluation-memory spike on a 24 GB GPU.

## Installation

PyTorch wheels may not yet support the system Python version. Use Python 3.12 explicitly:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

On Windows Server, activate the environment with PowerShell instead:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv sync --extra dev
```

The shell scripts under `scripts/` are bash-only. Every step they perform is also
available as a CLI subcommand, which runs unchanged on Windows.

## Hugging Face authentication

Put the access token in a single-line file named `HF_Access_Token.txt` in the repository
root. The file is git-ignored and read automatically by every command.

```powershell
Set-Content -Path HF_Access_Token.txt -Value 'hf_xxx' -NoNewline -Encoding utf8
```

The file takes precedence over `HF_TOKEN` in the environment, so a stale machine-scope
variable cannot silently shadow it. A token is optional: the default datasets and models
are public, and commands fall back to anonymous access. `fedicl-mqa doctor` reports which
source was used without printing the token.

## Selecting a GPU

Pass `--gpu 0` or `--gpu 1` to any subcommand. The flag restricts the process to that
device via `CUDA_VISIBLE_DEVICES`, leaving `hardware.device` in the config untouched so
the sealed configuration hash still matches. Editing `hardware.device` in the YAML to
pick a GPU would change the hash and invalidate prepared data and existing checkpoints.

## Evaluating arms

`evaluate-arm` runs one arm across every configured training seed on the test split:

```bash
fedicl-mqa evaluate-arm --config outputs/a5000/sealed_config.json --arm B1 --gpu 1
```

Seeds come from `experiment.training_seeds`. B0 and B1 do not consume a trained
checkpoint, so they run once and are recorded under `deterministic/` rather than per
seed. `evaluate-all` sweeps all eight arms the same way:

```bash
fedicl-mqa evaluate-all --config outputs/a5000/sealed_config.json --gpu 1
```

Both commands skip any evaluation whose `summary.json` already exists, so an interrupted
sweep resumes where it stopped. Pass `--force` to re-run everything.

Confirm the GPU before a full run:

```bash
nvidia-smi
fedicl-mqa doctor --config configs/a5000.yaml
```

## Reproducible data preparation

The CLI resolves mutable Hugging Face refs such as `main` to immutable commit SHAs and writes the
complete frozen configuration to `OUTPUT_DIR/sealed_config.json`. Subsequent commands fail closed if
the resolved configuration differs.

Choose one dataset in the config and prepare it:

```bash
fedicl-mqa prepare-data --config configs/a5000.yaml
fedicl-mqa audit-retrieval --config outputs/a5000/sealed_config.json
```

Run the small development configuration first:

```bash
fedicl-mqa prepare-data --config configs/a5000-smoke.yaml
fedicl-mqa audit-retrieval --config outputs/a5000-smoke/sealed_config.json
bash scripts/train_all.sh outputs/a5000-smoke/sealed_config.json 42
fedicl-mqa evaluate \
  --config outputs/a5000-smoke/sealed_config.json \
  --arm F1 \
  --seed 42 \
  --split test
```

The smoke configuration uses one seed and subsampled splits. It is for code validation only and is
not valid paper evidence.

Prepared data are stored per client as `fit.jsonl`, `support.jsonl`, `validation.jsonl` and
`test.jsonl`. The partition manifest records the immutable dataset SHA, data seed, subject-skew
weights, assignments and file hashes. Preparation fails if support leakage is detected.
`audit-retrieval` batch-encodes the frozen validation/test cohort, verifies that every query has
exactly five eligible train-only exemplars, and writes candidate IDs plus exclusion/expansion
diagnostics to `data/<dataset>/retrieval_audit.json`. It never runs the answer model or reads gold
labels for retrieval.

## Training and checkpoint selection

Train all paired seeds and select one global FL round:

```bash
bash scripts/train_all.sh outputs/a5000/sealed_config.json 42 43 44
```

The script performs:

1. five Local LoRA runs;
2. FedAvg through round 8 for every seed;
3. F0 validation at rounds 4, 6 and 8 for every seed;
4. selection of one global round from mean validation accuracy, with an earlier-round tie-break;
5. Centralized LoRA for `selected_round × local_epochs` epochs for every seed.

Every training command accepts `--resume auto`. Local/Centralized checkpoints include adapter,
optimizer, trainer counters and Python/NumPy/Torch/CUDA RNG state. FL stores a global checkpoint at
every round plus hash-verified client updates. If FL stops mid-round, completed client updates are
reused and only unfinished clients restart.

Checkpoint layout:

```text
outputs/a5000/checkpoints/<dataset>/
├── local/seed-42/client-0/checkpoint-.../
├── federated/seed-42/
│   ├── global/checkpoint-round-0000/
│   ├── global/checkpoint-round-0001/
│   ├── round-work-0001/client-0.safetensors
│   └── selected_round.json
└── centralized/seed-42/checkpoint-.../
```

Each checkpoint has SHA-256 hashes and the exact config hash, base model ID and immutable revision.
A checkpoint is rejected if any artifact or configuration differs.

## Evaluation arms

B0 and B1 are deterministic and run once:

```bash
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm B0 --split test
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm B1 --split test
```

Evaluate trained arms for every seed:

```bash
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm L0 --seed 42 --split test
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm L1 --seed 42 --split test
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm F0 --seed 42 --split test
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm F1 --seed 42 --split test
fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm C0 --seed 42 --split test
```

Build the leave-one-client-out weakness prior only from F0 validation predictions, then evaluate F2:

```bash
fedicl-mqa build-priors --config outputs/a5000/sealed_config.json --seed 42
fedicl-mqa evaluate \
  --config outputs/a5000/sealed_config.json \
  --arm F2 \
  --seed 42 \
  --split test
```

The CLI reads the prior matching the globally selected round. The F2 candidate pool comes from the
same retriever as F1; F2 changes only the ranking score.

Finally, after every test arm exists for all three seeds:

```bash
fedicl-mqa report --config outputs/a5000/sealed_config.json
```

The report contains the six primary pipeline-accuracy contrasts, 95% confidence intervals, Holm
adjusted p-values, conditional-likelihood effects and an `evaluator_dependent` flag when evaluators
reverse the effect direction.

## Tests

Logic tests do not download models or datasets:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Full GPU validation should start with the smoke config, inspect peak VRAM in each summary/telemetry
file, then freeze the full A5000 config before any reported test run.
