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

`uv sync` installs the default PyPI wheel for torch, which on Windows carries no CUDA
support. Reinstall it from the PyTorch index afterwards, matching the CUDA version that
`nvidia-smi` reports:

```powershell
uv pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A version string ending in `+cpu` means the CPU-only wheel is still installed. Repeat
this after any later `uv sync`, which resolves torch from PyPI again. On Linux the
default PyPI wheel already bundles CUDA and no extra step is needed.

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

Confirm the GPU and the torch build before a full run:

```powershell
nvidia-smi
fedicl-mqa doctor --config configs\a5000.yaml --gpu 1
```

`doctor` prints a `torch_build` block with the torch version, the CUDA version it was
built against, the visible device count and `CUDA_VISIBLE_DEVICES`. A version ending in
`+cpu` means CUDA is unavailable no matter what `nvidia-smi` reports, because the driver
and the installed wheel are independent layers.

## Full run, start to finish

One command runs every step, from downloading the dataset to writing the contrast
report:

```powershell
fedicl-mqa pipeline --config configs\a5000.yaml --gpu 1
```

It is resumable. Each step checks whether its output already exists, so re-running after
an interruption continues where it stopped rather than repeating hours of GPU work.
Training steps always run but resume from their own checkpoints and return quickly when
there is nothing left to do. `--force` re-runs everything.

The ten steps, in the order their outputs become available:

| # | Step | Considered done when |
| --- | --- | --- |
| 1 | prepare-data | `data/<dataset>/partition_manifest.json` exists |
| 2 | audit-retrieval | `data/<dataset>/retrieval_audit.json` exists |
| 3 | train-local | resumes from its checkpoints |
| 4 | train-federated | resumes from its checkpoints |
| 5 | evaluate-f0-validation | every seed x round summary exists |
| 6 | select-round | `training/<dataset>/federated/selected_round.json` exists |
| 7 | train-centralized | resumes from its checkpoints |
| 8 | build-priors | every seed's prior exists |
| 9 | evaluate-all | every arm has a test summary |
| 10 | report | `reports/<dataset>/contrasts.json` exists |

Step 8 depends on step 5, not on step 9: the leave-one-client-out prior is built only
from F0 validation predictions, never from test data.

### Watching a run

`pipeline_state.yaml` is rewritten whenever a step changes status, including before a
step starts, so a long run is watchable while it happens:

```powershell
Get-Content outputs\a5000\pipeline_state.yaml -Wait
```

Each entry carries the status, timestamps, duration and, for a failed step, the error:

```yaml
- index: 3
  name: train-local
  status: failed
  duration_seconds: 412.7
  error: 'RuntimeError: CUDA out of memory'
- index: 4
  name: train-federated
  status: pending
```

Training also logs progress per batch and per epoch, so the console shows position and
loss rather than going silent for hours:

```
01:20:31 local-client-0: epoch 1/1 batch 10/89 step 10 loss 1.3820
01:22:40 local-client-0: epoch 1/1 complete in 146.2s, step 44
01:22:43 local-client-0: saved checkpoint-epoch-0001-step-00000044
```

Pass `--quiet` for warnings only.

### Results

`arms_comparison.md` is rewritten each time an arm finishes, so the table is usable
while a sweep is still running. Rows are ordered by accuracy:

```
| arm | runs | pipeline_accuracy | position_macro_f1 | ...
| F2  | 3    | 0.5644 ± 0.0057   | 0.5544 ± 0.0057   | ...
| F0  | 3    | 0.5119 ± 0.0108   | 0.5019 ± 0.0108   | ...
| B1  | 1    | 0.3392            | 0.3292            | ...
```

`arms_comparison.json` holds the same figures plus every individual seed.
`arms/<dataset>/<ARM>/run_log.json` traces each run of one arm with its status,
duration and accuracy, including runs that were skipped or that failed.

### Running the steps individually

`scripts/train_all.sh` performs the training half on Linux; the PowerShell below is the
Windows equivalent and is what `pipeline` issues internally.

```powershell
$cfg = "outputs\a5000\sealed_config.json"
$gpu = 1

# 1. Freeze the data and audit the retrieval cohort.
fedicl-mqa prepare-data --config configs\a5000.yaml --gpu $gpu
fedicl-mqa audit-retrieval --config $cfg --gpu $gpu

# 2. Local and Federated LoRA, scoring F0 on validation at each candidate round.
foreach ($seed in 42, 43, 44) {
  fedicl-mqa train --config $cfg --mode local --seed $seed --resume auto --gpu $gpu
  fedicl-mqa train --config $cfg --mode federated --seed $seed --resume auto --gpu $gpu
  foreach ($round in 4, 6, 8) {
    fedicl-mqa evaluate --config $cfg --arm F0 --seed $seed `
      --split validation --round $round --gpu $gpu
  }
}

# 3. Select one global FL round from validation accuracy alone.
fedicl-mqa select-round --config $cfg --gpu $gpu

# 4. Centralized LoRA, then the F2 leave-one-client-out prior per seed.
fedicl-mqa train --config $cfg --mode centralized --all-seeds --resume auto --gpu $gpu
foreach ($seed in 42, 43, 44) {
  fedicl-mqa build-priors --config $cfg --seed $seed --gpu $gpu
}

# 5. Evaluate every arm on the test split, then build the contrast report.
fedicl-mqa evaluate-all --config $cfg --gpu $gpu
fedicl-mqa report --config $cfg --gpu $gpu
```

Step 1 reads `configs\a5000.yaml`; every later step reads
`outputs\a5000\sealed_config.json`. `prepare-data` resolves mutable Hub refs to commit
SHAs and writes that sealed file, and the remaining commands must read it so the config
hash matches.

Every training command accepts `--resume auto`, and the evaluation sweeps skip work
already on disk, so the whole sequence is safe to re-run after an interruption.

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

The same smoke run on Windows:

```powershell
$smoke = "outputs\a5000-smoke\sealed_config.json"
fedicl-mqa prepare-data --config configs\a5000-smoke.yaml --gpu 1
fedicl-mqa audit-retrieval --config $smoke --gpu 1
fedicl-mqa train --config $smoke --mode local --seed 42 --resume auto --gpu 1
fedicl-mqa train --config $smoke --mode federated --seed 42 --resume auto --gpu 1
fedicl-mqa evaluate-arm --config $smoke --arm F1 --gpu 1
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

This script requires bash. On Windows Server, use steps 2 to 4 of "Full run, start to
finish" above, which issue the same commands in the same order.

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

### Every arm at once

`evaluate-all` sweeps all eight arms across every configured training seed on the test
split. This is the command to run once training has finished:

```powershell
fedicl-mqa evaluate-all --config outputs\a5000\sealed_config.json --gpu 1
```

It is not a shortcut past training. Only B0 and B1 evaluate without a checkpoint; the
command stops at the first arm whose checkpoint or prior is missing, keeping whatever it
already finished. See "Prerequisites" below.

`evaluate-arm` does the same for one arm:

```powershell
fedicl-mqa evaluate-arm --config outputs\a5000\sealed_config.json --arm B1 --gpu 1
```

Both default to `--split test`. Seeds come from `experiment.training_seeds`, and the
arm's checkpoint family decides how many runs happen:

| Arms | Checkpoint family | Runs per arm | Output subdirectory |
| --- | --- | --- | --- |
| B0, B1 | none | 1 | `deterministic/` |
| L0, L1 | local | one per seed | `seed-42/`, `seed-43/`, `seed-44/` |
| F0, F1, F2 | federated | one per seed | `seed-<n>/` |
| C0 | centralized | one per seed | `seed-<n>/` |

B0 and B1 reset the adapter to its initial state rather than loading a trained
checkpoint, so every seed would repeat identical work. They run once and are recorded
under `deterministic/`, not per seed.

Both commands treat an existing `summary.json` as a finished run and skip it, printing
the accuracy already on disk. `summary.json` is written after `predictions.jsonl`, so a
run interrupted part-way is repeated rather than skipped, and an interrupted sweep
resumes where it stopped. `--force` re-runs everything including completed evaluations:

```powershell
fedicl-mqa evaluate-all --config outputs\a5000\sealed_config.json --gpu 1 --force
```

Arms are swept in alphabetical order: B0, B1, C0, F0, F1, F2, L0, L1.

### Prerequisites

| Needed first | Arms that depend on it |
| --- | --- |
| `prepare-data` | all |
| `audit-retrieval` | B1, L1, F1, F2 |
| `train --mode local` | L0, L1 |
| `train --mode federated` | F0, F1, F2 |
| `evaluate --arm F0 --split validation` at rounds 4, 6, 8 | `select-round`, then F0, F1, F2 |
| `select-round` | F0, F1, F2, C0 |
| `train --mode centralized` | C0 |
| `build-priors --seed <n>` | F2 |

`build-priors` depends on the F0 validation evaluations: the leave-one-client-out
weakness prior is built only from validation predictions, never from test data.

### Single evaluations

`evaluate` remains the single-run primitive that both sweep commands call underneath.
B0 and B1 reject `--seed`; every other arm requires it:

```powershell
fedicl-mqa evaluate --config outputs\a5000\sealed_config.json --arm B0 --split test
fedicl-mqa evaluate --config outputs\a5000\sealed_config.json --arm L0 --seed 42 --split test
fedicl-mqa evaluate --config outputs\a5000\sealed_config.json --arm F2 --seed 42 --split test
```

Unlike the sweep commands, `evaluate` always re-runs and overwrites. The CLI reads the
F2 prior matching the globally selected round. The F2 candidate pool comes from the same
retriever as F1; F2 changes only the ranking score.

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

In PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

These cover configuration, schema, retrieval, leakage, checkpointing, reporting, token
resolution and the evaluation sweeps. They need no GPU and no CUDA build of torch.

Full GPU validation should start with the smoke config, inspect peak VRAM in each summary/telemetry
file, then freeze the full A5000 config before any reported test run.
