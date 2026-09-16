# Train-time exemplars and inference-time ICL

Use `configs/a5000-medmcqa-train-icl.yaml` for the train/eval context factorial.
It adds seven arms to the 13-arm controlled MedMCQA study in a new output directory:
the 2 x 2 (train exemplars x eval exemplars) for Local, Federated and Centralized.
Existing configurations, hashes, arms and checkpoints remain compatible.

| Arm | Training | Exemplars per train prompt | Exemplars per eval prompt | Checkpoint |
| --- | --- | ---: | ---: | --- |
| L0 | Local LoRA, one epoch | 0 | 0 | `local` |
| L1 | Same training as L0 | 0 | 5 | Same as L0 |
| LT0 | Local LoRA, one epoch | 5 | 0 | `local-icl` |
| LT1 | Same training as LT0 | 5 | 5 | Same as LT0 |
| F0 | Federated LoRA, selected round R | 0 | 0 | `federated` |
| F1 | Same training as F0 | 0 | 5 | Same as F0 |
| FT0 | Federated LoRA, R rounds | 5 | 0 | `federated-icl` |
| FT1 | Same training as FT0 | 5 | 5 | Same as FT0 |
| C0 | Centralized LoRA, R epochs | 0 | 0 | `centralized` |
| C1 | Same training as C0 | 0 | 5 | Same as C0 |
| CT0 | Centralized LoRA, R epochs | 5 | 0 | `centralized-icl` |
| CT1 | Same training as CT0 | 5 | 5 | Same as CT0 |

Here “train with ICL” means fine-tuning on a prompt containing five labeled
demonstrations followed by a target question. All prompt tokens, including the
demonstration answers, are masked with `-100`. Only the target answer completion
and EOS contribute directly to the training loss. The prompt remains part of the
forward/backward computation. Evaluation does not update weights.

## Frozen exemplars and comparable training data

Before any training family runs, `audit-training-icl` creates
`data/medmcqa/training_icl_audit.json`. Each target retrieves five examples from
**its own client's support pool**, with the same dense retriever and
ID/question/hash/lexical/semantic/provenance exclusions used for eval. No validation
or test example supplies a demonstration. Retrieval does not inspect the target
answer. Demo IDs and order are frozen across epochs, FL rounds and training seeds.

The audit checks both k=0 and k=5 training lengths with the actual tokenizer, all
four possible target completions and padding to a multiple of eight. Targets with
insufficient eligible exemplars or excessive context length are excluded from a
**common fit cohort used by every training family**, including the k=0 baselines,
matched Local and Centralized. It records every excluded target and reason. It
does not truncate prompts, reduce k or filter by model accuracy. Validation/test
and support pools are unchanged by this training audit.

Training uses the same target IDs, order, seeds, nominal epochs and batch settings
within each pair. Five demonstrations cost more input tokens, compute and memory;
equal target exposures and optimizer updates do not imply equal FLOPs or wall time.
Support answers also appear in the training context of the new arms. Report this
additional conditioning explicitly; it is the intervention being tested.

The plan binds to the configuration and partition file manifest. Its content hash
is recorded in training-family protocol files and evaluation summaries. Checkpoint
resume additionally verifies the ordered demonstration-ID mapping. Different
families have separate optimizer/RNG/checkpoint paths. A stale plan or mismatched
training context fails rather than silently mixing checkpoints.

## FL round policy and contrasts

R is selected from F0 validation using the existing 4/6/8 candidate procedure.
The new federated family then trains for exactly R rounds; FT0/FT1 use the same
round-R checkpoint. The Centralized families train R epochs on the pooled fit
cohort; pooling the data does not pool the demonstrations, so each target keeps
the five exemplars frozen from its own client's support. This answers the effect of train context **at the baseline's
selected budget**. It does not independently optimize the round count of FT0/FT1.
L0/L1 and LT0/LT1 remain one-epoch Local pairs; comparing LT directly with FT would
still mix federation with unequal local epochs. LM0/LM1 retain the existing
matched-Local control for the k=0 training family.

The report adds twelve primary paired contrasts and applies Holm correction
jointly over all 22 primary contrasts:

| Difference | Question |
| --- | --- |
| LT0 − L0 | Does train context help Local when eval has no ICL? |
| LT1 − L1 | Does train context help Local when eval has ICL? |
| LT1 − LT0 | Does eval ICL help after Local training with exemplars? |
| FT0 − F0 | Does train context help FL when eval has no ICL? |
| FT1 − F1 | Does train context help FL when eval has ICL? |
| FT1 − FT0 | Does eval ICL help after FL training with exemplars? |
| C1 − C0 | Does eval ICL help Centralized trained without exemplars? |
| CT0 − C0 | Does train context help Centralized when eval has no ICL? |
| CT1 − C1 | Does train context help Centralized when eval has ICL? |
| CT1 − CT0 | Does eval ICL help after Centralized training with exemplars? |
| FT1 − F0 | Does the full train+eval exemplar system beat plain FL? |
| CT1 − C0 | Does the full train+eval exemplar system beat plain Centralized? |

Federated versus Centralized in each cell (C0 − F0, C1 − F1, CT0 − FT0,
CT1 − FT1) is reported as descriptive, as C0 − F0 always was: the pair differs
in data pooling, not in the exemplar intervention under test.
Positive/negative point estimates alone do not establish improvement or harm;
read the paired confidence intervals and adjusted p-values. These contrasts do
not constitute a separate statistical test of the train-by-eval interaction.

## Run

Start the new study in its separate output root; do not append `icl_training` to
an already sealed run or copy old checkpoints into the new families.

```powershell
uv run --no-sync fedicl-mqa pipeline --config configs/a5000-medmcqa-train-icl.yaml --gpu 1
```

The pipeline has 15 steps: the existing 11 plus training-context audit and
Local-ICL, Federated-ICL and Centralized-ICL training. It evaluates 20 arms
(56 final-test runs with three training seeds; B0/B1 each run once).

After the pipeline has prepared data, frozen the training plan and selected R,
individual new modes and eval arms are available:

```powershell
$cfg = "outputs/a5000-medmcqa-train-icl/sealed_config.json"
uv run --no-sync fedicl-mqa train --config $cfg --mode local-icl --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa train --config $cfg --mode federated-icl --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa train --config $cfg --mode centralized-icl --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LT0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LT1 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm FT0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm FT1 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm C1 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm CT0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm CT1 --gpu 1
```

Validation includes CPU tests with a tiny locally initialized Qwen3/LoRA model:
training with demonstrations, exact interrupted/resumed Local adapter equality,
and two-client FedAvg training/resume. This is functional validation, not an A5000
memory measurement or a full MedMCQA accuracy experiment.

## One-seed FL-only pilot

`configs/a5000-medmcqa-train-icl-pilot.yaml` is a one-seed, subsampled run
(train 25k, internal validation 6k, full 4,181-question final test) meant to
give a directional read on **FL with vs. without ICL** before the three-seed
study. A smaller validation sample (3k) fails the other-client coverage guard
(`Orthopaedics: 8 < 10`); 6k passes with a minimum of 15. Do not report pilot
numbers as the final experiment: one seed collapses the hierarchical bootstrap
to an item bootstrap, so the interval ignores training-seed variance.

Only the two federated families are trained. Nothing in the F-arms depends on
Local, matched Local or Centralized checkpoints; they do depend on the frozen
training plan and the validation-selected round. One resumable command runs the
nine steps in that order:

```powershell
uv run --no-sync fedicl-mqa pipeline --config configs/a5000-medmcqa-train-icl-pilot.yaml --gpu 1 --arms F0 F1 FT0 FT1
```

The same steps individually, should one need to be re-run by hand:

```powershell
$cfg = "configs/a5000-medmcqa-train-icl-pilot.yaml"
uv run --no-sync fedicl-mqa prepare-data       --config $cfg
uv run --no-sync fedicl-mqa audit-retrieval    --config $cfg --gpu 1
uv run --no-sync fedicl-mqa audit-training-icl --config $cfg --gpu 1

$cfg = "outputs/a5000-medmcqa-train-icl-pilot/sealed_config.json"
# k=0 training, 8 rounds; then pick R from validation at rounds 4/6/8.
uv run --no-sync fedicl-mqa train --config $cfg --mode federated --all-seeds --resume auto --gpu 1
foreach ($r in 4, 6, 8) {
  uv run --no-sync fedicl-mqa evaluate --config $cfg --arm F0 --seed 42 --split validation --round $r --gpu 1
}
uv run --no-sync fedicl-mqa select-round --config $cfg

# k=5 training for exactly R rounds (fine-tuning with five frozen exemplars per prompt).
uv run --no-sync fedicl-mqa train --config $cfg --mode federated-icl --all-seeds --resume auto --gpu 1

# Final test: non-ICL and ICL evaluation of both checkpoints.
foreach ($arm in "F0", "F1", "FT0", "FT1") {
  uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm $arm --gpu 1
}
uv run --no-sync fedicl-mqa report --config $cfg --arms F0 F1 FT0 FT1
```

`report --arms` writes `reports/medmcqa/contrasts-F0-F1-FT0-FT1.json`, marked
`"partial": true`, with Holm correction over the five surviving contrasts
(F0−F1, F0−FT0, F1−FT1, FT0−FT1, F0−FT1).

To add the matching Centralized 2 x 2 to the same pilot output directory, run
the pipeline again with all eight arms. Finished federated steps are skipped or
resume instantly; only `centralized` and `centralized-icl` train, the four new
arms are evaluated, and the report covers every FL and Centralized contrast plus
the four descriptive FL-versus-Centralized cells:

```powershell
uv run --no-sync fedicl-mqa pipeline --config configs/a5000-medmcqa-train-icl-pilot.yaml --gpu 1 --arms F0 F1 FT0 FT1 C0 C1 CT0 CT1
``` It never touches `contrasts.json`, so the
full pipeline can still be run later in the same output directory.
`arms_comparison.md` is updated after every `evaluate-arm` and gives point
estimates while the sweep is still running.

Reading the pilot: F1−F0 and FT1−FT0 answer whether inference-time exemplars
help at all; F0−FT0 and F1−FT1 answer whether training with exemplars helps.
If all four intervals cover zero, the small model is not using demonstrations
and the method rather than the seed count should change.
