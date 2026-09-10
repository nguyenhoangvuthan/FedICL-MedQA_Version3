# Train-time exemplars and inference-time ICL

Use `configs/a5000-medmcqa-train-icl.yaml` for the train/eval context factorial.
It adds four arms to the 13-arm controlled MedMCQA study in a new output directory.
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
round-R checkpoint. This answers the effect of train context **at the baseline's
selected budget**. It does not independently optimize the round count of FT0/FT1.
L0/L1 and LT0/LT1 remain one-epoch Local pairs; comparing LT directly with FT would
still mix federation with unequal local epochs. LM0/LM1 retain the existing
matched-Local control for the k=0 training family.

The report adds six primary paired contrasts and applies Holm correction jointly
over all 16 primary contrasts:

| Difference | Question |
| --- | --- |
| LT0 − L0 | Does train context help Local when eval has no ICL? |
| LT1 − L1 | Does train context help Local when eval has ICL? |
| LT1 − LT0 | Does eval ICL help after Local training with exemplars? |
| FT0 − F0 | Does train context help FL when eval has no ICL? |
| FT1 − F1 | Does train context help FL when eval has ICL? |
| FT1 − FT0 | Does eval ICL help after FL training with exemplars? |

Positive/negative point estimates alone do not establish improvement or harm;
read the paired confidence intervals and adjusted p-values. These contrasts do
not constitute a separate statistical test of the train-by-eval interaction.

## Run

Start the new study in its separate output root; do not append `icl_training` to
an already sealed run or copy old checkpoints into the new families.

```powershell
uv run --no-sync fedicl-mqa pipeline --config configs/a5000-medmcqa-train-icl.yaml --gpu 1
```

The pipeline has 14 steps: the existing 11 plus training-context audit, Local-ICL
training and Federated-ICL training. It evaluates 17 arms (47 final-test runs with
three training seeds; B0/B1 each run once).

After the pipeline has prepared data, frozen the training plan and selected R,
individual new modes and eval arms are available:

```powershell
$cfg = "outputs/a5000-medmcqa-train-icl/sealed_config.json"
uv run --no-sync fedicl-mqa train --config $cfg --mode local-icl --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa train --config $cfg --mode federated-icl --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LT0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LT1 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm FT0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm FT1 --gpu 1
```

Validation includes CPU tests with a tiny locally initialized Qwen3/LoRA model:
training with demonstrations, exact interrupted/resumed Local adapter equality,
and two-client FedAvg training/resume. This is functional validation, not an A5000
memory measurement or a full MedMCQA accuracy experiment.
