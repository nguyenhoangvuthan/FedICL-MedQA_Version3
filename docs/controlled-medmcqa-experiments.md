# Controlled MedMCQA experiments

This protocol addresses three limitations of the previous MedQA run: the single
`medicine` subject made every prior zero; Local used one epoch while FL used six;
and F2 changed both diversity and prior relative to F1.

The implementation is opt-in through `controls` in
`configs/a5000-medmcqa-controls.yaml`. Historical configurations omit this section;
their serialized configuration and hashes remain unchanged. Their eight-arm sweeps
and legacy min-max prior remain available. The new run uses a separate output root;
MedQA adapters must not be relabeled as MedMCQA adapters.

## Data and subject evidence

Use the native `subject_name` field from
[openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa/blob/main/README.md).
The dataset card's machine-readable schema encodes `cop` as zero-based A/B/C/D;
its prose example uses a different convention. The existing adapter follows the schema.
No subjects are fabricated by clustering, keyword rules, answers or test performance.

The source also contains the placeholder `Unknown` (3,045 train and 2 validation
rows in the public Dataset Viewer checked on 2026-09-10). Controlled preparation
excludes missing/placeholder native subjects **before** holdout, sample limits,
and client assignment. The fixed metadata-only exclusion policy also covers blank,
`general`, `n/a`, and `none` values, ignoring case and surrounding whitespace.
It never relabels these questions or uses answers/predictions to choose exclusions.
`source_subject_filter` in the subject audit records input/retained/excluded counts,
excluded labels, and all excluded IDs for each original source split. This audit
is also embedded in the hashed partition manifest. Legacy runs without `controls`
retain their previous behavior. Report results as applying to the cohort with
usable native subjects, not the entire original validation set.

The controlled loader explicitly requests only the source `train` and `validation`
splits. It does not consume official test, whose public labels may be masked.
It creates the following fixed roles, with no question ID overlap:

| Experiment role | Original source |
| --- | --- |
| Fit + retrieval support | Remaining 90% of official train |
| Validation: FL round selection and LOCO prior | 10% of official train, held out within subject |
| Final test | Official validation |

The split is determined by sorted IDs, native subjects and the data seed, before
client partitioning. It does not inspect answer labels or model outputs. Optional
sample limits apply **after** this split and a deterministic shuffle. A test limit
therefore refers to the official validation source. This is a development-set
benchmark, not an evaluation on the official test set; report that explicitly.

Different IDs can contain identical or near-identical questions. Controlled
preparation therefore applies `global_cross_split_exclusion_v1` after holdout and
sample limits, before client assignment and fit/support allocation:

1. Keep the selected final test questions fixed.
2. Remove internal validation questions overlapping final test.
3. Remove training questions overlapping final test or the retained validation.

Overlap uses the existing audit criteria: ID, normalized question,
question/options hash, provenance group, or question-token Jaccard at the configured
threshold (0.85 by default). All future clients participate in this global check;
both fit and support are protected. Correct answers, model predictions and test
accuracy are not used. A lossless rare-token candidate index is followed by exact
Jaccard scoring; the leakage threshold is not relaxed. Duplicates **within** a
split remain, so this is cross-split decontamination, not full deduplication.

The `decontamination` entry in `subject_audit.json` and the hashed partition
manifest records the policy, threshold, before/after counts, and every removed ID
with a matching protected ID and reason. One witness is recorded per excluded
question, so these counts are not counts of all duplicate pairs. Sample limits
are upper bounds: decontamination can reduce the retained train/validation sizes.
Subject coverage and global disjointness are checked after filtering; the original
per-client support leakage audit still runs after materialization.

CPU verification on source revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`
with the shipped controls config completed the entire `prepare-data` command from
the source Parquet files, including artifact hashes, subject coverage and all five
per-client leakage audits. The source transport was local Parquet in place of the
Hub loader; no GPU training or retrieval encoder was run.

| Role | Before decontamination | Removed | Retained |
| --- | ---: | ---: | ---: |
| Train (fit + support) | 161,800 | 9,490 | 152,310 |
| Internal validation | 17,977 | 5 | 17,972 |
| Final test | 4,181 | 0 | 4,181 |

The retained cohort has 20 subjects. Minimum other-client validation coverage is
31 questions per required subject (threshold 10). These are data preparation
checks, not model accuracy results.

Preparation freezes Hub revisions, records the source-role mapping in the hashed
partition manifest, and writes `data/medmcqa/subject_audit.json`. It rejects any
remaining missing subject metadata, fewer than two support subjects in a client, identical client
fit-subject distributions, or fewer than 10 other-client validation questions for
any subject used in a client's support. The audit records per-client/role counts
and maximum pairwise total-variation distance between fit-subject distributions.
This verifies observed subject heterogeneity; it does not establish real hospital heterogeneity.

A small smoke sample can fail these evidence requirements. Choose sample sizes and
minimum counts in a fresh config before running; do not loosen requirements after
examining test accuracy. A failed guard does not justify inventing subject labels
or adding random noise to prior weights.

## Prior and placebo

For target client c and each subject s occurring in its support pool:

`prior[c,s] = (errors on validation of clients other than c + 1) / (count + 2)`

This is a Beta(1,1)-smoothed error rate. It preserves the magnitude of observed
weakness differences without stretching small differences to [0,1] with min-max.
It is an other-client population weakness estimate, not a direct measurement of
the target client's own weakness. It excludes the target client's validation
predictions and all test predictions.

The builder checks exact validation IDs, gold labels, client, subject, seed,
configuration, round and prediction-file hash. Each client must have nonconstant
weights on its support subjects, and client vectors cannot all be identical.
`round-N.audit.json` records counts/errors, estimator, selected round, and SHA-256
of the prior and its validation sources. Evaluation verifies this provenance and
rejects `--subject-weights` overrides in controlled runs.

FS permutes weights across subjects **within each client**, deterministically from
the training seed and client ID. It preserves each client's weight multiset and
requires a changed assignment. The same permutation is used for all that client's
queries. This tests subject-to-weakness alignment; it is not a client-ID permutation.

## Arms and causal comparisons

All training uses k=0. All ICL arms use five eligible examples from local support.
The four reranking arms use the same candidate-pool construction and FL checkpoint.

| Arm | Checkpoint | Diversity beta | Subject prior gamma | Interpretation |
| --- | --- | --- | --- | --- |
| F1 | Selected FL | 0 | 0 | Relevance top-5 |
| FD | Same FL | 0.15 | 0 | Diversity only |
| FP | Same FL | 0 | 0.20 | Prior only |
| F2 | Same FL | 0.15 | 0.20 | Diversity + true prior |
| FS | Same FL | 0.15 | 0.20, shuffled | Diversity + placebo prior |
| LM0 | Matched Local | — | — | No inference-time ICL |
| LM1 | Same matched Local | 0 | 0 | Retrieval ICL |

B0, B1, L0, L1, F0 and C0 remain available. L0/L1 retain one-epoch Local behavior.
FD−F1 isolates diversity with prior disabled; FP−F1 isolates prior with diversity
disabled; F2−FD isolates prior with diversity fixed; F2−FP isolates diversity with
prior fixed. F2−FS checks alignment against a shuffled-subject placebo. F0−LM0
compares federation at matched local data exposures; F2−LM1 compares full systems.
These contrasts have different questions; F2−F1 alone cannot attribute a gain to prior.

The report applies Holm correction across its ten declared primary contrasts.
It preserves C0−F0 as descriptive and rejects mismatched training-seed coverage or
paired examples with inconsistent labels/client/subject. It retains the project's
existing item/seed bootstrap estimator. Three seeds on one data partition are still
limited evidence; the factorial layout itself does not establish significance.

Every reranked summary includes `reranking_activity`: number of queries with varying
candidate prior, changes from relevance top-5, and changes caused by turning on
prior with diversity held fixed. A nonconstant prior may still change no selected
examples at the fixed gamma. Report that null result; do not retune gamma on test.

## Matched Local training and resume

The global FL round R is selected once from mean F0 validation accuracy over all
training seeds (candidates 4, 6, 8). Matched Local then trains **R epochs/client**.
Changing `training.local_epochs` would also change each FL round, so it remains 1.

Matched Local starts from the base adapter and keeps its own optimizer/RNG state.
Its paths are `training/medmcqa/local-matched/round-R/seed-S/client-C/`.
LM0/LM1 resolve that exact selected-round path and verify completed epochs and total
example exposures. They cannot accidentally load L0/L1's one-epoch checkpoint or
an interrupted matched run. FL and Centralized budget checks also run before
controlled evaluation. Standard and matched Local remain independently resumable.

Matched means equal per-client example exposures and nominal optimizer-update
budgets to the selected FL round; it does not mean equal wall time, memory,
communication, or identical optimizer state transitions. Centralized can have a
slightly different update count because final minibatches are pooled differently.
The full FL sweep still pays for eight rounds and validation checkpoint selection.

## Run on the training machine

PowerShell, using the existing environment and CUDA setup:

```powershell
uv run --no-sync fedicl-mqa pipeline --config configs/a5000-medmcqa-controls.yaml --gpu 1
```

The new pipeline runs data preparation, retrieval audit, Local, FL, F0 validation,
round selection, priors, Centralized, matched Local, the 13-arm evaluation, and report.
Base arms run once; the other eleven arms run for all three seeds: 35 test evaluations.
Subject checks happen before training. Prior variation is checked after F0 validation;
no GPU training has been performed as part of this code change.

After a failure in `prepare-data`, rerun the same pipeline command with the fixed
code. The controlled completion marker (`subject_audit.json`) is written only
after the audits pass, so a failed attempt is retried and partial partition files
are rebuilt. Do not delete the sealed config, CUDA environment, or old MedQA
outputs. If a controlled partition already completed under an older preparation
policy, use a fresh `output_dir` for the new policy; explicit preparation refuses
to replace that completed partition and silently reuse its trained checkpoints.

To run only matched Local after round selection, then evaluate its paired arms:

```powershell
$cfg = "outputs/a5000-medmcqa-controls/sealed_config.json"
uv run --no-sync fedicl-mqa train --config $cfg --mode local-matched --all-seeds --resume auto --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LM0 --gpu 1
uv run --no-sync fedicl-mqa evaluate-arm --config $cfg --arm LM1 --gpu 1
```

To resume evaluation and write the controlled contrasts:

```powershell
uv run --no-sync fedicl-mqa evaluate-all --config $cfg --gpu 1
uv run --no-sync fedicl-mqa report --config $cfg
```

Use `pipeline` for the controlled sequence; the historical `train_all.sh` helper
retains its original eight-arm training workflow. An explicit `--fl-round` is
supported for standalone matched training, but LM evaluation always uses the
validation-selected round. Controlled summaries bind to that round and include
prediction hashes; stale results fail rather than being silently reused.
