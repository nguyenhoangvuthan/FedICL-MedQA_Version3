# FedICL-MQA — Core paper sections (reframed for publication fit)

Revision note (rev 2): this revision acts on three review findings. (a) The retrieval prior $w$ was underspecified — nothing forced its signal to originate at *other* clients, which made RQ2 unfalsifiable; it is now defined leave-one-client-out with three named controls (§4.2). (b) The specialty partition of rev 1 made subjects nearly disjoint across clients, which leaves a leave-one-client-out prior with no support to estimate from; the partition is now skew-based rather than disjoint (§5.2). (c) RQ4 promised a privacy–utility trade-off without deploying any privacy mechanism; it is downgraded to exposure characterization under an explicit threat model (§2, §7). The baseline list is reorganized into two ordered ladders that isolate the retrieval prior separately from the training protocol (§6.1).

## 1. Problem statement

Medical question answering under real hospital settings is inherently non-IID: different institutions have different specialties, patient populations, and label distributions. In this setting, small language models (SLMs) are attractive because they can be deployed under resource constraints, but they often underperform larger models and are sensitive to distribution shift.

Federated learning (FL) offers a way to leverage cross-institution supervision without centralizing private records, while in-context learning (ICL) can improve reasoning by conditioning on task-relevant demonstrations. However, neither mechanism is automatically sufficient in medical QA: retrieval quality can be poor, demonstrations may be noisy or irrelevant, and privacy leakage may still occur through prompts, adapters, or retrieved examples.

This motivates the central question of our study:

> Can federated LoRA adaptation combined with client-aware demonstration retrieval improve SLM performance on non-IID medical QA while controlling communication cost and prompt-level exposure?

## 2. Research questions

RQ1. Does federated LoRA adaptation with local demonstration retrieval improve accuracy over zero-shot, local-only LoRA, and standard FedAvg baselines on non-IID medical QA?

RQ2. Does the **federated** component of the retrieval prior contribute anything beyond generic similarity retrieval and beyond a purely local prior? This is tested against three controls — $\gamma=0$, local-only prior, and shuffled prior (§4.2, §6.1) — not against similarity retrieval alone.

RQ3. How sensitive is performance to non-IID skew, retrieval depth $k$, and prompt composition?

RQ4. Under the threat model stated in §7.1, does retrieval-ICL introduce a **measurable prompt-level exposure channel** that federated weight-sharing alone does not, and how does exposure scale with retrieval depth $k$?

**On the scope of RQ4.** Rev 1 phrased RQ4 as a "privacy–utility–communication trade-off". We retract that phrasing. A trade-off curve requires a privacy mechanism with a tunable knob (e.g. a DP budget $\varepsilon$); we deploy none, so there is no privacy axis along which to trade. We deliberately do **not** add DP-SGD to manufacture one: an $\varepsilon$ chosen without a calibrated accounting and clipping study yields neither a meaningful guarantee nor a competitive utility point, and a half-configured mechanism would make the claim weaker rather than stronger. What we can measure honestly is *exposure* — whether canaries planted in a local demonstration store reach the prompt and surface in the output — and that is what RQ4 now asks. DP and secure aggregation are named as future work in §7.4.

## 3. Novelty and contribution

Our contribution is not simply the combination of FL, LoRA, and retrieval. Instead, the key novelty is a client-aware federated retrieval mechanism for non-IID medical QA that is designed under the realistic constraints of SLM deployment and privacy auditing.

Specifically, our paper contributes the following:

1. A federated LoRA pipeline for medical QA that updates only adapter weights, reducing communication and preserving feasibility for resource-constrained environments.
2. A client-aware demonstration selection mechanism whose subject prior is computed **leave-one-client-out** from aggregated validation statistics — never from the receiving client's own data, and never from raw examples — together with the local and shuffled controls needed to show the federated component is what carries the effect.
3. A factorial evaluation protocol for FL × ICL × retrieval under non-IID subject skew, with conditional log-likelihood scoring for multiple-choice QA to avoid conflating generator quality with answer selection quality.
4. A prompt-level exposure audit using canary examples, conducted under an explicit threat model, to estimate whether retrieval-based ICL opens a leak channel that weight-level federation does not.

This framing is narrower and more defensible than claiming a novel general-purpose federated ICL method.

## 4. Problem formulation

Let each hospital or client $i$ hold a local dataset $D_i$ with non-IID medical questions and answer labels. The global objective is to improve the answer prediction function over the union of client distributions while keeping data local.

For each question $q$ and candidate answer option $o_k \in \{A, B, C, D\}$, the model computes the conditional log-likelihood:

$$
\hat{y} = \arg\max_{o_k} \log P(o_k \mid q, \text{prompt})
$$

This avoids the confounding problem of generating a free-form explanation and then matching it to an answer option by embedding similarity. In our setting, answer generation is not treated as a separate matching problem; each candidate option is scored directly under the model.

### 4.1 Demonstration scoring

For retrieval, each client selects demonstrations $d \in D_i$ using a relevance-and-redundancy score:

$$
s(d, q, i) = \alpha \cdot sim(e_d, e_q) - \beta \cdot redundancy(d, S) + \gamma \cdot w_{i, c(d)}
$$

where $sim(e_d, e_q)$ measures semantic relevance, $redundancy(d, S)$ penalizes overlap within the selected set $S$, $c(d)$ is the subject of demonstration $d$, and $w_{i,c}$ is the subject prior defined below.

### 4.2 The subject prior $w_{i,c}$ — definition and controls

Rev 1 described $w$ only as "aggregated from federated validation statistics". That is not a definition, and it hides the failure mode that makes RQ2 unfalsifiable: if client $i$'s own statistics enter $w_{i,\cdot}$, then a "federated prior" is a local prior wearing a federated label, and no contrast in the paper can tell the two apart.

**Statistics released by each client.** After a designated round $R_0$, each client $j$ evaluates the current global adapter on its own held-out validation split and releases, per subject $c$, only the pair

$$
\left( a_{j,c},\; n_{j,c} \right)
$$

where $a_{j,c}$ is accuracy on subject $c$ and $n_{j,c}$ is the number of validation items. No raw items, questions, answers, or embeddings leave the client. These two scalars per subject are themselves a disclosure and are accounted for in the threat model (§7.1).

**Leave-one-client-out deficit.** For client $i$ and subject $c$:

$$
\bar{e}_{-i,c} \;=\; \frac{\sum_{j \neq i} n_{j,c}\,\bigl(1 - a_{j,c}\bigr)}{\sum_{j \neq i} n_{j,c}}
$$

**Prior.** $w_{i,c}$ is $\bar{e}_{-i,c}$ standardized across subjects, so that $\gamma$ has a consistent scale across clients and rounds:

$$
w_{i,c} \;=\; \frac{\bar{e}_{-i,c} - \operatorname{mean}_{c'}\bigl(\bar{e}_{-i,c'}\bigr)}{\operatorname{std}_{c'}\bigl(\bar{e}_{-i,c'}\bigr)}
$$

The prior upweights demonstrations from subjects on which the model is weak **as measured at other institutions**. The index $i$ is excluded from the sum by construction, which is the property that makes the "federated" claim in RQ2 mean something.

**Freezing.** $w$ is computed once from the round-$R_0$ global adapter and then frozen for the remainder of the run. If $w$ were recomputed every round, retrieval and training would co-adapt within a single run and no contrast could attribute an effect to either one.

**The three controls.** Each replaces $w$ while leaving $\alpha$, $\beta$, $\gamma$, the retriever, and the prompt template untouched:

| Variant | Prior used | What it isolates |
|---|---|---|
| $\gamma = 0$ | none | Whether any prior beats pure top-$k$ similarity |
| Local prior | $w^{\text{loc}}_{i,c}$ from client $i$'s statistics only | Whether the gain needs *other* clients, or only self-knowledge |
| Federated (proposed) | $w_{i,c}$, leave-one-client-out | The claimed mechanism |
| Shuffled prior | $w_{i,\pi(c)}$ for a fixed random permutation $\pi$ over subjects | Whether the prior carries subject-specific information at all, or merely acts as a scoring perturbation |

The shuffled control is the falsification test. If the shuffled prior performs on par with the federated prior, then $\gamma \cdot w$ is functioning as a diversity or temperature term, not as transferred cross-institutional knowledge, and the paper's second contribution does not hold. This outcome must be reported as such.

## 5. Non-IID partition protocol

We intentionally use subject-structured partitioning rather than random IID splitting, because random partitioning does not reflect hospital heterogeneity.

### 5.1 Dataset choice

- Primary dataset: MedMCQA
- Secondary evaluation: MedQA-USMLE as a generalization benchmark
- We do not use PubMedQA as a primary dataset because its labeled size is limited relative to the required federated client partitioning and subject-aware analysis.

### 5.2 Partitioning strategy — skew, not disjoint

Rev 1 proposed assigning whole specialties to clients (surgery/anatomy/orthopedics to client 1, and so on). **That partition is incompatible with the prior defined in §4.2.** If subject $c$ lives entirely at client $i$, then $\sum_{j \neq i} n_{j,c} = 0$ and $\bar{e}_{-i,c}$ is undefined for exactly the subjects client $i$ actually retrieves from. A leave-one-client-out prior needs the held-out clients to have seen the subject.

We therefore partition by **skew rather than disjointness**: every client is dominated by a specialty cluster but retains a long tail of the remaining subjects, so each subject has support at more than one client.

- 5 simulated clients, each with a dominant specialty cluster (surgery/anatomy/orthopedics; medicine/pharmacology/physiology; pediatrics/obstetrics-gynecology; psychiatry/public health/preventive medicine; dental/ENT/ophthalmology and remaining subjects).
- Subject proportions drawn from a Dirichlet distribution with concentration $\rho$ over subjects per client, tilted toward each client's dominant cluster. $\rho$ is the knob for RQ3's skew sensitivity analysis.
- **Design check, verified before any training run:** $\min_{i,c \,:\, n_{i,c} > 0} \sum_{j \neq i} n_{j,c} \;\geq\; n_{\min}$, with $n_{\min}$ fixed in advance. Subjects failing this check are excluded from the prior (their $w_{i,c}$ is set to 0) and the exclusion rate is reported. Without this check the federated prior degrades silently into noise on precisely the subjects that matter most to each client.

We report subject-level distribution statistics — label entropy, subject proportions, and per-client subject support — to characterize the degree of skew. A near-disjoint partition is retained only as an RQ3 sensitivity point, explicitly labeled as a regime in which the federated prior is not estimable.

## 6. Experimental design

### 6.1 Baselines — two ordered ladders

Rev 1 listed baselines as a flat set that mixed retrieval variants with training protocols, so no single contrast isolated either one. They are reorganized into two ladders, run in the order given. Ladder A holds the training protocol fixed and varies the retrieval prior; ladder B holds retrieval fixed and varies the training protocol.

**Ladder A — retrieval prior (answers RQ2).** All four use the same base model and the same trained adapter; only the prior changes.

1. **$\gamma = 0$: top-$k$ similarity.** Pure relevance + redundancy, no prior. The reference point for every claim about $w$.
2. **Local weakness prior.** $w^{\text{loc}}$ from client $i$'s own validation statistics. Separates "knowing where the model is weak" from "knowing it from other institutions."
3. **Federated leave-one-client-out prior.** The proposed mechanism (§4.2).
4. **Shuffled federated prior.** Permutation control; the falsification test.

**Ladder B — training protocol (answers RQ1).**

5. **Local-only LoRA.** Each client trains alone. This is the realistic floor: what an institution achieves without collaborating. Federation's *benefit* is measured against this, and cannot be inferred from the centralized comparison.
6. **FedAvg-LoRA.** Standard federation, no ICL prior.
7. **FedAvg-LoRA + local adaptation.** Personalized FL. Personalization is a mature subfield with established methods, so the comparison must be against a named one rather than a bespoke variant; we use **Ditto** (Li et al., ICML 2021) as the reference personalization baseline.
8. **Centralized LoRA (upper bound), if compute allows.** Ceiling, not a deployable configuration — under the premise that motivates FL, pooling is unavailable. It bounds the gap; it does not establish benefit.

Zero-shot and random few-shot prompting are retained as untrained reference points beneath both ladders.

**Secondary (not on either ladder): public-corpus retrieval ICL.** Demonstrations drawn from a public medical corpus instead of proprietary local records. This is worth running — it tests whether local proprietary demonstrations are needed at all, which is a genuine deployment question — but it is *secondary*, because it does not bear on whether the federated prior has value. It answers "do we need private demos?", not "does cross-client signal help?". Schedule it after both ladders complete.

### 6.2 Hyperparameters and training protocol

- Model: Qwen2.5-3B-Instruct
- Parameter-efficiency: LoRA with rank $r=16$
- Aggregation: FedAvg on adapter weights only
- Number of simulated clients: 5
- Number of seeds: 5 for headline arms, 2 for auxiliary ablations
- Ladder A shares one trained adapter per seed across all four variants — the prior is applied at inference, so items 1–4 cost four evaluation passes, not four training runs
- $R_0$ (the round at which $w$ is computed and frozen) fixed in advance and reported
- Pilot run to choose effective FL round count; expected plateau around rounds 5–8

### 6.3 Metrics

Primary metrics:
- Accuracy
- Macro-F1
- ECE for calibration

Secondary metrics:
- Communication cost (MB per round), including the $(a_{j,c}, n_{j,c})$ statistics payload
- Latency per question
- Peak VRAM usage
- Number of tokens per prompt
- Retrieval quality (Recall@k, Precision@k, nDCG)
- Per-client accuracy, reported individually and not only as a mean — under skew, whether federation lifts every client or only the weak ones is a finding in itself

We do not use BLEU or ROUGE for MCQ evaluation because they are not meaningful for answer selection in multiple-choice settings.

## 7. Threat model and exposure audit

We do not claim that federated learning is privacy-preserving in a general sense. We state what is protected, against whom, and what we measure.

### 7.1 Threat model

**Assets.** Raw QA items and patient-derived content in each client's local store $D_i$.

**Adversaries in scope.**

| Adversary | Observes | Question asked |
|---|---|---|
| Honest-but-curious server | LoRA $A/B$ tensors each round; the $(a_{j,c}, n_{j,c})$ statistics | Do the released statistics themselves disclose client composition? |
| Curious client $j \neq i$ | The global adapter; the prior $w$ derived partly from $i$'s statistics | Can $j$ infer $i$'s subject distribution or weaknesses? |
| Prompt-output observer at client $i$ | Model outputs for submitted queries | Does a retrieved demonstration surface in the output? — the canary channel |

**Explicitly out of scope.** Malicious servers that craft adapters to induce extraction; gradient- or model-inversion attacks on adapter weights; network-level interception; client collusion; membership inference against the base model's pretraining corpus.

**What we claim.** Data locality at the weight level, and a *measurement* — not a bound — of prompt-level exposure. We provide no formal guarantee: there is no DP mechanism and no secure aggregation in this design.

### 7.2 Canary exposure test

- Insert synthetic QA canaries, with controlled distinctiveness, into a client's local demonstration store.
- Measure the **retrieval rate**: how often a canary enters the prompt at retrieval depth $k$.
- Measure the **surfacing rate**: how often generated output reproduces or reconstructs canary content, conditional on retrieval.
- Sweep $k$ to characterize how exposure scales with retrieval depth — this is the quantity RQ4 asks for.

Report retrieval rate and surfacing rate separately. Collapsing them hides which stage is responsible, and they have different mitigations: retrieval-stage exposure is addressable by filtering, generation-stage exposure is not.

### 7.3 Why this channel matters

Prompts can leak private content even when raw data never reach the server. FL preserves data locality at the weight level, but retrieval-ICL reintroduces a channel at the prompt level. This is realistic, under-examined in federated medical QA, and directly relevant to the trustworthiness of the design.

### 7.4 Limitations and future work

No differential privacy, no secure aggregation, no protection against the out-of-scope adversaries above. The released $(a_{j,c}, n_{j,c})$ statistics are a real disclosure surface that this work measures but does not defend. Adding DP to the adapter updates, or secure aggregation to the statistics channel, is future work — and doing it properly requires a privacy accounting and clipping study that is out of scope here, which is precisely why we do not present a partial version of it as a trade-off result.

## 8. Why this is a plausible contribution

The proposed work is viable because it addresses a genuinely difficult and practically relevant question:

> Under non-IID medical clients, can personalized federated retrieval and PEFT improve SLM performance without incurring unacceptable communication cost or prompt-level exposure?

This is more defensible than a broad statement such as "we propose a new federated ICL framework." The paper's contribution is narrower but more credible: it studies when this combination helps, under what skew and retrieval settings, and at what cost and exposure.

The contribution becomes especially compelling when the paper treats the effect as conditional rather than universal, and directly reports failure modes as well as successes.

## 9. Go / no-go criterion

Rev 1's criterion ("2–3 absolute points over strong baselines") did not name the baseline, which made it unfalsifiable — a gain over zero-shot would have satisfied it. Each claim is now tied to its own contrast:

| Claim | Contrast | Threshold |
|---|---|---|
| Federation is worth doing | item 6 vs item 5 (local-only) | 2–3 absolute points, stable CI |
| The prior helps | item 3 vs item 1 ($\gamma=0$) | 2–3 absolute points, stable CI |
| The *federated* part of the prior is what helps | item 3 vs item 2 (local) **and** item 3 vs item 4 (shuffled) | both positive with CI excluding 0 |

If the third row fails while the second holds, the honest conclusion is that a weakness prior helps and its federated provenance does not — the paper then reports a working retrieval heuristic, not a federated mechanism, and contribution 2 in §3 must be rewritten accordingly. If the first row fails, the paper shifts emphasis toward failure analysis and cost–exposure characterization rather than claiming a performance gain.

## 10. Framing for the final paper

A suitable paper framing is:

> A privacy-audited, client-aware federated retrieval framework for non-IID medical QA with small language models under LoRA-based adaptation.

This framing is precise, consistent with the data and compute constraints, and aligns with the experimental and privacy realities of the problem.
