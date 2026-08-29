# FedICL-MQA — Core paper sections (reframed for publication fit)

## 1. Problem statement

Medical question answering under real hospital settings is inherently non-IID: different institutions have different specialties, patient populations, and label distributions. In this setting, small language models (SLMs) are attractive because they can be deployed under resource constraints, but they often underperform larger models and are sensitive to distribution shift.

Federated learning (FL) offers a way to leverage cross-institution supervision without centralizing private records, while in-context learning (ICL) can improve reasoning by conditioning on task-relevant demonstrations. However, neither mechanism is automatically sufficient in medical QA: retrieval quality can be poor, demonstrations may be noisy or irrelevant, and privacy leakage may still occur through prompts, adapters, or retrieved examples.

This motivates the central question of our study:

> Can federated LoRA adaptation combined with client-aware demonstration retrieval improve SLM performance on non-IID medical QA while controlling communication cost and prompt-level leakage?

## 2. Research questions

RQ1. Does federated LoRA adaptation with local demonstration retrieval improve accuracy over zero-shot, local-only LoRA, and standard FedAvg baselines on non-IID medical QA?

RQ2. How much does client-aware retrieval contribute beyond generic similarity-based retrieval under skewed subject distributions?

RQ3. How sensitive is performance to non-IID partitioning, retrieval depth, and prompt composition?

RQ4. What is the privacy–utility–communication trade-off of the proposed design under a leakage audit?

## 3. Novelty and contribution

Our contribution is not simply the combination of FL, LoRA, and retrieval. Instead, the key novelty is a client-aware federated retrieval mechanism for non-IID medical QA that is designed under the realistic constraints of SLM deployment and privacy auditing.

Specifically, our paper contributes the following:

1. A federated LoRA pipeline for medical QA that updates only adapter weights, reducing communication and preserving feasibility for resource-constrained environments.
2. A client-aware demonstration selection mechanism that ranks local retrieval candidates using relevance, redundancy control, and a federated subject-prior signal derived from aggregated client performance profiles rather than raw examples.
3. A factorial evaluation protocol for FL × ICL × retrieval under non-IID subject partitions, with conditional log-likelihood scoring for multiple-choice QA to avoid conflating generator quality with answer selection quality.
4. A prompt-level leakage audit using canary examples to estimate whether retrieval-based ICL introduces a new privacy leak channel even when raw data remain local.

This framing is narrower and more defensible than claiming a novel general-purpose federated ICL method.

## 4. Problem formulation

Let each hospital or client $i$ hold a local dataset $D_i$ with non-IID medical questions and answer labels. The global objective is to improve the answer prediction function over the union of client distributions while keeping data local.

For each question $q$ and candidate answer option $o_k \in \{A, B, C, D\}$, the model computes the conditional log-likelihood:

$$
\hat{y} = \arg\max_{o_k} \log P(o_k \mid q, \text{prompt})
$$

This avoids the confounding problem of generating a free-form explanation and then matching it to an answer option by embedding similarity. In our setting, answer generation is not treated as a separate matching problem; each candidate option is scored directly under the model.

For retrieval, each client selects demonstrations $d \in D_i$ using a relevance-and-redundancy score:

$$
s(d, q, i) = \alpha \cdot sim(e_d, e_q) - \beta \cdot redundancy(d, S) + \gamma \cdot w_{subj(d)}^{(i)}
$$

where:
- $sim(e_d, e_q)$ measures semantic relevance between the question and a retrieved demonstration,
- $redundancy(d, S)$ penalizes demonstration overlap within the selected set,
- $w_{subj(d)}^{(i)}$ is a client-specific subject prior aggregated from federated validation statistics, not raw examples.

The key design choice is that the federated signal is based on aggregate statistics (e.g., subject-level performance profiles), not shared patient records or raw QA pairs.

## 5. Non-IID partition protocol

We intentionally use subject-structured partitioning rather than random IID splitting, because random partitioning does not reflect hospital heterogeneity.

### 5.1 Dataset choice

- Primary dataset: MedMCQA
- Secondary evaluation: MedQA-USMLE as a generalization benchmark
- We do not use PubMedQA as a primary dataset because its labeled size is limited relative to the required federated client partitioning and subject-aware analysis.

### 5.2 Partitioning strategy

We simulate five medical clients by grouping subjects into clinically coherent clusters, for example:
- Client 1: surgery, anatomy, orthopedics
- Client 2: medicine, pharmacology, physiology
- Client 3: pediatrics, gynecology and obstetrics
- Client 4: psychiatry, public health, preventive medicine
- Client 5: dental, ENT, ophthalmology, and remaining smaller subjects

This creates realistic heterogeneity in both label distributions and knowledge specialization. We further report subject-level distribution statistics, such as label entropy and subject proportions, to characterize the degree of non-IID skew.

We also include a sensitivity analysis using a Dirichlet-based partition for comparison, but the main benchmark is based on clinically meaningful specialty grouping rather than random IID splitting.

## 6. Experimental design

### 6.1 Baselines

We compare the proposed method against the following baselines:

- Zero-shot prompting
- Random few-shot prompting
- Local retrieval ICL
- Local-only LoRA fine-tuning
- FedAvg-LoRA without ICL
- FedAvg-LoRA with naive retrieval ICL
- Centralized LoRA training (upper bound)

This is necessary to isolate whether gains are coming from retrieval, federation, personalization, or all three combined.

### 6.2 Hyperparameters and training protocol

- Model: Qwen2.5-3B-Instruct
- Parameter-efficiency: LoRA with rank $r=16$
- Aggregation: FedAvg on adapter weights only
- Number of simulated clients: 5
- Number of seeds: 5 for headline arms, 2 for auxiliary ablations
- Pilot run to choose effective FL round count; expected plateau around rounds 5–8

### 6.3 Metrics

Primary metrics:
- Accuracy
- Macro-F1
- ECE for calibration

Secondary metrics:
- Communication cost (MB per round)
- Latency per question
- Peak VRAM usage
- Number of tokens per prompt
- Retrieval quality (Recall@k, Precision@k, nDCG)

We do not use BLEU or ROUGE for MCQ evaluation because they are not meaningful for answer selection in multiple-choice settings.

## 7. Privacy and leakage audit

We do not claim that federated learning is privacy-preserving in a general sense. Rather, we evaluate whether the system is data-local and whether the prompt-based retrieval layer introduces a measurable leakage channel.

Our privacy analysis focuses on a simple but informative canary test:

- insert synthetic QA canaries into a client’s local demonstration repository,
- measure whether those canaries are retrieved into the prompt,
- evaluate whether the generated output exposes or reconstructs the inserted sensitive content.

This is important because prompts can leak private content even when raw data are never transferred to the server. In other words, FL preserves data locality at the weight level, but retrieval-ICL can reintroduce leakage at the prompt level.

This is a realistic and under-examined channel in federated medical QA, and it is directly relevant to the trustworthiness of the proposed framework.

## 8. Why this is a plausible contribution

The proposed work is viable because it addresses a genuinely difficult and practically relevant question:

> Under non-IID medical clients, can personalized federated retrieval and PEFT improve SLM performance without incurring unacceptable communication and privacy costs?

This is more defensible than a broad statement such as “we propose a new federated ICL framework.” The paper’s contribution is narrower but more credible: it studies when this combination helps, under what skew and retrieval settings, and at what privacy and cost trade-off.

The contribution becomes especially compelling when the paper treats the effect as conditional rather than universal, and directly reports failure modes as well as successes.

## 9. Go / no-go criterion

We adopt a practical go/no-go checkpoint at the end of Week 2:

> If federated LoRA with personalized retrieval does not improve over strong baselines by at least 2–3 absolute points with stable confidence intervals, then the paper should shift emphasis toward failure analysis, cost–privacy trade-offs, or a more specific retrieval mechanism rather than claiming a broad performance gain.

This encourages a disciplined evaluation strategy and avoids overclaiming before the method has proven its utility.

## 10. Framing for the final paper

A suitable paper framing is:

> A privacy-audited, client-aware federated retrieval framework for non-IID medical QA with small language models under LoRA-based adaptation.

This framing is precise, consistent with the data and compute constraints, and aligns with the experimental and privacy realities of the problem.
