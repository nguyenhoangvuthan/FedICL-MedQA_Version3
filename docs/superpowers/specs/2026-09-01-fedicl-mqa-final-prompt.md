# FedICL-MQA — Prompt chốt thiết kế baseline FL × ICL

**Status:** Thiết kế đã chốt để lập implementation plan

**Ngày:** 2026-09-01

**Phạm vi:** Medical multiple-choice question answering bằng Small Language Model, LoRA, Federated Learning và In-Context Learning.

## 1. Mục tiêu thực nghiệm

Thiết kế và triển khai một protocol thực nghiệm có kiểm soát để trả lời bốn câu hỏi:

1. ICL có cải thiện một mô hình đã được Local LoRA adaptation hay không?
2. ICL có cải thiện một mô hình đã được FedAvg-LoRA adaptation hay không?
3. Query-dependent retrieval có tốt hơn một manifest gồm năm exemplar được chọn trước hay không?
4. Client-aware re-ranking có tạo thêm lợi ích so với single retriever chuẩn hay không?

Toàn bộ so sánh phải tách được ảnh hưởng của training, federation, exemplar context và re-ranking. Không được thay checkpoint khi chỉ muốn đo ảnh hưởng của ICL tại inference.

## 2. Phạm vi đã khóa

- Bài toán hiện tại là **text-only medical QA**.
- Loại hoàn toàn mục **Multimodal Representation Learning** khỏi proposed approach, experimental plan và contribution hiện tại.
- Các dataset chính dự kiến là MedQA/MedMCQA; giữ official train/validation/test split trước khi tạo client partition.
- Chỉ dùng **một local dense retriever** cho tất cả arm có retrieval.
- Mỗi prompt ICL có đúng `k = 5` exemplar hợp lệ.
- Exemplar chỉ được lấy từ repository tạo từ training split; validation và test không bao giờ là nguồn exemplar.

## 3. Single closure-constrained local retriever

Tên thống nhất của module:

> **Single closure-constrained local retriever**

Mỗi client `i` có đúng một encoder, một local index và một support repository. Tất cả arm dùng chung encoder revision, embedding procedure, index, candidate pool và manifest policy.

Với query `q`, trước tiên xây tập exemplar hợp lệ:

$$
\mathcal{A}_i(q)=\left\{d\in D_i^{support}\ \middle|\
\operatorname{id}(d)\ne\operatorname{id}(q),\
\operatorname{norm}(d_q)\ne\operatorname{norm}(q),\
\operatorname{sim}_{dup}(d_q,q)<\tau_{dup},\
\operatorname{group}(d)\ne\operatorname{group}(q)
\right\}.
$$

Sau khi loại toàn bộ candidate không hợp lệ, retriever chọn:

$$
R_i(q)=\operatorname{Top5}_{d\in\mathcal{A}_i(q)}
\operatorname{sim}_{rel}(q,d).
$$

`Top-5` vì vậy có nghĩa là **năm exemplar có relevance cao nhất trong tập đã loại exact duplicate và near-duplicate**, không phải cosine top-5 trực tiếp trên toàn bộ dữ liệu.

### 3.1 Quy tắc loại candidate

Một candidate không được dùng làm exemplar nếu vi phạm ít nhất một điều kiện:

- Trùng `example_id` với query.
- Trùng normalized question sau khi lowercase, chuẩn hóa Unicode, bỏ dấu câu thừa và chuẩn hóa whitespace.
- Trùng question–options hash.
- Thuộc cùng semantic near-duplicate cluster với query.
- Vượt ngưỡng lexical hoặc semantic duplicate đã khóa trên development workflow.
- Trùng source/provenance group với query khi dataset cung cấp metadata phù hợp.
- Xuất phát từ validation hoặc test split.

Không cấm một exemplar chỉ vì nó liên quan cùng chủ đề y khoa. Mục tiêu là loại leakage do câu trùng hoặc paraphrase gần như tương đương, trong khi vẫn giữ clinical relevance cần thiết cho ICL.

### 3.2 Retrieval procedure

1. Tìm top-50 candidate từ local support repository.
2. Áp toàn bộ closure constraint.
3. Nếu còn ít hơn năm exemplar, mở rộng lần lượt đến top-100 và toàn bộ local support repository.
4. Chọn đúng năm exemplar hợp lệ.
5. Nếu không đủ năm, đánh dấu capacity failure; không tự giảm `k` và không lấy dữ liệu từ client khác, validation hoặc test.
6. Trước khi seal cohort, refill từ training reserve pool hoặc loại query; báo cáo exclusion rate và lý do.
7. Retriever chỉ encode `question + answer options`; không được thấy gold label của query.

## 4. Data contract chống overfitting và leakage

Mỗi client có cấu trúc dữ liệu:

$$
D_i^{train}=D_i^{fit}\cup D_i^{support},
\qquad D_i^{fit}\cap D_i^{support}=\varnothing.
$$

- `fit`: dùng để train Local LoRA, Centralized LoRA hoặc FedAvg-LoRA.
- `support`: nguồn exemplar duy nhất.
- `validation`: chỉ dùng chọn hyperparameter, early stopping, FL round và khóa retrieval/re-ranking configuration.
- `test`: chỉ dùng đánh giá cuối cùng sau khi toàn bộ configuration đã freeze.

Các invariant bắt buộc:

$$
IDs(D^{support})\cap\left(IDs(D^{validation})\cup IDs(D^{test})\right)=\varnothing.
$$

Ngoài kiểm tra ID, phải kiểm tra disjointness ở các mức:

- normalized question;
- question–options hash;
- semantic near-duplicate cluster;
- source/provenance group.

Validation và test query vẫn được phép retrieve exemplar, nhưng exemplar của chúng chỉ đến từ `D_i^{support}` thuộc training split. Không dùng test để chọn `k`, duplicate threshold, prompt, checkpoint, retriever setting hoặc re-ranking weight.

## 5. Baseline arms

| Arm | Training | Inference context | Mục đích |
|---|---|---|---|
| **B0** | Frozen base model | Không exemplar (`k=0`) | Zero-shot floor |
| **B1** | Cùng frozen base model | Top-5 bằng single retriever | Hiệu quả pure retrieval-ICL |
| **L0** | Local LoRA trên `fit` | Không exemplar (`k=0`) | Local adaptation không ICL |
| **L1** | **Cùng checkpoint L0** | Top-5 bằng single retriever | Hiệu quả inference-time ICL sau Local LoRA |
| **LI0** | LoRA trên `fit` | Năm exemplar hợp lệ được chọn trước và seal trong manifest, không xếp hạng bằng query similarity | LoRA + exemplar-ICL control |
| **LI1** | **Cùng checkpoint LI0** | Top-5 query-specific bằng single retriever | Giá trị của retrieval so với exemplar manifest không retrieval |
| **F0** | FedAvg-LoRA trên `fit` | Không exemplar (`k=0`) | FL không ICL |
| **F1** | **Cùng checkpoint F0** | Top-5 bằng single retriever | Hiệu quả inference-time ICL trong FL |
| **F2** | **Cùng checkpoint F0** | Cùng retriever và candidate set với F1, sau đó client-aware re-ranking | Phương pháp đề xuất |
| **C0** | Centralized LoRA trên union của các `fit` set | Không exemplar (`k=0`) | Centralized training reference/upper bound |

### 5.1 Ý nghĩa của LI0

LI0 không dùng năm câu giống nhau một cách mù quáng cho toàn bộ query. Với mỗi query, năm exemplar được chọn deterministic từ tập hợp hợp lệ sau closure filtering, rồi toàn bộ mapping `query_id -> five exemplar_ids` được seal trước khi chạy model. Việc chọn không dùng relevance ranking của retriever. LI1 dùng chính support pool và closure rules đó nhưng bật query-dependent similarity ranking.

### 5.2 Client-aware re-ranking

F2 không tạo retriever thứ hai. F2 lấy candidate từ đúng single retriever của F1 và chỉ thay ranking score:

$$
s(d,q,i)=
\alpha\,\operatorname{sim}(q,d)
-\beta\,\operatorname{redundancy}(d,S)
+\gamma\,w_{\operatorname{subject}(d)}^{(i)}.
$$

Các trọng số `alpha`, `beta`, `gamma` chỉ được chọn bằng training/validation workflow và phải freeze trước test.

## 6. Primary contrasts

$$
\Delta_{Base\text{-}ICL}=Acc(B1)-Acc(B0)
$$

$$
\Delta_{Local\text{-}ICL}=Acc(L1)-Acc(L0)
$$

$$
\Delta_{Retrieval}=Acc(LI1)-Acc(LI0)
$$

$$
\Delta_{FL\text{-}ICL}=Acc(F1)-Acc(F0)
$$

$$
\Delta_{ClientAware}=Acc(F2)-Acc(F1)
$$

$$
\Delta_{FL}=Acc(F0)-Acc(L0)
$$

Các contrast ICL phải dùng paired predictions trên cùng test items. L0/L1, LI0/LI1 và F0/F1/F2 phải dùng cùng checkpoint tương ứng để không trộn ảnh hưởng của training với ảnh hưởng của inference context.

## 7. Evaluation contract

### 7.1 Primary metrics

- Overall multiple-choice accuracy.
- Macro-client accuracy.
- Worst-client accuracy.
- Per-subject accuracy.

### 7.2 Secondary metrics

- Calibration/ECE.
- Communication bytes mỗi round và toàn bộ FL run.
- Prompt-token P50/P95/max.
- Inference latency, throughput và peak VRAM.
- Retrieval exclusion, refill và capacity-failure rate.
- BLEU/ROUGE/BERTScore chỉ dùng như secondary metrics nếu có nhiệm vụ sinh free-form explanation hoặc rationale; không dùng để kết luận chất lượng chọn đáp án A/B/C/D.

### 7.3 Statistics

- Dùng paired seed IDs giữa các training arm cần so sánh.
- Dùng paired item bootstrap cho deterministic inference contrast.
- Dùng hierarchical paired bootstrap qua seed và evaluation unit cho trained arms.
- Báo cáo effect size và 95% confidence interval, không chỉ p-value.
- McNemar có thể dùng bổ sung cho hai prediction vector trên cùng test set.
- Freeze primary contrasts và multiple-comparison correction trước test.

## 8. Security và privacy trong plan

Giữ các thành phần sau trong plan nếu chúng thuộc phạm vi triển khai:

- Secure aggregation cho LoRA updates.
- Differential privacy với clipping norm, noise multiplier và báo cáo `(epsilon, delta)`.
- Encrypted communication channel.
- Privacy-aware local retrieval.
- Canary hoặc attack-based leakage evaluation.

Trong implementation plan phải phân biệt rõ trạng thái `planned`, `implemented` và `experimentally validated`. Final paper chỉ dùng claim **privacy-preserving** sau khi cơ chế tương ứng đã được triển khai và đánh giá; trước thời điểm đó mô tả hệ thống là **data-local by design** hoặc **privacy-oriented**.

## 9. Paper outline đã cập nhật

1. Introduction
2. Related Work
   - Medical Question Answering
   - Federated Learning for Healthcare NLP
   - In-Context Learning and Demonstration Retrieval
   - Small Language Models
3. Proposed Approach
   - Problem Formulation
   - Framework Overview
   - Single Closure-Constrained Local Retriever
   - In-Context Prompt Construction
   - Local and Federated LoRA Training
   - Client-Aware Demonstration Re-ranking
   - Secure Aggregation and Differential Privacy
   - Training Objective
4. Experiments
   - Datasets and Client Partitioning
   - Baseline Arms
   - Leakage-Controlled Manifest Construction
   - Metrics and Statistical Protocol
5. Results
   - Base and Local ICL Effects
   - Federated ICL Effect
   - Retrieval and Client-Aware Re-ranking Effects
   - Non-IID, Efficiency and Communication Analysis
   - Privacy Analysis
6. Discussion and Limitations
7. Conclusion

Không có mục Multimodal Representation Learning hoặc multimodal extension trong outline hiện tại.

## 10. Pre-run acceptance checklist

- [ ] Mọi exemplar đều có original split là `train`.
- [ ] Không có validation/test ID trong support repository.
- [ ] Không có overlap theo normalized text, hash, near-duplicate cluster hoặc provenance group.
- [ ] Mọi query trong sealed cohort có đúng năm exemplar hợp lệ.
- [ ] Retriever không encode gold label của query.
- [ ] Tất cả retrieval arm dùng đúng một encoder/index/revision.
- [ ] L0/L1, LI0/LI1 và F0/F1/F2 dùng cùng checkpoint trong từng nhóm.
- [ ] Prompt template và answer-scoring protocol giống nhau giữa các arm.
- [ ] Configuration, manifest và model revision đã được hash và freeze trước test.
- [ ] Test set không được dùng trong tuning hoặc debugging decision.
- [ ] Exclusion/refill/capacity-failure được lưu theo query và lý do.
- [ ] Multimodal đã được loại khỏi scope và outline.
