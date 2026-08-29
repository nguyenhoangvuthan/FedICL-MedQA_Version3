# FedICL-MQA — Thiết kế đã chốt (cập nhật 2026-08-11)

> Bản này thay thế bản phân tích cũ. Các quyết định dưới đây đã được chốt sau khi rà soát
> mâu thuẫn giữa hình kiến trúc (Jun 7), outline của advisor (PDF) và bản phản biện trước đó,
> đối chiếu với ràng buộc thực tế: **1× A5000 24GB, deadline ~1 tháng (nộp ~11/09/2026)**.

## 0. Bảng quyết định

| Hạng mục | Quyết định |
| --- | --- |
| Hướng bài | **B** — Factorial study làm thân bài + một cược nhỏ: client-aware demonstration selection |
| Model | **Qwen2.5-3B-Instruct** (1 model duy nhất) |
| Dataset chính | **MedMCQA** (21 subject labels → partition 5 client theo cụm chuyên khoa) |
| Dataset phụ | **MedQA-USMLE** (chỉ đánh giá generalization) |
| Bỏ | PubMedQA (1k labeled — quá mỏng); mục "Multimodal Representation Learning" trong outline; BLEU/ROUGE-L; `L_ICL`, `L_REG` |
| Số client | 5 |
| Seeds | **Phân tầng**: 5 seeds cho các arm headline, 2 seeds cho ablation phụ |
| Số FL rounds | Chốt bằng **pilot run** (kỳ vọng ~6, plateau thường ở round 5–8) |
| FL | FedAvg trên **LoRA adapters only** (r=16, ~15–30 MB/round) |
| Chấm MCQ | **Conditional log-likelihood** trên A/B/C/D — bỏ hẳn "sinh text → encoder match" |
| Privacy | 1 thí nghiệm duy nhất: **canary trong demonstration repository**; claim hạ xuống "data-local" |
| Go/no-go | **Cuối tuần 2** (không phải tuần 7 như bản cũ) |

## 1. Định vị bài báo và claim

Tên hệ thống giữ nguyên **FedICL-MQA** (khớp hình và outline của advisor), nhưng định nghĩa lại chính xác:

> FedICL-MQA là một framework **federated fine-tuning (LoRA) kết hợp retrieval-based
> in-context learning** cho SLM trên medical QA non-IID, trong đó các client chia sẻ
> **adapter updates và thống kê tổng hợp** — không bao giờ chia sẻ raw QA examples.

Câu claim chính (vừa đủ mạnh, không overclaim):

> We study whether federated LoRA adaptation combined with client-aware demonstration
> retrieval improves SLM performance on non-IID medical QA, while quantifying
> communication cost and a previously under-examined leakage channel in the prompt.

Ba contribution (thay cho 5 contribution overclaim trong outline):

1. **Client-aware demonstration selection** — cơ chế chọn demonstration dạng đóng,
   dùng thống kê liên bang (per-subject accuracy + subject histogram) aggregate qua server
   thay cho raw data. Đây là hiện thực hóa cụ thể của claim
   "context-aware knowledge-sharing" vốn có trong outline nhưng chưa có cơ chế.
2. **Giao thức đánh giá factorial FL × ICL trên partition non-IID theo chuyên khoa thật**
   (MedMCQA subjects), log-likelihood scoring, seeds phân tầng, McNemar + bootstrap CI.
3. **Phát hiện thực nghiệm về kênh rò rỉ mới**: FL bảo vệ weights nhưng retrieval-ICL
   mở kênh rò rỉ qua prompt (canary experiment), kèm đo communication/latency/VRAM.

Claim privacy: **"data-local collaborative learning"** — KHÔNG dùng chữ "privacy-preserving".

## 2. Kiến trúc — sửa mâu thuẫn cốt lõi của hình Jun 7

Hình Jun 7 vừa nói ICL (training-free) vừa có Local Training Objective + FedAvg trên θ.
Sửa bằng cách tách bạch hai vòng:

### 2.1 Vòng FL (có train)

- FedAvg trên **LoRA adapters only** (r=16, bf16). Payload ~15–30 MB/round
  thay vì ~7.6 GB full model → giảm ~200–800×, giữ được claim "efficient".
- Loss duy nhất: `L_QA` = next-token cross-entropy trên đáp án.
  **Bỏ `L_ICL` (Context Consistency) và `L_REG`** — không định nghĩa được thì không đưa vào.
- Sau mỗi round, mỗi client gửi thêm **vector accuracy theo subject trên local validation**
  (21 số thực — không phải data). Server aggregate thành *global weakness profile*.

### 2.2 Vòng inference (không train) — cơ chế cốt lõi

Mỗi client retrieve demonstrations từ repository local bằng điểm số dạng đóng:

$$
s(d, q, i) = \alpha \cdot \underbrace{\cos(e_d, e_q)}_{\text{relevance}}
- \beta \cdot \underbrace{\max_{d' \in S}\cos(e_d, e_{d'})}_{\text{redundancy (MMR)}}
+ \gamma \cdot \underbrace{w_{\mathrm{subj}(d)}}_{\text{federated prior}}
$$

- Điểm "federated" nằm ở $w_{\mathrm{subj}}$: client ưu tiên demonstration thuộc các subject
  mà **global model đang yếu** (theo weakness profile). Knowledge-sharing giữa các bệnh viện
  mà không chuyển một QA pair nào.
- Encoder: **MedCPT** (fallback: PubMedBERT-based sentence encoder) + FAISS flat index per client.
- $\alpha, \beta, \gamma$: grid search nhỏ trên local validation —
  không có tham số học bằng gradient, code được trong ~2 ngày.
- Chấm MCQ: $\hat{y} = \arg\max_{o \in \{A,B,C,D\}} \log P(o \mid \mathrm{prompt})$.

### 2.3 Đường lui cài sẵn

Nếu $\gamma$-term không thắng top-k similarity: bài vẫn đứng bằng contribution 2+3,
và kết quả âm của selector được báo cáo như một finding
("thống kê liên bang cấp corpus không đủ tín hiệu cho selection cấp câu hỏi").

## 3. Thiết kế thí nghiệm

### 3.1 Non-IID partition (MedMCQA)

- 5 client = 5 cụm chuyên khoa gộp từ 21 subjects, ví dụ:
  (1) Surgery + Anatomy + Orthopaedics; (2) Medicine + Pharmacology + Physiology;
  (3) Pediatrics + Gynaecology & Obstetrics; (4) Psychiatry + Social & Preventive Medicine;
  (5) Dental + ENT + Ophthalmology + còn lại. *(Cụm chính xác chốt khi nhìn phân phối thật.)*
- Báo cáo mức độ heterogeneity của partition (ví dụ JS-divergence giữa label/subject distributions).
- Sensitivity (ablation, 2 seeds): một partition Dirichlet để đối chiếu.
- MedQA-USMLE: **không partition** — chỉ dùng làm tập đánh giá thứ hai (generalization).

### 3.2 Các arm headline (5 seeds)

| # | Arm | FL | ICL |
| --- | --- | --- | --- |
| 1 | Zero-shot | – | – |
| 2 | Retrieval-ICL local (top-k similarity) | – | ✓ |
| 3 | FedAvg-LoRA | ✓ | – |
| 4 | FedAvg-LoRA + retrieval-ICL (top-k) | ✓ | ✓ |
| 5 | **FedAvg-LoRA + client-aware selection** | ✓ | ✓ (cơ chế mới) |
| 6 | Local-only LoRA (mỗi client tự train, không aggregate) | – | – |
| 7 | Centralized LoRA (upper bound) | – | – |

Arm 6 là baseline quan trọng nhất để chứng minh FL có ích — outline cũ thiếu, nay bắt buộc có.

### 3.3 Ablations (2 seeds)

- Số demonstrations k ∈ {2, 4, 8} và random demonstrations
- Retriever: BM25 vs generic encoder vs MedCPT
- FedProx vs FedAvg (một aggregation thay thế là đủ)
- Tách thành phần selector: bỏ MMR-term, bỏ $\gamma$-term

### 3.4 Metrics

- **Chính**: Accuracy, macro-F1 (per-dataset, per-client, per-subject)
- Calibration: ECE (rẻ, tính từ log-likelihood đã có)
- FL: communication MB/round, convergence curve (từ pilot + các run chính)
- Efficiency: latency/question, peak VRAM, tokens/prompt
- **Bỏ** BLEU/ROUGE-L — vô nghĩa với MCQ

### 3.5 Thống kê

- Mean ± SD qua seeds; bootstrap 95% CI trên test items
- **Paired McNemar** cho mọi so sánh accuracy chính (significance đến từ
  kích thước test set — MedMCQA dev 4.183 items, MedQA test 1.273 items — không phải từ seeds)
- Mỗi bảng ghi rõ số seeds

### 3.6 Privacy — một thí nghiệm duy nhất

Chèn canary (QA pair tổng hợp, không trùng phân phối) vào demonstration repository của từng
client → đo tỷ lệ canary bị retrieve vào prompt và tỷ lệ lộ ra generated output.
Luận điểm: *FL bảo vệ weights, nhưng retrieval-ICL mở lại một kênh rò rỉ ngay trong prompt.*
Chi phí ~1 ngày + ~5 GPU-hours.

## 4. Ngân sách compute (A5000 24GB)

Giả định 6 rounds (chốt lại sau pilot), ~3h/run FL:

| Nhóm | Ước tính |
| --- | --- |
| Pilot run (chốt số rounds, vẽ convergence) | ~5h |
| Headline FL arms (3,4,5) × 5 seeds | ~45h |
| Local-only + Centralized LoRA × 5 seeds | ~15h |
| Ablations × 2 seeds | ~36h |
| Arms inference-only (1,2) × 5 seeds | ~10h |
| Canary experiment | ~5h |
| **Tổng productive** | **~115–120h** |
| Buffer (debug, chạy lại) | ~180–230h còn lại (~60%) |

## 5. Timeline 4 tuần

- **Tuần 1**: Eval harness (log-likelihood scoring), zero-shot/ICL arms, FAISS/MedCPT,
  partition 5 cụm chuyên khoa, và **pilot FL run → chốt số rounds**.
  Phương pháp **đóng băng cuối tuần 1** — không phát minh thêm sau mốc này.
- **Tuần 2**: Chạy headline FL arms (3,4,6,7). **Go/no-go cuối tuần 2**:
  nếu FL và/hoặc ICL không cho tín hiệu ≥2–3 điểm với McNemar significant so với baseline mạnh
  → xoay framing sang "when does it help / failure analysis + cost + leakage" (bài vẫn đứng).
- **Tuần 3**: Client-aware selector (arm 5) + ablations + canary experiment.
- **Tuần 4**: Thống kê, error analysis theo subject, vẽ hình, viết bài. GPU chỉ chạy lại/bổ sung.

## 6. Thay đổi so với outline của advisor (cần trao đổi lại)

1. PubMedQA → MedMCQA (lý do: cần subject labels cho non-IID có ý nghĩa lâm sàng; PubMedQA 1k labeled quá mỏng cho 5 client).
2. Bỏ mục 3.3 "Multimodal Representation Learning" (dataset thuần text — mục này là template sót).
3. Metrics: bỏ BLEU/ROUGE-L, thay bằng accuracy/macro-F1/ECE + cost metrics.
4. 4 SLM → 1 SLM (Qwen2.5-3B-Instruct) do ngân sách compute; so sánh đa model đưa vào future work.
5. FedAvg trên full θ → LoRA-only aggregation (bắt buộc về mặt băng thông; củng cố claim "efficient").
6. Thêm baseline local-only LoRA (thiếu trong outline, bắt buộc để chứng minh giá trị của FL).
7. "Privacy-preserving" → "data-local" + 1 canary experiment thay cho các hộp DP/secure-agg chưa implement.
8. Retrieval strategy "Recency/Authority" trong hình: bỏ (MedMCQA/MedQA không có metadata này).

## 7. Ràng buộc phản biện giữ lại từ bản phân tích cũ

- **RAG/ICL không mặc nhiên có lợi**: có nghiên cứu cho thấy vanilla model thắng RAG khi
  retrieval precision thấp; medical ICL với demonstrations nhiễu có thể làm SLM tệ đi.
  → Framing luôn là "khi nào có lợi", không phải "luôn có lợi".
- **Data contamination**: MedQA/MedMCQA nhiều khả năng nằm trong pretraining của Qwen2.5.
  → Thêm 1 đoạn thảo luận + so sánh *relative* giữa các arm (mọi arm cùng model nên so sánh nội bộ vẫn valid).
- **FL không tự động là privacy**: adapter, prompt và retrieved demonstrations đều có thể rò rỉ.
  → Đã xử lý bằng claim "data-local" + canary experiment.
- Trích dẫn "Med-RISE cải thiện ~13 điểm" trong bản cũ **chưa xác minh được nguồn** —
  không đưa vào bài cho đến khi kiểm tra lại.

## 8. Tiêu đề đề xuất

1. FedICL-MQA: Federated LoRA and Client-Aware Demonstration Retrieval for Non-IID Medical QA with Small Language Models
2. When Does Federated In-Context Learning Help? A Study of SLMs on Non-IID Medical QA
3. Client-Aware Federated In-Context Learning for Non-IID Medical QA under Resource Constraints
