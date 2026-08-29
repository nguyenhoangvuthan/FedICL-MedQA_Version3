# FedICL-MQA — Pilot thu nhỏ: kiểm tra tính khả thi ICL centralized → federated

Status: Approved design, sẵn sàng cho implementation plan
Date: 2026-08-29
Liên quan: [fedicl_mqa_paper_core_sections.md](../../fedicl_mqa_paper_core_sections.md) (design full-scale gốc)

## 1. Mục tiêu

Kiểm tra tính khả thi của bài toán ICL (in-context learning) xuyên suốt pipeline từ centralized đến federated learning, dùng model cực nhỏ (≤1B) và quy mô dữ liệu nhỏ, **trước khi** đầu tư compute lớn cho full-scale paper (Qwen2.5-3B, MedMCQA 5-client, 120 GPU-giờ đã note trong nghiên cứu trước).

Đây là một feasibility pilot (architectural scope — code mới, không sửa flow có sẵn), không phải bản thu nhỏ 1:1 của full-scale design. Một số quyết định phương pháp luận (cách chọn đáp án, cách chia dữ liệu) **khác** với `fedicl_mqa_paper_core_sections.md` — được ghi chú rõ ở từng mục.

## 2. Hạ tầng & phạm vi

- Compute: Cloud GPU A5000 (đã cấp phát).
- 2 model: **Qwen2.5-0.5B-Instruct** và **SmolLM2-360M-Instruct**.
- 2 dataset: **MedQA-USMLE** (trắc nghiệm, có sẵn 4 lựa chọn) và **MedQuAD** (tự luận — được chuyển thành constructed 4-way MCQ, xem mục 6).
- Full factorial: 2 model × 2 dataset × 5 baseline (xem mục 3) = 20 ô thí nghiệm × 3 seed.
- Ước tính compute ban đầu: 15–25 GPU-giờ cho toàn bộ grid — **là giả thuyết**, sẽ đo lại sau khi chạy xong 1 ô đầu tiên. 3 seed gần như nhân ba chi phí training; với MedQuAD, độ dài sequence (câu trả lời dài + k=5 demo) nhiều khả năng là yếu tố chi phối chi phí hơn là số lượng mẫu.

## 3. Baseline (5 arm)

| # | Baseline | Train | Eval | Mục đích |
|---|---|---|---|---|
| 1 | Zero-shot | Không train | k=0, không demo | Sàn dưới cùng |
| 2 | **ICL-only** | Không train | k=5 demo (từ `test-support`) | Đo đóng góp thuần của retrieval-ICL, tách khỏi hiệu ứng SFT |
| 3 | train-k0 / eval-k5 | LoRA SFT, prompt trơn (không demo) | k=5 demo | SFT không tiếp xúc ICL lúc train |
| 4 | train-k5 / eval-k5 | LoRA SFT, mỗi prompt train có k=5 demo (leave-one-out từ `train-core`) | k=5 demo | SFT có tiếp xúc ICL lúc train |
| 5 | **Federated k=5** | 3 client mô phỏng, FedAvg trên LoRA adapter, mỗi client dùng k=5 demo lấy cục bộ | k=5 demo | So trực tiếp với #4 (cùng k) để cô lập đúng chi phí liên bang hóa |

Lưu ý lịch sử quyết định: ban đầu FL dùng k=3, nhưng vì điều đó khiến so sánh FL vs train-k5/eval-k5 nhiễu bởi hai biến (train method + k khác nhau), đã đổi FL sang k=5 để so sánh công bằng, thay vì thêm một baseline centralized-k3 riêng.

## 4. Chia dữ liệu

Thiết kế leave-one-out (không tách train-target / train-demo-pool riêng — một mẫu có thể vừa là target vừa là demo cho mẫu khác, miễn loại chính nó và near-duplicate khỏi candidate retrieval).

| Phần | Số lượng/dataset | Vai trò | Ràng buộc |
|---|---|---|---|
| `train-core` | 600 | Vừa là SFT target, vừa là demo pool (leave-one-out) | train-k0 và train-k5 tiếp xúc cùng 600 QA |
| `dev-query` / `dev-support` | 100 / 100 | Chọn checkpoint/hyperparameter (early-stopping theo dev accuracy) | Không bao giờ dùng để tính kết quả cuối |
| `test-query` / `test-support` | 300 / 300 | Đánh giá cuối cùng; `test-support` chỉ dùng để retrieval demo lúc test | **Tuyệt đối không dùng để chọn prompt, checkpoint, hoặc hyperparameter** |
| **Tổng** | **1400** | | |

- **Federated:** `train-core` chia đều 200/client (IID random split — quyết định có chủ đích: dùng IID trước để debug cơ chế FedAvg, non-IID theo subject để dành cho full-scale). Mỗi client chỉ retrieve leave-one-out trong 199 mẫu local còn lại của chính nó — không truy cập dữ liệu client khác (giữ đúng tinh thần data-locality).
- **Audit bắt buộc** (xem `data/audit_splits.py` ở mục 9): loại near-duplicate **xuyên suốt cả 5 phần**, không chỉ trong từng phần — vì một câu hỏi gần giống có thể xuất hiện lặp ở nguồn dữ liệu gốc.

## 5. Prompt & matching

Thay đổi lớn nhất so với bản nháp đầu: **candidate phải luôn xuất hiện trong prompt** (nhiều câu MedQA dạng "Which of the following…" vô nghĩa nếu model không thấy 4 lựa chọn).

```
prompt = question + 4 candidates (hiển thị rõ nhãn A/B/C/D) + k retrieved demos
→ SLM generate free-text
→ parse: tìm nhãn/text khớp exact-normalized với 1 trong 4 candidate
→ nếu parse thất bại → fallback: BioBERT embedding-match (mục 7)
```

Nhờ candidate luôn hiển thị trong prompt, mốc random-chance = 25% là hợp lệ để dùng làm sàn tham chiếu.

## 6. MedQuAD → constructed 4-way MCQ

- Distractor: **semi-hard negative**, không dùng random distractor cho thí nghiệm chính (chỉ dùng random cho smoke test nội bộ).
  - Cùng topic/source, độ dài tương đồng, gần về ngữ nghĩa — lấy qua BioBERT retrieval (near-neighbor), sau đó loại các ứng viên near-duplicate với gold.
- Candidate set **cố định một lần**, lưu vào `manifests/`, dùng chung cho mọi model/baseline/seed — không tạo lại ngẫu nhiên mỗi lần chạy.
- Vị trí gold trong 4 lựa chọn được **cân bằng ngẫu nhiên** A/B/C/D (seed cố định, tránh positional bias).
- Metric báo cáo phải gọi đúng tên **"constructed 4-way MedQuAD accuracy"**, không phải "open-ended MedQuAD QA accuracy" — tránh overclaim.

## 7. Hiệu chuẩn BioBERT (matcher fallback + MedQuAD distractor retrieval)

Encoder: `dmis-lab/biobert-base-cased-v1.2`. Đây là biomedical language model, **không phải** sentence-embedding model huấn luyện sẵn cho cosine similarity — cần hiệu chuẩn trước khi tin số liệu.

- Masked mean pooling: loại padding token và special token trước khi average.
- L2-normalize nhất quán cho mọi embedding (câu hỏi, demo, candidate).
- Chính sách xử lý MedQuAD answer dài quá 512 token (giới hạn BERT): cắt tại ranh giới câu gần 512 token nhất, không cắt giữa câu — ghi rõ trong code và log số câu bị cắt.
- Audit tay 50–100 output: so % BioBERT chọn đúng candidate với nhãn do người đọc xác định (`eval/calibrate_matcher.py`).
- Báo cáo tỷ lệ % trường hợp phải dùng embedding fallback (tức parse exact-match thất bại) — tỷ lệ cao tự nó là một finding cần nêu, không được ỉm đi.
- Thêm metric **gold similarity margin** = `sim(gold) − max(sim(negative))`, thông tin hơn so với chỉ mean similarity-to-gold.
- Accuracy đo được **không được diễn giải là năng lực QA thuần túy** cho tới khi matcher đã được kiểm chứng qua audit tay ở trên.

## 8. Success criteria

Tách rõ hai loại, không gộp chung một tiêu chí duy nhất:

**Operational (điều kiện cần để coi pipeline chạy được):**
- Toàn bộ grid (2 model × 2 dataset × 5 baseline × 3 seed) chạy xong không lỗi.
- Audit split xác nhận: không overlap, không near-duplicate xuyên phần.
- Kết quả tái lập được với cùng seed (trong dung sai hợp lý).

**Scientific (contrast định trước, báo cáo 95% CI qua bootstrap trên `test-query`, không dùng point estimate đơn lẻ):**
- `train-k5/eval-k5 − train-k0/eval-k5` — ICL lúc train có giúp không (giữ eval-k=5 cố định).
- `ICL-only(k5) − zero-shot` — retrieval-ICL tự thân đóng góp gì, không cần train.
- `FL-k5/eval-k5 vs train-k5/eval-k5` — chi phí liên bang hóa; **pass** nếu FL không thấp hơn quá **5 điểm phần trăm** (ngưỡng có thể điều chỉnh sau khi có số liệu thật), luôn báo cáo kèm CI.
- Cận dưới của CI 95% accuracy mỗi arm đã train phải > 25% (random-chance floor, hợp lệ vì candidate luôn trong prompt).
- `zero-shot ≤ train-k5/eval-k5` là **giả thuyết cần kiểm chứng**, không phải điều kiện go/no-go bắt buộc — SFT làm giảm chất lượng cũng là một kết quả khoa học hợp lệ, không phải "pipeline lỗi".

## 9. Cấu trúc code

```
Version_3/pilot/
  configs/                     # 1 config / ô thí nghiệm (model × dataset × baseline × seed)
  data/
    prepare_datasets.py        # tải, subsample, chia 5 phần (train-core/dev-*/test-*)
    partition_federated.py     # chia train-core 200/client (IID random), 3 client
    audit_splits.py            # kiểm tra overlap/near-duplicate xuyên phần, class/position balance
  retrieval/
    encoder.py                 # BioBERT wrapper: masked mean pool, L2-norm, chính sách 512-token
    retriever.py                # top-k cosine, leave-one-out exclusion (loại self + near-dup)
  train/
    lora_sft.py                 # LoRA SFT centralized (k0/k5 qua config)
    federated_lora.py           # FedAvg loop, 3 client, k=5 cục bộ
  eval/
    generate_and_match.py       # parse-first (exact-normalized) → BioBERT fallback
    calibrate_matcher.py        # kiểm chứng BioBERT matching bằng nhãn người (50–100 mẫu)
    metrics.py                  # accuracy, CI bootstrap, gold similarity margin, fallback rate
  run_pilot.py                  # chạy toàn bộ grid, ghi kết quả
  tests/                        # unit test: retrieval exclusion, prompt format, FedAvg aggregation
  manifests/                    # IDs, seed, hash, candidate set cố định (đặc biệt cho MedQuAD)
  results/*.json
```

## 10. Quyết định đã chốt qua trao đổi (tóm tắt để tránh lặp lại tranh luận)

- Compute: Cloud GPU A5000 đã cấp phát (không dùng chung ngân sách 120 GPU-giờ của full-scale).
- Model: Qwen2.5-0.5B-Instruct + SmolLM2-360M-Instruct — khác family để test generality.
- Dataset: MedQA-USMLE + MedQuAD (không dùng MedMCQA/PubMedQA cho pilot này).
- Eval method: sinh text tự do → parse trước → BioBERT embedding-match làm fallback (không dùng conditional log-likelihood scoring như full-scale design — đây là khác biệt phương pháp luận có chủ đích cho pilot, ECE bị loại khỏi metric vì không còn phù hợp).
- Retrieval demo: top-k cosine similarity qua BioBERT, chưa có redundancy/subject-prior term (để dành cho full-scale).
- FL: 3 client, IID random split (chủ đích tách riêng việc test cơ chế FedAvg khỏi việc test non-IID skew).
- Grid: full factorial, không rút gọn.
- FL k: đổi từ k=3 sang k=5 để so sánh công bằng 1-đối-1 với train-k5/eval-k5, thay vì thêm baseline centralized-k3 riêng.

## 11. Việc chưa quyết — cần làm rõ ở bước implementation plan

- Git: thư mục `Version_3` hiện chưa là git repo. Spec này được lưu file thường, **chưa commit**. Cần xác nhận có `git init` trước khi implement hay không.
- Hyperparameter cụ thể cho LoRA (rank, learning rate, epoch, số round FL) — mới chỉ có định hướng (rank nhỏ, ví dụ r=8), cần chốt số cụ thể ở implementation plan, có thể dùng `dev-query`/`dev-support` để chọn.
- Cơ chế near-duplicate detection cụ thể (ngưỡng lexical/semantic similarity nào được coi là "near-duplicate") — cần định nghĩa rõ trong `audit_splits.py`.
