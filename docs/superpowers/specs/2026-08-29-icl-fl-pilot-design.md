# FedICL-MQA — Pilot thu nhỏ: kiểm tra tính khả thi ICL centralized → federated

Status: Approved design (rev 2), sẵn sàng cho implementation plan
Date: 2026-08-29 (rev 2)
Liên quan: [fedicl_mqa_paper_core_sections.md](../../fedicl_mqa_paper_core_sections.md) (design full-scale gốc)

Rev 2 sửa một lỗ hổng cốt lõi của rev 1: Central và FL trước đây không đối chứng công bằng (retrieval pool khác kích thước), thuật ngữ "pure ICL effect" bị lạm dụng, chưa khóa data provenance, chưa có prompt/context contract, chưa khóa FL contract, evaluator chỉ báo một con số accuracy, và seed/thống kê gây lãng phí + overclaim. Toàn bộ được sửa trong rev này.

## 1. Mục tiêu

Kiểm tra tính khả thi của bài toán ICL (in-context learning) xuyên suốt pipeline từ centralized đến federated learning, dùng model cực nhỏ (≤1B) và quy mô dữ liệu nhỏ, **trước khi** đầu tư compute lớn cho full-scale paper (Qwen2.5-3B, MedMCQA 5-client, 120 GPU-giờ đã note trong nghiên cứu trước).

Đây là feasibility pilot (architectural scope), không phải bản thu nhỏ 1:1 của full-scale design. Một số quyết định phương pháp luận (cách chọn đáp án, cách chia dữ liệu) **khác** với `fedicl_mqa_paper_core_sections.md` — được ghi chú rõ ở từng mục.

## 2. Hạ tầng & phạm vi

- Compute: Cloud GPU A5000 (đã cấp phát), tách riêng khỏi ngân sách 120 GPU-giờ của full-scale.
- 2 model: **Qwen2.5-0.5B-Instruct** và **SmolLM2-360M-Instruct**.
- 2 dataset: **MedQA-USMLE** (trắc nghiệm, official splits) và **MedQuAD** (tự luận → constructed 4-way MCQ, mục 6).
- Thiết kế: **crossed model × dataset matrix trên 5 arm chính, với số seed riêng theo từng arm** (không còn gọi "full factorial 20 ô × 3 seed" — cách nói đó gây chạy lặp vô ích cho arm không train, xem mục 12).
- Ngoài 5 arm chính có 2 **diagnostic run** phạm vi hẹp (mục 9), không nằm trong crossed matrix.
- Tổng số run ước tính: 4 (zero-shot) + 4 (ICL-only) + 12 (train-k0/eval-k5) + 12 (demo-conditioned SFT) + 12 (Federated-k5) + 1 (diagnostic random-k5) + 1 (diagnostic Fed train-k0) = **46 run**.
- Ước tính compute ban đầu: 15–25 GPU-giờ — **là giả thuyết**, đo lại sau khi chạy xong 1 ô đầu tiên. Với MedQuAD, độ dài sequence (câu trả lời dài + k=5 demo) nhiều khả năng là yếu tố chi phối chi phí hơn số lượng mẫu.

## 3. Baseline (5 arm chính + 2 diagnostic)

| # | Arm | Train | Eval | Mục đích |
|---|---|---|---|---|
| 1 | Zero-shot | Không train | k=0, không demo | Sàn dưới cùng |
| 2 | ICL-only (retrieval-k5) | Không train | k=5 demo retrieval, client-scoped | Đóng góp của retrieval-ICL không cần train |
| 3 | train-k0 / eval-k5 | LoRA SFT, prompt trơn (không demo) | k=5 demo retrieval | SFT không tiếp xúc ICL lúc train |
| 4 | **demo-conditioned SFT (train-k5) / eval-k5** | LoRA SFT, mỗi prompt train có k=5 demo (leave-one-out, client-scoped) | k=5 demo retrieval | SFT có tiếp xúc ICL lúc train |
| 5 | Federated-k5 / eval-k5 | 3 client, FedAvg, mỗi client k=5 demo **chỉ từ pool cục bộ của chính client đó** | k=5 demo retrieval | So trực tiếp với #4 (matched-compute, mục 5) để cô lập đúng chi phí liên bang hóa |
| D1 | Diagnostic: ICL-only random-k5 | Không train | k=5 demo **random** (không retrieval) | Cô lập hiệu ứng chất lượng retrieval so với random demo — phạm vi hẹp: Qwen2.5-0.5B + MedQA, 1 run |
| D2 | Diagnostic: Fed train-k0 / eval-k5 | 3 client, FedAvg, **prompt trơn lúc train** (không demo) | k=5 demo retrieval | Đúng code path dự kiến của full-scale (FedAvg-LoRA rồi retrieval-ICL lúc inference, xem `fedicl_mqa_paper_core_sections.md`) — phạm vi hẹp: Qwen2.5-0.5B + MedQA, 1 run |

**Sửa thuật ngữ quan trọng:** arm #4 gọi là **demo-conditioned SFT (train-k5)**, không phải "pure ICL". Contrast `train-k5 − train-k0` đo hiệu ứng của cả gói gồm demo context, repeated exposure, và compute bổ sung — **không được diễn giải là hiệu ứng ICL thuần túy**.

**Lịch sử quyết định:** FL ban đầu dự định k=3, đổi sang k=5 để so khớp trực tiếp với arm #4 (matched-k). Nhưng việc so khớp k không đủ — cần matched-compute đầy đủ (mục 5).

## 4. Data provenance & partitioning

### 4.1 MedQA-USMLE

**Bảo toàn official train/dev/test splits** của MedQA repo — không pool rồi chia lại.

- `train-core` (600): subsample từ official train split.
- `dev-query` + `dev-support` (100 + 100): subsample từ official dev split.
- `test-query` + `test-support` (300 + 300): subsample từ official test split.

### 4.2 MedQuAD (không có official train/dev/test split sẵn)

Pipeline provenance bắt buộc theo đúng thứ tự:

1. Filter và log các câu missing/empty answer.
2. Group theo (source document/URL, focus/CUI, near-duplicate cluster).
3. **Group-split trước khi subsample** — chia nhóm (không chia lẻ từng item trong nhóm) thành train/dev/test-role ở cấp nhóm, đảm bảo một nguồn tài liệu không xuất hiện ở nhiều vai trò.
4. Subsample trong từng vai trò xuống đúng số lượng mục tiêu (600/200/600).
5. **Sau đó mới xây candidate** (distractor) — không xây candidate trước khi split.
6. Human-audit tính "chỉ có một đáp án đúng" trên candidate set đã xây.
7. **Seal manifest trước khi mở test** — đóng băng candidate set + split assignment, không sửa sau khi bắt đầu chạy test.

Tham khảo cấu trúc nguồn tại MedQuAD repository và official splits tại MedQA repository trong bước implementation.

### 4.3 Client-scoping — áp dụng cho TẤT CẢ các phần, không chỉ train-core

Đây là điểm sửa cốt lõi so với rev 1: rev 1 chỉ chia client cho `train-core` (FL), khiến centralized retrieve từ 599 mẫu trong khi FL chỉ có 199 mẫu/client — khác biệt không chỉ đến từ FedAvg mà còn từ kích thước pool.

- Gán `client_id` cố định (1 trong 3) cho **toàn bộ** `train-core`, `dev-query`, `dev-support`, `test-query`, `test-support` của cả hai dataset.
- Với MedQuAD, việc gán client tôn trọng ranh giới nhóm ở mục 4.2 (một nhóm nguồn không bị chia giữa các client).
- **Mọi arm có ICL** (kể cả centralized: ICL-only, train-k0/eval-k5, train-k5/eval-k5) chỉ retrieve demo từ support-pool **cùng client_id** với query — không chỉ FL mới bị giới hạn phạm vi retrieval.
- MedQuAD distractor cho dev/test-query chỉ lấy từ support cùng role + cùng client. Candidate source không được trùng nguồn của query, và không được trùng nguồn của bất kỳ demo nào đã retrieve vào cùng prompt đó (tránh shortcut chọn theo nguồn thay vì nội dung).

## 5. Matched-compute contract: Central vs Federated

Để contrast `Federated-k5 vs train-k5` (arm #5 vs #4) xứng đáng gọi là **matched-compute Central vs FedAvg**, bắt buộc:

1. **Precompute canonical prompt manifest**: với mỗi target example trong `train-core`, retrieve k=5 demo chỉ từ 199 mẫu còn lại **cùng client_id** — tính một lần, lưu vào `manifests/`.
2. **Centralized (arm #4) train trên chính union của các local prompt manifest đó** — tức là mỗi example khi train ở centralized dùng đúng demo set đã được retrieve theo phạm vi client của nó, chỉ khác là toàn bộ 600 example được gộp vào một optimizer duy nhất thay vì 3 client + FedAvg.
3. **FL (arm #5) phân phối đúng cùng manifest** theo client_id — không tính lại retrieval riêng cho FL.
4. Log và so khớp giữa Central và FL: tổng target exposures, tổng token xử lý, số optimizer update, effective batch size — hai arm chỉ được khác nhau ở cơ chế aggregation (FedAvg vs single pooled optimizer), không khác ở lượng dữ liệu/tính toán tiếp xúc.

## 6. Prompt & Context Contract

Chính sách 512-token ở mục 8 chỉ áp dụng cho BioBERT — cần contract riêng cho prompt của SLM (k=5 demo của MedQuAD có thể rất dài).

- **Một canonical textual prompt** dùng chung cho cả hai model (cùng text, chỉ khác phần chat-template bắt buộc của từng tokenizer).
- Thứ tự cố định: `system → 5 demos → query + candidates → final-answer instruction`.
- Format output chuẩn để parse: `Final answer: <A|B|C|D>` — instruct model chỉ trả lời đúng format này, không kèm giải thích dài.
- Decoding: `do_sample=false` (greedy), deterministic.
- **Ngân sách token** (đề xuất, cần xác nhận ở implementation plan): `max_input_tokens = 2048`, reserve 32–64 token cho output — áp dụng cố định cho cả hai model bất kể context window gốc lớn hơn, để giới hạn runtime/VRAM.
- Mỗi manifest được **tokenize thử bằng cả hai tokenizer** (Qwen2.5 và SmolLM2) trước khi seal; lấy số token lớn hơn giữa hai model để quyết định overflow.
- **100% mẫu trong crossed matrix cuối cùng phải giữ đúng `effective_k=5`** — không được truncate khác nhau theo model, không được âm thầm drop demo.
- Nếu một item overflow ngân sách token ở k=5: phải sửa representation (rút gọn demo/candidate) hoặc **loại item khỏi manifest trước khi seal**, và báo cáo **exclusion rate** theo dataset/model — không truncate ngầm như một fallback.
- **SFT loss masking**: chỉ tính loss trên target-answer span; toàn bộ phần prompt + demo đặt `label = -100`.

## 7. MedQuAD → constructed 4-way MCQ

- Distractor: **semi-hard negative** (không dùng random distractor cho thí nghiệm chính, chỉ dùng cho smoke test nội bộ) — cùng topic/source, độ dài tương đồng, gần về ngữ nghĩa qua BioBERT retrieval, loại near-duplicate của gold.
- Ràng buộc nguồn distractor: xem mục 4.3 (cùng role + cùng client, không trùng nguồn query/demo trong cùng prompt).
- Candidate set **cố định một lần sau khi seal manifest** (mục 4.2 bước 7), dùng chung cho mọi model/baseline/seed.
- Vị trí gold cân bằng ngẫu nhiên A/B/C/D, seed cố định.
- Metric báo cáo phải gọi đúng tên **"constructed 4-way MedQuAD accuracy"**, không phải "open-ended MedQuAD QA accuracy".

## 8. Hiệu chuẩn BioBERT

Encoder: `dmis-lab/biobert-base-cased-v1.2` — không phải sentence-embedding model huấn luyện sẵn cho similarity, cần hiệu chuẩn trước khi tin số liệu.

- Masked mean pooling (loại padding + special token), L2-normalize nhất quán.
- MedQuAD answer > 512 token (giới hạn BERT): cắt tại ranh giới câu gần 512 token nhất, log số câu bị cắt.
- Audit tay 50–100 output tổng quát (`eval/calibrate_matcher.py`).
- **Rủi ro circularity**: BioBERT vừa tạo semi-hard distractor (mục 7) vừa làm fallback judge (mục 9) — audit bắt buộc cả hai chiều: MedQuAD single-correctness (mục 4.2 bước 6) và evaluator agreement (mục 9).

## 9. Evaluator — phân rã 5 tầng, không chỉ báo một accuracy tổng

Giữ generate → parse → BioBERT-fallback làm evaluator chính, nhưng phải báo đủ 5 tầng, không gộp chung:

1. **Strict parsed-label accuracy** — chỉ tính trên các mẫu parse thành công (`Final answer: <X>` khớp exact-normalized).
2. **Fallback-assisted pipeline accuracy** — accuracy tổng, bao gồm cả các mẫu được BioBERT fallback resolve.
3. **Fallback rate** — % mẫu phải dùng fallback (parse thất bại). Tỷ lệ cao tự nó là một finding cần nêu rõ, không ỉm đi.
4. **Human agreement trên fallback/ambiguous outputs** — audit tay riêng cho tập con đã đi qua fallback (khác với audit tổng quát 50–100 mẫu ở mục 8).
5. **Conditional-likelihood accuracy và agreement matrix trên test** — chạy thêm một pass scoring kiểu full-scale gốc (`argmax P(option|prompt)`, không qua generation) trên test set, và báo ma trận đồng thuận giữa 3 phương pháp: parsed / fallback-assisted / conditional-likelihood.

Tên metric chính gọi là **"pipeline answer-selection accuracy"**, không gọi chung là "model QA accuracy" cho tới khi matcher đã qua ngưỡng agreement được đóng băng trước test (đóng băng ở bước implementation plan, dùng dev set).

## 10. FL contract — khóa trước khi chạy test

- Constant learning rate trong suốt các round (không schedule/decay).
- Client optimizer **reset sau mỗi round** (không giữ state giữa các round).
- Cả 3 client tham gia mỗi round (full participation).
- Central và FL dùng **cùng base model và cùng LoRA initialization** — khác biệt duy nhất là cơ chế aggregation.
- Aggregate LoRA parameter **weighted theo số target example** của mỗi client.
- Server **không giữ optimizer state** (pure parameter averaging).
- Local epochs, số round, batch size, gradient accumulation, LR search space: chọn trên dev set, sau đó **freeze** trước khi chạm vào test.
- Unit test bắt buộc:
  - Một client FL chạy đơn lẻ (1 client) phải cho kết quả gần tương đương centralized local training trên đúng dữ liệu client đó.
  - Weighted aggregation phải khớp kết quả tính tay trên một toy case.

## 11. Success criteria

**Operational:**
- Toàn bộ 46 run (mục 2) chạy xong không lỗi.
- Audit split xác nhận: không overlap/near-duplicate xuyên phần, xuyên client.
- Kết quả tái lập được với cùng seed.

**Scientific — hạ thành exploratory estimates** (không phải confirmatory claim), báo cáo kèm 95% CI qua bootstrap **hai tầng** (resample cả item và training-seed) trên `test-query`:
- `train-k5 − train-k0` (demo-conditioned SFT effect, giữ eval-k=5 cố định) — exploratory estimate.
- `ICL-only(retrieval-k5) − zero-shot` — exploratory estimate.
- `ICL-only(retrieval-k5) − ICL-only(random-k5)` — exploratory estimate, dùng diagnostic D1.
- `zero-shot ≤ train-k5` — **giả thuyết cần kiểm chứng**, không phải điều kiện go/no-go — SFT làm giảm chất lượng cũng là kết quả khoa học hợp lệ.

**Một điều kiện confirmatory duy nhất** — non-inferiority Federated-k5 vs train-k5 (matched-compute, mục 5):
- Margin **−5 điểm phần trăm phải được freeze trước khi chạm test** (không còn "có thể điều chỉnh sau khi có số liệu thật" như rev 1 — câu đó đã bị xóa).
- Kiểm tra bằng **cận dưới của CI một phía (one-sided)**, tính bằng bootstrap hai tầng có tính đến cả **item variance và training-seed variance**.
- Cận dưới CI 95% accuracy mỗi arm đã train phải > 25% (random-chance floor — hợp lệ vì candidate luôn hiển thị trong prompt, mục 6).

## 12. Seed policy theo arm (thay cho "full factorial × 3 seed")

Đúng tinh thần: **crossed model × dataset matrix trên 5 arm chính, số seed riêng theo từng arm**.

| Arm | Phạm vi | Seed |
|---|---|---|
| Zero-shot | 2 model × 2 dataset | 1 (deterministic, greedy) |
| ICL-only (retrieval-k5) | 2 model × 2 dataset | 1 (deterministic) |
| train-k0/eval-k5 | 2 model × 2 dataset | 3 |
| demo-conditioned SFT (train-k5)/eval-k5 | 2 model × 2 dataset | 3 |
| Federated-k5/eval-k5 | 2 model × 2 dataset | 3 |
| D1: ICL-only random-k5 | Qwen2.5-0.5B × MedQA-USMLE | 1 |
| D2: Fed train-k0/eval-k5 | Qwen2.5-0.5B × MedQA-USMLE | 1 |

## 13. Cấu trúc code

```
Version_3/pilot/
  configs/                        # 1 config / run
  data/
    prepare_medqa.py              # giữ official train/dev/test split
    prepare_medquad.py            # filter, group theo source/CUI/near-dup, group-split, subsample
    build_candidates.py           # xây candidate SAU split (MedQuAD semi-hard negative)
    assign_clients.py             # gán client_id cho toàn bộ train-core/dev-*/test-* (cả 2 dataset)
    audit_splits.py                # overlap/near-duplicate xuyên phần+client, class/position balance
    audit_medquad_correctness.py  # human-audit "chỉ một đáp án đúng"
  retrieval/
    encoder.py                    # BioBERT wrapper: masked mean pool, L2-norm, chính sách 512-token
    retriever.py                   # top-k cosine, leave-one-out + client-scoped exclusion
  prompt/
    build_manifest.py             # canonical prompt manifest (mục 5), dual-tokenizer overflow check
    template.py                    # canonical prompt template dùng chung 2 model
  train/
    lora_sft.py                    # LoRA SFT centralized (k0/k5 qua config, loss mask -100)
    federated_lora.py              # FedAvg loop, FL contract mục 10
  eval/
    generate_and_match.py          # parse-first (exact-normalized) → BioBERT fallback
    likelihood_score.py            # conditional log-likelihood scoring (tầng 5, mục 9)
    calibrate_matcher.py           # audit tay 50-100 mẫu tổng quát + audit tay fallback subset
    metrics.py                     # 5 tầng accuracy, CI bootstrap 2 tầng, gold similarity margin
  run_pilot.py                     # chạy crossed matrix + 2 diagnostic, ghi kết quả
  tests/                           # retrieval exclusion, prompt format/effective_k, FedAvg (mục 10)
  manifests/                       # IDs, seed, hash, candidate set, client_id, prompt manifest (seal trước test)
  results/*.json
```

## 14. Quyết định đã chốt qua trao đổi (tóm tắt, tránh lặp lại tranh luận)

- Compute: Cloud GPU A5000 đã cấp phát.
- Model: Qwen2.5-0.5B-Instruct + SmolLM2-360M-Instruct.
- Dataset: MedQA-USMLE (official splits) + MedQuAD (group-split theo provenance).
- Eval: generate → parse-first → BioBERT fallback, **cộng thêm** conditional-likelihood scoring làm tầng đối chiếu (không thay thế, để đối chiếu agreement).
- Retrieval demo: top-k cosine qua BioBERT, client-scoped ở mọi arm ICL (không chỉ FL).
- FL: 3 client, IID random split ở cấp client assignment (chủ đích tách test cơ chế FedAvg khỏi test non-IID skew), nhưng client-scoped retrieval là bắt buộc cho mọi arm.
- FL k: 5 (khớp arm #4), cộng matched-compute contract (mục 5) để so sánh công bằng thật sự — không chỉ khớp k.
- Thêm arm ICL-only (mục 3, #2) và 2 diagnostic (D1 random-k5, D2 Fed train-k0/eval-k5).
- Seed: theo arm, không đồng loạt 3 seed (mục 12).
- Non-inferiority margin FL vs Central: −5pp, freeze trước test, kiểm bằng one-sided CI lower bound với bootstrap 2 tầng (item + training-seed variance).

## 15. Việc chưa quyết — cần làm rõ ở bước implementation plan

- Git: đã init, đã push rev 1 lên `git@github.com:nguyenhoangvuthan/FedICL-MedQA_Version3.git` (branch `main`). Rev 2 này chưa commit/push — chờ xác nhận.
- Hyperparameter cụ thể cho LoRA (rank, learning rate, epoch, số round FL) — chọn trên dev set rồi freeze theo mục 10, số cụ thể chưa chốt.
- Ngưỡng near-duplicate detection cụ thể (lexical/semantic similarity threshold nào được coi là "near-duplicate") — cần định nghĩa rõ trong `data/audit_splits.py`.
- `max_input_tokens = 2048` (mục 6) là đề xuất mặc định, cần xác nhận hoặc điều chỉnh.
- Ngưỡng agreement cụ thể để "đóng băng" matcher trước khi gọi kết quả là "pipeline answer-selection accuracy" chính thức (mục 9) — chưa có số cụ thể, cần chốt ở implementation plan dựa trên audit tay trên dev.
