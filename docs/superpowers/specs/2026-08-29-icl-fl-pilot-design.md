# FedICL-MQA — Pilot thu nhỏ: kiểm tra tính khả thi ICL centralized → federated

Status: Approved design (rev 3), sẵn sàng cho implementation plan
Date: 2026-08-29 (rev 3)
Liên quan: [fedicl_mqa_paper_core_sections.md](../../fedicl_mqa_paper_core_sections.md) (design full-scale gốc)

Rev 3 vá 10 lỗ hổng còn lại sau rev 2: DAG tạo artifact chưa có thứ tự duy nhất (candidate/demo mâu thuẫn), retrieval chưa định nghĩa closure constraint (kể cả retriever thấy gold label), client-size bị hard-code, evaluator denominator có thể bị thổi phồng, non-inferiority tự nhận "confirmatory" nhưng không đủ chuẩn, estimand Central–FL mô tả sai (bỏ qua local optimizer steps), thiếu feasibility/systems metrics, compute gate đánh giá thấp workload, operational gate quá mỏng, và git metadata bị stale. Sau rev này, các hạn chế còn lại của thiết kế (3 client IID, 3 training seed, BioBERT circularity, khác biệt với full-scale) được coi là đã giới hạn đúng bằng khung exploratory — không còn là blocker cho implementation plan.

## 1. Mục tiêu

Kiểm tra tính khả thi của bài toán ICL (in-context learning) xuyên suốt pipeline từ centralized đến federated learning, dùng model cực nhỏ (≤1B) và quy mô dữ liệu nhỏ, **trước khi** đầu tư compute lớn cho full-scale paper (Qwen2.5-3B, MedMCQA 5-client, 120 GPU-giờ đã note trong nghiên cứu trước).

Đây là feasibility pilot (architectural scope), không phải bản thu nhỏ 1:1 của full-scale design. Một số quyết định phương pháp luận khác với `fedicl_mqa_paper_core_sections.md` — được ghi chú rõ ở từng mục.

## 2. Hạ tầng & phạm vi

- Compute: Cloud GPU A5000 (đã cấp phát), tách riêng khỏi ngân sách 120 GPU-giờ của full-scale.
- 2 model: **Qwen2.5-0.5B-Instruct** và **SmolLM2-360M-Instruct**.
- 2 dataset: **MedQA-USMLE** (trắc nghiệm, official splits) và **MedQuAD** (tự luận → constructed 4-way MCQ, mục 9).
- Thiết kế: **crossed model × dataset matrix trên 5 arm chính, với số seed riêng theo từng arm** — không dùng khung "full factorial × N seed".
- Ngoài 5 arm chính có 2 **diagnostic run** phạm vi hẹp (mục 3), không nằm trong crossed matrix.
- **46 là số experiment configuration, không phải tổng workload** — chưa tính tuning search, calibration/audit pass, retry, và pass eval thứ hai (conditional-likelihood scoring chạy riêng cho mọi arm đã train). Trình tự ước tính compute thật xem mục 16.

## 3. Baseline (5 arm chính + 2 diagnostic)

| # | Arm | Train | Eval | Mục đích |
|---|---|---|---|---|
| 1 | Zero-shot | Không train | k=0, không demo | Sàn dưới cùng |
| 2 | ICL-only (closure-constrained retrieval-k5) | Không train | k=5 demo, mục 6 | Đóng góp của retrieval-ICL không cần train |
| 3 | train-k0 / eval-k5 | LoRA SFT, prompt trơn (không demo) | k=5 demo, mục 6 | SFT không tiếp xúc ICL lúc train |
| 4 | **demo-conditioned SFT (train-k5) / eval-k5** | LoRA SFT, mỗi prompt train có k=5 demo (leave-one-out, closure-constrained) | k=5 demo, mục 6 | SFT có tiếp xúc ICL lúc train |
| 5 | Federated-k5 / eval-k5 | 3 client, full FedAvg protocol (mục 12), mỗi client k=5 demo chỉ từ pool cục bộ | k=5 demo, mục 6 | Matched-compute contrast với #4 (mục 7) |
| D1 | Diagnostic: ICL-only random-k5 | Không train | k=5 demo **random**, 1 manifest cố định | Smoke diagnostic (không phải estimate tổng quát) — Qwen2.5-0.5B + MedQA, 1 run |
| D2 | Diagnostic: Fed train-k0 / eval-k5 | 3 client, FedAvg, prompt trơn lúc train | k=5 demo, mục 6 | Đúng code path dự kiến của full-scale (FedAvg-LoRA rồi retrieval-ICL lúc inference) — Qwen2.5-0.5B + MedQA, 1 run |

**Thuật ngữ:** arm #4 gọi là **demo-conditioned SFT (train-k5)**, không phải "pure ICL". Contrast `train-k5 − train-k0` đo hiệu ứng của cả gói gồm demo context, repeated exposure, và compute bổ sung.

**Định dạng demo:** mỗi demo giữ đầy đủ `question + 4 options + gold label` — task-isomorphic với query, không rút gọn thành chỉ câu hỏi+đáp án tự do.

## 4. Data provenance & partitioning

### 4.1 MedQA-USMLE

**Bảo toàn official train/dev/test splits** của MedQA repo — không pool rồi chia lại.

- `train-core` (600): subsample từ official train split.
- `dev-query` + `dev-support` (100 + 100): subsample từ official dev split.
- `test-query` + `test-support` (300 + 300): subsample từ official test split.

Gọi đúng tên phần test là **"labeled in-domain support carved from the official test split"** — đây không phải standard untouched MedQA test evaluation, vì đã subsample + client-partition. Không so sánh trực tiếp con số với literature benchmark trên full official test set.

### 4.2 MedQuAD (không có official split sẵn)

1. Filter và log các câu missing/empty answer.
2. Group theo (source document/URL, focus/CUI, near-duplicate cluster).
3. **Group-split trước khi subsample** — chia nhóm (không chia lẻ item) thành train/dev/test-role.
4. Subsample trong từng vai trò xuống đúng số lượng mục tiêu (600/200/600), **có reserve pool** phía trên số mục tiêu để phục vụ refill (mục 5) mà không thu hẹp cohort cuối cùng.
5. Human-audit tính "chỉ có một đáp án đúng" trên các nhóm/ứng viên tiềm năng (trước khi candidate cuối được xây trong DAG ở mục 5).

Tham khảo cấu trúc nguồn tại MedQuAD repository và official splits tại MedQA repository ở bước implementation.

### 4.3 Client-scoping — áp dụng cho TẤT CẢ các phần

- Gán `client_id` cố định (1 trong 3) cho toàn bộ `train-core`, `dev-query`, `dev-support`, `test-query`, `test-support` của cả hai dataset, bằng **deterministic group-aware bin packing/stratification** — không giả định chia đều tuyệt đối. Với MedQuAD, việc gán tôn trọng ranh giới nhóm ở mục 4.2 (một nhóm nguồn không bị chia giữa các client).
- Kích thước pool cục bộ mỗi client gọi là `n_i` (không hard-code bằng 200); khi retrieve leave-one-out trong pool của chính nó, pool khả dụng là `n_i − 1`, **không giả định luôn bằng 199**.
- Query-ID và support-ID trong mỗi role/client phải **disjoint rõ ràng**, kiểm tra bằng script (mục 17).
- Kiểm tra đủ demo/distractor hợp lệ cho từng role/client **trước** khi bắt đầu candidate/retrieval (dùng reserve pool ở 4.2 bước 4 để refill nếu thiếu).
- **Mọi arm có ICL** (kể cả centralized) chỉ retrieve demo từ support-pool cùng `client_id` với query — không chỉ FL mới bị giới hạn phạm vi.

## 5. Artifact construction DAG — thứ tự duy nhất, bắt buộc

Đây là điểm sửa cốt lõi của rev 3: rev 2 mô tả candidate-building (mục 4.2 cũ) và demo-retrieval (mục 5 cũ) như hai quá trình tách rời, khiến ràng buộc "candidate không trùng nguồn demo" không thể enforce được (candidate được xây trước khi demo tồn tại). Từ rev 3, toàn bộ artifact được xây theo đúng một DAG:

```
raw + hash
  → split / group (mục 4.2)
  → client assignment (mục 4.3)
  → build candidate sets (mục 9 — bao gồm distractor semi-hard cho MedQuAD)
  → closure-constrained demo selection (mục 6)
  → build final prompts (mục 8 — canonical template)
  → dual-tokenizer fit check (mục 8)
  → human audit TRÊN ĐÚNG displayed text (không audit candidate abstract, audit chính text sẽ hiển thị cho model)
  → refill nếu lỗi (dùng reserve pool, mục 4.2)
  → seal cohort / candidate / prompt manifest
```

- **Refill = chạy lại toàn bộ vòng candidate → retrieval → fit → audit** cho item bị lỗi — không patch cục bộ một bước rồi giữ nguyên kết quả các bước khác (tránh trạng thái không nhất quán).
- Không có bước nào trong DAG được seal riêng lẻ trước bước trước nó hoàn tất cho toàn bộ cohort.
- Manifest chỉ seal sau khi toàn bộ DAG (kể cả audit + refill) đã pass cho 100% item còn lại trong cohort cuối.

## 6. Closure-constrained retrieval-k5

Đổi tên chính xác từ "cosine top-k" thành **closure-constrained retrieval-k5** — vì top-k thuần bị lọc thêm bởi các ràng buộc sau trước khi chấp nhận:

- **Retriever chỉ encode `question + candidates`, không được thấy gold label** — tránh retrieval học shortcut theo nội dung đáp án.
- Hard constraint khác nhau theo dataset:
  - **MedQuAD**: không lặp `example_id`, không lặp normalized-answer-hash, không lặp near-duplicate cluster (mục 4.2) trong cùng một prompt; source-document của query phải khác source-document của mọi demo được chọn.
  - **MedQA**: hard-ban question ID trùng/near-duplicate câu hỏi; **chỉ log** (không ban) option-text overlap, vì các lựa chọn phổ biến kiểu "none of the above" có thể lặp hợp lệ giữa các câu khác nhau.
- **Preflight bắt buộc**: trước khi seal, chứng minh mỗi query trong cohort còn đủ ≥5 demo hợp lệ sau khi áp toàn bộ closure constraint (nếu không đủ, dùng reserve pool để refill theo DAG mục 5, hoặc loại item — báo exclusion rate).
- **Log bắt buộc**: số demo bị closure filter loại theo từng query, rank gốc (trước filter) của các demo cuối cùng được giữ, và tỷ lệ query phải tìm sâu quá top-5/top-10 mới đủ 5 demo hợp lệ.

## 7. Matched-compute contract: Central vs Federated

**Estimand chính xác** (sửa từ rev 2, vốn nói sai là "chỉ khác aggregation"):

> Matched data/prompt/exposure comparison of pooled optimization versus the full FedAvg optimization protocol.

FL không chỉ khác Central ở cách gộp tham số cuối cùng — FL còn có local optimizer steps riêng theo từng client và optimizer reset mỗi round (client drift là hiệu ứng thuật toán thật, không chỉ là aggregation). Vì vậy contrast này so sánh hai **optimization protocol đầy đủ** dưới cùng dữ liệu/prompt/exposure, không phải một biến thể "giống hệt trừ một bước".

Để dữ liệu/prompt/exposure thật sự matched:

1. **Precompute canonical prompt manifest** qua DAG ở mục 5: với mỗi target trong `train-core`, closure-constrained retrieve k=5 demo chỉ từ pool cục bộ cùng `client_id` (kích thước `n_i − 1`).
2. **Centralized (arm #4) train trên union của chính các local prompt manifest đó** — mỗi example dùng đúng demo set đã retrieve theo phạm vi client của nó; khác biệt duy nhất với FL là 600 example được gộp vào một optimizer thay vì chạy qua full FedAvg protocol.
3. **FL (arm #5) phân phối đúng cùng manifest** theo `client_id` — không tính lại retrieval riêng.
4. Log và so khớp: tổng target exposures, tổng token xử lý, số optimizer update, effective batch size.

Khóa cấu trúc bổ sung (chi tiết hyperparameter cụ thể để dành implementation plan):

- Weighted FedAvg áp dụng **trực tiếp trên trainable LoRA tensor A/B** (không có tham số nào khác được aggregate).
- Cùng adapter initialization giữa Central và FL.
- Shared LoRA architecture (rank/alpha/target modules giống nhau giữa hai arm).
- Equal tuning budget hoặc shared search rule — Central và FL không được nhận lượng hyperparameter-search khác nhau.

## 8. Prompt & Context Contract

- **Một canonical textual prompt** dùng chung cho cả hai model (cùng text, chỉ khác chat-template bắt buộc của từng tokenizer).
- Thứ tự cố định: `system → 5 demos → query + candidates → final-answer instruction`.
- Demo hiển thị đầy đủ `question + 4 options + gold label` (mục 3) — task-isomorphic với phần query.
- Format output chuẩn: `Final answer: <A|B|C|D>` — instruct model chỉ trả lời đúng format này.
- Decoding: `do_sample=false` (greedy), deterministic.
- Ngân sách token đề xuất: `max_input_tokens = 2048`, reserve 32–64 token cho output — áp dụng cố định cho cả hai model (cần xác nhận ở implementation plan, mục 19).
- Mỗi manifest **tokenize thử bằng cả hai tokenizer** (Qwen2.5 và SmolLM2) trước khi seal; lấy số token lớn hơn giữa hai model để quyết định overflow.
- **100% mẫu trong crossed matrix cuối cùng phải giữ đúng `effective_k=5`** — không truncate khác nhau theo model, không âm thầm drop demo.
- Overflow ở k=5: sửa representation hoặc loại item khỏi manifest trước khi seal (theo DAG mục 5), báo cáo exclusion rate theo dataset/model.
- **SFT loss masking**: chỉ tính loss trên target-answer span; toàn bộ prompt + demo đặt `label = -100`.

## 9. MedQuAD → constructed 4-way MCQ

- Distractor: semi-hard negative — cùng topic, độ dài tương đồng, gần về ngữ nghĩa qua closure-constrained BioBERT retrieval (mục 6), loại near-duplicate của gold.
- **Ràng buộc nguồn** (đã sửa mâu thuẫn ở rev 2): candidate source không được trùng nguồn của query, và không được trùng nguồn của bất kỳ demo nào trong cùng prompt — ràng buộc "cùng topic" chỉ áp dụng ở mức chủ đề/ngữ nghĩa, không phải cùng source document. Việc này khả thi vì candidate-building xảy ra TRƯỚC demo-selection trong DAG (mục 5) nhưng demo-selection biết candidate set đã chọn để loại trùng nguồn — tức bước "closure-constrained demo selection" (mục 5) nhận candidate set làm input và áp constraint "≠ nguồn candidate" khi retrieve demo.
- Candidate set cố định một lần sau khi seal manifest (mục 5), dùng chung cho mọi model/baseline/seed.
- Vị trí gold cân bằng ngẫu nhiên A/B/C/D, seed cố định.
- Metric gọi đúng tên **"constructed 4-way MedQuAD accuracy"**, không phải "open-ended MedQuAD QA accuracy".

## 10. Hiệu chuẩn BioBERT

Encoder: `dmis-lab/biobert-base-cased-v1.2` — không phải sentence-embedding model huấn luyện sẵn cho similarity.

- Masked mean pooling (loại padding + special token), L2-normalize nhất quán.
- MedQuAD answer > 512 token: cắt tại ranh giới câu gần 512 token nhất, log số câu bị cắt.
- Audit tay 50–100 output tổng quát (`eval/calibrate_matcher.py`).
- **Circularity risk**: BioBERT vừa tạo semi-hard distractor (mục 9) vừa làm fallback judge (mục 11) vừa làm retriever (mục 6) — audit bắt buộc: MedQuAD single-correctness (mục 4.2 bước 5) và evaluator agreement (mục 11).

## 11. Evaluator — state machine 4 bước, 7 metric tách biệt

Sửa lỗi denominator của rev 2 (strict-parsed-accuracy tính trên tập con parse-thành-công có thể bị thổi phồng nếu model né câu khó bằng output sai format).

**State machine:**

1. Parse terminal label theo grammar đóng `Final answer: A|B|C|D`.
2. Nếu thất bại → exact-normalized whole-candidate-text match.
3. Nếu tiếp tục thất bại → BioBERT fallback (embedding-match).
4. Tie hoặc confidence thấp → **unresolved** (trạng thái cuối riêng, không ép về một đáp án).

**7 metric báo cáo riêng biệt:**

1. Parse coverage (% đạt bước 1).
2. Parsed-subset accuracy — **diagnostic only**, denominator = tập con parse thành công.
3. All-item strict accuracy — denominator = N (tổng), parse-failure tính là sai.
4. Fallback-assisted accuracy — denominator = N, unresolved tính là sai.
5. Fallback/unresolved rate.
6. Human agreement (trên tập fallback/ambiguous, tách khỏi audit tổng quát 50–100 mẫu ở mục 10).
7. Conditional-likelihood accuracy + agreement matrix giữa parsed/fallback-assisted/conditional-likelihood.

Tên metric chính gọi là **"pipeline answer-selection accuracy"**. Nếu matcher không qua ngưỡng agreement đóng băng trước test (mục 19), toàn bộ metric phải mang nhãn **"unvalidated pipeline diagnostic"**. Nếu pipeline-accuracy và conditional-likelihood-accuracy đảo dấu effect hoặc đảo thứ hạng arm ở cùng cell, kết luận phải ghi rõ **"evaluator-dependent"**, không chọn một phương pháp làm ground truth ngầm.

## 12. FL contract — khóa trước khi chạy test

- Constant learning rate trong suốt các round (không schedule/decay).
- Client optimizer reset sau mỗi round.
- Cả 3 client tham gia mỗi round (full participation).
- Central và FL dùng cùng base model và cùng LoRA initialization, cùng shared architecture (mục 7).
- Weighted FedAvg **trực tiếp trên LoRA tensor A/B**, weighted theo số target example của mỗi client.
- Server không giữ optimizer state (pure parameter averaging).
- Local epochs, số round, batch size, gradient accumulation, LR search space: chọn trên dev set, freeze trước khi chạm test, cùng ngân sách tuning với Central (mục 7).
- Unit test bắt buộc:
  - FL chạy với 1 client duy nhất phải cho kết quả gần tương đương centralized local training trên đúng dữ liệu client đó.
  - Weighted aggregation phải khớp kết quả tính tay trên một toy case.

## 13. Success criteria

### 13.1 Operational

- Toàn bộ run trong crossed matrix + 2 diagnostic chạy xong không lỗi.
- Manifest hash integrity — mọi input dùng để train/eval khớp hash đã seal.
- Query/support disjointness và closure constraint được verify bằng script, không chỉ theo thiết kế (mục 4.3, 6).
- Không có cross-client retrieval (verify runtime, không chỉ khẳng định trong thiết kế).
- `effective_k=5` đúng 100% mẫu trong crossed matrix (verify, không chỉ khẳng định).
- Test config đã freeze không bị sửa sau khi seal.
- Artifact completeness: mọi run xuất đủ file kỳ vọng.
- Deterministic replay: chạy lại cùng seed/config phải ra cùng kết quả (test thật, không chỉ giả định).
- Resume/idempotency: một run bị gián đoạn có thể resume mà không hỏng kết quả.
- Rerun kỹ thuật (technical rerun) chỉ hợp lệ khi cùng toàn bộ hash; mọi thay đổi protocol phải tạo **manifest version mới**, không ghi đè.
- Output mỗi run lưu: per-example prediction, config/manifest/git/model hash, timing + resource metrics — không chỉ `results/*.json` tổng hợp.

### 13.2 Scientific — toàn bộ exploratory (không còn mục "confirmatory")

Rev 2 tự nhận một contrast "confirmatory" (non-inferiority FL vs Central) nhưng không có primary model×dataset cell định trước, không power calculation, và chỉ 3 seed — không đủ chuẩn. Rev 3 hạ toàn bộ về exploratory, báo riêng từng cell model×dataset (không pool tùy ý):

- `train-k5 − train-k0` (demo-conditioned SFT effect, giữ eval-k=5 cố định).
- `ICL-only(retrieval-k5) − zero-shot`.
- `ICL-only(retrieval-k5) − ICL-only(random-k5)` — dùng D1, gọi rõ là **single-manifest random-k5 smoke diagnostic**, không phải estimate tổng quát của phân phối random-demo.
- `zero-shot ≤ train-k5` — giả thuyết cần kiểm chứng, không phải điều kiện go/no-go.
- `Δ = Federated-k5 − train-k5` (matched-compute, mục 7) — báo cáo như **descriptive engineering reference** kèm −5pp làm mốc tham chiếu, **không phải pass/fail gate**, không dùng từ "confirmatory".

**Phương pháp resampling** (95% CI, mọi contrast):

- Contrast không train (deterministic): paired item bootstrap trên `test-query`.
- Contrast có train: Central/FL và các seed dùng **paired seed ID** (seed=i của arm A khớp seed=i của arm B); resample paired theo cả seed-pair và evaluation unit (bootstrap hai tầng, paired chứ không độc lập).
- MedQuAD: resample ở cấp **source/near-duplicate group** trong từng client cố định (không resample ở cấp item thô, vì item không độc lập trong cùng nhóm nguồn).
- Cận dưới CI 95% accuracy mỗi arm đã train phải > 25% (random-chance floor — hợp lệ vì candidate luôn hiển thị trong prompt, mục 8).

## 14. Feasibility & systems metrics

Bổ sung bắt buộc — spec trước thiếu gần như toàn bộ metric hệ thống cần để quyết định scale full-scale:

- Per-client accuracy.
- Prompt token P50/P95/max theo dataset/model.
- Train/eval wall-clock.
- Throughput và latency.
- Peak VRAM.
- Adapter upload + server broadcast bytes/round (communication cost — nhất quán với metric đã nêu trong full-scale design gốc).
- Total communication (toàn bộ FL run).
- FL dev curve theo round (hội tụ hay không).
- Exclusion/refill/capacity-failure rate theo dataset/client/reason (sức khỏe vận hành của DAG mục 5).

## 15. Seed policy theo arm

| Arm | Phạm vi | Seed |
|---|---|---|
| Zero-shot | 2 model × 2 dataset | 1 (deterministic, greedy) |
| ICL-only (closure-constrained retrieval-k5) | 2 model × 2 dataset | 1 (deterministic) |
| train-k0/eval-k5 | 2 model × 2 dataset | 3 |
| demo-conditioned SFT (train-k5)/eval-k5 | 2 model × 2 dataset | 3 |
| Federated-k5/eval-k5 | 2 model × 2 dataset | 3 (paired seed ID với train-k5, mục 13.2) |
| D1: ICL-only random-k5 | Qwen2.5-0.5B × MedQA-USMLE | 1 (single-manifest smoke diagnostic) |
| D2: Fed train-k0/eval-k5 | Qwen2.5-0.5B × MedQA-USMLE | 1 |

## 16. Compute gate — trình tự khóa cứng (thay "đo 1 ô đầu tiên")

1. **Token census** trên manifest đã seal (mục 5) — phân phối token thật, không ước lượng.
2. **Benchmark worst-case**: Qwen2.5-0.5B (model lớn hơn) × MedQuAD (sequence dài hơn) × Federated-k5 (arm phức tạp/tốn nhất), chạy **cả hai evaluator** (generate+match và conditional-likelihood).
3. Áp tuning cap (ngân sách search đã khóa ở mục 7/12).
4. Ước tính lại toàn bộ workload thật (không chỉ 46 config — cộng cả tuning, calibration, audit, retry, eval pass thứ hai).
5. Chỉ mở crossed matrix đầy đủ nếu nằm trong budget; nếu không, cắt phạm vi trước khi chạy, không chạy tràn rồi cắt giữa chừng.

## 17. Cấu trúc code

```
Version_3/pilot/
  configs/                        # 1 config / run
  data/
    prepare_medqa.py              # giữ official train/dev/test split
    prepare_medquad.py            # filter, group theo source/CUI/near-dup, group-split, subsample + reserve pool
    assign_clients.py             # deterministic group-aware client assignment, n_i theo client
    audit_splits.py                # disjointness query/support, overlap/near-dup xuyên phần+client
    audit_medquad_correctness.py  # human-audit "chỉ một đáp án đúng"
  retrieval/
    encoder.py                    # BioBERT wrapper: masked mean pool, L2-norm, chính sách 512-token, chỉ encode question+candidates
    closure_retriever.py          # closure-constrained retrieval-k5 (mục 6): hard constraint theo dataset, preflight capacity, logging rank/exclusion
  prompt/
    build_manifest.py             # chạy toàn bộ DAG mục 5: candidate → demo → prompt → dual-tokenizer fit → audit → refill → seal
    template.py                    # canonical prompt template dùng chung 2 model
  train/
    lora_sft.py                    # LoRA SFT centralized (k0/k5 qua config, loss mask -100)
    federated_lora.py              # full FedAvg protocol, FL contract mục 12
  eval/
    generate_and_match.py          # state machine 4 bước (mục 11)
    likelihood_score.py            # conditional log-likelihood scoring
    calibrate_matcher.py           # audit tay tổng quát + audit tay fallback/ambiguous subset riêng
    metrics.py                     # 7 metric, agreement matrix, gold similarity margin
    resampling.py                  # paired item bootstrap, paired seed-pair bootstrap, MedQuAD group-level resampling
  benchmark/
    token_census.py                # mục 16 bước 1
    worst_case_bench.py            # mục 16 bước 2
  ops/
    manifest_hash.py               # hash + version manifest, seal enforcement
    resume.py                      # resume/idempotency cho run bị gián đoạn
  run_pilot.py                     # chạy crossed matrix + 2 diagnostic theo compute gate mục 16
  tests/                           # retrieval exclusion, prompt format/effective_k, FedAvg single-client sanity, weighted-aggregation toy case
  manifests/                       # candidate/demo/prompt manifest đã seal, hash, version log
  results/
    <run_id>/predictions.jsonl     # per-example prediction
    <run_id>/metrics.json          # 7 metric + feasibility metrics
    <run_id>/hashes.json           # config/manifest/git/model hash
```

## 18. Quyết định đã chốt qua trao đổi (tóm tắt, tránh lặp lại tranh luận)

- Compute: Cloud GPU A5000 đã cấp phát.
- Model: Qwen2.5-0.5B-Instruct + SmolLM2-360M-Instruct.
- Dataset: MedQA-USMLE (official splits, gọi đúng "labeled in-domain support carved from test split") + MedQuAD (group-split theo provenance).
- Eval: state machine 4 bước (parse → exact-match → BioBERT fallback → unresolved), cộng conditional-likelihood làm tầng đối chiếu — 7 metric tách biệt, không gộp một accuracy.
- Retrieval: closure-constrained retrieval-k5 (không phải cosine top-k thuần), client-scoped, retriever không thấy gold label.
- FL: 3 client, IID random split ở cấp client assignment, nhưng client-scoped retrieval + full FedAvg protocol là bắt buộc mọi arm.
- FL k: 5 (khớp arm #4), matched-compute contract (mục 7) với estimand chính xác "pooled optimization vs full FedAvg protocol" — không nói "chỉ khác aggregation".
- Arm: 5 chính + ICL-only + 2 diagnostic (D1 random-k5, D2 Fed train-k0/eval-k5).
- Seed: theo arm, paired seed ID giữa Central/FL.
- Toàn bộ scientific claim là exploratory; −5pp FL vs Central là descriptive engineering reference, không phải confirmatory gate.
- Artifact xây theo một DAG duy nhất (mục 5); mọi refill chạy lại toàn bộ vòng candidate→retrieval→fit→audit.
- Compute gate: token census → benchmark worst-case (cả 2 evaluator) → tuning cap → ước tính lại → mở grid.

## 19. Checklist bắt buộc cho implementation plan (không nhồi vào design spec)

- Git: đã init, commit + push (rewrite lịch sử để bỏ `Co-Authored-By`) — HEAD hiện tại = `origin/main` = `5b96af4`. Rev 3 (file này) chưa commit/push — chờ xác nhận.
- Dataset/model/BioBERT revision cụ thể + hash (pin version).
- Seed/RNG registry cụ thể (danh sách seed dùng cho từng arm, paired ID).
- LoRA module cụ thể (rank/alpha/dropout/target modules) và optimizer parameters.
- Ngưỡng near-duplicate detection cụ thể (lexical/semantic similarity threshold).
- Dependency/CUDA lock (requirements pin, driver version).
- Manifest/result JSON schema cụ thể.
- Benchmark/tuning limit cụ thể (giới hạn thời gian/số trial tuning trên dev).
- Test-access, retry, và resume command cụ thể.
- `max_input_tokens = 2048` (mục 8) — xác nhận hoặc điều chỉnh.
- Ngưỡng agreement cụ thể để đóng băng matcher trước khi gọi "pipeline answer-selection accuracy" chính thức (mục 11) — chốt dựa trên audit tay trên dev.
