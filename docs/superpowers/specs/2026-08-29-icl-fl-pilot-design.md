# FedICL-MQA — Pilot: vertical slice MedMCQA non-IID

Status: Approved design (rev 5), sẵn sàng cho implementation plan
Date: 2026-08-31 (rev 5)
Liên quan: [fedicl_mqa_paper_core_sections.md](../../fedicl_mqa_paper_core_sections.md) (full-scale design, rev 2)

Rev 5 đổi mục đích của pilot. Rev 1–4 xây một feasibility study về demo-conditioned SFT (centralized vs federated) trên MedQA + MedQuAD, IID 3 client, tới rev 4 là 8 arm / 196 configuration. Vấn đề: **pilot đó không chạm tới cơ chế mang novelty của bài** — full-scale rev 2 đặt đóng góp chính ở client-aware federated retrieval prior (`w_{i,c}` leave-one-client-out), mà rev 4 hoàn toàn không có `γ`, không có `w`. Pilot đang de-risk phần an toàn.

Rev 5 vì vậy thu pilot thành **một vertical slice dọc theo đúng code path của full-scale**: một model, một dataset (MedMCQA), partition non-IID skew 5 client, và một tập arm tối thiểu có chạm ladder A (§3). Toàn bộ protocol machinery của rev 3/rev 4 được giữ nguyên vì nó độc lập với arm matrix: artifact DAG có thứ tự duy nhất (§5), closure-constrained retrieval (§6), matched-compute contract (§8), FL contract (§11), operational gate (§12.1). Thứ bị bỏ là **phạm vi**, không phải **kỷ luật**.

Bị loại khỏi pilot ở rev 5: MedQuAD và toàn bộ constructed-4-way-MCQ, model thứ hai (SmolLM2), trục eval-k, arm random-k5, arm local-only-k0, và 8-arm crossed matrix. Những thứ này không mất — arm random và local-only đã trở thành mục 1 và mục 5 của ladder full-scale (§6.1 của doc kia), nên chạy chúng ở pilot là chạy hai lần.

## 1. Mục tiêu — ba câu hỏi feasibility

Pilot này **không ước lượng effect size**. Nó trả lời ba câu hỏi, theo thứ tự quan trọng:

**F1 — Code path có chạy end-to-end không?** `partition non-IID → client validation stats (a_{j,c}, n_{j,c}) → prior LOO → closure-constrained retrieval có prior → FedAvg-LoRA → conditional-likelihood eval`. Đây là đúng chuỗi của full-scale, không phải một xấp xỉ.

**F2 — Prior LOO có ước lượng được không?** Đây là câu hỏi quan trọng nhất và là lý do pilot tồn tại. Full-scale §5.2 yêu cầu design check `min_{i,c: n_{i,c}>0} Σ_{j≠i} n_{j,c} ≥ n_min`. Nếu check này trượt trên phần lớn cặp `(i,c)` dưới một partition skew thực tế, thì `w_{i,c}` không xác định được ở đúng những subject mà client quan tâm nhất — và **contribution 2 của paper không triển khai được**. Cần biết điều này trong một tuần, không phải ở tháng thứ ba.

**F3 — Chi phí thật mỗi round là bao nhiêu?** Wall-clock, VRAM, adapter bytes/round, cộng payload của kênh thống kê `(a_{j,c}, n_{j,c})` — để size ngân sách 120 GPU-giờ của full-scale.

Không có câu hỏi F nào cần 3+ seed hay CI hẹp. Kết luận của pilot là **operational**, không phải scientific (§12).

## 2. Hạ tầng & phạm vi

- Compute: Cloud GPU A5000 (đã cấp phát), tách riêng khỏi ngân sách 120 GPU-giờ của full-scale.
- **1 model: Qwen2.5-0.5B-Instruct.** Cùng họ với Qwen2.5-3B-Instruct của full-scale, nên chat template, tokenizer và code path chuyển thẳng lên được. SmolLM2-360M bị loại: nó chỉ phục vụ contrast cross-model, mà cross-model không nằm trong câu hỏi nào của §1.
- **1 dataset: MedMCQA** — primary dataset của full-scale. MedQA-USMLE là secondary generalization benchmark ở full-scale và **không** thuộc pilot.
- **5 client, partition non-IID skew** (§4.3) — khớp số client của full-scale, vì `n_min` design check phụ thuộc trực tiếp vào số client.
- 2 seed. Đủ để phát hiện run không ổn định; **không** đủ để ước lượng effect, và §12.2 cấm báo cáo effect size như một estimate.

## 3. Arm — vertical slice

| # | Arm | Train | Retrieval lúc eval | Chạm mục nào của full-scale |
|---|---|---|---|---|
| V1 | Zero-shot | Không train | k=0 | Sàn sanity |
| V2 | FedAvg-LoRA, `γ=0` | FedAvg-LoRA (§11) | closure-constrained top-k, **không prior** | **Ladder A mục 1** |
| V3 | FedAvg-LoRA, prior LOO | *cùng adapter V2* | closure-constrained + `γ·w_{i,c}` LOO | **Ladder A mục 3** |
| V4 | FedAvg-LoRA, prior shuffled | *cùng adapter V2* | closure-constrained + `γ·w_{i,π(c)}` | **Ladder A mục 4** |
| V5 | Local-only LoRA, `γ=0` | 5 client train độc lập = FedAvg với aggregation tắt (§8.2) | closure-constrained top-k, không prior | **Ladder B mục 5** |

**V2, V3, V4 dùng chung đúng một adapter mỗi seed** — prior áp ở inference, không ở training. Ba arm này vì vậy tốn **ba eval pass, không phải ba training run**. Đây là lý do ladder A rẻ và là lý do nó nên chạy trước ở cả pilot lẫn full-scale.

**Về V4 (shuffled prior)** — mục này không nằm trong yêu cầu "chạm mục 1 + mục 3", tôi thêm vào vì nó là inference-only trên adapter đã có, tức gần như miễn phí, và nó là **falsification test** của chính đóng góp số 2: nếu prior shuffled ngang prior LOO thì `γ·w` chỉ đang hoạt động như một nhiễu/temperature cho scorer chứ không truyền tri thức xuyên viện. Chạy nó ở pilot với giá một eval pass rẻ hơn nhiều so với phát hiện ra ở full-scale. V4 là nấc đầu tiên của thang cắt phạm vi (§15.3) nếu budget không cho phép.

**Không có trong pilot** (đã nằm ở ladder full-scale, chạy ở pilot là chạy hai lần): local prior (mục 2), FedAvg + local adaptation / Ditto (mục 7), centralized LoRA (mục 8), public-corpus retrieval, random few-shot.

## 4. Data provenance & non-IID partition

### 4.1 MedMCQA — nguồn và cách gọi tên

MedMCQA có `train` (~182k), `validation` (~4.2k, có nhãn), `test` (~6k, **nhãn bị giữ lại**). Vì test không có nhãn công khai:

- `train-core` + `val-support`: subsample từ official **train** split.
- `eval-query`: subsample từ official **validation** split.
- `stats-holdout`: subsample từ official train split, **tách rời `train-core`**, dùng riêng để tính `(a_{j,c}, n_{j,c})` ở §7.

Gọi đúng tên phần đánh giá là **"labeled evaluation carved from the official MedMCQA validation split"** — không phải "MedMCQA test accuracy". Không so sánh trực tiếp con số với leaderboard trên official test set.

Giữ metadata `subject_name` và `topic_name` của MedMCQA xuyên suốt — chúng là `c` trong `w_{i,c}` và không được drop ở bất kỳ bước nào của DAG (§5).

### 4.2 Bốn vai trò dữ liệu, disjoint bắt buộc

| Vai trò | Nguồn | Dùng cho |
|---|---|---|
| `train-core` | train split | SFT (V2, V5) |
| `val-support` | train split | Pool demo để retrieve |
| `stats-holdout` | train split | Tính `a_{j,c}`, `n_{j,c}` (§7) — **không** dùng để train, **không** dùng làm demo |
| `eval-query` | validation split | Query lúc eval |

`stats-holdout` phải tách khỏi cả `train-core` lẫn `val-support`. Nếu nó trùng `train-core`, `a_{j,c}` là training accuracy và prior đo sai thứ cần đo. Nếu nó trùng `val-support`, prior và demo pool tương quan nhau và contrast V3−V2 confounded. Verify bằng script (§16).

### 4.3 Partition non-IID skew — không disjoint

Theo full-scale §5.2: partition **skew chứ không disjoint**. Mỗi client có cụm specialty trội nhưng giữ long tail các subject còn lại.

- 5 client, mỗi client gắn một cụm subject trội của MedMCQA.
- Tỷ lệ subject mỗi client rút từ Dirichlet với concentration `ρ`, nghiêng về cụm trội của client đó. `ρ` cố định ở pilot (một giá trị duy nhất, ghi trong config); sweep `ρ` là việc của RQ3 full-scale.
- Gán deterministic, group-aware, tôn trọng ranh giới near-duplicate cluster — một nhóm câu hỏi gần trùng không bị chia giữa các client.
- Kích thước pool cục bộ mỗi client là `n_i`, **không hard-code**; leave-one-out trong pool của chính nó cho `n_i − 1`.
- **Mọi arm có retrieval** chỉ lấy demo từ `val-support` cùng `client_id` với query — kể cả V5.

Lý do không dùng partition disjoint của full-scale rev 1: nếu subject `c` chỉ tồn tại ở client `i` thì `Σ_{j≠i} n_{j,c} = 0` và `w_{i,c}` không xác định đúng ở những subject client `i` retrieve nhiều nhất. Partition disjoint và prior LOO loại trừ nhau về mặt toán học.

## 5. Artifact construction DAG — thứ tự duy nhất, bắt buộc

Giữ nguyên từ rev 3. Toàn bộ artifact xây theo đúng một DAG, không có bước nào seal riêng lẻ trước khi bước trước nó xong cho toàn bộ cohort:

```
raw + hash
  → split theo vai trò (§4.2) + group near-duplicate
  → client assignment non-IID skew (§4.3)
  → LOO support design check (§7.3)          ← gate: trượt thì dừng, không train
  → train FedAvg-LoRA tới round R_0
  → tính (a_{j,c}, n_{j,c}) trên stats-holdout → w_{i,c}, freeze (§7)
  → closure-constrained demo selection, có/không prior (§6)
  → build final prompts (§9 — canonical template)
  → tokenizer fit check
  → human audit TRÊN ĐÚNG displayed text
  → refill nếu lỗi (reserve pool)
  → seal cohort / prompt manifest
```

- **Refill = chạy lại toàn bộ vòng** retrieval → fit → audit cho item lỗi, không patch cục bộ một bước.
- Manifest chỉ seal sau khi toàn bộ DAG pass cho 100% item còn lại.
- Khác rev 4: DAG giờ có **hai gate cắt ngang** — LOO support check trước khi train, và freeze `w` sau round `R_0`. Cả hai đều nằm giữa DAG chứ không ở đầu, nên `run_pilot.py` phải xử lý được trạng thái "đã train một phần, chưa có prior".

## 6. Closure-constrained retrieval

Đổi tên chính xác từ "cosine top-k": top-k thuần bị lọc thêm bởi các ràng buộc sau trước khi chấp nhận.

- **Retriever chỉ encode `question + options`, không được thấy gold label.**
- Hard constraint (MedMCQA): ban question ID trùng và near-duplicate question trong cùng prompt; **chỉ log** (không ban) option-text overlap, vì các lựa chọn phổ biến có thể lặp hợp lệ giữa các câu khác nhau.
- Chỉ retrieve trong `val-support` cùng `client_id`.
- **Preflight bắt buộc**: trước khi seal, chứng minh mỗi query còn đủ ≥k demo hợp lệ sau toàn bộ closure constraint; nếu thiếu, refill theo DAG hoặc loại item và báo exclusion rate.
- **Log bắt buộc**: số demo bị closure filter loại theo từng query, rank gốc trước filter của các demo được giữ, tỷ lệ query phải tìm sâu quá top-k mới đủ.

### 6.1 Tích hợp prior vào scorer

V2/V3/V4 dùng đúng một scorer, khác nhau **chỉ ở số hạng prior**:

```
s(d, q, i) = α·sim(e_d, e_q) − β·redundancy(d, S) + γ·w*_{i, c(d)}
```

| Arm | `w*` | `γ` |
|---|---|---|
| V2 | — | 0 |
| V3 | `w_{i,c}` LOO (§7) | `γ > 0`, chọn trên dev |
| V4 | `w_{i,π(c)}`, `π` là hoán vị ngẫu nhiên cố định trên tập subject | cùng `γ` với V3 |

Ràng buộc engineering: ba variant là **ba mode của cùng module** `closure_retriever.py` (`--prior {none,loo,shuffled}`), không phải ba đường code. Closure constraint, pool hợp lệ, `α`, `β`, encoder và prompt template phải identical giữa ba mode — có unit test assert pool identity (§16). Nếu không dùng chung module, "khác nhau đúng một số hạng" trở thành quy ước phải tự kiểm bằng tay và sẽ trôi.

`γ` chọn trên dev **một lần cho V3**, và V4 dùng lại đúng giá trị đó. Tune `γ` riêng cho V4 sẽ phá vai trò control của nó.

## 7. Federated prior LOO — đường tính và design check

Đây là phần mới hoàn toàn ở rev 5 và là lý do pilot tồn tại.

### 7.1 Kênh thống kê

Sau round `R_0`, mỗi client `j` đánh giá global adapter hiện tại trên `stats-holdout` cục bộ của mình và release, theo từng subject `c`, **đúng hai vô hướng**:

```
( a_{j,c}, n_{j,c} )
```

Không có item thô, câu hỏi, đáp án hay embedding nào rời khỏi client. Payload này được đo và báo cáo như một phần của communication cost (§13) — nó không miễn phí và cũng không vô hại (§18).

### 7.2 Prior

```
ē_{-i,c} = Σ_{j≠i} n_{j,c}·(1 − a_{j,c}) / Σ_{j≠i} n_{j,c}

w_{i,c}  = ( ē_{-i,c} − mean_{c'} ē_{-i,c'} ) / std_{c'} ē_{-i,c'}
```

Chỉ số `i` bị loại khỏi tổng **theo cấu trúc**. Đây chính là tính chất làm cho chữ "federated" trong contrast V3−V2 có nghĩa; nếu thống kê của chính client `i` lọt vào `w_{i,·}` thì đó là local prior đội lốt federated và không contrast nào phân biệt được hai thứ.

**Freeze**: `w` tính một lần từ adapter ở round `R_0` rồi đóng băng tới hết run. Nếu tính lại mỗi round, retrieval và training co-adapt trong cùng một run và không contrast nào quy được hiệu ứng cho bên nào. `R_0` cố định trước, ghi trong config.

### 7.3 Design check — gate cứng trước khi train

```
min over (i,c) với n_{i,c} > 0 của  Σ_{j≠i} n_{j,c}  ≥  n_min
```

`n_min` cố định trước khi chạy. Subject nào trượt check thì set `w_{i,c} = 0` và **báo exclusion rate theo client**.

Đây là gate, không phải một metric để ngắm: nếu tỷ lệ `(i,c)` trượt vượt ngưỡng đã khai báo trước, pilot **dừng và báo F2 = fail** thay vì chạy tiếp — vì `w` khi đó là nhiễu trên đúng những subject quan trọng nhất với từng client, và mọi con số phía sau đều vô nghĩa. Đây là kết quả hợp lệ và có giá trị của pilot, không phải một thất bại vận hành.

## 8. Matched-compute contract

### 8.1 Estimand

> Matched data/prompt/exposure comparison of the full FedAvg optimization protocol versus the same protocol with aggregation disabled.

FL khác local-only không chỉ ở bước gộp tham số cuối: FL có optimizer reset mỗi round và client drift là hiệu ứng thuật toán thật. Contract dưới đây làm cho **đúng một biến** khác nhau giữa V2 và V5.

### 8.2 V5 (local-only) = FL protocol với aggregation tắt

Local-only **không** implement thành script riêng. Nó là `federated_lora.py --no-aggregate`: mỗi client giữ adapter của mình xuyên suốt, cùng số local epoch, cùng số round, cùng lịch optimizer reset, cùng LoRA init, cùng manifest như V2.

Hệ quả: contrast V2−V5 cô lập đúng bước aggregation. Tổng optimizer step, tổng token, exposure của từng client giống hệt nhau; thứ duy nhất khác là adapter có được trung bình hoá giữa các round hay không.

**Báo cáo**: per-client accuracy, và aggregate = trung bình có trọng số theo `|eval-query_i|`. Δ per-client báo riêng — dưới partition skew, federation nâng đều mọi client hay chỉ kéo client yếu lên (và có kéo client mạnh xuống không) là finding thật.

### 8.3 Khoá cấu trúc

- Weighted FedAvg **trực tiếp trên trainable LoRA tensor A/B**, không aggregate tham số nào khác.
- Cùng adapter initialization giữa V2 và V5.
- Shared LoRA architecture (rank/alpha/target modules).
- Equal tuning budget giữa V2 và V5.

## 9. Prompt & Context Contract

- **Một canonical textual prompt**, chỉ khác chat-template bắt buộc của tokenizer.
- Thứ tự cố định: `system → k demos → query + options → answer instruction`.
- Demo hiển thị đầy đủ `question + options + gold label` — task-isomorphic với query.
- Decoding cho conditional-likelihood: không sinh; scoring trực tiếp (§10).
- `max_input_tokens = 2048`, reserve 32–64 token cho output — xác nhận ở implementation plan.
- Manifest tokenize thử trước khi seal; overflow ở k: sửa representation hoặc loại item trước khi seal, báo exclusion rate.
- **100% mẫu trong manifest phải giữ đúng `effective_k`** khai báo của config — verifier so `effective_k == expected_k`, không hard-code kỳ vọng.
- **SFT loss masking**: chỉ tính loss trên target-answer span; toàn bộ prompt + demo đặt `label = -100`.

## 10. Evaluator — conditional log-likelihood là primary

Đổi so với rev 4. Full-scale §4 chọn conditional log-likelihood làm phương pháp answer-selection, nên pilot phải dùng đúng nó làm **primary**, không phải làm tầng đối chiếu:

```
ŷ = argmax_k  log P(o_k | q, prompt)
```

Không sinh free-form rồi match — chính là confound mà full-scale nêu.

Generate-and-match (state machine parse → exact-match → unresolved) giữ lại làm **agreement check trên một subsample**, không phải metric chính. Báo cáo:

1. Conditional-likelihood accuracy (**primary**), denominator = N.
2. Parse coverage của generate-and-match trên subsample.
3. Agreement matrix giữa conditional-likelihood và generate-and-match trên subsample.
4. Per-client accuracy (bắt buộc, §8.2).

Nếu hai phương pháp đảo thứ hạng arm, kết luận phải ghi rõ **"evaluator-dependent"**.

**Ghi chú về BioBERT**: ở rev 4, BioBERT vừa tạo distractor, vừa làm fallback judge, vừa làm retriever — circularity risk phải audit riêng. Rev 5 bỏ MedQuAD nên không còn constructed distractor, và conditional-likelihood primary nên không còn embedding fallback judge. BioBERT giờ **chỉ còn một vai trò là retrieval encoder**, và circularity risk gần như biến mất. Vẫn giữ: masked mean pooling, L2-normalize, audit tay 50–100 output.

## 11. FL contract — khóa trước khi chạy eval

- Constant learning rate suốt các round (không schedule/decay).
- Client optimizer reset sau mỗi round.
- Cả 5 client tham gia mỗi round (full participation).
- Weighted FedAvg trực tiếp trên LoRA tensor A/B, weighted theo số example của mỗi client.
- Server không giữ optimizer state (pure parameter averaging).
- Local epochs, số round, `R_0`, batch size, gradient accumulation, LR, `γ`: chọn trên dev, freeze trước khi chạm `eval-query`.
- Áp cho **V2 và V5** (V5 chỉ khác ở `--no-aggregate`).

Unit test bắt buộc:

- FL với 1 client / 1 round phải cho kết quả tương đương centralized training trên đúng dữ liệu client đó.
- Weighted aggregation khớp kết quả tính tay trên toy case.
- `--no-aggregate` với 5 client cho 5 adapter khác nhau, và tổng optimizer step mỗi client **khớp chính xác** với V2 cùng seed (assert bằng số).
- `closure_retriever.py`: pool hợp lệ trả về cho `--prior none|loo|shuffled` phải **identical set** với cùng query + manifest.
- `w_{i,c}` tính từ toy stats phải khớp tay, và phải **assert `i` không nằm trong tổng** (test trực tiếp tính chất LOO, không suy ra từ giá trị).

## 12. Success criteria

### 12.1 Operational

- Toàn bộ run của slice chạy xong không lỗi.
- Manifest hash integrity — mọi input train/eval khớp hash đã seal.
- Disjointness bốn vai trò dữ liệu (§4.2) verify bằng script.
- Không có cross-client retrieval (verify runtime).
- `effective_k == expected_k` 100% mẫu cho mọi config.
- **LOO design check (§7.3) chạy và ghi kết quả trước mọi training run.**
- **`w` được freeze tại `R_0`** — verify hash của `w` không đổi giữa các round sau `R_0`.
- Test config đã freeze không bị sửa sau khi seal.
- Deterministic replay: cùng seed/config ra cùng kết quả (test thật).
- Resume/idempotency: run gián đoạn resume được không hỏng kết quả.
- Rerun kỹ thuật chỉ hợp lệ khi cùng toàn bộ hash; đổi protocol phải tạo **manifest version mới**.
- Mỗi run lưu: per-example prediction, config/manifest/git/model hash, timing + resource metrics.

### 12.2 Feasibility — kết luận của pilot, không phải scientific claim

Pilot báo cáo **F1/F2/F3 của §1**, không báo cáo effect size như một estimate. Với 2 seed và một cell duy nhất, mọi Δ chỉ được trình bày như **directional observation kèm cả hai giá trị seed**, không CI, không kết luận về dấu.

| Câu hỏi | Tiêu chí |
|---|---|
| F1 | Chuỗi §5 chạy end-to-end, mọi gate pass, artifact đầy đủ |
| F2 | Tỷ lệ `(i,c)` trượt LOO check ≤ ngưỡng khai báo trước; nếu vượt → **F2 fail, báo cáo và dừng** |
| F3 | Có số đo thật cho wall-clock, VRAM, adapter bytes/round, payload thống kê |

Contrast được phép **quan sát** (không được gọi là estimate): `V3−V2` (prior có làm gì không), `V3−V4` (phần federated có làm gì không), `V2−V5` (aggregation có làm gì không). Cả ba đều để lên full-scale ước lượng.

## 13. Feasibility & systems metrics

- Per-client accuracy.
- Prompt token P50/P95/max.
- Train/eval wall-clock; throughput và latency; peak VRAM.
- Adapter upload + server broadcast bytes/round.
- **Payload kênh thống kê `(a_{j,c}, n_{j,c})` bytes/round** (mới ở rev 5).
- Total communication toàn run.
- FL dev curve theo round (hội tụ hay không).
- **Phân bố `n_{j,c}` và tỷ lệ trượt LOO check theo client/subject** (mới ở rev 5 — đây là output chính của F2).
- Exclusion/refill/capacity-failure rate theo client/reason.
- Label entropy và subject proportion mỗi client (đặc trưng hoá độ skew thật đạt được).

## 14. Seed & config registry

| Arm | Phạm vi | Đơn vị lặp |
|---|---|---|
| V1 Zero-shot | 1 cell | 1 (deterministic) |
| V2 FedAvg `γ=0` | 1 cell | 2 shared seed ID `{s1,s2}` |
| V3 FedAvg prior LOO | *cùng adapter V2* | cùng `{s1,s2}` — eval pass, không train |
| V4 FedAvg prior shuffled | *cùng adapter V2* | cùng `{s1,s2}`; hoán vị `π` cố định, ghi riêng trong registry |
| V5 Local-only | 1 cell × 5 client | cùng `{s1,s2}` |

- Một tập seed ID chung cho mọi arm đã train — V2 và V5 phải paired theo seed để contrast §8.2 hợp lệ.
- Hoán vị `π` của V4 nằm ở namespace riêng, không tái sử dụng số của training seed.

## 15. Compute

### 15.1 Ước tính

| Arm | Training run | Eval configuration |
|---|---|---|
| V1 | 0 | 1 |
| V2 | 2 (FedAvg, 5 client, R round) | 2 |
| V3 | 0 (dùng adapter V2) | 2 |
| V4 | 0 (dùng adapter V2) | 2 |
| V5 | 2 × 5 client = 10 (mỗi run trên `n_i` ≈ 1/5 dữ liệu) | 2 × 5 = 10 |
| **Tổng** | **12 run** | **17 configuration** |

So với rev 4 (84 training run / 196 configuration). Cộng: conditional-likelihood là eval pass duy nhất cho mọi arm, generate-and-match chỉ chạy trên subsample.

### 15.2 Gate

1. **Token census** trên manifest đã seal — phân phối token thật.
2. **LOO design check (§7.3)** — chạy trước, vì nếu trượt thì không cần benchmark gì thêm.
3. **Benchmark worst-case**: V2 (FedAvg 5 client) một seed, đo đủ §13.
4. Áp tuning cap (`γ`, LR, round — ngân sách đã khoá ở §11).
5. Chỉ mở slice đầy đủ nếu trong budget; nếu không, cắt theo §15.3 **trước** khi chạy.

### 15.3 Thang cắt phạm vi

1. Bỏ V4 (shuffled prior) — mất falsification test, phải nói rõ trong báo cáo.
2. V5: 2 seed → 1 seed.
3. Giảm `|eval-query|`, giữ nguyên số arm — thà đo thô mọi arm còn hơn đo kỹ vài arm.

**Không bao giờ cắt**: V2 và V3. Không có cặp đó thì pilot không chạm ladder A và mất lý do tồn tại.

## 16. Cấu trúc code

```
Version_3/pilot/
  configs/                        # 1 config / run
  data/
    prepare_medmcqa.py            # 4 vai trò (§4.2), giữ subject_name/topic_name
    partition_noniid.py           # Dirichlet skew 5 client, group-aware, deterministic
    audit_splits.py               # disjointness 4 vai trò, overlap/near-dup xuyên client
  prior/
    client_stats.py               # tính (a_{j,c}, n_{j,c}) trên stats-holdout
    loo_prior.py                  # ē_{-i,c} → w_{i,c}, freeze tại R_0, hoán vị π cho V4
    support_check.py              # §7.3 design check — GATE, chạy trước train
  retrieval/
    encoder.py                    # BioBERT: masked mean pool, L2-norm, chỉ encode question+options
    closure_retriever.py          # closure constraint + preflight + logging
                                  #   --prior {none,loo,shuffled}: V2/V3/V4 dùng CHUNG pool (§6.1)
  prompt/
    build_manifest.py             # DAG §5: retrieval → prompt → fit → audit → refill → seal
    template.py                   # canonical prompt template
  train/
    federated_lora.py             # FedAvg protocol, FL contract §11
                                  #   --no-aggregate: V5 local-only, cùng protocol trừ aggregation (§8.2)
  eval/
    likelihood_score.py           # PRIMARY: conditional log-likelihood
    generate_and_match.py         # agreement check trên subsample
    metrics.py                    # accuracy, per-client, agreement matrix
  benchmark/
    token_census.py
  ops/
    manifest_hash.py              # hash + version manifest, seal enforcement
    resume.py
  run_pilot.py                    # 5 arm theo compute gate §15, xử lý trạng thái "train xong R_0, chưa có prior"
  tests/                          # closure exclusion, pool identity 3 mode prior, LOO excludes-self,
                                  # --no-aggregate step-count match, FedAvg toy case, effective_k
  manifests/
  results/
    <run_id>/predictions.jsonl
    <run_id>/metrics.json
    <run_id>/hashes.json
    <run_id>/prior_stats.json     # (a_{j,c}, n_{j,c}), w_{i,c}, kết quả support check
```

## 17. Quyết định đã chốt

- Compute: Cloud GPU A5000, tách khỏi ngân sách full-scale.
- **1 model (Qwen2.5-0.5B-Instruct), 1 dataset (MedMCQA), 5 client non-IID skew, 2 seed.**
- Pilot là **vertical slice theo code path full-scale**, không phải bản thu nhỏ của một feasibility matrix.
- **Kết luận của pilot là feasibility (F1/F2/F3), không phải effect size.** 2 seed là đủ cho mục đích đó và không đủ cho mục đích khác — §12.2 cấm vượt rào.
- **F2 (prior LOO có ước lượng được không) là câu hỏi chính**; §7.3 là gate cứng, trượt thì dừng và đó là kết quả hợp lệ.
- Partition **skew chứ không disjoint** — disjoint làm prior LOO không xác định (§4.3).
- `w` freeze tại `R_0`; `stats-holdout` tách khỏi cả `train-core` lẫn `val-support`.
- V2/V3/V4 chung một adapter, khác nhau chỉ ở mode prior của **cùng một module** retriever.
- V5 = `federated_lora.py --no-aggregate`, không phải script riêng.
- **Conditional log-likelihood là primary evaluator**, khớp full-scale §4; generate-and-match hạ xuống agreement check.
- Giữ nguyên từ rev 3/rev 4: artifact DAG thứ tự duy nhất, closure-constrained retrieval, matched-compute contract, FL contract, operational gate, manifest hash + resume.
- Bỏ khỏi pilot: MedQuAD, constructed 4-way MCQ, SmolLM2, trục eval-k, random-k5, local-only-k0, 8-arm matrix.

## 18. Checklist bắt buộc cho implementation plan

- Git: đã init, commit + push; lịch sử đã rewrite để bỏ `Co-Authored-By`. **Không ghi commit hash vào tài liệu này** — nó là metadata tự tham chiếu nằm trong chính file được commit nên sẽ stale sau mỗi lần commit (rev 3 mắc đúng lỗi này). Provenance tra bằng `git log`; hash của run thuộc `results/<run_id>/hashes.json`.
- MedMCQA revision + hash; BioBERT revision + hash (pin version).
- Kích thước cụ thể của `train-core`, `val-support`, `stats-holdout`, `eval-query`.
- **`ρ` (Dirichlet concentration) và cụm subject trội của từng client** — cố định, ghi trong config.
- **`n_min` và ngưỡng tỷ lệ trượt cho F2** — phải chốt **trước** khi chạy, nếu không §7.3 không còn là gate.
- **`R_0`** — round tính và freeze `w`.
- `γ` search space trên dev; `α`, `β` cố định hay tune.
- Seed registry: `{s1,s2}` chung cho V2/V5; namespace riêng cho hoán vị `π`.
- LoRA rank/alpha/dropout/target modules; optimizer parameters — dùng chung V2 và V5.
- Ngưỡng near-duplicate detection.
- Dependency/CUDA lock.
- Manifest/result JSON schema — phải mang `arm_id`, `prior_mode`, `seed_id`, `client_id`, `subject` để join được mà không parse tên file.
- `max_input_tokens = 2048` và `k` — xác nhận hoặc điều chỉnh.
- Kích thước subsample cho agreement check (§10).

## 19. Quan hệ với full-scale

Pilot này ánh xạ 1-1 vào full-scale rev 2: V2→ladder A mục 1, V3→mục 3, V4→mục 4, V5→ladder B mục 5. Không có arm nào của pilot nằm ngoài ladder full-scale, nên không có công nào bị bỏ đi khi scale lên.

Những gì full-scale có mà pilot **cố ý không có**, và phải chạy ở full-scale chứ không suy ra từ pilot: local prior (ladder A mục 2), Ditto (ladder B mục 7), centralized upper bound (mục 8), public-corpus retrieval, MedQA-USMLE generalization, sweep `ρ` và `k` cho RQ3, canary exposure audit cho RQ4, và mọi ước lượng effect size kèm CI.

Narrative của paper chốt trước submission, không phải trước implementation. Pilot này không đóng cửa lựa chọn nào — nó chỉ trả lời liệu cơ chế trung tâm có triển khai được hay không.
