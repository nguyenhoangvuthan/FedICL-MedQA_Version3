# FedICL-MQA — Prompt chốt thiết kế baseline FL × ICL

**Status:** Thiết kế đã chốt để lập implementation plan

**Ngày:** 2026-09-01

**Phạm vi:** Medical multiple-choice question answering bằng Small Language Model, LoRA, Federated Learning và In-Context Learning.

## 1. Mục tiêu thực nghiệm

Thiết kế và triển khai một protocol thực nghiệm có kiểm soát để trả lời ba câu hỏi:

1. ICL có cải thiện một mô hình đã được Local LoRA adaptation hay không?
2. ICL có cải thiện một mô hình đã được FedAvg-LoRA adaptation hay không?
3. Client-aware re-ranking có tạo thêm lợi ích so với single retriever chuẩn hay không?

Ba main baseline để đối chiếu là **Centralized SLM** (C0), **Federated SLM without ICL** (F0) và **Non-federated ICL** (L1, kèm B1 làm đối chiếu training-free).

Toàn bộ so sánh phải tách được ảnh hưởng của training, federation, exemplar context và re-ranking. Không được thay checkpoint khi chỉ muốn đo ảnh hưởng của ICL tại inference.

## 2. Phạm vi đã khóa

- Bài toán hiện tại là **text-only medical QA**.
- Loại hoàn toàn mục **Multimodal Representation Learning** khỏi proposed approach, experimental plan và contribution hiện tại.
- Toàn bộ evaluation chạy trên **native four-way multiple-choice answer selection**: MedQA và MedMCQA, dùng option gốc của dataset. Không có track constructed MCQ và không có dataset free-text.
- Giữ official train/validation/test split trước khi tạo client partition.
- Chỉ dùng **một local dense retriever** cho tất cả arm có retrieval.
- Mỗi prompt ICL có đúng `k = 5` exemplar hợp lệ.
- Exemplar chỉ được lấy từ repository tạo từ training split; validation và test không bao giờ là nguồn exemplar.
- Kết quả **không** được dùng để claim năng lực open-ended QA. Quy tắc phát ngôn ở mục 7.6.

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
| **F0** | FedAvg-LoRA trên `fit` | Không exemplar (`k=0`) | FL không ICL |
| **F1** | **Cùng checkpoint F0** | Top-5 bằng single retriever | Hiệu quả inference-time ICL trong FL |
| **F2** | **Cùng checkpoint F0** | Cùng retriever và candidate set với F1, sau đó client-aware re-ranking | Phương pháp đề xuất |
| **C0** | Centralized LoRA trên union của các `fit` set | Không exemplar (`k=0`) | Pooled-training reference; **không** giả định là upper bound trước khi đo |

Vai trò theo ba main baseline: **C0** = Centralized SLM, **F0** = Federated SLM without ICL, **L1** = Non-federated ICL (B1 là biến thể training-free của cùng baseline đó). **F2** là phương pháp đề xuất; **F1** là ablation bắt buộc để tách đóng góp của re-ranking khỏi đóng góp của ICL. **B0** và **L0** là floor tương ứng của B1 và L1, giữ lại vì cần thiết để đọc hai contrast ICL.

### 5.1 Client-aware re-ranking

F2 không tạo retriever thứ hai. F2 lấy candidate từ đúng single retriever của F1 và chỉ thay ranking score:

$$
s(d,q,i)=
\alpha\,\operatorname{sim}(q,d)
-\beta\,\operatorname{redundancy}(d,S)
+\gamma\,w_{\operatorname{subject}(d)}^{(i)}.
$$

Các trọng số `alpha`, `beta`, `gamma` chỉ được chọn bằng training/validation workflow và phải freeze trước test.

### 5.2 Seed và determinism policy

Tách hai loại randomness. Nhầm lẫn giữa chúng làm hỏng toàn bộ contrast ở mục 6.

**Data seed — cố định, dùng chung cho mọi arm.** Một seed duy nhất chi phối client partition, tách `fit`/`support` và near-duplicate clustering. Seed này freeze một lần trước mọi arm và **không bao giờ biến thiên theo arm**: nếu nó đổi giữa các arm thì mọi $\Delta$ bị confound bởi khác biệt dữ liệu chứ không phải khác biệt phương pháp.

**Training seed — biến thiên, paired ID giữa các arm.** Chi phối LoRA init, data order, dropout và client sampling mỗi round. Seed $s$ của arm A phải khớp seed $s$ của arm B trong mọi contrast có training.

| Arm | Loại | Training seed | Ghi chú |
|---|---|---|---|
| **B0**, **B1** | Inference-only | Không có; chạy đúng một lần | Frozen base, greedy decoding |
| **L0** → **L1** | Trained | $s\in\{1,2,3\}$, mỗi seed sinh **N checkpoint** (một cho mỗi client) | L1 dùng lại checkpoint L0 của cùng $s$ và cùng client |
| **F0** → **F1**, **F2** | Trained | $s\in\{1,2,3\}$, paired ID với L0 và C0 | F1/F2 dùng lại checkpoint F0 của cùng $s$ |
| **C0** | Trained | $s\in\{1,2,3\}$, paired ID với F0 | |

Ba training seed là mức tối thiểu để có hierarchical bootstrap; nếu compute cho phép, nâng lên năm cho các headline arm và giữ ba cho ablation. Số seed thực dùng phải ghi trong mọi bảng kết quả.

**Determinism tại inference — áp dụng cho mọi arm:**

- Greedy decoding, temperature 0, không sampling.
- Encoder revision, index build và query đều pin và deterministic.
- Với mỗi cặp arm dùng chung checkpoint, khác biệt output phải thuần do inference context.

**Hệ quả cho thống kê ở mục 7.5:**

- $\Delta_{Base\text{-}ICL}$: cả hai arm inference-only, không có seed → paired item bootstrap thuần.
- $\Delta_{Local\text{-}ICL}$, $\Delta_{FL\text{-}ICL}$, $\Delta_{ClientAware}$: chung checkpoint nhưng có training seed → hierarchical paired bootstrap qua seed-pair và item.
- $\Delta_{FL}$, $\Delta_{Central}$, $\Delta_{System}$: hai checkpoint khác nhau → hierarchical paired bootstrap, bắt buộc paired seed ID.

### 5.3 Training contract

**Trạng thái: đã chốt.** Chỉ `R` còn được chọn trên validation trong search space đã khóa dưới đây; mọi giá trị khác cố định trước khi chạy.

| Tham số | Giá trị | Lý do |
|---|---|---|
| `N` client | 5 | Gộp subject label thành 5 cụm chuyên khoa; khớp thiết kế trước |
| `fit:support` | MedQA **70:30** · MedMCQA **80:20** | Cân giữa sức mạnh training và độ dày kho exemplar |
| LoRA `r` | 16 | Chuẩn cho SLM; giữ payload ~15–30 MB/round |
| LoRA alpha / dropout | 32 / 0.05 | Khóa giống hệt giữa L0, F0, C0 |
| Local epochs `E` | 1 | Giảm client drift dưới non-IID |
| FL rounds `R` | search {4, 6, 8}, chọn trên validation | Chính là đại lượng $R_{90}$ ở mục 7.3 đo |
| C0 epochs | $R \times E$ của $R$ thắng cuộc | Khớp target exposures với F0 |
| Training seeds | 3 giá trị: **42, 43, 44** | 21 training run tổng cộng |

Số training run: $(N + 1 + 1)\times 3 = 21$ — Local sinh $N$ checkpoint mỗi seed, FedAvg và Centralized mỗi arm một checkpoint mỗi seed.

Danh sách seed phải **giống hệt nhau** giữa L0, F0 và C0 để paired seed ID ở mục 5.2 thi hành được. Giá trị cụ thể của seed không quan trọng; điều quan trọng là cùng một danh sách và được ghi lại.

**Kiến trúc LoRA khóa chung.** `r`, alpha, dropout và target modules giống hệt nhau giữa L0, F0 và C0. Khác một tham số nào trong nhóm này thì $\Delta_{FL}$ và $\Delta_{Central}$ không còn đo protocol nữa mà đo kiến trúc.

**FL protocol:**

- Constant learning rate xuyên suốt các round, không schedule, không decay.
- Client optimizer state reset sau mỗi round.
- Full participation: cả $N$ client tham gia mọi round.
- Chỉ LoRA adapter được truyền lên và phát xuống; base weight không bao giờ rời client.
- Aggregation là FedAvg có trọng số theo $|D_i^{fit}|$.

**Matched-compute giữa C0 và F0.** Ba đại lượng phải log cho cả hai arm:

- `total_target_exposures` — tổng số lần model nhìn thấy một training example, cộng qua toàn hệ thống.
- `total_optimizer_updates` — tổng số bước cập nhật tham số.
- `effective_batch_size`.

C0 khớp F0 ở **`total_target_exposures`** bằng cách train $R\times E$ epoch trên union các `fit` set. Hai đại lượng còn lại **không** khớp đồng thời được — C0 train trên tập lớn gấp $N$ lần nên số optimizer step mỗi epoch cao hơn — và báo cáo phải nói rõ khớp ở đại lượng nào.

**Ngân sách tuning bằng nhau.** Central và FL dùng cùng số lượng cấu hình thử trên validation. Không được tune FL kỹ hơn Central rồi kết luận về $\Delta_{Central}$.

**Early stopping.** Tiêu chí trên validation cho mọi arm trained; với FL, "early stopping" chính là việc chọn $R$ trong search space, và phải chọn xong trước khi chạm test.

## 6. Primary contrasts

$$
\Delta_{Base\text{-}ICL}=Acc(B1)-Acc(B0)
$$

$$
\Delta_{Local\text{-}ICL}=Acc(L1)-Acc(L0)
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

$$
\Delta_{System}=Acc(F2)-Acc(L1)
$$

$\Delta_{System}$ là headline comparison giữa phương pháp đề xuất và main baseline **Non-federated ICL**. Đây là contrast **system-level**: F2 và L1 khác nhau ở cả training protocol lẫn re-ranking, nên nó trả lời "toàn hệ thống có tốt hơn không", không phải "yếu tố nào tạo ra khác biệt". Bốn contrast một-yếu-tố ở trên mới là phần phân rã. Báo cáo $\Delta_{System}$ kèm B1 làm đối chiếu training-free.

Sáu $\Delta$ trên là **primary contrasts** và là family duy nhất chịu multiple-comparison correction.

Ngoài ra báo cáo một contrast phụ, **descriptive-only**:

$$
\Delta_{Central}=Acc(C0)-Acc(F0)
$$

$\Delta_{Central}$ không phải primary contrast, không nằm trong correction family, và **không** có pass/fail gate. Ba ràng buộc khi báo cáo nó:

- **Gọi đúng estimand.** C0 và F0 khác nhau ở *pooled optimization so với full FedAvg protocol* — số optimizer step, batch composition và tính liên tục của optimizer state đều khác — chứ không phải "chỉ khác bước aggregation".
- **Kèm matched-compute accounting.** Báo số optimizer step, số lần model nhìn thấy mỗi example, và wall-clock của cả hai arm theo mục 7.3. Không có các con số này thì $\Delta_{Central}$ không diễn giải được.
- **Không gọi C0 là upper bound trước khi đo.** Centralized không được đảm bảo thắng federated, nhất là ở quy mô client nhỏ và dữ liệu non-IID.

C0 không có ICL counterpart, nên hiệu ứng ICL trong chế độ centralized là **unmeasured**; nêu rõ điều này ở Limitations thay vì suy diễn từ $\Delta_{FL\text{-}ICL}$.

Các contrast ICL phải dùng paired predictions trên cùng test items. B0/B1, L0/L1 và F0/F1/F2 phải dùng cùng checkpoint tương ứng để không trộn ảnh hưởng của training với ảnh hưởng của inference context.

Mọi contrast được tính **riêng cho từng dataset** (MedQA, MedMCQA). Khi cần một con số tổng hợp, macro-average các $\Delta$ qua dataset theo mục 7.4.

## 7. Evaluation contract

### 7.1 Answer-scoring protocol

Một protocol duy nhất, áp dụng cho MedQA và MedMCQA. Mọi arm dùng protocol giống hệt nhau — đó là điều kiện để các $\Delta$ ở mục 6 đọc được.

$$
\text{Question} + \text{A/B/C/D}
\ \rightarrow\
\text{SLM sinh final-answer text}
\ \rightarrow\
\text{matcher}
\ \rightarrow\
\text{A/B/C/D}
$$

Ví dụ: option `B. Metformin`, model sinh `Metformin`, matcher ánh xạ về `B`.

Matcher là state machine, mỗi bước chỉ chạy khi bước trước thất bại:

1. Parse terminal label theo grammar đóng `Final answer: A|B|C|D`.
2. Exact-normalized whole-candidate-text match.
3. Semantic fallback bằng embedding match.
4. Tie hoặc confidence dưới ngưỡng → **unresolved**, là trạng thái cuối riêng, **không** ép về một đáp án.

Matcher, ngưỡng của nó, encoder revision và prompt template phải giống hệt nhau giữa các arm và freeze trước test.

### 7.2 Primary metric

$$
Acc_{\mathrm{pipeline}}
=
\frac{\#\{\text{generated answer được match đúng option}\}}{N}
$$

Denominator là **$N$ = toàn bộ item của cell**, không phải tập con parse thành công. Parse failure và unresolved tính là **sai**. Đây là điểm bắt buộc: accuracy tính trên tập con parse-thành-công bị thổi phồng khi model né câu khó bằng output sai format.

Tên gọi chính thức của metric này là **pipeline answer-selection accuracy**. Toàn bộ $Acc(\cdot)$ trong sáu contrast ở mục 6 là $Acc_{\mathrm{pipeline}}$.

Ngoài ra báo cáo tách theo client:

- Macro-client accuracy.
- Worst-client accuracy.
- Per-subject accuracy.

### 7.3 Secondary metrics

**Evaluator diagnostics:**

- Conditional-likelihood accuracy trên bốn option (denominator = $N$).
- Exact-match coverage (tỷ lệ dừng ở bước 1–2).
- Semantic-fallback rate (tỷ lệ phải dùng bước 3).
- Unresolved rate.
- Agreement matrix giữa pipeline accuracy và conditional-likelihood accuracy.
- **Position-bias macro-F1**: macro-F1 trên bốn nhãn vị trí A/B/C/D, kèm phân phối vị trí gold của từng dataset. Micro-F1 trùng với accuracy; khoảng cách giữa macro-F1 và accuracy phản ánh thiên lệch của model về một vị trí nhất định, sau khi đã trừ đi phần do dataset vốn lệch. Đây là diagnostic, không phải headline metric.

**Chất lượng dự đoán:** Calibration/ECE.

**Retrieval health và chất lượng:**

- Retrieval exclusion, refill và capacity-failure rate theo client và lý do.
- Recall@k / nDCG của retriever trên tập có relevance annotation, nếu có.

**Efficiency và federation:**

- **Communication cost**: bytes uplink (adapter upload) và downlink (server broadcast) mỗi round, cộng tổng của toàn bộ FL run. Chỉ truyền LoRA adapter, không truyền base weight.
- **Convergence rate**, báo cáo bằng hai đại lượng: (a) đường validation accuracy theo round, và (b) $R_{90}$ = số round nhỏ nhất để đạt 90% mức accuracy plateau của chính run đó. Không dùng từ "converged" nếu đường cong chưa phẳng trong ít nhất hai round cuối.
- Prompt-token P50/P95/max.
- Inference latency, throughput và peak VRAM.
- Train/eval wall-clock và số optimizer step, để đối chiếu matched-compute giữa C0 và F0.

**Privacy** (chỉ khi cơ chế tương ứng ở mục 8 đã triển khai): canary retrieve rate, canary leak rate trong generated output, `(epsilon, delta)`.

BLEU/ROUGE-L/BERTScore **không** dùng cho answer-selection; xem mục 7.7.

### 7.4 Báo cáo theo dataset, không gộp micro-accuracy

Kết quả phải báo riêng từng dataset:

- Native MedQA accuracy.
- Native MedMCQA accuracy.

Không gộp sample của hai dataset thành một micro-accuracy: MedQA và MedMCQA khác nhau về nguồn, độ khó và kích thước, nên micro-average sẽ để MedMCQA lấn át. Khi cần một con số tổng hợp, chỉ **macro-average các effect** $\Delta$ ở mục 6 qua dataset, không macro-average accuracy tuyệt đối.

### 7.5 Statistics

- Dùng paired seed IDs giữa các training arm cần so sánh.
- Dùng paired item bootstrap cho deterministic inference contrast.
- Dùng hierarchical paired bootstrap qua seed và evaluation unit cho trained arms.
- Báo cáo effect size và 95% confidence interval, không chỉ p-value.
- McNemar có thể dùng bổ sung cho hai prediction vector trên cùng test set.
- Freeze primary contrasts và multiple-comparison correction trước test. Correction family là đúng sáu primary contrast ở mục 6; $\Delta_{Central}$ và mọi secondary metric nằm ngoài family và được báo cáo descriptive.
- Nếu pipeline accuracy và conditional-likelihood accuracy **đảo dấu effect hoặc đảo thứ hạng arm** trong cùng một cell, kết luận của cell đó phải mang nhãn **"evaluator-dependent"**; không im lặng chọn một evaluator làm ground truth.
- Toàn bộ kết quả được framing **exploratory**, không dùng từ "confirmatory" và không đặt pass/fail gate trên effect size.

### 7.6 Quy tắc phát ngôn

Được phép claim:

> We evaluate a unified free-text generation and answer-matching pipeline on native four-way medical MCQ benchmarks. This benchmark evaluates answer selection rather than open-ended question answering.

**Không** được claim:

> The system is validated on both MCQ and open-ended QA.

Model có sinh free-text, nhưng nó sinh **trong khi nhìn thấy bốn option**, nên output là hành vi chọn đáp án chứ không phải trả lời tự luận. Muốn claim thứ hai, cần một evaluation riêng trên câu hỏi không hiển thị A/B/C/D, chấm bằng metric riêng: token-F1, semantic answer similarity, medical-concept agreement, human factuality audit. Track này là **secondary analysis**, không nằm trong sáu contrast chính ở mục 6.

### 7.7 Ánh xạ từ metric list gốc của proposal

Danh sách metric trong proposal ban đầu được xử lý như sau. Không metric nào bị bỏ im lặng.

| Metric gốc | Quyết định | Nằm ở đâu |
|---|---|---|
| **Accuracy** | ✅ **Primary** | $Acc_{\mathrm{pipeline}}$, mục 7.2, denominator $=N$ |
| **F1-score** | ⚠️ Hạ xuống **diagnostic** | Position-bias macro-F1, mục 7.3 |
| **BLEU** | ❌ **Loại** | — |
| **ROUGE-L** | ❌ **Loại** | — |
| **BERTScore** | ❌ **Loại khỏi protocol chính** | Chỉ dùng được ở track free-form tùy chọn, mục 7.6 |
| **Communication Cost** | ✅ **Secondary** | Mục 8.3, tách uplink/downlink/tổng |
| **Convergence Rate** | ✅ **Secondary** | Mục 8.3, đường cong theo round + $R_{90}$ |

Lý do loại ba metric sinh văn bản:

- Sau mục 2, **toàn bộ evaluation là four-way answer selection**. Output cần chấm là một nhãn trong `{A,B,C,D}`, không phải một đoạn văn. BLEU và ROUGE-L đo n-gram overlap giữa hai chuỗi; với output `Metformin` so với reference `Metformin` chúng luôn bằng 1 hoặc 0 và mang đúng thông tin mà accuracy đã mang, chỉ nhiễu hơn.
- BERTScore trên chuỗi một-vài-token không ổn định và có thể cho điểm cao cho một option **sai nhưng gần nghĩa** — tức thưởng cho câu trả lời sai chỉ vì nó nằm cùng vùng ngữ nghĩa với đáp án đúng.
- Ba metric này chỉ có nghĩa nếu có output tự luận thật để chấm. Điều kiện đó chỉ tồn tại ở track free-form tùy chọn ở mục 7.6, và ngay tại đó cũng nên ưu tiên token-F1, semantic answer similarity và medical-concept agreement hơn BLEU/ROUGE-L.

Nếu reviewer hoặc hội đồng yêu cầu giữ BLEU/ROUGE-L, cách đúng là **mở track free-form ở mục 7.6**, không phải áp chúng lên MCQ.

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
- [ ] B0/B1, L0/L1 và F0/F1/F2 dùng cùng checkpoint trong từng nhóm.
- [ ] Prompt template và answer-scoring protocol giống nhau giữa các arm.
- [ ] LoRA `r`/alpha/dropout/target modules giống hệt nhau giữa L0, F0 và C0.
- [ ] Ngân sách tuning của Central và FL bằng nhau; $R$ đã chốt trên validation trước khi chạm test.
- [ ] `total_target_exposures`, `total_optimizer_updates` và `effective_batch_size` đã log cho C0 và F0.
- [ ] Configuration, manifest và model revision đã được hash và freeze trước test.
- [ ] Test set không được dùng trong tuning hoặc debugging decision.
- [ ] Exclusion/refill/capacity-failure được lưu theo query và lý do.
- [ ] Multimodal đã được loại khỏi scope và outline.
- [ ] Matcher, ngưỡng fallback và evaluator protocol đã freeze trước test.
- [ ] Data seed đã freeze và dùng chung cho mọi arm; không arm nào chạy trên partition/manifest khác.
- [ ] Training seed theo mục 5.2 đã khai báo; paired seed ID map được giữa L0, F0 và C0.
- [ ] Decoding deterministic (greedy, temperature 0) và encoder revision đã pin cho mọi arm.
- [ ] Denominator của mọi accuracy là N, parse-failure và unresolved tính là sai.

## 11. Post-run reporting checklist

Chạy **sau** khi eval test xong. Đây là kiểm tra tính đầy đủ và trung thực của báo cáo, không phải gate chặn.

- [ ] Sáu primary contrast ở mục 6 đều có effect size kèm 95% CI, báo riêng theo dataset.
- [ ] Mọi bảng ghi rõ số training seed đã dùng.
- [ ] $\Delta_{Central}$ báo kèm matched-compute accounting và không mang ngôn ngữ upper bound hay pass/fail.
- [ ] Parse coverage, semantic-fallback rate và unresolved rate đã báo cáo cho mọi cell.
- [ ] Agreement giữa pipeline accuracy và conditional-likelihood accuracy đã báo cáo; cell nào đảo dấu hoặc đảo thứ hạng đã dán nhãn "evaluator-dependent".
- [ ] Không có micro-accuracy gộp qua dataset trong bất kỳ bảng nào.
- [ ] Claim về open-ended QA không xuất hiện, trừ khi track free-form ở mục 7.6 đã thực sự chạy.
- [ ] Claim privacy vẫn ở mức "data-local by design" nếu cơ chế tương ứng chưa được đánh giá.
- [ ] Config/manifest/model hash trong log khớp bản đã freeze ở mục 10.
- [ ] Toàn bộ kết luận mang framing exploratory.
