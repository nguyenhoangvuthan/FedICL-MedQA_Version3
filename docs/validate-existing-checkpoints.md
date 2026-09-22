# Chọn LoRA adapter từ checkpoint đã có, không train lại

## Lấy kết quả đúng epoch 3

Không cần chạy `validate-checkpoints` nếu đã muốn cố định epoch 3. Nạp trực tiếp
adapter đã lưu và chạy đánh giá trên test:

```bash
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm F0 --seed 42 --epoch 3 --gpu 0
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm C0 --seed 42 --epoch 3 --gpu 0
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm L0 --seed 42 --epoch 3 --gpu 0
```

Thay đường dẫn config bằng run thực tế trên server. Đổi seed để lấy các seed còn
lại; dùng F1/C1/L1 cho đánh giá ICL nếu arm được config hỗ trợ. Các arm train ICL
FT0/FT1/CT0/CT1/LT0/LT1 cũng nhận `--epoch` và nạp family tương ứng.

Với FL một local epoch mỗi round, epoch 3 là adapter global sau round 3.
Centralized/Local yêu cầu checkpoint **cuối epoch 3**, không lấy checkpoint đang
ở giữa epoch. Local theo cấu hình gốc chỉ train một epoch/client, nên lệnh L0
chỉ chạy được khi thực sự có checkpoint cuối epoch 3. Nếu checkpoint chưa từng
được lưu hoặc đã bị xóa, lệnh báo thiếu; không tự train hay thay bằng epoch khác.

Kết quả được ghi riêng tại
`arms/<dataset>/<ARM>/seed-42/test/epoch-3/summary.json` và `predictions.jsonl`.
Không dùng đồng thời `--epoch` với `--round` hay `--checkpoint best-validation`.

## Tự chọn checkpoint theo validation

Chạy từ thư mục repo trên server chứa checkpoint và dữ liệu đã chuẩn bị. Dùng đúng
`sealed_config.json` của lần train đó; trường `experiment.output_dir` phải trỏ tới
output đang chứa checkpoint. Các lệnh dưới đây dùng `outputs/a5000` làm ví dụ;
thay bằng output thực tế nếu checkpoint thuộc lần chạy khác.

```bash
uv run --no-sync fedicl-mqa validate-checkpoints --config outputs/a5000/sealed_config.json --mode all --all-seeds --gpu 0
```

`--mode all` xử lý Local, Federated và Centralized. Để chạy riêng, dùng
`--mode local`, `--mode federated` hoặc `--mode centralized`; để chỉ xử lý seed 42,
thay `--all-seeds` bằng `--seed 42`. Các family `local-icl`, `federated-icl`,
`centralized-icl` và `local-matched` cũng được hỗ trợ qua `--mode` riêng.
Với `local-matched`, có thể dùng `--fl-round 4|6|8` để chỉ đúng thư mục checkpoint.

Lệnh nạp adapter đã lưu và tính validation loss, không gọi training, backward hay
optimizer. Local dùng validation riêng của từng client; Federated và Centralized
dùng validation gộp. Metric là negative log-likelihood trung bình theo số token
được giám sát trong câu trả lời (kể cả EOS), không tính prompt/padding. Tất cả family
dùng prompt không ICL khi chọn checkpoint để cố định tiêu chí lựa chọn; đây không
phải metric accuracy của đánh giá cuối.

Lệnh xét mọi checkpoint train hợp lệ còn trên đĩa, gồm checkpoint cuối epoch,
cuối round và checkpoint theo step. Round 0 chưa train bị loại. Loss thấp nhất
thắng; nếu bằng nhau, chọn checkpoint có tiến độ train sớm hơn. Vì vậy FL có thể
chọn cả round 1, 2, 3, 5 hoặc 7. Nếu chỉ còn một checkpoint thì chỉ có một ứng viên;
không thể phục hồi epoch đã bị cơ chế retention xóa nếu không train lại.

Trong mỗi thư mục checkpoint, lệnh ghi:

- `validation_selection.json`: loss, epoch/round/step, hash checkpoint và tập
  validation, trạng thái hoàn tất và checkpoint được chọn.
- `best_validation_checkpoint.txt`: tên checkpoint thắng. Adapter vẫn nằm tại
  `<thư mục checkpoint>/<tên được chọn>/adapter/`, không tạo bản sao trọng số.

Checkpoint, optimizer, `last_checkpoint.txt` và `best_checkpoint.txt` cũ được giữ
nguyên. Có thể chạy lại cùng lệnh sau khi bị ngắt: các điểm đã tính được dùng lại
khi hash checkpoint và tập validation còn khớp. Checkpoint hỏng được ghi cảnh báo
và bỏ qua; checkpoint khác config/model/seed/family bị từ chối.

Đánh giá adapter đã chọn trên test (ví dụ seed 42):

```bash
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm L0 --seed 42 --checkpoint best-validation --gpu 0
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm F0 --seed 42 --checkpoint best-validation --gpu 0
uv run --no-sync fedicl-mqa evaluate --config outputs/a5000/sealed_config.json --arm C0 --seed 42 --checkpoint best-validation --gpu 0
```

Đổi sang L1/F1/C1 nếu muốn đánh giá với ICL (C1 yêu cầu config bật family đó).
Các arm có prior theo round như F2/FP/FS không dùng tùy chọn này vì prior cũ thuộc
checkpoint đã chọn theo protocol cũ.

Kết quả mới nằm tại
`arms/<dataset>/<ARM>/seed-<n>/test/best-validation/`, có metadata của checkpoint
thực sự được nạp. Lệnh `evaluate` mặc định, các sweep, pipeline và báo cáo so sánh
theo protocol cũ vẫn dùng lựa chọn cũ. Kết quả `best-validation` là phân tích riêng
vì mỗi family có thể chọn ngân sách train khác nhau; không tự ghép vào bảng causal
contrasts yêu cầu ngân sách khớp nhau. Validation selection giảm nguy cơ chọn
adapter cuối đã overfit, nhưng mức cải thiện thực tế cần đo trên test.
