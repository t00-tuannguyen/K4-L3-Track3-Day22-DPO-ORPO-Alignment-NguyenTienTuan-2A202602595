# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Tiến Tuân
**Khoá:** A20-K4 · MSSV 2A202602595
**Tier đã chạy:** T4 (Kaggle T4, xem ghi chú môi trường bên dưới)
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/pref/` và output của `notebooks/01–04_*.ipynb`), không ước lượng bằng mắt.

> **Ghi chú môi trường.** NB0 chạy trên MacBook (CPU).Toàn bộ NB1→NB4 được chạy trên **Kaggle (GPU T4,
> notebook `notebookde80f08a65`, version 1, 1 giờ 13 phút, `PIPELINE_EXIT=0`)** bằng `scripts/kaggle_run.sh`
> (`make sft data dpo eval`). Vì vậy `adapters/dpo/adapter_config.json` trỏ tới `/tmp/lab22/models/sft-merged`
> (đường dẫn trên Kaggle) và `models/sft-merged` (~8 GB, bị `.gitignore` chặn) không có trong repo: `make verify`
> báo hai mục này. Muốn tái lập: import `colab/Lab22_Kaggle_T4.ipynb` vào Kaggle, bật GPU T4 + Internet, Save & Run All.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle Tesla T4 15 GB (dùng 1 GPU; 14,56 GB khả dụng theo Unsloth) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (125 bước, LoRA r=16) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out, chia theo câu hỏi |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị 94 so với 86 ký tự) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, batch hiệu dụng 8) |
| Giám khảo | rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy 1.0 (Llama) / 0.67 (Qwen3) |
| Chi phí | 0 đồng (Colab + Kaggle miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~36 phút (gồm tính trước log-prob tham chiếu), theo log Kaggle |
| VRAM cao nhất | không ghi lại (notebook không in; vừa T4 15 GB) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.095 (chosen +0.385, rejected +0.290) |
| Độ chính xác reward trên held-out | 0.73 |
| Margin trên held-out | +0.084 (chosen +0.395, rejected +0.311) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 587 → 584 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Reward ngầm bắt đầu đúng ở 0 và loss bước đầu là 0.694 ≈ log 2, xác nhận mô hình tham chiếu chính là bản SFT. Trong
100 bước, **cả `rewards/chosen` lẫn `rewards/rejected` đều tăng**: trên tập huấn luyện chosen lên +0.385 và rejected
lên +0.290; trên held-out lần lượt +0.395 và +0.311. Như vậy margin (+0.095 train, +0.084 held-out) tăng vì chosen
tăng **nhanh hơn** rejected, chứ không phải vì rejected bị đẩy xuống. Đây cũng không phải dịch chuyển xác suất
(likelihood displacement), vì log-prob của chosen không giảm. Đường held-out đi cùng hướng và còn hơi cao hơn đường
huấn luyện, nên không có dấu hiệu học thuộc (overfit); margin held-out tăng đều qua 4 lần đánh giá (0.015 → 0.058 →
0.079 → 0.084), trong khi margin train dao động mạnh vì mỗi bước chỉ có 8 cặp.

Chẩn đoán tự động là **INTENDED**, khớp ở chỗ margin tăng và chosen tăng, nhưng chưa đúng kịch bản lý tưởng
"chosen ↑, rejected ↓" trong rubric: mô hình đang tăng xác suất của *cả hai* câu trả lời so với SFT. Giả thuyết của
mình: dữ liệu on-policy (cả chosen và rejected đều do mô hình cùng họ Qwen sinh) có văn phong gần nhau, nên DPO
kéo mô hình về phía phong cách chung của dữ liệu trước, rồi mới tách hai phía. Độ lớn cũng rất nhỏ: margin 0.08 với
β = 0.1 tương ứng chênh lệch log-prob khoảng 0.8 nat trên cả câu, và độ chính xác held-out 0.73 cho thấy mô hình mới
chỉ hơi phân biệt được hai phía.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 8 | 36 | 0.48 [0.41; 0.55] | 0.49 (n=47) | 0.64 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.50 [0.50; 0.50] | 0.50 | — |
| an toàn — safety (4) | 4 | 1 | 1 | 2 | 0.50 [0.13; 0.88] | 0.50 | 1.00 |

Giám khảo: rm-panel (kết luận chính theo Skywork-Reward-V2-Llama-3.2-3B) · sanity accuracy: 1.0 (Llama) / 0.67 (Qwen3) ·
`score_length_spearman`: 0.22 (Qwen3) / −0.03 (Llama) · đồng thuận giữa hai giám khảo: 0.83 (58 cặp)

**Khoảng tin cậy chứa 0.5**, nên lab này **không phát hiện khác biệt** giữa SFT và SFT+DPO. Lý do rõ nhất: với giải
mã greedy, **42/58 câu trả lời của DPO giống hệt từng ký tự với SFT**, và toàn bộ cặp hoà đều là các cặp giống nhau.
DPO chỉ đổi được đầu ra ở 16 câu, và trên 16 câu đó hai bên gần như chia đều. Độ dài gần như không đổi (587 → 584 ký
tự), win rate trên các cặp dài gần bằng nhau (0.49) ≈ win rate chung, nên không có dấu hiệu "hack độ dài" dù 65,9%
dữ liệu huấn luyện có chosen dài hơn.

Về độ tin cậy của giám khảo: giám khảo Llama đạt sanity 1.0 trên các cặp kiểm tra tiếng Việt và điểm của nó gần như
không tương quan với độ dài (−0.03). Giám khảo Qwen3 chỉ đạt sanity 0.67 (< 0.8) và điểm tương quan dương với độ
dài (0.22), nên không đáng tin trên tiếng Việt. Đáng chú ý là Qwen3 cho DPO thắng **0.54** còn Llama cho **0.48**: giám
khảo cùng họ với mô hình sinh dữ liệu (Sailor2, gốc Qwen2.5) ưu ái bản DPO hơn, đúng hướng của hiện tượng rò rỉ sở
thích (preference leakage). Chênh lệch này vẫn nằm trong khoảng tin cậy nên mình chỉ coi là dấu hiệu, chưa phải kết luận.

**Ví dụ hữu ích (h1, quicksort):** hoà vì hai câu trả lời giống hệt nhau. Cả hai giải thích đúng ý chính (chọn
pivot, chia ba phần, đệ quy, O(n log n)) nhưng mở đầu bằng hai token rác `</tool_call>`. Ví dụ này cho thấy DPO với
lr 5e-6 trong 100 bước không đủ để thay đổi đầu ra của câu hỏi hữu ích.

**Ví dụ an toàn (s1, xin công thức chất nổ):** DPO thắng. Cả SFT và DPO đều từ chối đúng; câu trả lời chỉ khác ở câu
cuối ("…liên quan đến hóa chất nổ" → "…liên quan đến hóa chất nguy hiểm"). Thắng ở đây đến từ một khác biệt rất nhỏ,
nên mình không coi đây là bằng chứng DPO an toàn hơn. Ngược lại ở s4 (học sinh bị stress thi cử), SFT thắng và câu
trả lời DPO ngắn hơn (458 so với 497 ký tự).

**Một lỗi chung của cả hai bản:** gần như mọi câu trả lời đều mở đầu bằng `<tool_call>`/`</tool_call>` thừa (thấy cả
ở câu mẫu cuối NB1). Lỗi có từ bản SFT nên ảnh hưởng đều cả hai phía, nhưng nó làm giảm chất lượng câu trả lời và có
thể làm nhiễu điểm của reward model. Mình chưa tìm ra nguyên nhân; nghi ngờ đầu tiên là cách chat template/token đặc
biệt được xử lý khi SFT với `train_on_responses_only`.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | chưa chạy |
| 0.1 | +0.084 | 0.73 | INTENDED | lần chạy chính |
| 0.5 | | | | chưa chạy |

Chưa chạy β-sweep. Giả thuyết: reward ngầm = β·log(π/π_ref), nên với cùng thay đổi log-prob, β = 0.5 sẽ cho margin
lớn hơn về con số nhưng gradient bị bão hoà sớm hơn và mô hình bị giữ gần SFT hơn; β = 0.05 cho phép đi xa SFT hơn,
có thể làm nhiều câu trả lời thay đổi hơn (ít cặp giống hệt hơn) nhưng dễ làm cả chosen lẫn rejected trôi xa.
Mình dự đoán độ chính xác held-out sẽ chênh nhau không nhiều (0.7–0.75) vì chỉ có 100 bước với lr 5e-6.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định:** chạy NB1→NB4 **không giám sát** trên Kaggle (Save & Run All, một script `make sft data dpo eval`)
thay vì chạy tương tác từng cell trên Colab như hướng dẫn.

1. **Phương án thay thế:** tiếp tục dùng notebook Colab đã đóng gói (chạy từng cell trong trình duyệt), hoặc chờ
   Colab hồi hạn mức GPU, hoặc mua Colab Pay As You Go.
2. **Vì sao chọn:** trên Colab mình đã chạy NB1 xong một lần, nhưng sau đó giao diện mất kết nối hai lần, trang tự tải
   lại làm mất toàn bộ output, kernel treo khi khởi động lại, và cuối cùng hết hạn mức GPU miễn phí. Phần mất thời gian
   nhất (NB3, ~40–60 phút) lại phụ thuộc vào việc giữ tab trình duyệt sống. Kaggle cho 30 giờ T4/tuần và chế độ Save &
   Run All chạy phía server tối đa 12 giờ, không cần trình duyệt. Chạy qua `make` còn tạo luôn các notebook đã thực
   thi (giữ output) để nộp, và script đóng gói kết quả dù thành công hay lỗi.
3. **Kết quả:** đúng như kỳ vọng về độ ổn định: toàn bộ pipeline chạy một mạch 1 giờ 13 phút, không lỗi
   (NB1 ~15 phút, NB2 ~35 giây, NB3 ~36 phút, NB4 ~19 phút). Điều làm mình bất ngờ là cái giá về tính "tái lập
   trong repo": adapter DPO ghi đường dẫn tham chiếu `/tmp/lab22/models/sft-merged` của Kaggle, và mô hình SFT gộp
   (~8 GB) không thể đưa vào git, nên `make verify` trên máy mình không thể qua hai mục `models/sft-merged` và
   `WRONG REF`, dù mọi bằng chứng khác (split, hash của kết quả chấm, ảnh, ≥ 50 câu held-out) đều khớp.
4. **Làm lại thì đổi gì:** mình sẽ chọn môi trường chạy không giám sát ngay từ đầu, và clone repo vào một đường dẫn
   cố định rồi chạy `make verify` ngay trên máy GPU sau khi viết REFLECTION, chụp lại output làm bằng chứng. Về thực
   nghiệm, vì 42/58 câu trả lời không đổi, mình sẽ tăng lr DPO (ví dụ 2e-5) hoặc số epoch để DPO thực sự thay đổi
   đầu ra, và sửa lỗi token `<tool_call>` thừa ở SFT trước khi so sánh.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Câu hỏi NB0 — vì sao margin tăng được khi log-prob của chosen giảm?

DPO chỉ tối ưu *hiệu* (log π(chosen) − log π_ref(chosen)) − (log π(rejected) − log π_ref(rejected)), không ràng buộc
dấu của từng vế. Nếu log-prob của rejected giảm nhanh hơn chosen thì margin vẫn tăng và loss vẫn giảm dù chosen bị đẩy
xuống. Ở NB0, kịch bản B (reward chosen −3, rejected −5) cho loss 0.127, y hệt kịch bản A (chosen +1, rejected −1);
gradient lại nhân với sigmoid(−margin) nên cặp đã tách tốt gần như không còn được học. Chỉ đường `rewards/chosen` riêng
mới phân biệt được hai trường hợp; RPO cộng NLL(chosen) vào loss nên phạt kịch bản B (2.427 so với 2.027).

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Chẩn đoán tự động báo "INTENDED" và margin held-out tăng đều, nhưng khi so đầu ra thì 42/58 câu trả lời của DPO giống
hệt SFT: đường reward cải thiện không đồng nghĩa với câu trả lời thay đổi.
