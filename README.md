# ffd — detector CLIP ViT-L/14 (224) + LoRA (CompFFD, v2)

So sánh hai arm huấn luyện trên FF++: `raw` (ảnh raw train) và `raw_aug` (raw train + mọi ảnh tăng cường `rep`/`seam`
của preprocessing). Đánh giá mức frame trên FF++ val, FF++ test và Celeb-DF.
Trạng thái: code v2 mới chạy test CPU (CLIP tí hon, dữ liệu tổng hợp MINH HỌA); **chưa chạy thử trên GPU**.

## Cài đặt

Venv riêng cho detector (PowerShell, trong thư mục `detector/`):

```powershell
py -3.14 -m venv .venv                            # máy GPU: Python 3.14; máy local: py -3.15
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements-gpu.txt     # máy GPU; máy local không GPU: requirements-cpu.txt
python -c "import torch, transformers, cv2; print(torch.__version__, torch.cuda.is_available())"
```

- `requirements-gpu.txt` (torch cu130, Python 3.14) và `requirements-cpu.txt` (torch CPU, Python 3.15, chỉ chạy test)
  đã ghi phiên bản torch; `requirements.txt` là phần chung, không cài trực tiếp. Ghi chú chọn bản CUDA nằm đầu file.
- Không cài torchvision. Detector độc lập với preprocessing: chỉ đọc dữ liệu trên đĩa.
- Mỗi terminal mới: activate lại rồi gọi `python -m ffd.cli ...` (không dùng `py`).

### Config

```powershell
Copy-Item config.example.yaml config.yaml
```

Sửa `data_root` = `out_root` của preprocessing (thư mục chứa `ffpp/` và `celebdf/`). `ckpt_root` mặc định
`checkpoints/clip`. Mọi khóa đổi được tạm thời bằng `--set k=v` (vd `--set num_workers=8`); khóa lạ -> lỗi.
Không có pilot (bỏ 16/11/2026): `main` dùng đúng `lr`, `p_jpeg` (và mọi khóa khác) trong `config.yaml`.
Giá trị trong `config.example.yaml`:

- `lr: 1.0e-4`: theo run thử `lr1e-4_p0.5`, AUC frame val 0.9954 sau 3 epoch (16/11).
- `p_jpeg: 0.5`.
- `grad_checkpointing: false`: tắt nhanh hơn ~21%, VRAM ~8.8 GB (smoke 16/11). Khóa này **không vào
  `config_hash`** (không đổi phép toán, chỉ đổi bộ nhớ/tốc độ), cùng `data_root`, `ckpt_root`, `num_workers`,
  `run_name`.

Mọi khóa khác vào `config_hash`: đổi giữa các lần chạy main thì run cũ báo lỗi (xóa thư mục run hoặc đổi `ckpt_root`).

### Test

```powershell
python -m pytest -q                                                                  # CPU, MINH HỌA
$env:FFD_REAL_CONFIG = "config.yaml"; python -m pytest -q --run-gpu tests/test_smoke_gpu.py   # máy GPU, dữ liệu thật
```

## Thứ tự chạy: smoke -> main

PowerShell, thư mục `detector/`, đã activate `.venv`.

1. **smoke** (GPU, ViT-L thật): vài bước train, in giây/bước, VRAM đỉnh, `model_commit`; chép 16 ảnh `rep` + 16 ảnh
   `seam` vào `checkpoints/clip/smoke/samples/` để duyệt bằng mắt.
   ```powershell
   python -m ffd.cli smoke --config config.yaml --steps 50 --set num_workers=8
   ```
   **Bắt buộc trước main:** chép `model_commit` in ra vào `model_revision` của `config.yaml` để ghim phiên bản
   model (`model_revision` vào `config_hash`; ghim sau khi main đã chạy sẽ làm run cũ báo lỗi config khác).
2. **main**: train + eval `raw` và `raw_aug` với `lr`, `p_jpeg` lấy từ `config.yaml`, rồi so sánh (`compare.json`),
   in bảng tổng hợp và ghi `checkpoints/clip/summary.md`. Thiếu `<data_root>/celebdf/manifest.parquet`
   (**phải tiền xử lý Celeb-DF trước main**) -> lỗi.
   ```powershell
   python -m ffd.cli main --config config.yaml --set num_workers=8
   if ($LASTEXITCODE -ne 0) { throw "main lỗi" }
   ```

Mỗi epoch in một dòng (stderr), vd
`[raw] epoch 3: loss 0.1234 | val ACC 0.9500 AUC 0.9900 PREC 0.9600 REC 0.9700 F1 0.9650 BCE 0.1500 | best e3, bad 0/5`
(số MINH HỌA). Sau eval mỗi run in bảng `dataset, n_frames, ACC, AUC, PREC, RECALL, F1, AUC_video` ra stdout.

`main` chạy lại được sau khi bị ngắt: run có `result.json` đủ dataset được bỏ qua (thiếu dataset, hoặc
`celebdf` bị `skipped` mà nay đã có dữ liệu -> chỉ eval lại); có `done.flag` thì chỉ eval; có `last.pt` thì train
tiếp (resume); chưa có gì thì train mới. Run cũ có `config_hash` khác config hiện tại -> lỗi
(xóa thư mục run hoặc đổi `ckpt_root`).

Lệnh lẻ (khi cần):

```powershell
python -m ffd.cli train --config config.yaml --set arm=raw_aug [--resume | --force]   # một run; không dùng cả hai cờ
python -m ffd.cli eval --run checkpoints/clip/raw [--datasets val,ffpp_test,celebdf] [--force] [--data-root <data_root trên máy này>]
python -m ffd.cli compare --ckpt-root checkpoints/clip
```

`eval`: `--datasets` mặc định `val,ffpp_test,celebdf`; đã có `result.json` thì phải thêm `--force` (ghi đè).
Run cũ có `run_name` bắt đầu bằng `pilot/` (tạo trước 16/11) chỉ được chấm `val` (chống leakage).

`--no-progress` (mọi lệnh) tắt thanh tiến trình ở stderr; đặt **sau** tên lệnh, vd
`python -m ffd.cli main --no-progress --config config.yaml`.

## Bố cục `checkpoints/clip/`

```
checkpoints/clip/
  smoke/smoke.json, smoke/samples/{rep,seam}_00..15.png
  raw/  raw_aug/          # run chính
  compare.json
  summary.md              # bảng tổng hợp cuối main
```

Mỗi thư mục run: `run_info.json` (config đầy đủ, `config_hash`, `model_commit`, phiên bản thư viện, sha256 dữ liệu),
`train_log.csv` (mỗi epoch một dòng, cột: `epoch, step, train_loss, lr, val_auc_frame, val_bce, val_acc,
val_precision, val_recall, val_f1, frac_real, frac_rep, frac_seam, frac_jpeg, seconds`; chỉ số val ngưỡng logit >= 0,
chỉ để theo dõi; log cũ 11 cột khi resume được ghi lại với header mới, ô thiếu để trống), `last.pt` (để resume), `best.pt` (epoch tốt nhất theo AUC frame val, hòa trong
1e-4 thì BCE val thấp hơn), `done.flag` (`epochs_run`, `best_epoch`, `stopped_early` = dừng vì patience trước `max_epochs`, `total_steps`),
`scores_<dataset>.parquet` (logit từng frame), `result.json`, `result.md` (cùng bảng in ra lúc eval dạng Markdown:
tiêu đề run, `best_epoch`/`epochs_run`, bảng ACC/AUC/PREC/RECALL/F1/AUC_video từng dataset; `ffpp_test` thêm bảng
AUC theo method).

## `result.json` (một run, chấm `best.pt`)

- `arm`, `config` (model, commit, `lr`, `p_jpeg`, `max_epochs`, `patience`, LoRA, sha256 dữ liệu), `best_epoch`,
  `epochs_run`, `stopped_early`, `total_steps`.
- Mỗi dataset (`val`, `ffpp_test`, `celebdf`): `auc` (AUC frame, lớp dương = fake), `auc_ci95`
  (bootstrap 2000 lần trên frame, seed 0), `acc`, `f1`, `recall`, `precision`, `confusion_matrix` (ngưỡng xác suất
  0.5), `n_frames`, `n_videos`, `n_videos_dropped` (video bị preprocessing loại hẳn theo `drops.parquet`; không tính video lỗi khi
  tiền xử lý), `auc_video` (tham khảo).
  `ffpp_test` thêm `auc_per_method`. Thiếu dữ liệu Celeb-DF -> `{"skipped": ...}` (chỉ khi gọi `eval` lẻ; main
  từ chối chạy). Chấm Celeb-DF -> `config.manifest_sha256.celebdf` = sha256 manifest Celeb-DF đã chấm.

## `compare.json` (raw_aug so với raw)

- `ffpp_test`, `celebdf`: `auc_raw`, `auc_raw_aug`, `diff` = AUC(raw_aug) − AUC(raw), `diff_ci95` (paired bootstrap
  2000 lần trên frame, seed 0), `delong_p` (DeLong ghép cặp). Hai arm phải chấm cùng tập frame và nhãn.
- `total_steps`, `epochs_run`, `best_epoch` của từng arm; `note`.

`summary.md` (cuối `main`, cũng in ra stdout): mỗi (arm, dataset) một dòng ACC, AUC, PREC, RECALL, F1; rồi với
`ffpp_test`, `celebdf`: AUC raw, AUC raw_aug, hiệu, CI95, p DeLong (lấy từ `compare.json`).

## Ghi chú Exploratory

`compare.json` luôn ghi: *"Exploratory: 1 run/arm, không prereg; frame trong cùng video tương quan nên CI và p-value
lạc quan."* Chạy 1 run/arm (seed 0) và không pre-register là **quyết định có chủ ý** của chủ đề tài ngày 15/11/2026
(ngoại lệ với nguyên tắc 4, 5 của project). Kết quả chỉ dùng để định hướng, không dùng làm claim.

Chống leakage: `lr` chọn từ run thử trên FF++ val, epoch tốt nhất chỉ chọn trên FF++ val; FF++ test và Celeb-DF chỉ được đọc khi eval
`best.pt`. **Không chỉnh gì sau khi xem test.**
