# ffd — detector CLIP ViT-L/14 (224) + LoRA (CompFFD, v3)

So sánh bốn arm huấn luyện trên FF++ (thứ tự cố định, arm mới thêm sau): `raw` (ảnh raw train), `raw_aug` (raw train +
mọi ảnh tăng cường `rep`/`seam` của preprocessing), `raw_aug_filt` (như `raw_aug` nhưng lọc `rep`/`seam` yếu rồi cân bằng
lại seam, xem mục "Hai arm v3") và `raw_aug_filt_loc` (như `raw_aug_filt` + head dự đoán vị trí, loss `L_loc`).
Đánh giá mức frame trên FF++ val, FF++ test và Celeb-DF. Spec: `../spec_v3_loc_filter_2026-11-17.md` mục 4.4.
Trạng thái: code mới chạy test CPU (CLIP tí hon, dữ liệu tổng hợp MINH HỌA); **chưa chạy thử trên GPU**.

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

Mặc định trong `TrainConfig` cũng là `lr` 1e-4, `grad_checkpointing` false. Mọi khóa khác vào `config_hash`: đổi
giữa các lần chạy main thì run cũ báo lỗi (xóa thư mục run hoặc đổi `ckpt_root`). `main` in khối config hiệu lực
(`data_root`, `ckpt_root`, `lr`, `p_jpeg`, `batch_size`, `max_epochs`, `patience`, `seed`, `grad_checkpointing`,
`model_name`, `model_revision`, `num_workers`) ra stdout trước khi train: **đọc lại khối này trước khi để chạy**.

**Nếu `config.yaml` chép từ mẫu cũ** (trước 16/11), sửa tay vì khóa ghi rõ trong file thắng mặc định:

- `lr: 1.0e-4` (mẫu cũ ghi `2.0e-4`);
- `grad_checkpointing: false` (mẫu cũ ghi `true`);
- `model_revision`: ghim commit HF (`model_commit` in ra từ `smoke`);
- xóa mọi khóa hay chú thích liên quan pilot (khóa lạ -> `load_train_config` báo lỗi).

**Run cũ không dùng tiếp được:** commit 16/11 bỏ `grad_checkpointing` khỏi `config_hash`, nên mọi thư mục run tạo
trước đó (vd `checkpoints/clip/pilot/`, `raw/`/`raw_aug/` dở dang) có `config_hash` khác -> `main`/`eval` báo lỗi.
Xóa các thư mục đó hoặc đặt `ckpt_root` mới.

### Test

```powershell
python -m pytest -q                                                                  # CPU, MINH HỌA
$env:FFD_REAL_CONFIG = "config.yaml"; python -m pytest -q --run-gpu tests/test_smoke_gpu.py   # máy GPU, dữ liệu thật
```

## Hai arm v3: `raw_aug_filt`, `raw_aug_filt_loc`

Hằng số gắn arm nằm trong code (`ffd/data/augfilter.py`), **không** phải khóa `TrainConfig`:

- **Quy tắc lọc** (cùng cho `rep` và `seam`): giữ dòng có `r >= 0.75` và `mask_frac_final >= τ_m`, với `τ_m` = median
  `mask_frac_final` của **mọi dòng `rep` chưa lọc** (toàn cục). Lọc làm rỗng `rep` hoặc `seam` -> lỗi rõ ràng.
- **Cân bằng lại** sau lọc: `p_real = S/(R+S)` so với `p_fake = P/(F+P)` (R, F = số ảnh raw train nhãn 0/1; S, P =
  số seam/rep sau lọc); bên lớn hơn bị bớt, bỏ theo khóa xác định (SHA-1 `item_seed`, cùng preprocessing).
- **`raw_aug_filt_loc`**: thêm head vị trí trên patch token; `loss = BCE_cls + λ · L_loc`, **λ = 1.0**. `L_loc` = trung
  bình theo ảnh của BCE nhãn mềm trên lưới `image_size // patch_size`; nhãn mềm = tỉ lệ pixel mask trong mỗi ô patch
  (lật ảnh thì lật cả mask). Ảnh có giám sát: `rep` (nhãn từ mask), `seam` và real raw (toàn 0); fake raw không có.
  Batch không có ảnh giám sát -> `L_loc = 0`.
- **Cần mask trên máy train:** arm `raw_aug_filt_loc` đọc mask của ảnh `rep` -> phải chép
  `<data_root>/ffpp/aug/train/masks/` lên SSD cùng với `aug/train` trước khi chạy. Mask không nằm trong `data_hashes` (chỉ kiểm tra tồn tại), nên chép nguyên thư mục (vd. `robocopy /E`) và không sửa nó. Thiếu file mask của một dòng `rep`
  đã lọc -> `FileNotFoundError` nêu đường dẫn ngay lúc dựng tập train (không lỗi giữa epoch). Test và eval không dùng mask.
- Cột `train_loss` luôn là BCE_cls trung bình (mọi arm; arm `_loc` tối ưu BCE_cls + λ·L_loc nhưng không log tổng). Cột `train_loc_loss` trong `train_log.csv`: `L_loc` trung bình theo ảnh có giám sát của epoch (chỉ arm `_loc`; arm khác để trống).
- `run_info.json` của hai arm này có thêm `arm_constants` (`filter_min_r`, `filter_mask`, + `loc_lambda` ở arm loc) và
  `aug_stats` (`tau_m`, số rep/seam trước/sau lọc và sau cân bằng, `n_real`, `n_fake` = R, F; `p_real`, `p_fake` = tỉ lệ seam/rep trước cân bằng; `p_real_after`, `p_fake_after` = sau cân bằng); `evaluate` chép cả hai vào
  `result.json`. Run `raw_aug_filt*` trên đĩa mà `arm_constants` thiếu hoặc khác hằng số hiện tại -> `ValueError` ở
  mọi đường (bỏ qua, eval lại, resume): xóa thư mục run hoặc đổi `ckpt_root`. Arm `raw`, `raw_aug` không đổi gì.

## Arm phụ `raw_rep_filt` (spec v5, ngoài `main`)

`raw_aug_filt` bỏ mọi ảnh `seam`: lọc + cân bằng y hệt, rồi bỏ seam, nên tập rep giống hệt `raw_aug_filt`. Mục đích: xem
bỏ seam thì real Celeb-DF có bớt bị đoán fake không. `arm_constants` thêm `drop_seam: true`; `aug_stats` thêm
`n_seam_dropped`. Không có head loc. Không thuộc `ARMS` nên `main` không chạy; chạy riêng:

```
python -m ffd.cli train --config config.yaml --set arm=raw_rep_filt --set num_workers=8
python -m ffd.cli eval --run checkpoints/clip/raw_rep_filt
python -m ffd.realfp --config config.yaml --a raw_aug_filt --b raw_rep_filt   # ckpt_root lấy từ config.yaml
```

`ffd.realfp` ghi `checkpoints/clip/realfp_<a>__<b>.json` và `.md`. Ngưỡng t mỗi arm = TPR 0.95 trên fake FF++ val
(`scores_val.parquet` bắt buộc); chỉ số chính = tỉ lệ frame real Celeb-DF có logit >= t, diff b − a, CI95 bootstrap
trên video real (B = 4000, seed 0), `verdict` ∈ `seam_hurts_real` (cận trên < 0), `inconclusive`, `seam_helps_real`
(cận dưới > 0). Phụ: bản theo video, real FF++ test, AUC video Celeb-DF / FF++ test (bootstrap phân tầng), tỉ lệ ở
logit >= 0, theo nguồn real (Celeb-real, YouTube-real). `verdict` chỉ có nghĩa cho cặp mặc định
`raw_aug_filt → raw_rep_filt`; cặp khác thì `.md` ghi cảnh báo.

## Cache đặc trưng và probe (spec v6, ngoài `main`)

Spec: `../spec_v6_probe_cache_2026-11-18.md`. Trả lời: CLIP **đóng băng** + head tuyến tính có xếp hạng real/fake
Celeb-DF tốt hơn `raw` (LoRA) không. Mọi lệnh dùng `--config config.yaml` (lấy `data_root`, `ckpt_root`, `model_name`,
`model_revision`, `num_workers`); không thêm khóa vào `TrainConfig`.

**1. Cache (GPU, một lần):**

```
python -m ffd.features --config config.yaml --check-against checkpoints/clip/raw
```

Ghi `features/clip/<commit>/<dataset>/{cls.npy, index.parquet, state.json, meta.json}` cho `train` (raw train),
`val`, `ffpp_test`, `celebdf`. `cls.npy` = memmap fp16 `[N, 25, 1024]`: slot 0..23 = CLS của `hidden_states[1..24]`
(chưa LayerNorm), slot 24 = `pooler_output` (đầu vào head của `raw`). Cả bốn dataset ≈ **9.7 GB**. Ảnh qua
`eval_transform`, không tăng cường. Ngắt giữa chừng thì chạy lại lệnh, tiếp từ khối cuối đã ghi (`state.json`);
`meta.json` chỉ ghi khi xong (kèm kiểm 16 frame tính lại, lệch pooler ≥ 1e-2 -> lỗi). Thư mục cache có commit, manifest,
transform hoặc N khác -> `ValueError` (xóa thư mục hoặc đổi `--feat-root`). `--check-against` so thứ tự frame với
`scores_*.parquet` của run đó (lệch -> lỗi). `--datasets val,celebdf` để chạy một phần.

## Thứ tự chạy: smoke -> main

PowerShell, thư mục `detector/`, đã activate `.venv`.

1. **smoke** (GPU, ViT-L thật): vài bước train, in giây/bước, VRAM đỉnh, `model_commit`; chép 16 ảnh `rep` + 16 ảnh
   `seam` vào `checkpoints/clip/smoke/samples/` để duyệt bằng mắt.
   ```powershell
   python -m ffd.cli smoke --config config.yaml --steps 50 --set num_workers=8
   ```
   **Bắt buộc trước main:** chép `model_commit` in ra vào `model_revision` của `config.yaml` để ghim phiên bản
   model (`model_revision` vào `config_hash`; ghim sau khi main đã chạy sẽ làm run cũ báo lỗi config khác).
2. **main**: train + eval cả 4 arm (`raw`, `raw_aug`, `raw_aug_filt`, `raw_aug_filt_loc`) với `lr`, `p_jpeg` lấy từ
   `config.yaml`; arm đã có `result.json` thì bỏ qua (nên chỉ chạy thêm hai arm mới nếu `raw`/`raw_aug` đã xong). Rồi
   so sánh (`compare.json` + 4 file `compare_<a>__<b>.json`), in bảng tổng hợp và ghi `checkpoints/clip/summary.md`. Thiếu `<data_root>/celebdf/manifest.parquet`
   (**phải tiền xử lý Celeb-DF trước main**) -> lỗi.
   ```powershell
   python -m ffd.cli main --config config.yaml --set num_workers=8
   if ($LASTEXITCODE -ne 0) { throw "main lỗi" }
   ```

Mỗi epoch in một dòng (stderr), vd
(số MINH HỌA):

```
[raw] epoch 3: loss 0.1234 | val ACC 0.9500 AUC 0.9900 PREC 0.9600 REC 0.9700 F1 0.9650 BCE 0.1500 | best e3, bad 0/5
```
 Sau eval mỗi run in bảng `dataset, n_frames, ACC, AUC, PREC, RECALL, F1, AUC_video` ra stdout.

`main` chạy lại được sau khi bị ngắt: run có `result.json` đủ dataset được bỏ qua (thiếu dataset, hoặc
`celebdf` bị `skipped` mà nay đã có dữ liệu -> chỉ eval lại); có `done.flag` thì chỉ eval; có `last.pt` thì train
tiếp (resume); chưa có gì thì train mới. Run cũ có `config_hash` khác config hiện tại -> lỗi
(xóa thư mục run hoặc đổi `ckpt_root`).

Lệnh lẻ (khi cần):

```powershell
python -m ffd.cli train --config config.yaml --set arm=raw_aug [--resume | --force]   # một run; không dùng cả hai cờ
python -m ffd.cli eval --run checkpoints/clip/raw [--datasets val,ffpp_test,celebdf] [--force] [--data-root <data_root trên máy này>]
python -m ffd.cli compare --ckpt-root checkpoints/clip   # compare.json + mọi cặp có đủ hai run (cặp thiếu run: bỏ qua, có ghi chú)
python -m ffd.cli diagnose --ckpt-root checkpoints/clip [--n 2000] [--num-workers 8]   # sau main
```

`diagnose` (không train lại): đọc `result.json`, `scores_*.parquet`, `train_log.csv` của `raw` và `raw_aug`, chấm
`best.pt` của cả hai trên một mẫu ảnh `rep`/`seam` (train) và ảnh real gốc cùng frame, ghi
`checkpoints/clip/diagnose/{diagnose.md, aug_scores.parquet, low_rep_grid.png}` và in `diagnose.md`. Mục:
AUC theo method; phân bố logit theo lớp (FF++ test, Celeb-DF); tỉ lệ đoán fake của `rep` theo `r` và theo diện tích
mask; độ tách rep–seam, rep–real, seam–real; best_epoch và đường AUC val. Chỉ để hiểu nguyên nhân, không dùng để
chỉnh tham số (đã xem kết quả test).

`eval`: `--datasets` mặc định `val,ffpp_test,celebdf`; đã có `result.json` thì phải thêm `--force` (ghi đè).
Run cũ có `run_name` bắt đầu bằng `pilot/` (tạo trước 16/11) chỉ được chấm `val` (chống leakage).

`--no-progress` (mọi lệnh) tắt thanh tiến trình ở stderr; đặt **sau** tên lệnh, vd
`python -m ffd.cli main --no-progress --config config.yaml`.

## Bố cục `checkpoints/clip/`

```
checkpoints/clip/
  smoke/smoke.json, smoke/samples/{rep,seam}_00..15.png
  raw/  raw_aug/  raw_aug_filt/  raw_aug_filt_loc/   # run chính
  compare.json                                      # raw_aug so với raw
  compare_<a>__<b>.json                             # 4 cặp (xem dưới)
  summary.md              # bảng tổng hợp cuối main
```

Mỗi thư mục run: `run_info.json` (config đầy đủ, `config_hash`, `model_commit`, phiên bản thư viện, sha256 dữ liệu),
`train_log.csv` (mỗi epoch một dòng, cột: `epoch, step, train_loss, lr, val_auc_frame, val_bce, val_acc,
val_precision, val_recall, val_f1, frac_real, frac_rep, frac_seam, frac_jpeg, seconds, train_loc_loss`; chỉ số val ngưỡng logit >= 0,
chỉ để theo dõi; log cũ 11 cột khi resume được ghi lại với header mới, ô thiếu để trống), `last.pt` (để resume),
`best.pt` (epoch tốt nhất theo AUC frame val, hòa trong 1e-4 thì BCE val thấp hơn), `done.flag` (`epochs_run`,
`best_epoch`, `stopped_early` = dừng vì patience trước `max_epochs`, `total_steps`),
`scores_<dataset>.parquet` (logit từng frame), `result.json`, `result.md` (cùng bảng in ra lúc eval dạng Markdown:
tiêu đề run, `best_epoch`/`epochs_run`, bảng ACC/AUC/PREC/RECALL/F1/AUC_video từng dataset; `ffpp_test` thêm bảng
AUC theo method).

## `result.json` (một run, chấm `best.pt`)

- `arm`, `config` (model, commit, `lr`, `p_jpeg`, `max_epochs`, `patience`, LoRA, sha256 dữ liệu), `best_epoch`,
  `epochs_run`, `stopped_early`, `total_steps`; arm `raw_aug_filt*` thêm `arm_constants`, `aug_stats` (chép từ `run_info.json`).
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

## `compare_<a>__<b>.json` (bốn cặp v3)

Cùng định dạng và cách tính với `compare.json` (ghép cặp theo frame, bootstrap seed 0, DeLong), nhưng khóa theo tên arm:
`auc_<a>`, `auc_<b>`, `diff` = AUC(b) − AUC(a), `diff_ci95`, `delong_p`; `total_steps`, `epochs_run`, `best_epoch`
theo từng arm; `note`. Bốn cặp (`compare.PAIRS`): `raw__raw_aug_filt`, `raw__raw_aug_filt_loc`,
`raw_aug__raw_aug_filt`, `raw_aug_filt__raw_aug_filt_loc` (tên file: `compare_raw__raw_aug_filt.json`, ...).
`compare.json` vẫn là (raw, raw_aug), không đổi định dạng.

`summary.md` (cuối `main`, cũng in ra stdout): 4 arm × 3 dataset, mỗi dòng ACC, AUC, PREC, RECALL, F1; rồi 5 bảng so
sánh (`compare.json` + 4 cặp) cho `ffpp_test`, `celebdf`: AUC hai arm, hiệu, CI95, p DeLong.

## Ghi chú Exploratory

Mọi file `compare*.json` luôn ghi: *"Exploratory: 1 run/arm, không prereg; frame trong cùng video tương quan nên CI và p-value
lạc quan."* Hai arm v3 cũng vậy (1 run/arm, label exploratory). Chạy 1 run/arm (seed 0) và không pre-register là **quyết định có chủ ý** của chủ đề tài ngày 15/11/2026
(ngoại lệ với nguyên tắc 4, 5 của project). Kết quả chỉ dùng để định hướng, không dùng làm claim.

Chống leakage: `lr` chọn từ run thử trên FF++ val, epoch tốt nhất chỉ chọn trên FF++ val; FF++ test và Celeb-DF
chỉ được đọc khi eval `best.pt`. **Không chỉnh gì sau khi xem test.**
