# ffd: detector backbone + LoRA (CompFFD)

Backbone chọn bằng config, không sửa code: mỗi backbone là một thư mục `backbones/<tên>/` (xem mục Backbone). Mặc định CLIP ViT-B/16.

So sánh ba arm huấn luyện trên FF++ (`ffd.config.ARMS`, thứ tự cố định):

| run | arm | dữ liệu train |
|---|---|---|
| R0 | `raw` | ảnh raw train (đối chứng) |
| R1 | `raw_aug` | raw train + mọi ảnh tăng cường `rep` (nhãn 1) và `seam` (nhãn 0) của preprocessing |
| R2 | `raw_rep` | như `raw_aug` nhưng bỏ hết `seam` (cùng tập rep) |

Không lọc, không cân bằng lại ở detector: preprocessing đã kiểm rep thấy được, seam sạch và cân seam/rep theo từng
target. Đánh giá trên FF++ val, FF++ test và Celeb-DF; so sánh mức video và tiêu chí ghi trước ở
`../ke_hoach_run_v2_2026-11-26.md`. Bản detector trước (dữ liệu PNG 224, arm `raw_aug_filt`, `raw_aug_filt_loc`,
`raw_rep_filt`, lệnh `compare`) giữ ở tag `detector_v1`.
Trạng thái: code mới chạy test CPU (CLIP và DINOv2 tí hon, dữ liệu tổng hợp MINH HỌA); **chưa chạy thử trên GPU**.

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

### Backbone

```
backbones/
  clip_vit_b16/        mặc định: openai/clip-vit-base-patch16 (86M, 12 tầng, dim 768)
    config.yaml        config đầy đủ (backbone + train + máy)
    README.md          model, số đo smoke, ghi chú tham số
    checkpoints/       ckpt_root của backbone này (nội dung không lên git)
  clip_vit_l14/        openai/clip-vit-large-patch14 (backbone của tag detector_v1)
```

Lệnh chạy từ thư mục `detector/` và trỏ `--config` vào config của backbone. Đường dẫn `ckpt_root` tương đối với
`detector/`.

```powershell
$cfg = "backbones/clip_vit_b16/config.yaml"      # đổi backbone = đổi dòng này
```

Sửa `data_root` trong config = `out_root` của preprocessing (thư mục chứa `ffpp/` và `celebdf/`), hoặc thêm
`--set data_root=...`. Mọi khóa đổi được tạm thời bằng `--set k=v`; khóa lạ -> lỗi.

**Thêm backbone mới:** chép một thư mục trong `backbones/`, đổi tên, rồi sửa khối backbone trong config của backbone:

| khóa | ý nghĩa |
|---|---|
| `model_class` | class trong `transformers` (vd `CLIPVisionModel`, `Dinov2Model`) |
| `model_name`, `model_revision` | checkpoint HF và commit ghim (`model_commit` smoke in ra) |
| `image_mean`, `image_std` | chuẩn hóa ảnh RGB thang [0, 1] của backbone (vd ImageNet cho DINOv2) |
| `lora_targets` | tên module attention để gắn LoRA (CLIP, DINOv2 trong transformers 5.x: `q_proj`, `v_proj`) |
| `ckpt_root` | `backbones/<tên>/checkpoints` |

Điều kiện: backbone trả `hidden_states` (token 0 = CLS) và `pooler_output`, config có `num_hidden_layers`,
`hidden_size`; ảnh vào 224×224. SigLIP (không có CLS token) không thỏa. Chuẩn hóa nằm trong model
(`ffd/backbone.py::Normalize`); transform chỉ trả ảnh RGB [0, 1]. Mọi khóa backbone vào `config_hash`.

**Dữ liệu:** ảnh JPEG q95 ở kích thước crop gốc, không vuông (cột `w`, `h`, `jpeg_q` trong manifest). Detector resize
khi nạp (`ffd/data/transforms.py::to_size`): kéo thẳng về 224×224, INTER_AREA khi thu nhỏ cả hai chiều, INTER_CUBIC
trong các trường hợp còn lại; JPEG lúc train áp sau resize. `main` từ chối dữ liệu thiếu các cột trên.

Tham số train (giống nhau ở mọi backbone; lý do từng giá trị ghi trong config của backbone):

- `lr: 1.0e-4`: theo run thử `lr1e-4_p0.5` trên ViT-L, AUC frame val 0.9954 sau 3 epoch (16/11). Chưa chỉnh lại cho
  ViT-B; muốn kiểm lr thì chỉ so trên FF++ val (`ke_hoach_run_v2_2026-11-26.md` mục 6).
- `max_epochs: 15` (27/11): v1 ViT-L dừng ở epoch 8–13, best 4–8, nên lịch cosine 30 epoch không bao giờ giảm lr.
- `p_jpeg: 0.5`, `batch_size` 32, LoRA r16/alpha16, `patience` 5, `warmup_frac` 0.05.
- `grad_checkpointing`, `num_workers`: theo máy và backbone; **không vào `config_hash`** (không đổi phép toán), cùng
  `data_root`, `ckpt_root`, `run_name`.

Mọi khóa khác vào `config_hash`: đổi giữa các lần chạy main thì run cũ báo lỗi (xóa thư mục run hoặc đổi
`ckpt_root`). `main` in khối config hiệu lực (`data_root`, `ckpt_root`, `lr`, `p_jpeg`, `batch_size`, `max_epochs`,
`patience`, `seed`, `grad_checkpointing`, `model_name`, `model_revision`, `num_workers`) ra stdout trước khi train:
**đọc lại khối này trước khi để chạy**.

### Test

```powershell
python -m pytest -q                                                                  # CPU, MINH HỌA
$env:FFD_REAL_CONFIG = $cfg; python -m pytest -q --run-gpu tests/test_smoke_gpu.py   # máy GPU, dữ liệu thật
```

## Thứ tự chạy: smoke -> main

PowerShell, thư mục `detector/`, đã activate `.venv`.

1. **smoke** (GPU, backbone thật): vài bước train, in giây/bước, VRAM đỉnh, `model_commit`; chép 16 ảnh `rep` + 16 ảnh
   `seam` vào `<ckpt_root>/smoke/samples/` để duyệt bằng mắt. Ảnh giờ phải giải mã JPEG và resize: xem giây/bước để
   chọn `num_workers`.
   ```powershell
   python -m ffd.cli smoke --config $cfg --steps 50
   ```
   **Bắt buộc trước main:** chép `model_commit` in ra vào `model_revision` của config của backbone để ghim phiên bản
   model (`model_revision` vào `config_hash`; ghim sau khi main đã chạy sẽ làm run cũ báo lỗi config khác).
2. **main**: kiểm dữ liệu, train + eval R0 → R1 → R2, rồi so sánh và áp tiêu chí.
   ```powershell
   python -m ffd.cli main --config $cfg [--B 4000]
   if ($LASTEXITCODE -ne 0) { throw "main lỗi" }
   ```
   - Kiểm trước khi train: manifest FF++ raw và Celeb-DF có `w`, `h`, `jpeg_q` (thiếu Celeb-DF -> lỗi); run đã có trên
     đĩa mà `data_hashes` khác dữ liệu hiện tại -> lỗi.
   - In khối config hiệu lực (`data_root`, `ckpt_root`, `lr`, `p_jpeg`, `batch_size`, `max_epochs`, `patience`,
     `seed`, `grad_checkpointing`, `model_name`, `model_revision`, `num_workers`): **đọc lại trước khi để chạy**.
   - Sau ba run: `videoboot` (gốc `raw`: AUC video, FPR@TPR95, hiệu kèm CI95), `realfp_raw__raw_aug` (mô tả),
     `realfp_raw_aug__raw_rep` (tiêu chí R2), Spearman(face_px, logit) trên frame real Celeb-DF (kiểm cơ chế, mô tả).
   - Ghi `summary.{json,md}` và in `summary.md`:
     - R1 − R0: `aug_helps` (cận dưới CI95 ΔAUC video Celeb-DF > 0) / `aug_hurts` (cận trên < 0) / `null`; kèm
       ΔAUC video FF++ test ≥ −0.005.
     - R2 − R1: verdict của `realfp` (`seam_helps_real` / `seam_hurts_real` / `inconclusive`).
     - `run_R3`: chỉ khi R1 `aug_helps` và FF++ đạt.

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
python -m ffd.cli train --config $cfg --set arm=raw_rep [--resume | --force]   # một run; không dùng cả hai cờ
python -m ffd.cli eval --run backbones/clip_vit_b16/checkpoints/raw [--datasets val,ffpp_test,celebdf] [--force] [--data-root <data_root trên máy này>]
python -m ffd.videoboot --config $cfg --base raw --arms raw_aug raw_rep
python -m ffd.realfp --config $cfg --a raw_aug --b raw_rep
python -m ffd.cli diagnose --ckpt-root backbones/clip_vit_b16/checkpoints [--n 2000] [--num-workers 8]   # sau main
```

`ffd.realfp`: ngưỡng t mỗi arm = TPR 0.95 trên fake FF++ val; chỉ số chính = tỉ lệ frame real Celeb-DF có logit >= t,
diff b − a, CI95 bootstrap trên video real (B = 4000, seed 0). `verdict` chỉ có nghĩa cho cặp `raw_aug → raw_rep`;
cặp khác thì `.md` ghi cảnh báo. Phụ: bản theo video, real FF++ test, AUC video Celeb-DF / FF++ test, theo nguồn real.

`diagnose` (không train lại): đọc `result.json`, `scores_*.parquet`, `train_log.csv` của `raw` và `raw_aug`, chấm
`best.pt` của cả hai trên một mẫu ảnh `rep`/`seam` (train) và ảnh real gốc cùng frame, ghi
`backbones/clip_vit_b16/checkpoints/diagnose/{diagnose.md, aug_scores.parquet, low_rep_grid.png}` và in `diagnose.md`. Mục:
AUC theo method; phân bố logit theo lớp (FF++ test, Celeb-DF); tỉ lệ đoán fake của `rep` theo `r` và theo diện tích
mask; độ tách rep–seam, rep–real, seam–real; best_epoch và đường AUC val. Chỉ để hiểu nguyên nhân, không dùng để
chỉnh tham số (đã xem kết quả test).

`eval`: `--datasets` mặc định `val,ffpp_test,celebdf`; đã có `result.json` thì phải thêm `--force` (ghi đè).
Run cũ có `run_name` bắt đầu bằng `pilot/` (tạo trước 16/11) chỉ được chấm `val` (chống leakage).

`--no-progress` (mọi lệnh) tắt thanh tiến trình ở stderr; đặt **sau** tên lệnh, vd
`python -m ffd.cli main --no-progress --config $cfg`.

## Bố cục `<ckpt_root>` (vd `backbones/clip_vit_b16/checkpoints/`)

```
backbones/clip_vit_b16/checkpoints/
  smoke/smoke.json, smoke/samples/{rep,seam}_00..15.jpg
  raw/  raw_aug/  raw_rep/                           # run chính
  videoboot.{json,md}                               # gốc raw, arm raw_aug và raw_rep
  realfp_raw__raw_aug.{json,md}, realfp_raw_aug__raw_rep.{json,md}
  summary.{json,md}                                 # tiêu chí và kết luận của main
```

Mỗi thư mục run: `run_info.json` (config đầy đủ, `config_hash`, `model_commit`, phiên bản thư viện, sha256 dữ liệu),
`train_log.csv` (mỗi epoch một dòng, cột: `epoch, step, train_loss, lr, val_auc_frame, val_bce, val_acc,
val_precision, val_recall, val_f1, frac_real, frac_rep, frac_seam, frac_jpeg, seconds`; chỉ số val ngưỡng logit >= 0,
chỉ để theo dõi; log cũ 11 cột khi resume được ghi lại với header mới, ô thiếu để trống), `last.pt` (để resume),
`best.pt` (epoch tốt nhất theo AUC frame val, hòa trong 1e-4 thì BCE val thấp hơn), `done.flag` (`epochs_run`,
`best_epoch`, `stopped_early` = dừng vì patience trước `max_epochs`, `total_steps`),
`scores_<dataset>.parquet` (logit từng frame), `result.json`, `result.md` (cùng bảng in ra lúc eval dạng Markdown:
tiêu đề run, `best_epoch`/`epochs_run`, bảng ACC/AUC/PREC/RECALL/F1/AUC_video từng dataset; `ffpp_test` thêm bảng
AUC theo method).

## `result.json` (một run, chấm `best.pt`)

- `arm`, `config` (model, commit, `lr`, `p_jpeg`, `max_epochs`, `patience`, LoRA, sha256 dữ liệu), `best_epoch`,
  `epochs_run`, `stopped_early`, `total_steps`; arm `raw_aug`, `raw_rep` thêm `aug_stats` (chép từ `run_info.json`).
- Mỗi dataset (`val`, `ffpp_test`, `celebdf`): `auc` (AUC frame, lớp dương = fake), `auc_ci95`
  (bootstrap 2000 lần trên frame, seed 0), `acc`, `f1`, `recall`, `precision`, `confusion_matrix` (ngưỡng xác suất
  0.5), `n_frames`, `n_videos`, `n_videos_dropped` (video bị preprocessing loại hẳn theo `drops.parquet`; không tính video lỗi khi
  tiền xử lý), `auc_video` (tham khảo).
  `ffpp_test` thêm `auc_per_method`. Thiếu dữ liệu Celeb-DF -> `{"skipped": ...}` (chỉ khi gọi `eval` lẻ; main
  từ chối chạy). Chấm Celeb-DF -> `config.manifest_sha256.celebdf` = sha256 manifest Celeb-DF đã chấm.

## Cache đặc trưng và probe (spec v6, ngoài `main`)

Spec: `../spec_v6_probe_cache_2026-11-18.md`. Trả lời: CLIP **đóng băng** + head tuyến tính có xếp hạng real/fake
Celeb-DF tốt hơn `raw` (LoRA) không. Mọi lệnh dùng `--config $cfg` (lấy `data_root`, `ckpt_root`, `model_name`,
`model_revision`, `num_workers`); không thêm khóa vào `TrainConfig`.

**1. Cache (GPU, một lần):**

```
python -m ffd.features --config $cfg --check-against backbones/clip_vit_b16/checkpoints/raw
```

Ghi `<data_root>/features/clip/<commit>/<dataset>/{cls.npy, index.parquet, state.json, meta.json}` cho `train` (raw train),
`val`, `ffpp_test`, `celebdf`. `cls.npy` = memmap fp16 `[N, L+1, D]` (ViT-B/16: `[N, 13, 768]`): slot 0..L-1 = CLS của `hidden_states[1..L]`
(chưa LayerNorm), slot L = `pooler_output` (đầu vào head của `raw`). Cả bốn dataset ≈ **3.8 GB** với ViT-B/16
(ước tính; ViT-L đo được 9.7 GB). Ảnh qua
`eval_transform`, không tăng cường. Ngắt giữa chừng thì chạy lại lệnh, tiếp từ khối cuối đã ghi (`state.json`);
`meta.json` chỉ ghi khi xong (kèm kiểm: tính lại batch đầu với cùng `--batch-size` lúc trích, lệch pooler > 1e-2 · max(1, |giá trị|) -> lỗi). Thư mục cache
có commit, manifest, transform, N hoặc `precision` khác -> `ValueError` (xóa thư mục hoặc đổi `--feat-root`).
`--check-against` kiểm `model_commit` và thứ tự frame với `scores_*.parquet` của run đó **trước khi forward** (lệch ->
lỗi ngay, không mất thời gian GPU). `--datasets val,celebdf` để chạy một phần.

Tốc độ: mỗi dataset một DataLoader (không dựng lại worker mỗi khối). Trên CUDA mặc định bật **TF32** (mantissa 10 bit,
bằng độ chính xác fp16 khi lưu; nhanh hơn fp32 thuần khoảng 2–3 lần trên GPU Ampere trở lên); `--no-tf32` để tính fp32
thuần. Sau khối đầu (4096 frame) lệnh in `frame/s` và số phút còn lại của dataset đó: xem dòng này để biết thời gian thật
(ước lượng 30 phút trong spec chưa đo).

**2. Probe (CPU hoặc GPU, đọc cache):**

```
python -m ffd.probe --config $cfg --arm probe_multi --seed 0          # tương tự probe_last, probe_gate; seed 0, 1, 2
python -m ffd.probe --config $cfg --arm probe_layer --layer all       # L probe một tầng (ViT-B: 12) + probe_layers.md
```

| arm | đầu vào | head |
|---|---|---|
| `probe_last` | slot 24 (`pooler_output`) | tuyến tính |
| `probe_multi` | CLS tầng 8, 12, 16, 20, 24 (residual), nối | tuyến tính |
| `probe_gate` | CLS cả 24 tầng | trọng số softmax theo tầng -> tuyến tính |
| `probe_layer` | một tầng (`--layer l` hoặc `all`) | tuyến tính; **bảng chỉ mô tả, không chọn tầng** |

Chuẩn hóa mean/std theo chiều trên train. Head khởi tạo 0. BCE, tỉ lệ lớp tự nhiên, AdamW lr 1e-3, batch 4096, tối đa
50 epoch, patience 5 theo BCE val; weight decay chọn trong {1e-4, 1e-3, 1e-2} theo BCE val. FF++ test và Celeb-DF chỉ
mở sau khi chốt weight decay. Thư mục cache = `--feat-root` (mặc định `<data_root>/features/clip`, cạnh dữ liệu; không làm lệch `data_hashes`) / commit (`model_revision` nếu là
hash 40 ký tự). Run `backbones/clip_vit_b16/checkpoints/<arm>_s<seed>/` (`probe_layer<l>_s<seed>`): `run_info.json` (slot, hằng số,
meta cache, wd chọn, `alpha` của gate), `train_log.csv`, `head.pt`, `scores_{val,ffpp_test,celebdf}.parquet` (cùng cột,
cùng thứ tự với `ffd.evaluate`, nên `ffd.realfp` dùng được), `result.json`. Có `result.json` thì bỏ qua (`--force` để
chạy lại); `run_info.json` có hằng số/slot/cache khác -> `ValueError`.

**3. So sánh với `raw` (mức video):**

```
python -m ffd.videoboot --config $cfg --base raw --arms probe_last_s0 probe_multi_s0 probe_gate_s0
python -m ffd.realfp --config $cfg --a raw --b probe_multi_s0
```

`ffd.videoboot` ghi `backbones/clip_vit_b16/checkpoints/videoboot_<base>.{json,md}` (`--name` để đổi tên; `main` dùng tên `videoboot`): AUC
video và FPR@TPR95 video từng arm trên FF++ test và Celeb-DF, hiệu arm − gốc kèm CI95 (bootstrap phân tầng theo lớp,
B = 4000, seed 0), tương quan điểm video real giữa gốc và arm.
Chỉ số chính spec v6: ΔAUC video Celeb-DF của `probe_multi_s0`; đồng chính: ΔFPR@TPR95. Tiêu chí ở spec mục 4.
Lưu ý: CI của ΔFPR@TPR95 rất rộng.

## Ghi chú Exploratory

`summary.json`, `realfp_*.json`, `videoboot.json` đều ghi nhãn exploratory. Chạy 1 run/arm (seed 0) và không pre-register là **quyết định có chủ ý** của chủ đề tài ngày 15/11/2026
(ngoại lệ với nguyên tắc 4, 5 của project). Kết quả chỉ dùng để định hướng, không dùng làm claim.

Chống leakage: `lr` chọn từ run thử trên FF++ val, epoch tốt nhất chỉ chọn trên FF++ val; FF++ test và Celeb-DF
chỉ được đọc khi eval `best.pt`. **Không chỉnh gì sau khi xem test.**
