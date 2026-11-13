# ffd — detector CLIP ViT-L/14-336 + LoRA (CompFFD)

So sánh 2 arm huấn luyện (`raw`, `raw+rep`) trên FF++ (đánh giá trên FF++ test). `raw` dùng `ffpp_raw/train`; `raw+rep` thêm
`ffpp_aug/train`: ảnh thay thế bộ phận (lớp fake) và real có đường nối (lớp real), ρ = 0.5 cho cả hai lớp
(spec `../spec_code_augmentation_can_bang_duong_noi_2026-11-11.md`).
Chế độ `compffd` (E0) đã bỏ ngày 11/11/2026.
Trạng thái: `smoke` và pilot 100 step đã chạy trên GPU (RTX 5060 Ti) với dữ liệu thật (13/11/2026); chưa có run thật.

## Cài đặt

Venv riêng cho detector (PowerShell, chạy trong thư mục `detector/`):

```powershell
py -3.14 -m venv .venv                            # máy GPU: Python 3.14; máy local: py -3.15
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements-gpu.txt     # máy GPU; máy local không GPU: requirements-cpu.txt
python -c "import torch, transformers, cv2; print(torch.__version__, torch.cuda.is_available())"
```

- Torch đã nằm trong file requirements theo máy, không cài riêng nữa:
  - `requirements-gpu.txt`: `torch==2.14.1+cu130`, cho máy GPU có driver CUDA 13.1 (`nvidia-smi`) và Python 3.14. Bản build CUDA của torch không được vượt `CUDA Version` của driver (`cu132` cần driver >= 13.2).
  - `requirements-cpu.txt`: `torch==2.15.0+cpu`, cho máy local (Python 3.15, chỉ chạy test). Không dùng 2.14.1 ở đây vì 2.14.1+cpu trên Python 3.15 làm DataLoader worker chết.
  - Đổi driver hoặc Python thì chọn lại `cuXXX` và phiên bản có wheel tại `https://download.pytorch.org/whl/<cuXXX>/torch/`.
- `requirements.txt` là phần chung (không có torch), được hai file trên gọi qua `-r`; không cài trực tiếp. Detector **độc lập với preprocessing** (từ 13/11/2026): không import `compsynth_ff`, không cần cài `../preprocessing`; chỉ đọc dữ liệu trên đĩa theo mục "Hợp đồng dữ liệu" dưới đây.
- Không cài torchvision (ffd không dùng; lệch phiên bản với torch -> lỗi `torchvision::nms`).
- Trên máy GPU, dòng cuối phải in `2.14.1+cu130 True`. Mỗi terminal mới: activate lại rồi chạy `python -m ffd.cli ...` (không dùng `py`, `py` có thể gọi Python chung của máy).
- Venv cài theo hướng dẫn cũ (torch cài tay, `-e ../preprocessing`): chạy lại `python -m pip install -r requirements-gpu.txt`; `compsynth-ff` còn sót trong venv không ảnh hưởng (gỡ bằng `pip uninstall compsynth-ff` nếu muốn).

### Config

Chép `config.example.yaml` thành `config.yaml` rồi sửa `ffpp_root` = `out_root` của preprocessing (thư mục chứa `ffpp_raw/`, `ffpp_aug/`). Config cũ còn khóa `ffpp_config` sẽ bị từ chối (khóa không hợp lệ): đổi thành `ffpp_root`.

### Chạy test

```powershell
python -m pytest -q                                   # CPU, dữ liệu tổng hợp MINH HỌA, ~5 phút; kỳ vọng 134 passed, 2 skipped
$env:FFD_REAL_CONFIG = "config.yaml"; python -m pytest -q --run-gpu tests/test_smoke_gpu.py   # máy GPU, dữ liệu thật
```

Test không cần preprocessing: fixture `tests/synth_ffpp.py` tự sinh `ffpp_root` tổng hợp đúng hợp đồng dữ liệu dưới đây.

## Hợp đồng dữ liệu (preprocessing -> detector)

Detector chỉ phụ thuộc vào file trên đĩa dưới `ffpp_root`; đổi format ở preprocessing thì phải sửa `ffd/data/ffpp.py`.

| File | Cột bắt buộc | Ghi chú |
|---|---|---|
| `ffpp_raw/manifest.parquet` | `split, method, tid, video_id, frame_idx, keep, path, config_hash` | `method = "real"` là nhãn real; chỉ dòng `keep` được đọc; video không còn frame keep vẫn là video kỳ vọng khi eval |
| `ffpp_aug/manifest.parquet` | `kind, tid, frame_idx, method, regions, skipped, path, config_hash` | `kind ∈ {rep, seam}`; bỏ dòng `skipped != ""`; mọi `tid` phải có `split == "train"` trong `ffpp_raw` (khác -> `LeakageError`) |
| ảnh | `path` tương đối với `ffpp_root` | 336×336 BGR; raw là PNG (chưa nén JPEG), aug là JPEG (đã nén một lần) |

Mỗi manifest phải có đúng một giá trị `config_hash` (khác -> `StaleArtifactError`). Detector không tính lại hash này từ config preprocessing; nó chỉ ghi hash vào `run_info.json` lúc train và so lại lúc eval. `jpeg_q` lúc train nằm trong `config.yaml` của detector (mặc định `[75, 95]`) và nên trùng `jpeg_q` của preprocessing.

## Thứ tự chạy

Cần có sẵn `<ffpp_root>/ffpp_raw` và `<ffpp_root>/ffpp_aug` (`ffpp_root` trong `config.yaml` = `out_root` của preprocessing; cách tạo: `../preprocessing/README.md`).

Mọi lệnh chạy trong PowerShell, thư mục `detector/`, đã activate `.venv`. `num_workers` không vào hash/prereg, đổi tự do bằng `--set`; máy GPU hiện tại (RTX 5060 Ti 16GB, dữ liệu trên HDD) đo nhanh nhất ở `num_workers=8`.

1. **smoke** (GPU, ViT-L thật): đo VRAM đỉnh và giây/step, in `model_commit`, chép 16 ảnh `rep` + 16 ảnh `seam` của `ffpp_aug/train` vào `runs/smoke/smoke_samples/` để duyệt bằng mắt.
   ```powershell
   python -m ffd.cli smoke --config config.yaml --steps 50 --set num_workers=8
   ```
   Kết quả 13/11/2026 (RTX 5060 Ti, batch 32, `grad_checkpointing: true`): 1.72 s/step, VRAM đỉnh 4.1 GB, `model_commit = ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. Đo riêng: GPU 1.64 s/step với dữ liệu giả, loader 1.38 s/batch với 8 worker (đọc từ HDD), nên GPU là giới hạn chính.
2. **Pilot cả hai arm** (seed 0, chỉ xem FF++ val) để chốt `lr`, `total_steps`, `eval_every`, `ckpt_every`:
   ```powershell
   foreach ($m in "raw", "raw+rep") {
     python -m ffd.cli train --config config.yaml --pilot --set mode=$m --set seed=0 --set num_workers=8 --set total_steps=3000 --set eval_every=150 --force
     if ($LASTEXITCODE -ne 0) { throw "pilot $m lỗi" }
   }
   ```
   - Pilot cả hai arm vì `raw+rep` mỗi step chỉ thấy khoảng (1−ρ) ảnh FF++ gốc, có thể hội tụ chậm hơn; chốt `total_steps` chỉ từ `raw` sẽ thiên về `raw`. Hai arm vẫn dùng **chung** một `total_steps` trong prereg (cùng ngân sách huấn luyện).
   - Khoảng 85 phút train mỗi arm (1.7 s/step) cộng 20 lượt val.
   - `--pilot` ghi vào `<runs_root>_pilot/` (không đè run thật), cho phép chưa có prereg; kết quả pilot không bao giờ vào report. Không có `--pilot` mà thiếu prereg thì `train` từ chối. `--force` xóa pilot cũ cùng arm/seed.
   - Đọc `runs_pilot/aug/raw/seed0/val_curve.jsonl` và `runs_pilot/aug/raw+rep/seed0/val_curve.jsonl`. Lr theo cosine về 0 ở `total_steps`, nên đường cong phẳng ở cuối pilot chưa phải là hội tụ. Chọn `total_steps` sao cho **cả hai** đường cong đã đạt đỉnh hoặc thôi tăng (> ~0.005 mỗi lượt val); AUC còn tăng đến cuối -> chạy pilot dài hơn (vd `total_steps=6000`). Nên chạy lại pilot ở `total_steps` đã chọn để xác nhận.
   - `eval_every`: khoảng 10–20 lượt val cho cả run (val chấm toàn bộ FF++ val).
   - Thử `lr` khác: thêm `--set lr=<giá trị>` (pilot chỉ xem val, chạy bao nhiêu lần cũng được). `lr` cũng chốt chung cho hai arm.
   - Pilot 100 step (13/11/2026): AUC video val 0.683 -> 0.916 (step 20 -> 100), chưa đủ để chốt.
3. **Viết `configs/prereg_aug.yaml`** (`Copy-Item configs\prereg_aug.example.yaml configs\prereg_aug.yaml`) rồi **không bao giờ sửa lại**. Prereg phải ghi **đủ** mọi khóa dưới đây (thiếu khóa nào -> lỗi, không lấy ngầm từ `config.yaml`). Khóa trong prereg (`rho`, `lr`, `weight_decay`, `warmup_frac`, `total_steps`, `eval_every`, `batch_size`, `grad_accum`, `selfblend` (code train không dùng, chỉ giữ để hash prereg không đổi), `model_name`, `model_revision`, `head`, `tap_layers`, `grad_checkpointing`, `lora_*`) không override được bằng `--set`; file prereg chứa khóa khác -> lỗi. Hash prereg được kiểm lại lúc eval.
   - **Ghim `model_revision`** = commit `smoke` in ra (hiện tại `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`); `train` không `--pilot` từ chối khi `model_revision` rỗng. Không dùng tên nhánh (`main`) vì nó trôi theo thời gian.
   - `grad_checkpointing: true` (đã chốt 13/11/2026). `batch_size` phải chẵn (mỗi batch đúng nửa real, nửa fake).
4. **Train thật**: 2 arm × 3 seed, cùng prereg; các run chỉ khác `mode` và `seed`.
   ```powershell
   foreach ($m in "raw", "raw+rep") { foreach ($s in 0, 1, 2) {
     python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=$m --set seed=$s --set num_workers=8
     if ($LASTEXITCODE -ne 0) { throw "train $m seed$s lỗi" }
   } }
   ```
   - Chạy từng run: `python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=raw --set seed=0 --set num_workers=8`.
   - Run dừng giữa chừng: thêm `--resume`. Xóa kết quả cũ để chạy lại: `--force` (xóa cả `scores_*.parquet`, `eval_info.json`).
   - Thiếu seed nào thì `report` liệt kê dưới CẢNH BÁO.
   - `run_info.json` ghi `model_revision`, `model_commit` (commit đã resolve), `ffpp_raw_hash`, `ffpp_aug_hash` (cột `config_hash` của `ffpp_raw/manifest.parquet` và `ffpp_aug/manifest.parquet`, do preprocessing ghi; ghi cả hai cho mọi arm); `eval` đọc lại từ manifest và từ chối nếu lệch.
5. **eval** cả 6 run (chỉ chạm test ở bước này; **không chỉnh gì sau khi xem test**):
   ```powershell
   foreach ($m in "raw", "raw+rep") { foreach ($s in 0, 1, 2) {
     python -m ffd.cli eval --run runs/aug/$m/seed$s --datasets ffpp_test
     if ($LASTEXITCODE -ne 0) { throw "eval $m seed$s lỗi" }
   } }
   ```
   `--datasets` chỉ nhận `ffpp_test`. Eval đọc ảnh bằng `num_workers` lưu trong `run_info.json`.
   **Eval trên máy khác máy train:** thêm `--ffpp-root <thư mục ffpp_root trên máy này>` và `--prereg <prereg trên máy này>`. Đường dẫn được khác; `config_hash` trong hai manifest và nội dung prereg phải trùng lúc train, lệch -> từ chối.
6. **report**: `python -m ffd.cli report --runs-root runs` → `runs/report/`: `main.md`, `delong.csv`, `summary.parquet`. `main.md` có AUC video và AUC frame; CI bootstrap và DeLong (`delong.csv`, phương sai Obuchowski 1997) đều gom cụm theo `tid` video nguồn (ffpp_test). Dưới mục CẢNH BÁO: seed/arm thiếu (arm kỳ vọng `raw`, `raw+rep` vắng hoàn toàn cũng được nêu), run không prereg, config arm lệch ngoài `mode/seed/runs_root/num_workers`, scores không khớp `best.pt` hiện tại.

## Thanh tiến trình

`train` (step/tổng, loss, AUC val tốt nhất; resume chạy tiếp từ step đã lưu), lượt chấm val giữa chừng, `eval` và `smoke` đều có thanh tqdm ghi ra stderr. Khi stderr chuyển ra file thì in thưa mỗi 30 s. Tắt bằng `--no-progress`. Theo dõi chi tiết vẫn ở `train_log.jsonl` và `val_curve.jsonl`.

## Cảnh báo

**Không chỉnh gì sau khi xem FF++ test.** Mọi ngưỡng và hyperparameter chốt trên FF++ val trước khi eval.

## Hạn chế đã biết

- Test tự động chạy trên CPU với dữ liệu tổng hợp (MINH HỌA) và CLIP tí hon ngẫu nhiên. ViT-L thật mới chạy `smoke` và pilot 100 step trên máy GPU; `tests/test_smoke_gpu.py` chưa chạy.
- Test local chạy torch 2.15.0+cpu / Python 3.15; máy GPU dùng torch 2.14.1+cu130 / Python 3.14 (đã chạy `smoke` và pilot ngày 13/11/2026; bộ test chưa chạy trên tổ hợp này).
- Detector không tự phát hiện việc sửa config preprocessing mà chưa chạy lại stage: nó chỉ kiểm mỗi manifest có một `config_hash` và hash lúc eval trùng lúc train. Giữ đồng bộ dữ liệu với config là việc của preprocessing.
- Chưa có đánh giá cross-dataset: lệnh `prep-cross` (Celeb-DF v2) đã bỏ ngày 12/11/2026.
- Mục fake không bốc được mask (`fallback=True`) dùng fake gốc của method đã bốc; theo dõi bằng QC Q9 ở preprocessing.
- Bốc method đều theo xác suất trên các method dùng được của từng frame.
- Arm `raw+rep` chỉ dùng ảnh tăng cường khi frame có **cả** `rep` lẫn `seam`; thiếu một trong hai thì cả hai lớp dùng ảnh gốc (`fallback=True`). Vì vậy ρ hiệu dụng ≈ ρ × (tỉ lệ frame có đủ cặp) < 0.5; train chưa ghi tỉ lệ này.
- Mục `rep` mang method đã bốc lúc `build-aug` (`meta["method"]`, đã loại cặp `(tid, method)` lệch hàng), không phải method bốc lúc train.
- Ảnh gốc `ffpp_raw` lúc train luôn qua đúng một lần JPEG q ~ U(`jpeg_q`), mặc định (75, 95); val/test (`eval_transform`) không nén thêm. Lệch này giống nhau cho cả hai arm.
