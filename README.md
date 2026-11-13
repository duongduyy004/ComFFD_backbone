# ffd — detector CLIP ViT-L/14-336 + LoRA (CompFFD)

So sánh 2 arm huấn luyện (`raw`, `raw+rep`) trên FF++ (đánh giá trên FF++ test). `raw` dùng `ffpp_raw/train`; `raw+rep` thêm
`ffpp_aug/train`: ảnh thay thế bộ phận (lớp fake) và real có đường nối (lớp real), ρ = 0.5 cho cả hai lớp
(spec `../spec_code_augmentation_can_bang_duong_noi_2026-11-11.md`).
Chế độ `compffd` (E0) đã bỏ ngày 11/11/2026.
Trạng thái: chưa chạy thử trên GPU/dữ liệu thật.

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
python -m pytest -q                                   # CPU, dữ liệu tổng hợp MINH HỌA, ~4 phút; kỳ vọng 132 passed, 2 skipped
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

1. **smoke** (GPU, ViT-L thật): đo VRAM đỉnh và giây/step, chép 16 ảnh `rep` + 16 ảnh `seam` của `ffpp_aug/train` vào `runs/smoke/smoke_samples/` (`rep_XX.jpg`, `seam_XX.jpg`) để duyệt bằng mắt.
   ```bash
   python -m ffd.cli smoke --config config.yaml --steps 50
   ```
2. **Pilot arm `raw`** (1 seed, chỉ xem FF++ val) để chốt `lr`, `total_steps`, `eval_every`, `ckpt_every`:
   ```bash
   python -m ffd.cli train --config config.yaml --pilot --set mode=raw --set seed=0
   ```
   `--pilot` ghi vào `<runs_root>_pilot/` (thư mục anh em, không đè `runs/aug/raw/seed0` của run thật) và cho phép chưa có prereg. Kết quả pilot không bao giờ đưa vào report. Không có `--pilot` mà thiếu prereg thì `train` từ chối.
3. **Viết `configs/prereg_aug.yaml`** (mẫu: `configs/prereg_aug.example.yaml`) rồi **không bao giờ sửa lại**. Prereg phải ghi **đủ** mọi khóa dưới đây (thiếu khóa nào -> lỗi, không lấy ngầm từ `config.yaml`). Khóa trong prereg (`rho`, `lr`, `weight_decay`, `warmup_frac`, `total_steps`, `eval_every`, `batch_size`, `grad_accum`, `selfblend` (code train không dùng, chỉ giữ để hash prereg không đổi), `model_name`, `model_revision`, `head`, `tap_layers`, `grad_checkpointing`, `lora_*`) không override được bằng `--set`; file prereg chứa khóa khác -> lỗi. Hash prereg được kiểm lại lúc eval.
   **Ghim `model_revision`** (commit HF của `model_name`, `smoke` in ra) trong prereg: `train` không `--pilot` từ chối khi `model_revision` rỗng. `batch_size` phải chẵn (mỗi batch đúng nửa real, nửa fake).
4. **2 run** (seed 0, cùng prereg; hai run chỉ khác `mode`):
   1. `raw` (chỉ `ffpp_raw/train`):
      ```bash
      python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=raw --set seed=0
      ```
   2. `raw+rep` (thêm `ffpp_aug/train`):
      ```bash
      python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=raw+rep --set seed=0
      ```
   Một seed không đo được dao động giữa các lần train; trước khi đưa vào paper nên chạy thêm `--set seed=1`, `--set seed=2` cho mỗi arm (không cần train lại seed 0). Đến lúc đó `report` liệt kê seed 1, 2 còn thiếu dưới CẢNH BÁO.
   Dùng `--resume` để tiếp tục run dừng giữa chừng, `--force` để xóa kết quả cũ chạy lại (xóa cả `scores_*.parquet`, `eval_info.json`).
   `run_info.json` ghi `model_revision`, `model_commit` (commit đã resolve), `ffpp_raw_hash`, `ffpp_aug_hash` (cột `config_hash` của `ffpp_raw/manifest.parquet` và `ffpp_aug/manifest.parquet`, do preprocessing ghi; ghi cả hai cho mọi arm); `eval` đọc lại từ manifest và từ chối nếu lệch.
5. **eval** cả 2 run (chỉ chạm test ở bước này):
   ```bash
   python -m ffd.cli eval --run runs/aug/raw/seed0 --datasets ffpp_test
   python -m ffd.cli eval --run runs/aug/raw+rep/seed0 --datasets ffpp_test
   ```
   `--datasets` chỉ nhận `ffpp_test`.
   **Eval trên máy khác máy train:** thêm `--ffpp-root <thư mục ffpp_root trên máy này>` và `--prereg <prereg trên máy này>`. Đường dẫn được khác; `config_hash` trong hai manifest và nội dung prereg phải trùng lúc train, lệch -> từ chối.
6. **report**: `python -m ffd.cli report --runs-root runs` → `main.md`, `delong.csv`, `summary.parquet`. `main.md` có AUC video và AUC frame; CI bootstrap và DeLong (`delong.csv`, phương sai Obuchowski 1997) đều gom cụm theo `tid` video nguồn (ffpp_test). Dưới mục CẢNH BÁO: seed/arm thiếu (arm kỳ vọng `raw`, `raw+rep` vắng hoàn toàn cũng được nêu), run không prereg, config arm lệch ngoài `mode/seed/runs_root/num_workers`, scores không khớp `best.pt` hiện tại.

## Thanh tiến trình

`train` (step/tổng, loss, AUC val tốt nhất; resume chạy tiếp từ step đã lưu), lượt chấm val giữa chừng, `eval` và `smoke` đều có thanh tqdm ghi ra stderr. Khi stderr chuyển ra file thì in thưa mỗi 30 s. Tắt bằng `--no-progress`. Theo dõi chi tiết vẫn ở `train_log.jsonl` và `val_curve.jsonl`.

## Cảnh báo

**Không chỉnh gì sau khi xem FF++ test.** Mọi ngưỡng và hyperparameter chốt trên FF++ val trước khi eval.

## Hạn chế đã biết

- Mới test trên CPU với dữ liệu tổng hợp (MINH HỌA) và CLIP tí hon ngẫu nhiên; chưa chạy ViT-L thật, chưa có FF++ trên máy.
- Test local chạy torch 2.15.0+cpu / Python 3.15; máy GPU dùng torch 2.14.1+cu130 / Python 3.14 (tổ hợp chưa test, mới kiểm `pip --dry-run`). Chạy `smoke` trước khi train thật.
- Detector không tự phát hiện việc sửa config preprocessing mà chưa chạy lại stage: nó chỉ kiểm mỗi manifest có một `config_hash` và hash lúc eval trùng lúc train. Giữ đồng bộ dữ liệu với config là việc của preprocessing.
- Chưa có đánh giá cross-dataset: lệnh `prep-cross` (Celeb-DF v2) đã bỏ ngày 12/11/2026.
- Mục fake không bốc được mask (`fallback=True`) dùng fake gốc của method đã bốc; theo dõi bằng QC Q9 ở preprocessing.
- Bốc method đều theo xác suất trên các method dùng được của từng frame.
- Arm `raw+rep` chỉ dùng ảnh tăng cường khi frame có **cả** `rep` lẫn `seam`; thiếu một trong hai thì cả hai lớp dùng ảnh gốc (`fallback=True`). Vì vậy ρ hiệu dụng ≈ ρ × (tỉ lệ frame có đủ cặp) < 0.5; train chưa ghi tỉ lệ này.
- Mục `rep` mang method đã bốc lúc `build-aug` (`meta["method"]`, đã loại cặp `(tid, method)` lệch hàng), không phải method bốc lúc train.
- Ảnh gốc `ffpp_raw` lúc train luôn qua đúng một lần JPEG q ~ U(`jpeg_q`), mặc định (75, 95); val/test (`eval_transform`) không nén thêm. Lệch này giống nhau cho cả hai arm.
