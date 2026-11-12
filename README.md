# ffd — detector CLIP ViT-L/14-336 + LoRA (CompFFD)

So sánh 2 arm huấn luyện (`raw`, `raw+rep`) trên FF++ → Celeb-DF v2. `raw` dùng `ffpp_raw/train`; `raw+rep` thêm
`ffpp_aug/train`: ảnh thay thế bộ phận (lớp fake) và real có đường nối (lớp real), ρ = 0.5 cho cả hai lớp
(spec `../spec_code_augmentation_can_bang_duong_noi_2026-11-11.md`).
Chế độ `compffd` (E0) đã bỏ ngày 11/11/2026.
Trạng thái: chưa chạy thử trên GPU/dữ liệu thật.

## Cài đặt

```bash
pip install -e ../preprocessing[gpu] -e .
```

`[gpu]` kéo `facexlib` (RetinaFace) cho `prep-cross`. Train và smoke cần CUDA.

## Thứ tự chạy

1. **Preprocessing** (trong `../preprocessing`): extract → select → realign → export-raw → build-aug. Ra `out_root/ffpp_raw` và `out_root/ffpp_aug`.
2. **prep-cross cdf2** (GPU, một lần):
   ```bash
   python -m ffd.cli prep-cross --dataset cdf2 --root /data/Celeb_DFv2 --out /data/cross --ffpp-config ../preprocessing/config.yaml
   ```
3. **smoke** (GPU, ViT-L thật): đo VRAM đỉnh và giây/step, chép 16 ảnh `rep` + 16 ảnh `seam` của `ffpp_aug/train` vào `runs/smoke/smoke_samples/` (`rep_XX.jpg`, `seam_XX.jpg`) để duyệt bằng mắt.
   ```bash
   python -m ffd.cli smoke --config config.yaml --steps 50
   ```
4. **Pilot arm `raw`** (1 seed, chỉ xem FF++ val) để chốt `lr`, `total_steps`, `eval_every`, `ckpt_every`:
   ```bash
   python -m ffd.cli train --config config.yaml --pilot --set mode=raw --set seed=0
   ```
   `--pilot` ghi vào `<runs_root>_pilot/` (thư mục anh em, không đè `runs/aug/raw/seed0` của run thật) và cho phép chưa có prereg. Kết quả pilot không bao giờ đưa vào report. Không có `--pilot` mà thiếu prereg thì `train` từ chối.
5. **Viết `configs/prereg_aug.yaml`** (mẫu: `configs/prereg_aug.example.yaml`) rồi **không bao giờ sửa lại**. Prereg phải ghi **đủ** mọi khóa dưới đây (thiếu khóa nào -> lỗi, không lấy ngầm từ `config.yaml`). Khóa trong prereg (`rho`, `lr`, `weight_decay`, `warmup_frac`, `total_steps`, `eval_every`, `batch_size`, `grad_accum`, `selfblend` (code train không dùng, chỉ giữ để hash prereg không đổi), `model_name`, `model_revision`, `head`, `tap_layers`, `grad_checkpointing`, `lora_*`) không override được bằng `--set`; file prereg chứa khóa khác -> lỗi. Hash prereg được kiểm lại lúc eval.
   **Ghim `model_revision`** (commit HF của `model_name`, `smoke` in ra) trong prereg: `train` không `--pilot` từ chối khi `model_revision` rỗng. `batch_size` phải chẵn (mỗi batch đúng nửa real, nửa fake).
6. **6 run** = 2 arm × seed 0,1,2:
   ```bash
   for arm in raw raw+rep; do for s in 0 1 2; do
     python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=$arm --set seed=$s
   done; done
   ```
   Dùng `--resume` để tiếp tục run dừng giữa chừng, `--force` để xóa kết quả cũ chạy lại (xóa cả `scores_*.parquet`, `eval_info.json`).
   `run_info.json` ghi `model_revision`, `model_commit` (commit đã resolve), `ffpp_config_hash`, `ffpp_stage_hash` (stage `aug` của compsynth_ff, gồm cả trường extract + select; dùng cho cả arm `raw`); `eval` kiểm lại và từ chối nếu lệch.
7. **eval** từng run (chỉ chạm test ở bước này):
   ```bash
   python -m ffd.cli eval --run runs/aug/raw/seed0 --datasets ffpp_test,cdf2 --cross-root cdf2=/data/cross
   ```
   `--datasets` chỉ nhận `ffpp_test`, `cdf2`.
   **Eval trên máy khác máy train:** thêm `--ffpp-config <config FF++ trên máy này>` và `--prereg <prereg trên máy này>`. Đường dẫn (`data_root`, `out_root`, `splits_dir`) được khác; mọi tham số khác và nội dung prereg phải trùng lúc train, lệch -> từ chối.
8. **report**: `python -m ffd.cli report --runs-root runs` → `main.md`, `delong.csv`, `summary.parquet`. `main.md` có AUC video và AUC frame; CI bootstrap và DeLong (`delong.csv`, phương sai Obuchowski 1997) đều gom cụm theo `tid` video nguồn (ffpp_test). Dưới mục CẢNH BÁO: seed/arm thiếu (arm kỳ vọng `raw`, `raw+rep` vắng hoàn toàn cũng được nêu), run không prereg, config arm lệch ngoài `mode/seed/runs_root/num_workers`, scores không khớp `best.pt` hiện tại.

## Thanh tiến trình

`train` (step/tổng, loss, AUC val tốt nhất; resume chạy tiếp từ step đã lưu), lượt chấm val giữa chừng, `eval` (mỗi dataset), `prep-cross` (theo video) và `smoke` đều có thanh tqdm ghi ra stderr. Khi stderr chuyển ra file thì in thưa mỗi 30 s. Tắt bằng `--no-progress`. Theo dõi chi tiết vẫn ở `train_log.jsonl` và `val_curve.jsonl`.

## Cảnh báo

**Không chỉnh gì sau khi xem Celeb-DF v2 / FF++ test.** Mọi ngưỡng và hyperparameter chốt trên FF++ val trước khi eval. Dữ liệu cross-dataset không dùng để tinh chỉnh dưới bất kỳ hình thức nào.

Quy ước nhãn Celeb-DF v2: trong `List_of_testing_videos.txt`, nhãn `1` được giả định là real (đã kiểm lúc `prep-cross` theo thư mục video); lệch với thư mục -> `CrossListError`, dừng, không đoán.

## Hạn chế đã biết

- Mới test trên CPU với dữ liệu tổng hợp (MINH HỌA) và CLIP tí hon ngẫu nhiên; chưa chạy ViT-L thật, chưa có FF++ hay Celeb-DF trên máy.
- Mục fake không bốc được mask (`fallback=True`) dùng fake gốc của method đã bốc; theo dõi bằng QC Q9 ở preprocessing.
- Bốc method đều theo xác suất trên các method dùng được của từng frame.
- Arm `raw+rep` chỉ dùng ảnh tăng cường khi frame có **cả** `rep` lẫn `seam`; thiếu một trong hai thì cả hai lớp dùng ảnh gốc (`fallback=True`). Vì vậy ρ hiệu dụng ≈ ρ × (tỉ lệ frame có đủ cặp) < 0.5; train chưa ghi tỉ lệ này.
- Mục `rep` mang method đã bốc lúc `build-aug` (`meta["method"]`, đã loại cặp `(tid, method)` lệch hàng), không phải method bốc lúc train.
- Ảnh train luôn qua đúng một lần JPEG q ~ U(75, 95); val/test (`eval_transform`) không nén thêm. Lệch này giống nhau cho cả hai arm.
