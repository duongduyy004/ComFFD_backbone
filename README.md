# ffd — detector CLIP ViT-L/14-336 + LoRA (CompFFD)

So sánh 3 arm huấn luyện (`raw`, `raw+comp`, `raw+sb`) và `compffd` trên FF++ → Celeb-DF v2.
Trạng thái: chưa chạy thử trên GPU/dữ liệu thật.

## Cài đặt

```bash
pip install -e ../preprocessing[gpu] -e .
```

`[gpu]` kéo `facexlib` (RetinaFace) cho `prep-cross`. Train và smoke cần CUDA.

## Thứ tự chạy

1. **Preprocessing** (trong `../preprocessing`): extract → select → render val/test. Ra `out_root` có `frames.parquet`, `splits/`, manifest.
2. **prep-cross cdf2** (GPU, một lần):
   ```bash
   python -m ffd.cli prep-cross --dataset cdf2 --root /data/Celeb_DFv2 --out /data/cross --ffpp-config ../preprocessing/config.yaml
   ```
3. **smoke** (GPU, ViT-L thật): đo VRAM đỉnh và giây/step, xuất 16 ảnh `comp` + 16 ảnh `sb` vào `runs/smoke/smoke_samples/` để duyệt tham số `T` của self-blend bằng mắt.
   ```bash
   python -m ffd.cli smoke --config config.yaml --steps 50
   ```
4. **Pilot arm `raw`** (1 seed, chỉ xem FF++ val) để chốt `lr`, `total_steps`, `eval_every`, `ckpt_every`:
   ```bash
   python -m ffd.cli train --config config.yaml --pilot --set mode=raw --set seed=0
   ```
   `--pilot` ghi vào `<runs_root>_pilot/` (thư mục anh em, không đè `runs/aug/raw/seed0` của run thật) và cho phép chưa có prereg. Kết quả pilot không bao giờ đưa vào report. Không có `--pilot` mà thiếu prereg thì `train` từ chối.
5. **Viết `configs/prereg_aug.yaml`** (mẫu: `configs/prereg_aug.example.yaml`) rồi **không bao giờ sửa lại**. Khóa trong prereg (`rho`, `lr`, `weight_decay`, `warmup_frac`, `total_steps`, `eval_every`, `batch_size`, `selfblend`, `lora_*`) không override được bằng `--set`; hash prereg được kiểm lại lúc eval.
6. **9 run** = 3 arm × seed 0,1,2:
   ```bash
   for arm in raw raw+comp raw+sb; do for s in 0 1 2; do
     python -m ffd.cli train --config config.yaml --set prereg=configs/prereg_aug.yaml --set mode=$arm --set seed=$s
   done; done
   ```
   Dùng `--resume` để tiếp tục run dừng giữa chừng, `--force` để xóa kết quả cũ chạy lại.
7. **eval** từng run (chỉ chạm test ở bước này):
   ```bash
   python -m ffd.cli eval --run runs/aug/raw/seed0 --datasets ffpp_test,compsynth_test,cdf2 --cross-root cdf2=/data/cross
   ```
   `--datasets` chỉ nhận `ffpp_test`, `compsynth_test`, `cdf2`.
8. **report**: `python -m ffd.cli report --runs-root runs` → `main.md`, `per_cell.md`, `delong.csv`, `summary.parquet`. Seed thiếu được liệt kê dưới mục CẢNH BÁO.

## Cảnh báo

**Không chỉnh gì sau khi xem Celeb-DF v2 / FF++ test.** Mọi ngưỡng và hyperparameter chốt trên FF++ val trước khi eval. Dữ liệu cross-dataset không dùng để tinh chỉnh dưới bất kỳ hình thức nào.

Quy ước nhãn Celeb-DF v2: trong `List_of_testing_videos.txt`, nhãn `1` được giả định là real (đã kiểm lúc `prep-cross` theo thư mục video); lệch với thư mục -> `CrossListError`, dừng, không đoán.

## Hạn chế đã biết

- Mới test trên CPU với dữ liệu tổng hợp (MINH HỌA) và CLIP tí hon ngẫu nhiên; chưa chạy ViT-L thật, chưa có FF++ hay Celeb-DF trên máy.
- Revision của model CLIP chưa được ghim (`model_name` trỏ thẳng `openai/clip-vit-large-patch14-336`).
- Mode `compffd` không đổi σ/q theo seed train.
- Bốc method đều theo xác suất trên các method dùng được của từng frame.
