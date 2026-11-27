# Backbone CLIP ViT-L/14

`config.yaml` trong thư mục này là config đầy đủ để train/eval với backbone này. Checkpoint và kết quả ghi vào
`checkpoints/` cạnh file này (khóa `ckpt_root`; nội dung không đưa lên git).

## Model

- `openai/clip-vit-large-patch14`, class `CLIPVisionModel`: khoảng 304M tham số, 24 tầng, dim 1024, patch 14 (256 token ở 224).
- Chuẩn hóa CLIP, LoRA r16/alpha16 trên `q_proj`, `v_proj`.

## Lệnh (PowerShell, từ thư mục `detector/`, đã activate `.venv`)

Sửa `data_root` trong `config.yaml` theo máy (hoặc thêm `--set data_root=...` vào mỗi lệnh).

```powershell
python -m ffd.cli smoke --config backbones/clip_vit_l14/config.yaml --steps 50   # in giây/bước, VRAM, model_commit
python -m ffd.cli main  --config backbones/clip_vit_l14/config.yaml              # 3 arm + videoboot + realfp + summary
```

Sau smoke: chép `model_commit` vào `model_revision` của `config.yaml`. Kết quả tổng hợp ở
`checkpoints/summary.md`.

## Số đo

Smoke 16/11 trên dữ liệu v1 (bản code ở tag `detector_v1`): 0,526 giây/bước, VRAM đỉnh 8,77 GB với
`batch_size` 32, `grad_checkpointing` false. Chưa đo lại với code hiện tại và dữ liệu v2.

## Ghi chú

- Backbone của tag `detector_v1`. Config này chạy ViT-L với code hiện tại (dữ liệu v2, 3 arm), không tái tạo
  đúng run v1: `max_epochs` 15 thay vì 30, ảnh v2 kích thước gốc.
- Checkpoint v1 cũ không dùng lại được ở đây: `config_hash` khác (khóa backbone mới, `max_epochs`).
