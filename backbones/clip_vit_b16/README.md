# Backbone CLIP ViT-B/16 (mặc định)

`config.yaml` trong thư mục này là config đầy đủ để train/eval với backbone này. Checkpoint và kết quả ghi vào
`checkpoints/` cạnh file này (khóa `ckpt_root`; nội dung không đưa lên git).

## Model

- `openai/clip-vit-base-patch16`, class `CLIPVisionModel`: khoảng 86M tham số, 12 tầng, dim 768, patch 16 (196 token ở 224).
- Chuẩn hóa CLIP, LoRA r16/alpha16 trên `q_proj`, `v_proj`.

## Lệnh (PowerShell, từ thư mục `detector/`, đã activate `.venv`)

Sửa `data_root` trong `config.yaml` theo máy (hoặc thêm `--set data_root=...` vào mỗi lệnh).

```powershell
python -m ffd.cli smoke --config backbones/clip_vit_b16/config.yaml --steps 50   # in giây/bước, VRAM, model_commit
python -m ffd.cli main  --config backbones/clip_vit_b16/config.yaml              # 3 arm + videoboot + realfp + summary
```

Sau smoke: chép `model_commit` vào `model_revision` của `config.yaml`. Kết quả tổng hợp ở
`checkpoints/summary.md`.

## Số đo

Chưa đo (chưa chạy smoke). Ghi giây/bước, VRAM đỉnh và `model_commit` vào đây sau smoke.

## Ghi chú

- Mặc định từ 27/11, đổi từ ViT-L/14 cho nhẹ hơn. Không chọn ViT-B/32: patch 32 quá thô cho artifact cục bộ.
- `lr` 1e-4 giữ từ ViT-L, chưa chỉnh lại. Muốn kiểm lr thì chỉ so trên FF++ val (xem
  `CompFFD/ke_hoach_run_v2_2026-11-26.md`, mục 6).
- `max_epochs` 15, `num_workers` 8 (lý do ghi trong `config.yaml`).
- Kết quả ViT-B không so trực tiếp với kết quả ViT-L của v1.
