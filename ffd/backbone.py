"""Trunk vision đóng băng + LoRA (peft); trả CLS và patch token nhiều tầng. Backbone chọn hoàn toàn bằng config.

Khóa config quyết định backbone (xem `backbones/<tên>/config.yaml`):
- `model_class`: tên class trong `transformers` (vd `CLIPVisionModel`, `Dinov2Model`, `ViTModel`).
- `model_name`, `model_revision`: checkpoint HF và commit ghim.
- `image_mean`, `image_std`: chuẩn hóa ảnh của backbone (RGB, thang [0, 1]). Chuẩn hóa nằm trong trunk
  (`Normalize`), transform chỉ trả ảnh RGB [0, 1].
- `lora_targets`: tên module attention của backbone (CLIP và DINOv2 trong transformers 5.x: `q_proj`/`v_proj`;
  kiểm bằng `model.named_modules()` khi thêm backbone mới).
Điều kiện với backbone: output có `hidden_states` (token 0 = CLS) và `pooler_output`; config có
`num_hidden_layers`, `hidden_size`. SigLIP (không có CLS token) không thỏa.

`vision_config` cho phép test dựng model nhỏ ngẫu nhiên (không tải trọng số).
`self.commit` = commit HF đã resolve (`resolve_commit`), hoặc "local" khi dựng từ vision_config.
Trạng thái: chưa chạy thử trên GPU.
"""
from __future__ import annotations

import torch
from torch import nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def resolve_commit(model_name, revision=None):
    """Commit HF đã resolve, lấy từ đường dẫn cache `.../snapshots/<commit>/config.json`.

    transformers >= 5 không còn giữ `config._commit_hash` (luôn None). Thư mục local -> "local:<path>".
    Không resolve được -> ValueError (không để None lọt vào run_info, vì eval so None == None sẽ qua).
    """
    from pathlib import Path

    if Path(model_name).is_dir():
        return f"local:{Path(model_name).resolve()}"
    from huggingface_hub import hf_hub_download

    p = Path(hf_hub_download(model_name, "config.json", revision=revision))
    if p.parent.parent.name != "snapshots":
        raise ValueError(f"không resolve được commit HF của {model_name!r} (đường dẫn cache lạ: {p})")
    return p.parent.name


def model_cls(model_class: str):
    """Class `transformers.<model_class>`; tên sai -> ValueError."""
    import transformers

    cls = getattr(transformers, model_class, None)
    if cls is None or not hasattr(cls, "from_pretrained"):
        raise ValueError(f"model_class {model_class!r} không phải model trong transformers")
    return cls


def load_vision_model(model_class, model_name, model_revision=None, vision_config=None):
    """(model, commit). vision_config (test) -> model ngẫu nhiên từ config đó, commit "local"."""
    cls = model_cls(model_class)
    if vision_config is not None:
        return cls(vision_config), "local"
    return cls.from_pretrained(model_name, revision=model_revision), resolve_commit(model_name, model_revision)


class Normalize(nn.Module):
    """(x - mean) / std theo kênh, x là RGB [0, 1]. Buffer không lưu vào state_dict (lấy lại từ config)."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean_t, self.std_t = tuple(float(v) for v in mean), tuple(float(v) for v in std)
        self.register_buffer("mean", torch.tensor(self.mean_t).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.std_t).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class LoraTrunk(nn.Module):
    def __init__(self, model_name, lora_r, lora_alpha, lora_dropout, lora_targets, tap_layers,
                 grad_checkpointing, vision_config=None, model_revision=None, model_class="CLIPVisionModel",
                 image_mean=CLIP_MEAN, image_std=CLIP_STD):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        vit, self.commit = load_vision_model(model_class, model_name, model_revision, vision_config)
        self.norm = Normalize(image_mean, image_std)
        self.dim = vit.config.hidden_size
        self.num_layers = vit.config.num_hidden_layers
        self.tap_layers = tuple(tap_layers)
        bad = [l for l in self.tap_layers if not 1 <= l <= self.num_layers]
        if bad:
            raise ValueError(f"tap_layers ngoài [1, {self.num_layers}]: {bad}")
        for p in vit.parameters():
            p.requires_grad_(False)
        if grad_checkpointing:
            vit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vit.enable_input_require_grads()
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=list(lora_targets), bias="none")
        self.base = get_peft_model(vit, lcfg)

    def forward(self, x):
        want = bool(self.tap_layers)
        o = self.base(pixel_values=self.norm(x), output_hidden_states=want)
        patch = {l: o.hidden_states[l][:, 1:] for l in self.tap_layers} if want else {}
        return {"cls": o.pooler_output, "patch": patch}


class NormalizedVision(nn.Module):
    """Backbone đóng băng (không LoRA) nhận ảnh RGB [0, 1]; dùng cho cache đặc trưng (`ffd.features`)."""

    def __init__(self, vit, image_mean, image_std):
        super().__init__()
        self.vit, self.norm = vit, Normalize(image_mean, image_std)
        self.config = vit.config

    def forward(self, pixel_values, **kw):
        return self.vit(pixel_values=self.norm(pixel_values), **kw)
