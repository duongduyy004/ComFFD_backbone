"""Trunk CLIP ViT vision đóng băng + LoRA (peft) trên q_proj/v_proj; trả CLS và patch token nhiều tầng.

`vision_config` cho phép test dựng model nhỏ ngẫu nhiên (không tải trọng số).
`model_revision` truyền vào `from_pretrained(revision=...)`; `self.commit` = commit HF đã resolve
(`resolve_commit`, đọc từ đường dẫn cache HF), hoặc "local" khi dựng từ vision_config.
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import torch
from torch import nn


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


class ClipLoraTrunk(nn.Module):
    def __init__(self, model_name, lora_r, lora_alpha, lora_dropout, lora_targets, tap_layers,
                 grad_checkpointing, vision_config=None, model_revision=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPVisionModel

        if vision_config is not None:
            vit = CLIPVisionModel(vision_config)
            self.commit = "local"
        else:
            vit = CLIPVisionModel.from_pretrained(model_name, revision=model_revision)
            self.commit = resolve_commit(model_name, model_revision)
        self.dim = vit.config.hidden_size
        self.grid = vit.config.image_size // vit.config.patch_size
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

    @property
    def post_layernorm(self):
        """LayerNorm cuối của ViT (transformers 5: ở CLIPVisionModel; bản cũ: dưới .vision_model)."""
        m = self.base.base_model.model
        return getattr(m, "vision_model", m).post_layernorm

    def forward(self, x, patch_tokens: bool = False):
        """patch_tokens=True thêm "tokens" = post_layernorm(last_hidden_state[:, 1:]) [B, G*G, dim] (row-major).
        Mặc định False: đường cls/patch không đổi."""
        want = bool(self.tap_layers)
        o = self.base(pixel_values=x, output_hidden_states=want)
        patch = {l: o.hidden_states[l][:, 1:] for l in self.tap_layers} if want else {}
        out = {"cls": o.pooler_output, "patch": patch}
        if patch_tokens:
            out["tokens"] = self.post_layernorm(o.last_hidden_state[:, 1:])
        return out
