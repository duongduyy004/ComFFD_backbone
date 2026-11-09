"""Trunk CLIP ViT vision đóng băng + LoRA (peft) trên q_proj/v_proj; trả CLS và patch token nhiều tầng.

`vision_config` cho phép test dựng model nhỏ ngẫu nhiên (không tải trọng số).
Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

import torch
from torch import nn


class ClipLoraTrunk(nn.Module):
    def __init__(self, model_name, lora_r, lora_alpha, lora_dropout, lora_targets, tap_layers,
                 grad_checkpointing, vision_config=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPVisionModel

        if vision_config is not None:
            vit = CLIPVisionModel(vision_config)
        else:
            vit = CLIPVisionModel.from_pretrained(model_name)
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
        o = self.base(pixel_values=x, output_hidden_states=want)
        patch = {l: o.hidden_states[l][:, 1:] for l in self.tap_layers} if want else {}
        return {"cls": o.pooler_output, "patch": patch}
