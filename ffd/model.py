"""Detector = ClipLoraTrunk + head. Chỉ LoRA và head là tham số huấn luyện.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from torch import nn

from .backbone import ClipLoraTrunk
from .heads import build_head


class Detector(nn.Module):
    def __init__(self, cfg, vision_config=None):
        super().__init__()
        self.cfg = cfg
        self.trunk = ClipLoraTrunk(cfg.model_name, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout,
                                   cfg.lora_targets, cfg.tap_layers, cfg.grad_checkpointing,
                                   vision_config=vision_config, model_revision=cfg.model_revision)
        self.head = build_head(cfg.head, self.trunk.dim)  # dim lấy từ hidden_size, không hardcode
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad)
        n_head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        assert n_train == n_lora + n_head, (n_train, n_lora, n_head)

    def forward(self, x):
        return self.head(self.trunk(x))

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if expected - set(sd) or set(sd) - set(self.state_dict()):
            raise KeyError(f"state không khớp: thiếu {sorted(expected - set(sd))[:3]}, thừa {sorted(set(sd) - set(self.state_dict()))[:3]}")
