"""Detector = ClipLoraTrunk + head. Chỉ LoRA và head là tham số huấn luyện.

Trạng thái: chưa chạy thử.
"""
from __future__ import annotations

from torch import nn

from .backbone import ClipLoraTrunk
from .config import LOC_ARMS
from .heads import build_head
from .heads.loc import LocHead


class Detector(nn.Module):
    def __init__(self, cfg, vision_config=None):
        super().__init__()
        self.cfg = cfg
        self.trunk = ClipLoraTrunk(cfg.model_name, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout,
                                   cfg.lora_targets, cfg.tap_layers, cfg.grad_checkpointing,
                                   vision_config=vision_config, model_revision=cfg.model_revision)
        self.head = build_head(cfg.head, self.trunk.dim)  # dim lấy từ hidden_size, không hardcode
        self.use_loc = cfg.arm in LOC_ARMS
        self.grid = self.trunk.grid  # image_size // patch_size của ViT
        self.loc_head = LocHead(self.trunk.dim) if self.use_loc else None
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_lora = sum(p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad)
        n_head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        n_loc = sum(p.numel() for p in self.loc_head.parameters() if p.requires_grad) if self.use_loc else 0
        assert n_train == n_lora + n_head + n_loc, (n_train, n_lora, n_head, n_loc)

    def forward(self, x):
        """Dict của head phân loại ("logit"); arm loc thêm "patch_logit" [B, G, G]."""
        t = self.trunk(x, patch_tokens=True) if self.use_loc else self.trunk(x)
        out = self.head(t)
        if self.use_loc:
            out["patch_logit"] = self.loc_head(t["tokens"], self.grid)
        return out

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if expected - set(sd) or set(sd) - set(self.state_dict()):
            raise KeyError(f"state không khớp: thiếu {sorted(expected - set(sd))[:3]}, thừa {sorted(set(sd) - set(self.state_dict()))[:3]}")
