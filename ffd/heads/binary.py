"""Head nhị phân real/fake trên CLS.

Head tính ở fp32 (tắt autocast cho riêng head, CLS ép .float()) nên logit trả về luôn float32, không bị lượng
tử hóa bf16 khi trunk chạy dưới autocast. Trạng thái: chưa chạy thử."""
import torch
import torch.nn.functional as F
from torch import nn

from . import register_head


@register_head("binary")
class BinaryHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, out):
        cls = out["cls"]
        with torch.autocast(device_type=cls.device.type, enabled=False):
            return {"logit": self.fc(cls.float()).squeeze(-1)}

    def loss(self, pred, targets):
        y = targets["L0"].to(pred["logit"].dtype)
        l = F.binary_cross_entropy_with_logits(pred["logit"], y)
        return l, {"bce": float(l.detach())}
