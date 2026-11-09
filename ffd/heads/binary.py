"""Head nhị phân real/fake trên CLS. Trạng thái: chưa chạy thử."""
import torch.nn.functional as F
from torch import nn

from . import register_head


@register_head("binary")
class BinaryHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, out):
        return {"logit": self.fc(out["cls"]).squeeze(-1)}

    def loss(self, pred, targets):
        y = targets["L0"].to(pred["logit"].dtype)
        l = F.binary_cross_entropy_with_logits(pred["logit"], y)
        return l, {"bce": float(l.detach())}
