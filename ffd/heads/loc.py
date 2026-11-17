"""Head vị trí theo patch (arm raw_aug_filt_loc): một Linear(dim, 1) trên patch token, và loss L_loc.

Head tính ở fp32 (tắt autocast, token ép .float()) như BinaryHead. Không đăng ký trong registry head phân loại
(không thay head CLS, chỉ là nhánh phụ của Detector). Trạng thái: chưa chạy thử trên GPU."""
import torch
import torch.nn.functional as F
from torch import nn


class LocHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, tokens, grid):
        """tokens [B, G*G, dim] (row-major: p -> hàng p // G, cột p % G) -> logit [B, G, G] float32."""
        b, n, _ = tokens.shape
        if n != grid * grid:
            raise ValueError(f"số patch token {n} != grid^2 ({grid}^2)")
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.fc(tokens.float()).squeeze(-1).reshape(b, grid, grid)


def loc_loss(patch_logit, target, has_loc):
    """L_loc = trung bình theo ảnh có giám sát của trung bình BCE nhãn mềm trên patch. Không ảnh nào có giám sát -> 0
    (vẫn nối đồ thị autograd để backward không lỗi)."""
    has = has_loc.to(device=patch_logit.device, dtype=torch.bool)
    if not bool(has.any()):
        return patch_logit.sum() * 0.0
    t = target.to(device=patch_logit.device, dtype=patch_logit.dtype)
    per_img = F.binary_cross_entropy_with_logits(patch_logit, t, reduction="none").mean(dim=(1, 2))
    return per_img[has].mean()
