"""Test trunk CLIP+LoRA, head registry, Detector (model tí hon, không cần mạng).

Trạng thái: chưa chạy thử.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from ffd.backbone import ClipLoraTrunk  # noqa: E402
from ffd.config import TrainConfig  # noqa: E402
from ffd.heads import HEADS, build_head, register_head  # noqa: E402
from ffd.model import Detector  # noqa: E402


def cfg(**kw):
    base = dict(ffpp_config="x.yaml", runs_root="runs", mode="raw", seed=0, total_steps=1,
                eval_every=1, ckpt_every=1, lora_r=4, lora_alpha=8, head="binary",
                grad_checkpointing=False)
    base.update(kw)
    return TrainConfig(**base)


def snapshot_base(d):
    return {n: p.detach().clone() for n, p in d.named_parameters() if "lora_" not in n and not n.startswith("head.")}


def test_tap_layer_indices(tiny):
    m = ClipLoraTrunk(**tiny, tap_layers=(8, 16, 24))
    x = torch.randn(2, 3, 336, 336)
    out = m(x)
    hs = m.base(x, output_hidden_states=True).hidden_states
    assert torch.allclose(out["patch"][16], hs[16][:, 1:]) and out["patch"][8].shape[1] == 576
    assert out["cls"].shape == (2, 32) and set(out["patch"]) == {8, 16, 24}


def test_no_tap_layers(tiny):
    out = ClipLoraTrunk(**tiny, tap_layers=())(torch.randn(1, 3, 336, 336))
    assert out["patch"] == {} and out["cls"].shape == (1, 32)


def test_only_lora_and_head_trainable(tiny):
    d = Detector(cfg(head="binary"), vision_config=tiny["vision_config"])
    names = [n for n, p in d.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n or n.startswith("head.") for n in names)
    assert any("lora_" in n for n in names) and any(n.startswith("head.") for n in names)


def test_base_frozen_after_step(tiny):
    d = Detector(cfg(), vision_config=tiny["vision_config"])
    before = snapshot_base(d)
    opt = torch.optim.SGD([p for p in d.parameters() if p.requires_grad], lr=0.1)
    out = d(torch.randn(2, 3, 336, 336))
    loss, _ = d.head.loss(out, {"L0": torch.tensor([0.0, 1.0])})
    loss.backward()
    opt.step()
    after = snapshot_base(d)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
    # LoRA/head thật sự đã đổi
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in d.named_parameters() if p.requires_grad)


def test_trainable_state_roundtrip(tiny):
    torch.manual_seed(0)  # cùng trọng số base ngẫu nhiên (thật thì cùng pretrained)
    d1 = Detector(cfg(), vision_config=tiny["vision_config"])
    torch.manual_seed(0)
    d2 = Detector(cfg(), vision_config=tiny["vision_config"])
    with torch.no_grad():  # lora_B khởi tạo 0 -> làm cho d1 khác d2 rõ ràng
        for p in d1.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    sd = d1.trainable_state_dict()
    assert all("lora_" in k or k.startswith("head.") for k in sd)
    x = torch.randn(1, 3, 336, 336)
    assert not torch.allclose(d1(x)["logit"], d2(x)["logit"])
    d2.load_trainable(sd)
    assert torch.allclose(d1(x)["logit"], d2(x)["logit"])


def test_grad_checkpointing_runs(tiny):
    kw = dict(tiny, grad_checkpointing=True)
    m = ClipLoraTrunk(**kw, tap_layers=(24,))
    m.train()
    out = m(torch.randn(1, 3, 336, 336))
    out["cls"].sum().backward()
    assert any(p.grad is not None for n, p in m.named_parameters() if "lora_" in n)


def test_head_registry_and_binary_loss():
    assert "binary" in HEADS
    h = build_head("binary", 32)
    out = h({"cls": torch.randn(3, 32), "patch": {}})
    assert out["logit"].shape == (3,)
    loss, logs = h.loss(out, {"L0": torch.tensor([0.0, 1.0, 1.0])})
    assert loss.ndim == 0 and isinstance(logs, dict)

    @register_head("dummy_t")
    class D(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()

    assert HEADS["dummy_t"] is D
    del HEADS["dummy_t"]
    with pytest.raises(KeyError):
        build_head("khong_co", 4)


@pytest.mark.gpu
def test_real_vit_l_loads():
    m = ClipLoraTrunk("openai/clip-vit-large-patch14-336", 16, 16, 0.0, ("q_proj", "v_proj"), (8, 16, 24), False)
    out = m(torch.randn(1, 3, 336, 336))
    assert out["cls"].shape == (1, 1024) and out["patch"][8].shape == (1, 576, 1024)
