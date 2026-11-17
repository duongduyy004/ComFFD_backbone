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
    base = dict(data_root="x", lora_r=4, lora_alpha=8, head="binary", grad_checkpointing=False)
    base.update(kw)
    return TrainConfig(**base)


def snapshot_base(d):
    return {n: p.detach().clone() for n, p in d.named_parameters() if "lora_" not in n and not n.startswith("head.")}


def test_tap_layer_indices(tiny):
    m = ClipLoraTrunk(**tiny, tap_layers=(8, 16, 24))
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    hs = m.base(x, output_hidden_states=True).hidden_states
    assert torch.allclose(out["patch"][16], hs[16][:, 1:]) and out["patch"][8].shape[1] == 256
    assert out["cls"].shape == (2, 32) and set(out["patch"]) == {8, 16, 24}


def test_no_tap_layers(tiny):
    out = ClipLoraTrunk(**tiny, tap_layers=())(torch.randn(1, 3, 224, 224))
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
    out = d(torch.randn(2, 3, 224, 224))
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
    x = torch.randn(1, 3, 224, 224)
    assert not torch.allclose(d1(x)["logit"], d2(x)["logit"])
    d2.load_trainable(sd)
    assert torch.allclose(d1(x)["logit"], d2(x)["logit"])


def test_grad_checkpointing_runs(tiny):
    kw = dict(tiny, grad_checkpointing=True)
    m = ClipLoraTrunk(**kw, tap_layers=(24,))
    m.train()
    out = m(torch.randn(1, 3, 224, 224))
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
    m = ClipLoraTrunk("openai/clip-vit-large-patch14", 16, 16, 0.0, ("q_proj", "v_proj"), (8, 16, 24), False)
    out = m(torch.randn(1, 3, 224, 224))
    assert out["cls"].shape == (1, 1024) and out["patch"][8].shape == (1, 256, 1024)


def test_detector_checkpointing_follows_cfg(tiny):
    assert TrainConfig(data_root="x").grad_checkpointing is False  # mặc định tắt từ 16/11
    d = Detector(cfg(grad_checkpointing=True), vision_config=tiny["vision_config"])
    assert d.trunk.base.base_model.model.encoder.gradient_checkpointing is True
    d.train()
    d(torch.randn(1, 3, 224, 224))["logit"].sum().backward()
    grads = [p.grad for n, p in d.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    d2 = Detector(cfg(), vision_config=tiny["vision_config"])
    assert d2.trunk.base.base_model.model.encoder.gradient_checkpointing is False


def test_binary_head_fp32_under_autocast(tiny):
    """I7: dưới autocast bf16, head vẫn tính fp32 và logit trả về float32 (không lượng tử hóa bf16)."""
    head = build_head("binary", 32)
    cls = torch.randn(3, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.nn.Linear(32, 1)(cls).dtype == torch.bfloat16  # đối chứng: autocast CPU có hạ Linear xuống bf16
        out = head({"cls": cls.bfloat16()})
    assert out["logit"].dtype == torch.float32
    d = Detector(cfg(), vision_config=tiny["vision_config"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert d(torch.randn(2, 3, 224, 224))["logit"].dtype == torch.float32


def test_from_pretrained_gets_revision_and_records_commit(tiny, monkeypatch):
    """I2 (không mạng): revision truyền vào from_pretrained và resolve_commit; commit lấy từ resolve_commit
    (transformers 5 không còn config._commit_hash)."""
    import ffd.backbone
    from transformers import CLIPVisionModel

    seen = {}

    def fake(name, **kw):
        seen.update(kw, name=name)
        return CLIPVisionModel(tiny["vision_config"])

    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", staticmethod(fake))
    monkeypatch.setattr(ffd.backbone, "resolve_commit", lambda n, r: f"sha-of-{n}@{r}")
    d = Detector(cfg(model_revision="c0ffee"))
    assert seen == {"name": "openai/clip-vit-large-patch14", "revision": "c0ffee"}
    assert d.trunk.commit == "sha-of-openai/clip-vit-large-patch14@c0ffee"
    assert Detector(cfg(), vision_config=tiny["vision_config"]).trunk.commit == "local"


def test_resolve_commit_from_cache_path(tmp_path, monkeypatch):
    """transformers 5 bỏ config._commit_hash -> commit lấy từ đường dẫn snapshots/<commit>/; đường dẫn lạ -> lỗi."""
    import huggingface_hub
    from ffd.backbone import resolve_commit

    snap = tmp_path / "models--org--m" / "snapshots" / "c0ffee123"
    snap.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(snap / "config.json"))
    assert resolve_commit("org/m", "main") == "c0ffee123"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(tmp_path / "x" / "config.json"))
    with pytest.raises(ValueError, match="commit"):
        resolve_commit("org/m")
    assert resolve_commit(str(tmp_path)).startswith("local:")


# ---- head vị trí (arm raw_aug_filt_loc) ----
def test_detector_params_old_arms(tiny):
    """raw/raw_aug/raw_aug_filt: không có loc_head; tham số train = LoRA + head như trước; trunk không trả tokens."""
    for arm in ("raw", "raw_aug", "raw_aug_filt"):
        d = Detector(cfg(arm=arm), vision_config=tiny["vision_config"])
        assert d.loc_head is None
        names = [n for n, p in d.named_parameters() if p.requires_grad]
        assert names and all("lora_" in n or n.startswith("head.") for n in names)
        out = d(torch.randn(1, 3, 224, 224))
        assert set(out) == {"logit"}
        assert "tokens" not in d.trunk(torch.randn(1, 3, 224, 224))


def test_detector_loc_outputs(tiny):
    d = Detector(cfg(arm="raw_aug_filt_loc"), vision_config=tiny["vision_config"])
    out = d(torch.randn(2, 3, 224, 224))
    assert out["logit"].shape == (2,) and out["patch_logit"].shape == (2, 16, 16)
    assert out["patch_logit"].dtype == torch.float32
    sd = d.trainable_state_dict()
    assert {"loc_head.fc.weight", "loc_head.fc.bias"} <= set(sd)
    d2 = Detector(cfg(arm="raw_aug_filt_loc"), vision_config=tiny["vision_config"])
    d2.load_trainable(sd)
    with pytest.raises(KeyError):  # state của arm cũ thiếu loc_head -> strict
        d2.load_trainable({k: v for k, v in sd.items() if not k.startswith("loc_head.")})


def test_trunk_tokens_row_major(tiny):
    m = ClipLoraTrunk(**tiny, tap_layers=())
    x = torch.randn(1, 3, 224, 224)
    out = m(x, patch_tokens=True)
    ref = m.base(pixel_values=x).last_hidden_state[:, 1:]
    assert out["tokens"].shape == (1, 256, 32) and not torch.allclose(out["tokens"], ref)
    assert torch.allclose(out["tokens"], m.post_layernorm(ref))
    head = Detector(cfg(arm="raw_aug_filt_loc"), vision_config=tiny["vision_config"]).loc_head
    pl = head(out["tokens"], 16)
    assert torch.allclose(pl[0, 3, 5], head.fc(out["tokens"][0, 3 * 16 + 5]).squeeze())  # hàng = p // G, cột = p % G


def test_loc_loss_values():
    from ffd.heads.loc import loc_loss

    z = torch.zeros(2, 2, 2, requires_grad=True)
    t = torch.tensor([[[1.0, 0.0], [0.5, 0.0]], [[1.0, 1.0], [1.0, 1.0]]])
    none = loc_loss(z, t, torch.tensor([False, False]))
    assert float(none.detach()) == 0.0
    none.backward()  # vẫn nối autograd
    # logit 0 -> BCE = ln2 ở mọi patch, bất kể nhãn mềm
    ln2 = float(torch.log(torch.tensor(2.0)))
    assert float(loc_loss(z, t, torch.tensor([True, False])).detach()) == pytest.approx(ln2)
    # tính tay: ảnh 0, logit [[2,-1],[0,0]] với nhãn [[1,0],[0.5,0]]
    lg = torch.tensor([[[2.0, -1.0], [0.0, 0.0]], [[5.0, 5.0], [5.0, 5.0]]])
    sp = lambda v: float(torch.nn.functional.softplus(torch.tensor(v)))
    img0 = (sp(-2.0) + sp(-1.0) + ln2 + ln2) / 4  # BCE(z,1)=softplus(-z); BCE(z,0)=softplus(z); z=0 -> ln2
    assert float(loc_loss(lg, t, torch.tensor([True, False]))) == pytest.approx(img0, rel=1e-5)
    # trung bình theo ảnh (không theo patch): ảnh 1 toàn 1 với logit 5
    img1 = sp(-5.0)
    assert float(loc_loss(lg, t, torch.tensor([True, True]))) == pytest.approx((img0 + img1) / 2, rel=1e-5)
