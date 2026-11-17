"""diagnose trên hai run tí hon (CPU, dữ liệu tổng hợp MINH HỌA): đủ file và đủ mục. Trạng thái: chưa chạy thử trên GPU."""
import functools

import pandas as pd
import pytest

pytest.importorskip("torch")

from ffd import cli, experiment, progress  # noqa: E402
from ffd.diagnose import diagnose  # noqa: E402


def test_diagnose_outputs(data_root, tiny, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(progress, "ENABLED", progress.ENABLED)
    kw = dict(vision_config=tiny["vision_config"], allow_cpu=True, device="cpu")
    monkeypatch.setattr(experiment, "run_main", functools.partial(experiment.run_main, **kw))
    ck = tmp_path / "ck"
    c = tmp_path / "c.yaml"
    c.write_text(f"data_root: {data_root.as_posix()}\nckpt_root: {ck.as_posix()}\nmodel_name: local-tiny\n"
                 "lora_r: 4\nlora_alpha: 8\nnum_workers: 0\n", encoding="utf-8")
    assert cli.main(["main", "--config", str(c), "--set", "max_epochs=1", "--no-progress"]) == 0
    capsys.readouterr()
    p = diagnose(ck, n_per_kind=50, batch_size=16, num_workers=0, device="cpu", vision_config=tiny["vision_config"])
    text = p.read_text(encoding="utf-8")
    for h in ("## 1. AUC theo method", "## 2. Phân bố logit", "## 3 và 5.", "## 4. Quá trình train", "rep theo độ đậm r"):
        assert h in text
    s = pd.read_parquet(ck / "diagnose" / "aug_scores.parquet")
    assert set(s.kind) == {"rep", "seam", "real"} and {"logit_raw", "logit_raw_aug"} <= set(s.columns)
    assert (ck / "diagnose" / "low_rep_grid.png").exists()
    assert "Chẩn đoán raw_aug" in capsys.readouterr().out


def test_diagnose_requires_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        diagnose(tmp_path)
