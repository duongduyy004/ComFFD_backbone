"""Test thanh tiến trình detector. Trạng thái: chưa chạy thử trên GPU."""
from ffd import progress
from ffd.cli import build_parser


def test_bar_writes_to_stderr_not_stdout(capsys):
    for _ in progress.bar(range(3), desc="demo", unit="x"):
        pass
    out = capsys.readouterr()
    assert "demo" in out.err and "3/3" in out.err and out.out == ""


def test_bar_initial_for_resume(capsys):
    pb = progress.bar(total=10, initial=4, desc="train")
    pb.update(1)
    pb.close()
    assert "5/10" in capsys.readouterr().err


def test_disabled_prints_nothing(capsys, monkeypatch):
    monkeypatch.setattr(progress, "ENABLED", False)
    for _ in progress.bar(range(3), desc="demo"):
        pass
    assert capsys.readouterr().err == ""


def test_cli_no_progress_flag_on_every_command():
    p = build_parser()
    for argv in (["train", "--config", "x.yaml"], ["eval", "--run", "r", "--datasets", "ffpp_test"],
                 ["report", "--runs-root", "r"], ["smoke", "--config", "x.yaml"]):
        assert p.parse_args(argv + ["--no-progress"]).no_progress
        assert not p.parse_args(argv).no_progress
