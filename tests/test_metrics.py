import json

import pandas as pd
import pytest

from ffd.metrics import auc_per_method, frame_block, threshold_metrics


def test_frame_block_known_values():
    df = pd.DataFrame({"video_id": ["a", "a", "b", "b"], "label": [0, 0, 1, 1], "logit": [-2., 1., 3., -1.]})
    b = frame_block(df, n_videos_dropped=1)
    assert b["confusion_matrix"] == [[1, 1], [1, 1]] and b["acc"] == 0.5 and b["auc"] == 0.75
    assert (b["n_frames"], b["n_videos"], b["n_videos_dropped"]) == (4, 2, 1)
    assert set(b) == {"auc", "acc", "f1", "recall", "precision", "confusion_matrix", "n_frames", "n_videos",
                      "n_videos_dropped", "auc_video", "auc_ci95"}
    assert b["auc_video"] == 1.0 and b["recall"] == 0.5 and b["precision"] == 0.5 and b["f1"] == 0.5
    assert len(b["auc_ci95"]) == 2
    json.dumps(b)  # phải serialise được


def test_threshold_metrics_matches_frame_block():
    y, s = [0, 0, 1, 1, 1], [-2., 0., 3., -1., 0.5]  # logit 0 -> dương (ngưỡng >= 0)
    m = threshold_metrics(y, s)
    assert m == {"acc": 0.6, "precision": pytest.approx(2 / 3), "recall": pytest.approx(2 / 3),
                 "f1": pytest.approx(2 / 3)}
    b = frame_block(pd.DataFrame({"video_id": list("abcde"), "label": y, "logit": s}), 0)
    assert all(b[k] == m[k] for k in m)
    assert threshold_metrics([0, 0], [-1., -2.]) == {"acc": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_auc_per_method():
    df = pd.DataFrame({
        "video_id": list("abcdef"),
        "label": [0, 0, 1, 1, 1, 1],
        "method": ["real", "real", "m1", "m1", "m2", "m2"],
        "logit": [0., 2., 1., 3., -1., 5.],
    })
    r = auc_per_method(df)
    # m1 vs real: dương {1,3}, âm {0,2}: cặp thắng 3/4; m2: dương {-1,5}: 2/4
    assert r == {"m1": pytest.approx(0.75), "m2": pytest.approx(0.5)}
    assert all(type(v) is float for v in r.values())
