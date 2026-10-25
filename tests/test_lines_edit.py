import json

import numpy as np
import pytest

from foot_overlay.counting import CountLine, export_lines_json, import_lines_json
from foot_overlay.session import Session, write_result
from foot_overlay.suspects import SuspectSettings, find_suspects
from foot_overlay.tracking import TrackingResult, TrackPoint


def make_session(tmp_path, width=300, height=200):
    frame = np.full((height, width, 3), 100, dtype=np.uint8)
    tracks = {
        1: [TrackPoint(i, 280 - i * 10, 100) for i in range(26)],  # 右→左
        2: [TrackPoint(i, 20 + i * 10, 150) for i in range(26)],  # 左→右
    }
    out = tmp_path / "r"
    write_result(out, tmp_path / "v.mp4", TrackingResult(frame, 10.0, 26, tracks), tracks, smooth=5, min_points=10)
    return Session(out)


def test_in_is_left_to_right_of_start_to_end():
    # 始点(150,0)→終点(150,199)（画面で下向き）に立つと、右手は画像の左（-x）。IN = 右→左の移動
    ln = CountLine(150, 0, 150, 199)
    assert ln.in_direction() == pytest.approx((-1.0, 0.0))
    assert ln.reversed().in_direction() == pytest.approx((1.0, 0.0))


def test_names_reverse_and_move(tmp_path):
    s = make_session(tmp_path)
    s.add_line(CountLine(150, 0, 150, 199))
    c = s.counts[0]
    assert (c.in_count, c.out_count) == (1, 1)
    s.rename_line(0, " 改札口 ")
    assert s.lines[0].label(0) == "改札口"
    s.reverse_line(0)
    assert s.lines[0].name == "改札口"  # 反転しても名前は残る
    s.move_endpoint(0, 0, 150, 120)  # 始点を下げて、上の人（y=100）を数えないようにする
    assert s.counts[0].total == 1
    s.save()
    data = json.loads((s.dir / "count_lines.json").read_text(encoding="utf-8"))
    assert data["lines"][0]["name"] == "改札口"
    assert "IN" in (s.dir / "counts.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    # 開き直しても名前が残る
    assert Session(s.dir).lines[0].name == "改札口"


def test_json_roundtrip_with_scaling(tmp_path):
    lines = [CountLine(100, 50, 100, 150, "入口"), CountLine(10, 20, 30, 40)]
    path = tmp_path / "lines.json"
    export_lines_json(path, lines, 400, 200, "a.mp4")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image"] == {"width": 400, "height": 200}
    assert [d["name"] for d in data["lines"]] == ["入口", "L2"]
    back, note = import_lines_json(path, 400, 200)
    assert note == "" and back[0] == lines[0]
    scaled, note = import_lines_json(path, 800, 100)  # 大きさが違えば比で合わせる
    assert note and (scaled[0].x1, scaled[0].y1, scaled[0].y2) == (200, 25, 75)
    assert scaled[0].in_direction() == pytest.approx(lines[0].in_direction())


def test_import_fixes_hand_edited_direction(tmp_path):
    path = tmp_path / "lines.json"
    export_lines_json(path, [CountLine(100, 50, 100, 150, "入口")], 400, 200)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lines"][0]["in_direction"] = {"x": 1.0, "y": 0.0}  # 向きだけ手で書き換えた
    path.write_text(json.dumps(data), encoding="utf-8")
    ln = import_lines_json(path, 400, 200)[0][0]
    assert ln.in_direction() == pytest.approx((1.0, 0.0))


def test_import_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        import_lines_json(path, 100, 100)


def test_repeat_seconds_controls_duplicate_warning(tmp_path):
    s = make_session(tmp_path)
    # 人1 が線(x=150)をまたいで1.5秒後に戻る
    pts = [TrackPoint(i, 100 + 10 * i, 60) for i in range(10)] + [TrackPoint(10 + i, 190 - 10 * i, 60) for i in range(10)]
    s.tracks[3] = pts
    s._tracks_xy = None
    s.set_lines([CountLine(150, 0, 150, 199)])
    s.set_repeat_sec(0.5)
    assert s.evaluations[0].duplicates == 0
    s.set_repeat_sec(3.0)
    assert s.evaluations[0].duplicates == 1


def test_suspect_settings_switch_rules_on_and_off():
    tracks = {k: np.stack([np.linspace(60, 560, 50), np.full(50, 200.0 + k)], axis=1) for k in range(12)}
    jump = tracks[0].copy()
    jump[25:] += [0, 150]
    tracks[99] = jump
    assert "jump" in find_suspects(tracks, 640, 480, 10).get(99, [])
    off = SuspectSettings(jump_on=False)
    assert "jump" not in find_suspects(tracks, 640, 480, 10, off).get(99, [])
    assert SuspectSettings.from_dict(off.to_dict()) == off
