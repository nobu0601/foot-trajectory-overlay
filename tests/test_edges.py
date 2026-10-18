from foot_overlay.drawing import default_line_width, outline_width
from foot_overlay.tracking import (
    SPLIT_ID_BASE,
    TrackPoint,
    _renumber_split_ids,
    foot_outside_frame,
    split_at_frame_edges,
)

W, H = 640, 480


def test_box_cut_by_bottom_or_side_edge_means_feet_outside():
    assert foot_outside_frame(100, 300, 160, 479, W, H)  # 下端で切れている
    assert foot_outside_frame(0, 100, 40, 300, W, H)  # 左端で切れている
    assert foot_outside_frame(600, 100, 639, 300, W, H)  # 右端で切れている
    assert not foot_outside_frame(100, 100, 160, 400, W, H)  # 全身が映っている


def test_split_at_frame_edges_ends_line_and_starts_new_one_on_return():
    pts = [TrackPoint(0, 300, 400), TrackPoint(1, 300, 450), TrackPoint(2, 300, 479), TrackPoint(3, 300, 479),
           TrackPoint(4, 310, 460), TrackPoint(5, 320, 420)]
    out = split_at_frame_edges({5: pts}, W, H)
    assert [p.frame for p in out[5]] == [0, 1]
    assert [p.frame for p in out[6]] == [4, 5]  # 戻ってきた後は別の線
    assert len(out) == 2


def test_split_ids_are_renumbered_after_normal_ids():
    tracks = {1: [], 7: [], SPLIT_ID_BASE: [], SPLIT_ID_BASE + 1: []}
    assert sorted(_renumber_split_ids(tracks)) == [1, 7, 8, 9]


def test_thin_lines_have_no_outline():
    assert outline_width(1) == 0
    assert outline_width(2) == 3
    assert default_line_width(768, 432) == 1
