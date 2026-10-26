import numpy as np

from foot_overlay.resume import POINTS_NAME, PROGRESS_NAME, Checkpoint, progress_info
from foot_overlay.session import write_result
from foot_overlay.tracking import TrackingResult, TrackPoint, _record, _stitch

SETTINGS = {"model": "yolo11n.pt", "conf": 0.3}


def make_video(tmp_path, name="動画.mp4", size=1000):
    video = tmp_path / name
    video.write_bytes(b"x" * size)
    return video


def pts(tid, frames, x0=10.0):
    return [TrackPoint(f, x0 + f, 50.0) for f in frames]


def test_commit_and_load_appends_only_new_points(tmp_path):
    video = make_video(tmp_path)
    folder = tmp_path / "out"
    ck = Checkpoint(folder, video, SETTINGS)
    ck.start_fresh()
    tracks = {1: pts(1, range(0, 5)), 2: pts(2, range(3, 5), x0=200)}
    ck.commit(5, 20, tracks, 100000)
    size_after_first = (folder / POINTS_NAME).stat().st_size
    tracks[1] += pts(1, range(5, 8))
    ck.commit(8, 20, tracks, 100001)
    # 2回目は増えた3点だけ追記される
    assert (folder / POINTS_NAME).read_text().count("\n") == 10
    assert (folder / POINTS_NAME).stat().st_size > size_after_first

    saved = Checkpoint(folder, video, SETTINGS).load()
    assert saved.frames_done == 8 and saved.total_frames == 20 and saved.next_split_id == 100001
    assert [f for f, _, _ in saved.tracks[1]] == list(range(8))
    assert len(saved.tracks[2]) == 2
    assert progress_info(folder, video)["percent"] == 40


def test_points_written_after_last_commit_are_dropped(tmp_path):
    """電源断で、保存しきれなかった（progress.json に記録されていない）点が残っていても使わない。"""
    video = make_video(tmp_path)
    folder = tmp_path / "out"
    ck = Checkpoint(folder, video, SETTINGS)
    ck.start_fresh()
    ck.commit(3, 10, {1: pts(1, range(3))}, 100000)
    with (folder / POINTS_NAME).open("a") as f:
        f.write("1,3,13.00,50.00\n1,4,1")  # 途中で切れた書きかけの行
    ck2 = Checkpoint(folder, video, SETTINGS)
    saved = ck2.load()
    assert len(saved.tracks[1]) == 3
    # 続けて保存しても壊れない
    ck2.commit(5, 10, {1: pts(1, range(5))}, 100000)
    assert len(Checkpoint(folder, video, SETTINGS).load().tracks[1]) == 5


def test_other_video_or_settings_are_not_resumed(tmp_path):
    video = make_video(tmp_path)
    folder = tmp_path / "out"
    ck = Checkpoint(folder, video, SETTINGS)
    ck.start_fresh()
    ck.commit(3, 10, {1: pts(1, range(3))}, 100000)
    assert Checkpoint(folder, video, {**SETTINGS, "conf": 0.5}).load() is None
    other = make_video(tmp_path, "別.mp4", size=999)
    assert Checkpoint(folder, other, SETTINGS).load() is None
    assert progress_info(folder, other) is None
    (folder / PROGRESS_NAME).write_text("{壊れた", encoding="utf-8")
    assert Checkpoint(folder, video, SETTINGS).load() is None


def test_finished_result_removes_progress_files(tmp_path):
    video = make_video(tmp_path)
    folder = tmp_path / "out"
    ck = Checkpoint(folder, video, SETTINGS)
    ck.start_fresh()
    ck.commit(3, 10, {1: pts(1, range(3))}, 100000)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = TrackingResult(first_frame=frame, fps=10, total_frames=10, tracks={1: pts(1, range(10))})
    write_result(folder, video, result, result.tracks, smooth=1, min_points=1)
    assert not (folder / PROGRESS_NAME).exists() and not (folder / POINTS_NAME).exists()
    assert progress_info(folder, video) is None


def test_stitch_connects_people_across_resume(tmp_path):
    """再開後の新しい追跡IDが、同じ位置にいた保存済みの線につながる。"""
    w, h = 800, 600
    result = TrackingResult(first_frame=np.zeros((h, w, 3), np.uint8), fps=10, total_frames=40)
    # 保存済み: ID 3 は右へ、ID 7 は下へ歩いている（フレーム 0〜19）
    result.tracks = {
        3: [TrackPoint(f, 100 + 5 * f, 300) for f in range(20)],
        7: [TrackPoint(f, 500, 100 + 4 * f) for f in range(20)],
    }
    result._id_offset = 7
    # フレーム 15〜19 を新しい追跡器で解析し直した結果（ID は 1, 2 から振り直されている）
    overlap = {
        f: [(1, 500 + 1.0, 100 + 4 * f, False), (2, 100 + 5 * f - 2.0, 300, False), (5, 700, 50, False)]
        for f in range(15, 20)
    }
    _stitch(result, overlap, w, h)
    for f in range(20, 25):
        _record(result, f, [(1, 500, 100 + 4 * f, False), (2, 100 + 5 * f, 300, False), (5, 700, 50, False)], w, h)
    assert [p.frame for p in result.tracks[3]] == list(range(25))
    assert [p.frame for p in result.tracks[7]] == list(range(25))
    # 保存済みの線と重ならない新しい人は、保存済みのIDと重ならない新しいIDになる
    assert set(result.tracks) == {3, 7, 5 + 7}
