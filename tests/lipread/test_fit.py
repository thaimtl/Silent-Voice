"""fit() and the emotion fitter's frame clock, without the lip-reading weights."""
import json

import cv2
import numpy as np

from lipread.emotion_fit import clip_frames
from lipread.source import ReplaySource, fit
from lipread.types import Frame


def test_fit_shrinks_1080p_and_leaves_a_small_frame():
    big = Frame(t=0.0, image=np.zeros((1080, 1920, 3), np.uint8), index=0)
    shrunk = fit(big, (1280, 720))
    assert shrunk.image.shape[:2] == (720, 1280) and shrunk.t == 0.0
    wide = Frame(t=1.0, image=np.zeros((1024, 1280, 3), np.uint8))
    fitted = fit(wide, (1280, 720))
    assert fitted.image.shape[:2] == (720, 900)
    small = Frame(t=2.0, image=np.zeros((360, 640, 3), np.uint8))
    assert fit(small, (1280, 720)) is small
    assert fit(big, None) is big


def test_clip_frames_follow_replay_source(tmp_path):
    session = tmp_path / 'session_01'
    session.mkdir()
    video = session / 'c.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (1920, 1080))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((1080, 1920, 3), i * 10, np.uint8))
    writer.release()
    ts = np.arange(12, dtype=np.float64) / 30.0
    ts[8:] += 0.5
    np.save(session / 'c_video_ts.npy', ts)
    (session / 'manifest.jsonl').write_text(json.dumps({
        'id': 'c', 'video': 'c.avi', 'video_ts': 'c_video_ts.npy', 'clip_t0': 0.0}) + '\n')
    src = ReplaySource('c', realtime=False, root=str(session))
    try:
        expected = [frame.t + src.t_offset for frame in src.frames()]
    finally:
        src.close()
    got = list(clip_frames(str(tmp_path), 'session_01', {'id': 'c'}))
    assert [frame.recorder_t for frame in got] == expected
    assert len(got) != 12
    assert all(frame.image.shape[:2] == (720, 1280) for frame in got)
