"""E2 re-read loop: 100 ms tick schedule, caps, end-of-stream read, frame times, timing_ms, the tick loop's failure
handling (stall, thread death, failed ticks, backlog), the crop thread's ownership of source/cropper, and the real
model on the sample.

    venv-live/bin/python -m pytest tests/lipread/test_run.py -q
"""
import queue
import threading
import time

import numpy as np
import pytest

import lipread
from bus import now, decode, encode, validate
from lipread.endpoint import Segmenter
from lipread.model import LiveModel, Reading
from lipread.run import End, Runner, crop_thread, due_after, make_source, parse_args, rebased, serve, shifted
from bus import TIMING_KEYS, TOPK
from lipread.source import fit
from lipread.types import FPS, Crop, Frame

from .conftest import needs_cuda, needs_sample, needs_weights


class FakeModel:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def read(self, rois, keep_enc=False, cache=None):
        T = len(rois)
        n = len(self.calls)
        self.calls.append(T)
        if n in self.fail or 'all' in self.fail:
            raise RuntimeError(f'read {n} failed')
        ids = np.tile(np.arange(TOPK, dtype=np.int32), (T, 1))
        logp = np.tile(-np.arange(TOPK, dtype=np.float32) - 0.1, (T, 1))
        return Reading(ids, logp, np.full(T, -0.1, np.float32), 'HI', 1.0, 0.1, None, 0.4, 0.6)


def crops(n, t0=100.0, times=None):
    ts = times if times is not None else [t0 + i / FPS for i in range(n)]
    return [Crop(t=t, roi96=np.full((96, 96), i % 256, np.uint8), mouth_open=0.1, face_ok=True)
            for i, t in enumerate(ts)]


def run(cs, tick_ms=100, model=None):
    sent = []
    r = Runner(model or FakeModel(), sent.append, Segmenter(never_end=True), tick_ms=tick_ms)
    stats = [s for c in cs for s in r.on_crops([(c, 5.0)])]
    assert [s.T for s in stats] == [len(t.frame_t) for t in sent]
    return r, sent


def test_due_points():
    pts, n = [], 0
    for _ in range(8):
        n = due_after(n)
        pts.append(n)
    assert pts == [3, 5, 8, 10, 13, 15, 18, 20]
    assert [due_after(n, 200) for n in (0, 4, 5, 9)] == [5, 5, 10, 10]
    assert due_after(18) == 20 and due_after(19) == 20 and due_after(20) == 23


def test_ticks_every_100ms_and_a_last_read_at_the_end():
    r, sent = run(crops(37))
    assert [len(t.frame_t) for t in sent] == [3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35]
    assert [s.T for s in r.finish()] == [37] and r.segmenter.segment is None
    assert len(sent[-1].frame_t) == 37
    for t in sent:
        validate(*decode(encode(t)))
    assert sent[-1].frame_t[0] == sent[-1].seg_start == 100.0


def test_tick_ms_200():
    _, sent = run(crops(37), tick_ms=200)
    assert [len(t.frame_t) for t in sent] == [5, 10, 15, 20, 25, 30, 35]


def test_timing_ms_filled():
    r, sent = run(crops(10))
    for t in sent:
        tm = t.timing_ms
        assert tuple(tm) == TIMING_KEYS and tm['T'] == len(t.frame_t) and isinstance(tm['T'], int)
        assert tm['mediapipe'] == 5.0 and tm['frontend'] == 0.4 and tm['conformer'] == 0.6 and tm['ctc'] == 0.1
        assert all(isinstance(tm[k], float) and tm[k] >= 0 for k in TIMING_KEYS[1:])
        assert tm['total'] <= r.stats[0].total_ms + 1e3
        topic, msg = decode(encode(t))
        assert validate(topic, msg)['timing_ms']['T'] == tm['T']
    r = Runner(FakeModel(), [].append, Segmenter(never_end=True))
    cs = crops(5)
    for c, ms in zip(cs, (1.0, 2.0, 3.0, 10.0, 20.0)):
        r.on_crops([(c, ms)])
    assert [s.mediapipe_ms for s in r.stats] == [2.0, 15.0]


def test_cap_carries_2s_with_the_same_frame_times():
    times = [100 + ((i + 1) // 2) * 0.08 for i in range(230)]
    r, sent = run(crops(230, times=times))
    seg1 = [t for t in sent if t.seg_id == 1]
    seg2 = [t for t in sent if t.seg_id == 2]
    assert len(seg1[-1].frame_t) == 200
    assert len(seg2[0].frame_t) == 50
    assert np.array_equal(seg2[0].frame_t[:50], seg1[-1].frame_t[150:])
    assert seg2[0].frame_t[0] == times[150] + 0.04
    assert seg2[0].seg_start == seg2[0].frame_t[0]
    for t in sent:
        validate(*decode(encode(t)))
    assert np.all(np.diff(np.concatenate([seg1[-1].frame_t, seg2[-1].frame_t[50:]])) > 0)


def test_frame_times_strictly_increase_over_repeated_frames():
    r, sent = run(crops(5, times=(1.0, 1.0, 1.0, 1.08, 1.2)))
    r.finish()
    ft = sent[-1].frame_t
    assert ft.tolist() == pytest.approx([1.0, 1.04, 1.08, 1.12, 1.2])
    assert np.all(np.diff(ft) > 0)
    assert np.array_equal(sent[0].frame_t, ft[:3])


def test_backlog_drains_then_one_tick_on_the_newest_state():
    sent = []
    r = Runner(FakeModel(), sent.append, Segmenter(never_end=True))
    cs = crops(20)
    stats = [s for i in range(0, 20, 4) for s in r.on_crops([(c, 1.0) for c in cs[i:i + 4]])]
    assert [len(t.frame_t) for t in sent] == [4, 8, 12, 16, 20]
    assert [s.backlog for s in stats] == [3] * 5


def test_unk_and_eos_are_not_in_the_text():
    m = LiveModel.__new__(LiveModel)
    m.token_list = ['<blank>', '<unk>', '▁HI', '▁THERE', '<eos>']
    assert m.text(np.array([1, 2, 2, 0, 1, 3, 4, 1])) == 'HI THERE'
    assert m.text(np.array([1, 1, 4])) == ''


def q_of(items):
    q = queue.Queue()
    for x in items:
        q.put(x)
    return q


def make_runner(model=None):
    sent = []
    return Runner(model or FakeModel(), sent.append, Segmenter(never_end=True)), sent


def test_serve_clean_end_and_thread_death():
    r, sent = make_runner()
    assert serve(r, q_of([(c, 1.0) for c in crops(7)] + [End()]), live_mode=False) == 0
    assert [len(t.frame_t) for t in sent] == [3, 5, 7]
    r, sent = make_runner()
    code = serve(r, q_of([(c, 1.0) for c in crops(7)] + [End(RuntimeError('boom'))]), live_mode=False)
    assert code == 1 and [len(t.frame_t) for t in sent] == [3, 5, 7] and r.segmenter.segment is None


def test_serve_skips_a_failed_tick_and_gives_up_after_10_in_a_row():
    r, sent = make_runner(FakeModel(fail={0}))
    assert serve(r, q_of([(c, 1.0) for c in crops(7)] + [End()]), live_mode=False) == 0
    assert [len(t.frame_t) for t in sent] == [4, 5, 7]
    r, sent = make_runner(FakeModel(fail={'all'}))
    assert serve(r, q_of([(c, 1.0) for c in crops(40)] + [End()]), live_mode=False) == 1
    assert len(r.model.calls) == 10 and not sent


def test_serve_live_drains_the_backlog():
    r, sent = make_runner()
    assert serve(r, q_of([(c, 1.0) for c in crops(7)] + [End()]), live_mode=True, stall_s=1.0) == 0
    assert [len(t.frame_t) for t in sent] == [7] and r.stats[0].backlog == 6


def test_serve_ends_the_segment_on_a_camera_stall():
    r, sent = make_runner()
    q = q_of([(c, 1.0) for c in crops(4)])

    def later():
        time.sleep(0.3)
        assert r.segmenter.segment is None
        for c in crops(3, t0=200.0):
            q.put((c, 1.0))
        q.put(End())
    th = threading.Thread(target=later)
    th.start()
    assert serve(r, q, live_mode=True, stall_s=0.1) == 0
    th.join()
    assert [(t.seg_id, len(t.frame_t)) for t in sent] == [(1, 4), (2, 3)]


class FakeSource:
    def __init__(self, n, fail_at=None):
        self.n, self.fail_at, self.closed_by = n, fail_at, None

    def frames(self):
        for i in range(self.n):
            if i == self.fail_at:
                raise OSError('camera unplugged')
            yield Frame(t=10.0 + i / FPS, image=np.zeros((4, 4, 3), np.uint8), index=i)

    def close(self):
        self.closed_by = threading.current_thread().name


class FakeCropper:
    last_face = None

    def __init__(self):
        self.closed_by = None

    def push(self, f):
        return [Crop(t=f.t, roi96=np.zeros((96, 96), np.uint8), mouth_open=0.0, face_ok=True)]

    def flush(self):
        return []

    def close(self):
        self.closed_by = threading.current_thread().name


@pytest.mark.parametrize('fail_at', [None, 3])
def test_crop_thread_owns_and_closes_source_and_cropper(fail_at):
    src, cr, q, stop = FakeSource(6, fail_at), FakeCropper(), queue.Queue(), threading.Event()
    th = threading.Thread(target=crop_thread, args=(src, cr, q, stop), name='crop')
    th.start()
    th.join(5)
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert src.closed_by == cr.closed_by == 'crop'
    if fail_at is None:
        assert len(items) == 7 and items[-1] == End()
    else:
        assert len(items) == 4 and isinstance(items[-1].err, OSError)
    assert all(isinstance(c, Crop) and ms >= 0 for c, ms in items[:-1])


def test_crop_thread_stops_on_stop():
    src, cr, q, stop = FakeSource(10 ** 6), FakeCropper(), queue.Queue(), threading.Event()
    th = threading.Thread(target=crop_thread, args=(src, cr, q, stop, None, True), name='crop')
    th.start()
    while q.qsize() < 5:
        time.sleep(0.001)
    stop.set()
    th.join(5)
    assert not th.is_alive() and src.closed_by == 'crop'
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert items[-1] == End() and abs(items[0][0].t - now()) < 5


def test_rebased_moves_replay_times_onto_the_bus_clock():
    fs = [Frame(t=20.4 + i / FPS, image=None, index=i) for i in range(3)]
    t0 = now()
    out = list(rebased(fs))
    assert abs(out[0].t - t0) < 0.5
    assert np.diff([f.t for f in out]) == pytest.approx([1 / FPS] * 2)



def test_recorder_clock_keeps_the_recordings_times():
    fs = [Frame(t=i / FPS, image=None, index=i) for i in range(3)]       # ReplaySource: ts[i] - ts[0]
    assert [f.t for f in shifted(fs, 5000.0)] == pytest.approx([5000.0 + i / FPS for i in range(3)])
    assert parse_args(['--replay', 'x.mp4', '--recorder-clock']).recorder_clock
    assert not parse_args(['--replay', 'x.mp4']).recorder_clock

@needs_weights
@needs_cuda
@needs_sample
def test_real_model_on_the_sample(sample_video, face_model):
    from lipread.crop import LiveCropper
    from lipread.source import ReplaySource
    src = ReplaySource(sample_video, lipread.SAMPLE_START, lipread.SAMPLE_END, realtime=False)
    cropper = LiveCropper(face_model, k=0)
    sent = []
    runner = Runner(LiveModel(warmup=(25,)), sent.append, Segmenter())
    for f in rebased(src.frames()):
        for c in cropper.push(f):
            runner.on_crops([(c, 5.0)])
    runner.finish()
    src.close()
    cropper.close()
    assert sent and len({t.seg_id for t in sent}) == 1
    for t in sent:
        validate(*decode(encode(t)))
        tm = t.timing_ms
        assert tm['T'] == len(t.frame_t) and tm['frontend'] > 0 and tm['conformer'] > 0
        assert tm['frontend'] + tm['conformer'] + tm['ctc'] <= tm['total'] + 0.5
    assert len(sent[-1].frame_t) == 192
    words = set(sent[-1].greedy.split())
    assert {'BORN', 'NEW', 'YORK', 'CHICAGO', 'GRADUATED', 'SMITH', 'COLLEGE'} <= words, sent[-1].greedy
    steps = np.diff([len(t.frame_t) for t in sent])
    assert set(steps[:-1].tolist()) == {2, 3}
    print(f'\n{len(sent)} ticks, last: {sent[-1].greedy}')


class EncModel(FakeModel):
    def read(self, rois, keep_enc=False, cache=None):
        r = super().read(rois)
        return Reading(r.topk_ids, r.topk_logp, r.blank_logp, r.greedy, r.enc_ms, r.ctc_ms,
                       ('enc', len(rois)) if keep_enc else None, r.frontend_ms, r.conformer_ms)


class FakeFinal:
    def __init__(self):
        self.got = []

    def submit(self, seg_id, rois, enc=None):
        self.got.append((seg_id, len(rois), enc))


def test_cap_and_end_go_to_the_final_pass_reusing_the_last_read():
    final = FakeFinal()
    r = Runner(EncModel(), lambda t: None, Segmenter(never_end=True), final=final)
    for c in crops(230):
        r.on_crops([(c, 5.0)])
    r.finish()
    assert [(s, T) for s, T, _ in final.got] == [(1, 200), (2, 80)]
    assert final.got[0][2] == ('enc', 200) and final.got[1][2] == ('enc', 80)


@needs_weights
@needs_cuda
@needs_sample
def test_frontend_cache_matches_whole_segment_reads(sample_video, face_model):
    """E3 step 2: the front end once per frame gives the same greedy text as re-encoding the whole segment, on every
    tick of the 100 ms schedule, including a cap (new segment, new cache)."""
    from lipread.crop import LiveCropper
    from lipread.source import ReplaySource
    src = ReplaySource(sample_video, lipread.SAMPLE_START, lipread.SAMPLE_END, realtime=False)
    cropper = LiveCropper(face_model, k=0)
    cs = [c for f in rebased(src.frames()) for c in cropper.push(f)]
    src.close()
    cropper.close()
    model = LiveModel(warmup=(25,))
    out = {}
    for cached in (False, True):
        sent = []
        r = Runner(model, sent.append, Segmenter(never_end=True, cap_s=4.0, carry_s=1.0), fe_cache=cached)
        for c in cs:
            r.on_crops([(c, 5.0)])
        r.finish()
        out[cached] = sent
    assert len(out[True]) == len(out[False]) > 40 and len({t.seg_id for t in out[True]}) >= 2
    assert [t.greedy for t in out[True]] == [t.greedy for t in out[False]]
    assert all(np.array_equal(a.topk_ids[:, 0], b.topk_ids[:, 0]) for a, b in zip(out[True], out[False]))


@needs_weights
@needs_cuda
def test_encode_matches_the_encoders_own_forward():
    """LiveModel's split encode (front end, then embed + Conformer timed apart) is the espnet Encoder.forward."""
    import torch
    m = LiveModel(warmup=())
    rois = np.random.default_rng(0).integers(0, 256, (30, 96, 96), dtype=np.uint8)
    with torch.inference_mode():
        ref, _ = m.encoder(m.reader.to_tensor(rois).unsqueeze(0).to(m.device, m.dtype), None)
        assert torch.allclose(m.encode(rois), ref[0], atol=1e-4, rtol=1e-4)


def test_size_flags_reach_the_camera(monkeypatch):
    import lipread.source
    got = {}
    monkeypatch.setattr(lipread.source, 'CameraSource', lambda cam, **kw: got.update(cam=cam, **kw) or 'cam')
    args = parse_args([])
    assert args.size == (1920, 1080) and args.process_size == (1280, 720)
    assert make_source(args) == 'cam' and got == {'cam': 0, 'width': 1920, 'height': 1080, 'settings': False}
    args = parse_args(['--cam', '2', '--size', '1280x720', '--process-size', '0', '--camera-settings'])
    make_source(args)
    assert got == {'cam': 2, 'width': 1280, 'height': 720, 'settings': True} and args.process_size is None
    with pytest.raises(SystemExit):
        parse_args(['--size', '1280'])


def test_large_frames_are_shrunk_before_the_cropper():
    seen = []

    class SizeCropper(FakeCropper):
        def push(self, f):
            seen.append(f.image.shape)
            return super().push(f)

    class BigSource(FakeSource):
        def frames(self):
            for i, (w, h) in enumerate(((1920, 1080), (640, 360), (1280, 1024))):
                yield Frame(t=10.0 + i / FPS, image=np.zeros((h, w, 3), np.uint8), index=i)

    q = queue.Queue()
    crop_thread(BigSource(3), SizeCropper(), q, threading.Event(), process_size=(1280, 720))
    assert seen == [(720, 1280, 3), (360, 640, 3), (720, 900, 3)]
    small = Frame(t=0.0, image=np.zeros((360, 640, 3), np.uint8))
    assert fit(small, (1280, 720)) is small and fit(small, None) is small


def test_a_failing_emotion_submit_never_stops_the_crops():
    class BadEmotion:
        calls = 0

        def submit(self, t, image, face):
            BadEmotion.calls += 1
            raise ValueError('emotion broke')

    q = queue.Queue()
    crop_thread(FakeSource(6), FakeCropper(), q, threading.Event(), emotion=BadEmotion())
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert len(items) == 7 and items[-1] == End() and BadEmotion.calls == 6


def test_a_failed_read_in_a_backlog_loses_no_crop():
    model = FakeModel()
    r = Runner(model, [].append, Segmenter(never_end=True, cap_s=2.0, carry_s=1.0))
    cs = crops(54)
    for c in cs[:49]:
        r.on_crops([(c, 1.0)])
    model.fail = {len(model.calls)}
    with pytest.raises(RuntimeError):
        r.on_crops([(c, 1.0) for c in cs[49:]])
    seg = r.segmenter.segment
    assert seg.seg_id == 2 and [c.t for c in seg.crops] == [c.t for c in cs[25:]]


def test_a_failed_last_read_still_reaches_the_final_pass():
    final = FakeFinal()
    model = FakeModel()
    r = Runner(model, [].append, Segmenter(never_end=True), final=final)
    for c in crops(19):
        r.on_crops([(c, 1.0)])
    model.fail = {len(model.calls)}
    with pytest.raises(RuntimeError):
        r.finish()
    assert [(s, T, enc) for s, T, enc in final.got] == [(1, 19, None)]
    assert r.segmenter.segment is None


def test_crop_times_reset_while_no_segment_is_open():
    r = Runner(FakeModel(), [].append, Segmenter())
    for c in crops(50):
        r.on_crops([(c, 7.0)])
    assert r.segmenter.segment is None and r.crop_ms == []


def test_stream_segments_flag_starts():
    assert Segmenter.stream().never_end and parse_args(['--segments', 'stream']).segments == 'stream'
