"""E6 final pass: submit never blocks, queue policy, cam.final valid on the bus, clean close, and the real beam + LM on
the sample (the fixes greedy misses).

    venv-live/bin/python -m pytest tests/lipread/test_final.py -q
"""
import threading
import time
from collections import Counter

import numpy as np
import pytest

import lipread
from bus import CamFinal, decode, encode, validate
from lipread.final import FinalPass, RemoteBeam

from .conftest import needs_cuda, needs_sample, needs_weights


class SlowBeam:
    """Fake beam: sleeps, records what it decoded; seg ids come from the rois' first pixel."""

    def __init__(self, s=0.2, out=(('HELLO WORLD', -1.5), ('HELLO WORD', -2.5))):
        self.s, self.out, self.calls = s, out, []
        self.started = threading.Event()

    def __call__(self, rois, enc):
        self.started.set()
        self.calls.append(int(rois[0, 0, 0]))
        time.sleep(self.s)
        return self.out


def rois(seg_id, T=50):
    return np.full((T, 96, 96), seg_id, np.uint8)


def make(beam, **kw):
    sent = []
    return FinalPass(None, sent.append, beam=beam, **kw), sent


def test_submit_never_blocks_while_busy():
    beam = SlowBeam(0.3)
    fp, sent = make(beam, max_pending=2)
    ms = []
    for i in range(1, 8):
        t0 = time.perf_counter()
        assert fp.submit(i, rois(i))
        ms.append((time.perf_counter() - t0) * 1000)
        if i == 1:
            assert beam.started.wait(1.0)
    assert max(ms) < 5, ms
    assert fp.close(timeout=3.0)


def test_queue_keeps_the_newest_and_counts_drops():
    beam = SlowBeam(0.2)
    fp, sent = make(beam, max_pending=2)
    fp.submit(1, rois(1))
    assert beam.started.wait(1.0)
    for i in (2, 3, 4, 5):
        fp.submit(i, rois(i))
    assert fp.wait_idle(3.0)
    assert fp.dropped == [2, 3]
    assert beam.calls == [1, 4, 5]
    assert [m.seg_id for m in sent] == [1, 4, 5]
    assert [s['seg_id'] for s in fp.stats] == [1, 4, 5]
    assert all(s['T'] == 50 and s['beam_ms'] >= 150 and s['queued_ms'] >= 0 for s in fp.stats)
    assert fp.stats[1]['queued_ms'] >= 150
    assert fp.close(1.0)


def test_cap_then_continuation_both_get_a_final():
    """A capped 8 s segment and its continuation closing while the first beam still runs: nothing dropped."""
    beam = SlowBeam(0.2)
    fp, sent = make(beam)
    fp.submit(1, rois(1, 200))
    fp.submit(2, rois(2, 60))
    assert fp.close(3.0)
    assert [m.seg_id for m in sent] == [1, 2] and fp.dropped == []


def test_published_finals_are_valid_on_the_bus():
    outs = [(('HELLO WORLD', -1.5), ('HELLO WORD', -2.5)), (('', -3.0),), (("IT'S 1943", -0.25),)]
    for out in outs:
        fp, sent = make(SlowBeam(0.0, out))
        fp.submit(7, rois(7))
        assert fp.close(2.0)
        assert len(sent) == 1 and isinstance(sent[0], CamFinal)
        topic, msg = decode(encode(sent[0], seq=1))
        validate(topic, msg)
        assert topic == 'cam.final' and msg['seg_id'] == 7
        assert [(n['text'], n['score']) for n in msg['nbest']] == list(out)


def test_close_finishes_or_drops_and_refuses_new_work():
    beam = SlowBeam(0.3)
    fp, sent = make(beam)
    for i in (1, 2, 3):
        fp.submit(i, rois(i))
    assert beam.started.wait(1.0)
    t0 = time.perf_counter()
    fp.close(timeout=0.45)
    assert time.perf_counter() - t0 < 0.6
    assert 3 in fp.dropped and not fp.submit(4, rois(4))
    fp.thread.join(1.0)
    assert not fp.thread.is_alive()
    assert [m.seg_id for m in sent] == [1, 2]


def test_a_failing_beam_is_logged_and_the_worker_goes_on():
    calls = []

    def beam(r, enc):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('boom')
        return (('OK', -1.0),)

    fp, sent = make(beam)
    fp.submit(1, rois(1))
    fp.submit(2, rois(2))
    assert fp.close(2.0)
    assert fp.errors == [1] and [m.seg_id for m in sent] == [2]


def sample_rois(sample_video, face_model):
    from lipread.crop import LiveCropper
    from lipread.source import ReplaySource
    src = ReplaySource(sample_video, lipread.SAMPLE_START, lipread.SAMPLE_END, realtime=False)
    cropper = LiveCropper(face_model, k=0)
    crops = [c for f in src.frames() for c in cropper.push(f)] + list(cropper.flush())
    src.close()
    cropper.close()
    return np.stack([c.roi96 for c in crops])


def check_sample_nbest(nbest):
    texts = [t for t, _ in nbest]
    scores = [s for _, s in nbest]
    assert 1 <= len(nbest) <= 5 and len(set(texts)) == len(texts)
    assert scores == sorted(scores, reverse=True) and all(np.isfinite(scores))
    assert any('AND RAISED MOSTLY' in t for t in texts), texts
    assert any('NINETEEN FORTY THREE' in t for t in texts), texts
    truth = Counter(lipread.SAMPLE_TEXT.split())
    hit = sum((truth & Counter(texts[0].split())).values())
    assert hit >= 15, (hit, texts[0])


def test_a_child_that_dies_while_loading_fails_fast():
    from tests.lipread import fake_children
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match='died while starting'):
        RemoteBeam(target=fake_children.dies_on_start, start_timeout=60)
    assert time.perf_counter() - t0 < 15


def test_a_child_that_dies_is_started_again_on_the_next_segment():
    from tests.lipread import fake_children
    beam = RemoteBeam(target=fake_children.echo)
    try:
        assert beam(enc=np.zeros((5, 4), np.float32)) == (('HI', -1.0),)
        with pytest.raises(RuntimeError, match='died'):
            beam(enc=np.zeros((0, 4), np.float32))
        assert beam(enc=np.zeros((5, 4), np.float32)) == (('HI', -1.0),) and beam.restarts == 1
    finally:
        beam.close()
    assert not beam.proc.is_alive()


def test_close_with_a_big_job_queued_for_a_dead_child_does_not_hang():
    from tests.lipread import fake_children
    beam = RemoteBeam(target=fake_children.echo)
    with pytest.raises(RuntimeError):
        beam(enc=np.zeros((0, 4), np.float32))
    beam.jobs.put(np.zeros((200, 768), np.float32))
    t0 = time.perf_counter()
    beam.close()
    assert time.perf_counter() - t0 < 5 and not beam.proc.is_alive()


@needs_weights
@needs_cuda
@needs_sample
def test_real_beam_on_the_sample(sample_video, face_model):
    """The default design: the tick's encoder output -> beam + LM in the CPU process. Then the in-process GPU Beam on
    the rois gives the same 5-best."""
    from lipread.final import Beam
    from lipread.model import LiveModel
    x = sample_rois(sample_video, face_model)
    assert len(x) == 192
    model = LiveModel(warmup=(25,))
    r = model.read(x, keep_enc=True)
    assert r.enc is not None and tuple(r.enc.shape) == (192, model.reader.model.adim)
    assert model.read(x).enc is None
    sent = []
    fp = FinalPass(model, sent.append)
    t0 = time.perf_counter()
    assert fp.submit(3, x, enc=r.enc)
    assert (time.perf_counter() - t0) * 1000 < 5
    assert fp.close(timeout=30.0)
    assert len(sent) == 1 and sent[0].seg_id == 3
    validate(*decode(encode(sent[0])))
    check_sample_nbest(sent[0].nbest)
    gpu = Beam(model)(x)
    check_sample_nbest(gpu)
    assert gpu[0][0] == sent[0].nbest[0][0]
    print(f'\nbeam {fp.stats[0]["beam_ms"]:.0f} ms (process, CPU), best: {sent[0].nbest[0]}')


def test_load_ctc_head_takes_only_the_ctc_tensors(tmp_path):
    import torch
    from lipread.final import load_ctc_head

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ctc = torch.nn.Module()
            self.ctc.ctc_lo = torch.nn.Linear(3, 2)
            self.decoder = torch.nn.Linear(3, 2)

    adapter, base = Net(), Net()
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(1.0)
    torch.save(adapter.state_dict(), tmp_path / 'a.pth')
    decoder_before = base.decoder.weight.clone()
    assert load_ctc_head(base, str(tmp_path / 'a.pth')) == 2
    assert torch.equal(base.ctc.ctc_lo.weight, adapter.ctc.ctc_lo.weight)
    assert torch.equal(base.decoder.weight, decoder_before)
    torch.save({'encoder.embed.weight': torch.zeros(1)}, tmp_path / 'b.pth')
    assert load_ctc_head(base, str(tmp_path / 'b.pth')) == 0
