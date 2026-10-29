"""E6 final pass: submit never blocks, queue policy, cam.final valid on the bus, clean close, and the real beam + LM on
the sample (the fixes greedy misses). GPU rescoring: publishes before submit returns, ranks with the live model's CTC
head, and finds the same fixes on the sample; the CPU child loads the face adapter's CTC head.

    venv-live/bin/python -m pytest tests/lipread/test_final.py -q
"""
import threading
import time
from collections import Counter

import numpy as np
import pytest

import lipread
from bus import CamFinal, decode, encode, validate
from lipread.final import FinalPass, GpuFinal, RemoteBeam, Rescore

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


def test_gpu_final_publishes_before_submit_returns_and_survives_a_failure():
    calls = []

    def rescore(rois, enc):
        calls.append(len(enc))
        if len(calls) == 1:
            raise RuntimeError('boom')
        return (('HELLO WORLD', -1.5), ('HELLO WORD', -2.5))

    sent = []
    fp = GpuFinal(None, sent.append, rescore=rescore)
    assert fp.submit(1, rois(1), enc=np.zeros((50, 4), np.float32))
    assert fp.submit(2, rois(2), enc=np.zeros((60, 4), np.float32))
    assert fp.errors == [1] and [m.seg_id for m in sent] == [2]
    validate(*decode(encode(sent[0], seq=1)))
    assert sent[0].nbest == (('HELLO WORLD', -1.5), ('HELLO WORD', -2.5))
    assert [(s['seg_id'], s['T']) for s in fp.stats] == [(2, 50)]
    assert fp.close() and not fp.submit(3, rois(3), enc=np.zeros((5, 4), np.float32)) and len(sent) == 1


class FlatDecoder:
    """Attention decoder with the same output for every token: rescoring then ranks by CTC alone."""

    def parameters(self):
        import torch
        return iter([torch.zeros(1)])

    def __call__(self, ys_in, mask, mem, cache):
        import torch
        return torch.zeros((*ys_in.shape, 6)), None


class CtcHead:
    """CTC head whose log-probs favour one label path, frame by frame (blank 0)."""

    def __init__(self, path):
        self.path = path

    def log_softmax(self, x):
        import torch
        logp = torch.full((1, len(self.path), 6), -9.0)
        logp[0, torch.arange(len(self.path)), torch.tensor(self.path)] = -0.01
        return logp


class TinyModel:
    """LiveModel's surface as Rescore uses it: tokens <blank> HI THERE YOU NOW <eos>."""

    def __init__(self, path):
        from types import SimpleNamespace
        import torch
        self.token_list = ['<blank>', '▁HI', '▁THERE', '▁YOU', '▁NOW', '<eos>']
        self.device = torch.device('cpu')
        self.ctc = CtcHead(path)
        self.reader = SimpleNamespace(model=SimpleNamespace(eos=5, adim=4, decoder=FlatDecoder()))


def test_rescore_ranks_with_the_live_models_ctc_head():
    """An adapter that retrains the CTC head (recipe C) changes what the GPU final pass reads: it uses model.ctc."""
    enc = np.zeros((6, 4), np.float32)
    kw = dict(lm_weight=0.0, att_beam=0, ctc_weight=1.0, per_frame=4)
    a = Rescore(TinyModel([1, 0, 2, 2, 0, 0]), **kw)(enc=enc)
    b = Rescore(TinyModel([3, 0, 4, 4, 0, 0]), **kw)(enc=enc)
    assert a[0][0] == 'HI THERE' and b[0][0] == 'YOU NOW'
    for nb in (a, b):
        texts, scores = [t for t, _ in nb], [s for _, s in nb]
        assert 1 <= len(nb) <= 5 and len(set(texts)) == len(texts)
        assert scores == sorted(scores, reverse=True) and all(np.isfinite(scores))
    assert Rescore(TinyModel([0]), **kw)(enc=np.zeros((0, 4), np.float32)) == (('', 0.0),)


@needs_weights
def test_the_cpu_child_loads_the_adapters_ctc_head(tmp_path):
    """A face adapter whose CTC head allows only one label: the child's best reading then has nothing else."""
    import queue as q
    import signal
    import torch
    from lipflow.vsr import LipReader
    from lipread.final import _child
    tokens = LipReader(device='cpu', use_lm=False, personal=False).token_list
    the = tokens.index('▁THE')
    bias = torch.full((len(tokens),), -1000.0)
    bias[0] = bias[the] = 0.0
    path = tmp_path / 'ctc_only_the.pth'
    torch.save({'ctc.ctc_lo.weight': torch.zeros((len(tokens), 768)), 'ctc.ctc_lo.bias': bias}, path)
    enc = np.random.default_rng(0).normal(size=(40, 768)).astype(np.float32)
    kw = dict(beam_size=5, nbest=5, lm_weight=0.0, ctc_weight=0.1)
    old = signal.getsignal(signal.SIGINT)
    texts = {}
    try:
        for name, fa in (('base', None), ('adapter', str(path))):
            jobs, results = q.Queue(), q.Queue()
            jobs.put(enc)
            jobs.put(None)
            _child(jobs, results, kw, 'cpu', 2, 0, fa)
            assert results.get_nowait()[0] == 'ready'
            texts[name] = [t for t, _ in results.get_nowait()[0]]
    finally:
        signal.signal(signal.SIGINT, old)
    assert set(texts['adapter'][0].split()) <= {'THE'}, texts
    assert texts['base'][0] != texts['adapter'][0], texts


@needs_weights
@needs_cuda
@needs_sample
def test_gpu_rescoring_on_the_sample(sample_video, face_model):
    """The tick's encoder output -> GPU rescoring on the tick thread: cam.final before submit returns, with the fixes
    greedy misses."""
    from lipread.model import LiveModel
    x = sample_rois(sample_video, face_model)
    model = LiveModel(warmup=(25,))
    r = model.read(x, keep_enc=True)
    sent = []
    fp = GpuFinal(model, sent.append)
    t0 = time.perf_counter()
    assert fp.submit(3, x, enc=r.enc)
    ms = (time.perf_counter() - t0) * 1000
    assert len(sent) == 1 and sent[0].seg_id == 3
    validate(*decode(encode(sent[0])))
    nbest = sent[0].nbest
    texts, scores = [t for t, _ in nbest], [s for _, s in nbest]
    assert 1 <= len(nbest) <= 5 and len(set(texts)) == len(texts)
    assert scores == sorted(scores, reverse=True) and all(np.isfinite(scores))
    assert 'AND RAISED MOSTLY' in texts[0] and 'FORTY THREE' in texts[0], texts
    truth = Counter(lipread.SAMPLE_TEXT.split())
    assert sum((truth & Counter(texts[0].split())).values()) >= 15, texts[0]
    assert fp.rescore(x) == nbest
    print(f'\nGPU rescoring {ms:.0f} ms (T {len(x)}), best: {sent[0].nbest[0]}')
