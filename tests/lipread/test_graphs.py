"""CUDA-graph reads (LiveModel graphs=True) against the eager reads they replace, at every captured length 1..200, fp16
and fp32: the same top-1 id per frame and top-k / blank log-probabilities within 1e-3 (measured: identical) of the
cached eager read the live tick does (Runner fe_cache=True) and of a fresh whole-segment read; a kept encoder output
survives later replays; an uncaptured length reads eagerly.

    venv-live/bin/python -m pytest tests/lipread/test_graphs.py -q -s      # -s prints the worst difference per range
"""
import gc

import numpy as np
import pytest

from bus import encode
from lipread.endpoint import Segmenter
from lipread.model import FrontendCache, LiveModel
from lipread.run import Runner
from lipread.types import FPS, Crop

from .conftest import needs_cuda, needs_weights

pytestmark = [needs_weights, needs_cuda]

N = 200
TOL = 1e-3


@pytest.fixture(scope='module')
def rois():
    from lipread.bench.tick_bench import sample_rois
    x, what = sample_rois(N)
    print(f'\n{what}')
    return x


@pytest.fixture(scope='module', params=[True, False], ids=['fp16', 'fp32'])
def models(request):
    import torch
    graph = LiveModel(half=request.param)
    print(f'\n{"fp16" if request.param else "fp32"}: captured {len(graph.graphs)} lengths in {graph.warmup_s:.1f} s, '
          f'reserved {torch.cuda.memory_reserved() / 2 ** 20:.0f} MiB')
    eager = LiveModel(half=request.param, warmup=(), graphs=False)
    yield graph, eager
    del graph, eager
    gc.collect()
    torch.cuda.empty_cache()


def gap(a, b):
    """(same top-1 at every frame, max |difference| over the top-k and blank log-probabilities)."""
    same = np.array_equal(a.topk_ids[:, 0], b.topk_ids[:, 0])
    return same, float(max(np.abs(a.topk_logp - b.topk_logp).max(), np.abs(a.blank_logp - b.blank_logp).max()))


def report(name, rows):
    """rows: (T, same, dlogp). Prints the worst per 25-frame range; returns the lengths over the bar."""
    for lo in range(1, N + 1, 25):
        part = [r for r in rows if lo <= r[0] < lo + 25]
        if part:
            print(f'  {name} T {lo:3d}-{part[-1][0]:3d}: top-1 same {sum(r[1] for r in part)}/{len(part)}, '
                  f'max |dlogp| {max(r[2] for r in part):.2e}')
    return [(T, same, d) for T, same, d in rows if not same or d >= TOL]


def test_every_length_matches_the_cached_and_the_whole_eager_read(models, rois):
    """Every length, a frame at a time (the front-end cache finalises one frame per read) and as a fresh whole read."""
    graph, eager = models
    assert sorted(graph.graphs) == sorted(graph.fe_graphs) == list(range(1, N + 1))
    gc_, ec = FrontendCache(), FrontendCache()
    cached, whole = [], []
    for T in range(1, N + 1):
        cached.append((T, *gap(graph.read(list(rois[:T]), cache=gc_), eager.read(list(rois[:T]), cache=ec))))
        whole.append((T, *gap(graph.read(rois[:T]), eager.read(rois[:T]))))
    bad = report('vs cached eager', cached) + report('vs whole eager', whole)
    assert not bad, f'over the bar (T, top-1 same, max |dlogp|): {bad}'


def ticks(model, rois):
    sent = []
    r = Runner(model, sent.append, Segmenter(never_end=True, cap_s=N / FPS + 1), fe_cache=True)
    for i, x in enumerate(rois):
        r.on_crops([(Crop(t=100.0 + i / FPS, roi96=x, mouth_open=0.0, face_ok=True), 0.0)])
    r.finish()
    for t in sent:
        encode(t)
    return sent


def test_live_ticks_match_the_eager_ticks(models, rois):
    """The tick loop as live (Runner, front-end cache on, 100 ms schedule): every CamTick agrees."""
    graph, eager = models
    a, b = ticks(graph, rois), ticks(eager, rois)
    assert [len(t.frame_t) for t in a] == [len(t.frame_t) for t in b] and len(a) > 70
    rows = [(len(x.frame_t), *gap(x, y)) for x, y in zip(a, b)]
    bad = report('ticks', rows)
    assert not bad, f'over the bar (T, top-1 same, max |dlogp|): {bad}'
    assert [x.greedy for x in a] == [y.greedy for y in b]


def test_a_kept_encoder_output_survives_later_replays(models, rois):
    import torch
    graph, eager = models
    r = graph.read(rois[:50], keep_enc=True)
    kept = r.enc.clone()
    graph.read(rois[:60])
    graph.read(rois[::-1][:50].copy())
    assert torch.equal(r.enc, kept)
    ref = eager.read(rois[:50], keep_enc=True).enc
    assert r.enc.shape == ref.shape and r.enc.dtype == ref.dtype
    assert (r.enc.float() - ref.float()).abs().max().item() < 1e-2
    assert graph.read(rois[:50]).enc is None


def test_an_uncaptured_length_reads_eagerly_and_leaves_the_graphs_intact(models, rois):
    """T = 201 is not captured: it reads eagerly, also continuing a front-end cache the graphs filled; the graphs still
    read right after it."""
    graph, eager = models
    x = np.concatenate([rois, rois[:1]])
    assert len(x) not in graph.graphs
    same, d = gap(graph.read(x), eager.read(x))
    assert same and d < TOL
    gc_, ec = FrontendCache(), FrontendCache()
    for T in (100, 200, 201):
        same, d = gap(graph.read(list(x[:T]), cache=gc_), eager.read(list(x[:T]), cache=ec))
        assert same and d < TOL, (T, d)
    for T in (1, 100, 200):
        same, d = gap(graph.read(rois[:T]), eager.read(rois[:T]))
        assert same and d < TOL, (T, d)


def test_a_failed_capture_falls_back_to_eager_reads(monkeypatch, rois):
    def no_memory(self, lengths):
        self.graphs[1] = 'half captured'
        raise RuntimeError('CUDA out of memory. Tried to allocate 2.00 GiB\nmore detail')
    monkeypatch.setattr(LiveModel, '_capture', no_memory)
    model = LiveModel(half=True, warmup=(1, 50))
    assert model.graphs == {} and model.graphs_error == 'CUDA out of memory. Tried to allocate 2.00 GiB'
    eager = LiveModel(half=True, warmup=(1, 50), graphs=False)
    a, b = model.read(rois[:50]), eager.read(rois[:50])
    assert np.array_equal(a.topk_logp, b.topk_logp)
