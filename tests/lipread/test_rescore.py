"""lipread/rescore.py: the CTC prefix beam against exact CTC probabilities, the combined score, and the attention and
LM scores of a batch against each hypothesis scored alone.

    venv-live/bin/python -m pytest tests/lipread/test_rescore.py -q
"""
import itertools

import numpy as np
import pytest

from lipread.rescore import LOG0, attention_scores_multi, combined, ctc_prefix_beam, lm_scores

from .conftest import needs_cuda, needs_weights


def topk_of(logp):
    """(T, V) log-probs -> the read's (topk ids, topk logp, blank logp) with every id in the top-k."""
    ids = np.argsort(-logp, axis=1)
    return ids, np.take_along_axis(logp, ids, axis=1), logp[:, 0]


def exact_ctc(logp):
    """Every label sequence's CTC log-prob by summing all T-frame paths (blank 0)."""
    T, V = logp.shape
    out = {}
    for path in itertools.product(range(V), repeat=T):
        seq = tuple(c for k, c in enumerate(path) if c and (k == 0 or c != path[k - 1]))
        out[seq] = np.logaddexp(out.get(seq, -np.inf), sum(logp[t, c] for t, c in enumerate(path)))
    return out


def test_prefix_beam_with_a_wide_beam_is_exact_ctc():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4))
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    got = ctc_prefix_beam(*topk_of(logp), beam=1000, per_frame=4, blank_skip=0.0)
    got = [(seq, s) for seq, s in got if s > LOG0 / 2]                # the beam also keeps impossible prefixes
    want = exact_ctc(logp)
    assert [s for _, s in got] == sorted((s for _, s in got), reverse=True)
    assert len(got) == len(want)
    for seq, score in got:
        assert score == pytest.approx(want[seq], abs=1e-9)


def test_a_repeated_label_needs_a_blank_between():
    near0, low = np.log(0.98), np.log(0.01)
    logp = np.full((3, 3), low)
    logp[[0, 1, 2], [1, 0, 1]] = near0              # A blank A
    best = ctc_prefix_beam(*topk_of(logp), blank_skip=0.0)[0][0]
    assert best == (1, 1)
    logp[1] = low
    logp[1, 1] = near0                              # A A A
    assert ctc_prefix_beam(*topk_of(logp), blank_skip=0.0)[0][0] == (1,)


def test_frames_that_are_surely_blank_add_no_label():
    logp = np.full((3, 3), np.log(1e-5))
    logp[:, 0] = np.log(1 - 2e-5)
    assert [h for h, _ in ctc_prefix_beam(*topk_of(logp))] == [()]


def test_combined_weighs_ctc_attention_lm_and_length():
    hyps = [(1,), (1, 2)]
    ctc, att, lm = np.array([-1.0, -2.0]), np.array([-3.0, -2.0]), np.array([-1.0, -5.0])
    assert combined(hyps, ctc, att, None, 1.0).tolist() == [-1.0, -2.0]
    assert combined(hyps, ctc, att, None, 0.0).tolist() == [-3.0, -2.0]
    assert combined(hyps, ctc, att, None, 0.5, bonus=1.0).tolist() == [-1.0, 0.0]
    assert combined(hyps, ctc, att, lm, 0.5, w_lm=1.0).tolist() == [-3.0, -7.0]


@pytest.fixture(scope='module')
def reader():
    from lipflow.vsr import LipReader
    return LipReader(device='cuda', use_lm=False, personal=False)


class Model:
    def __init__(self, reader):
        self.reader = reader


@needs_cuda
@needs_weights
def test_a_batch_scores_each_hypothesis_as_alone(reader):
    import torch
    from lipread.final import load_lm
    m = reader.model
    dev = next(m.decoder.parameters()).device
    torch.manual_seed(0)
    encs = torch.randn(2, 30, m.decoder.decoders[0].size, device=dev)
    hyps = [[(), (40, 41, 42)], [(7,), (7, 8, 9, 10, 11), (300,)]]
    batch = attention_scores_multi(Model(reader), encs, hyps)
    alone = [attention_scores_multi(Model(reader), encs[b:b + 1], [[h]])[0][0] for b in range(2) for h in hyps[b]]
    assert np.allclose(np.concatenate(batch), alone, atol=1e-4)
    assert all(np.all(s < 0) for s in batch)
    lm = load_lm(len(reader.token_list)).to(dev)
    batch = lm_scores(lm, hyps, m.eos)
    alone = [lm_scores(lm, [[h]], m.eos)[0][0] for hs in hyps for h in hs]
    assert np.allclose(np.concatenate(batch), alone, atol=1e-4)
