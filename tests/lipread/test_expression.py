"""Expression network path of the emotion tag, with a fake network so no model file is needed."""
import os
import time

import numpy as np
import pytest

from bus import EMOTIONS, Emotion, decode, encode, validate
from lipread import emotion_fit
from lipread.emotion import NetTagger, EmotionWorker, RESTING
from lipread.expression import (CLASSES, MODEL_PATH, MODEL_SHA256, N_FEATURES, ExpressionNet, Head, face_box,
                                generic_probs)

from .test_emotion import tick_with

PTS = np.array([[100, 100], [300, 120], [200, 330]], np.float32)
IMG = np.zeros((720, 1280, 3), np.uint8)
BLEND = np.zeros(52, np.float32)


class FakeNet:
    """logits() returns whatever expression is set: a big logit on that AffectNet class."""

    def __init__(self):
        self.expr, self.calls = 'Neutral', 0

    def logits(self, image, pts):
        self.calls += 1
        z = np.zeros(len(CLASSES), np.float32)
        z[CLASSES.index(self.expr)] = 6.0
        return z


def toy_head(margin=0.5, hold_s=0.5, tau_s=0.3):
    """Happiness -> happy, Anger -> sad (Austin's frowns read as Anger), Surprise -> surprised."""
    labels = ('neutral', 'happy', 'sad', 'surprised')
    coef = np.zeros((4, N_FEATURES), np.float32)
    for i, c in enumerate(('Neutral', 'Happiness', 'Anger', 'Surprise')):
        coef[i, CLASSES.index(c)] = 3.0
    return Head(labels, np.zeros(N_FEATURES), np.ones(N_FEATURES), coef, np.array([1.0, 0, 0, 0]), margin, hold_s,
                tau_s, 'toy')


def run(tagger, net, script, fps=15.0, t0=0.0):
    """script: [(seconds, AffectNet class or None for no face)] -> [(t, Emotion)]."""
    out, t = [], t0
    for secs, expr in script:
        for _ in range(int(round(secs * fps))):
            if expr is not None:
                net.expr = expr
            out.append((t, tagger.push(t, IMG, None if expr is None else (t, PTS, BLEND))))
            t += 1 / fps
    return out


def labels_after(out, t):
    return {e.label for ti, e in out if ti >= t}


def test_head_roundtrip_and_probs(tmp_path):
    h = toy_head()
    h.save(tmp_path / 'h.npz')
    g = Head.load(tmp_path / 'h.npz')
    assert g.labels == h.labels and (g.margin, g.hold_s, g.tau_s) == (0.5, 0.5, 0.3) and g.info == 'toy'
    p = g.predict(np.random.default_rng(0).normal(size=(5, N_FEATURES)))
    assert p.shape == (5, 4) and np.allclose(p.sum(1), 1)
    with pytest.raises(ValueError):
        Head(('happy',), np.zeros(N_FEATURES), np.ones(N_FEATURES), np.zeros((1, N_FEATURES)), [0], .5, .5, .3)


def test_face_box_square_clipped_and_rejects_tiny():
    x0, y0, x1, y1 = face_box(PTS, 1280, 720)
    assert abs((x1 - x0) - (y1 - y0)) <= 1 and x0 >= 0 and y0 >= 0
    assert face_box(np.array([[0, 0], [10, 10]], np.float32), 1280, 720) is None
    assert face_box(np.array([[1270, 700], [1400, 800]], np.float32), 1280, 720) is None


def test_generic_probs_order():
    z = np.zeros(len(CLASSES), np.float32)
    z[CLASSES.index('Happiness')] = 5
    p = generic_probs(z)
    assert p.argmax() == 1 and np.isclose(p.sum(), 1)       # neutral, happy, sad, angry, surprised


def test_baseline_then_switch_with_hold():
    net = FakeNet()
    tg = NetTagger(net, toy_head())
    out = run(tg, net, [(3.2, 'Neutral'), (1.0, 'Neutral'), (2.0, 'Happiness')])
    assert labels_after(out[:60], 0) == {'neutral'}
    happy_t = next(t for t, e in out if e.label == 'happy')
    assert 4.2 + 0.5 <= happy_t <= 4.2 + 1.0                # hold 0.5 s after the smoothed prob crosses the margin
    assert out[-1][1].label == 'happy'


def test_frown_reads_as_sad_and_surprise():
    net = FakeNet()
    tg = NetTagger(net, toy_head())
    out = run(tg, net, [(3.2, 'Neutral'), (2.0, 'Anger'), (2.0, 'Neutral'), (2.0, 'Surprise')])
    seq = [e.label for _, e in out]
    assert 'sad' in seq and seq[-1] == 'surprised' and 'angry' not in seq


def test_baseline_removes_a_resting_bias():
    """A face that always reads slightly 'happy' at rest stays neutral: the baseline subtracts it."""
    net = FakeNet()
    tg = NetTagger(net, toy_head())
    out = run(tg, net, [(3.2, 'Happiness'), (3.0, 'Happiness')])
    assert labels_after(out, 3.3) == {'neutral'}


def test_brief_expression_is_ignored():
    net = FakeNet()
    tg = NetTagger(net, toy_head(hold_s=0.5))
    out = run(tg, net, [(3.2, 'Neutral'), (0.2, 'Surprise'), (2.0, 'Neutral')])
    assert labels_after(out, 0) == {'neutral'}


def test_no_face_holds_then_decays_to_neutral():
    net = FakeNet()
    tg = NetTagger(net, toy_head())
    out = run(tg, net, [(3.2, 'Neutral'), (2.0, 'Happiness'), (0.5, None), (6.0, None)])
    no_face = [(t, e.label) for t, e in out if t >= 5.2]
    assert no_face[0][1] == 'happy' and no_face[-1][1] == 'neutral'


def test_generic_mode_without_head_and_reset():
    net = FakeNet()
    tg = NetTagger(net, None)
    out = run(tg, net, [(3.2, 'Neutral'), (3.0, 'Anger')])
    assert out[-1][1].label == 'angry'
    tg.reset_baseline()
    assert tg.push(10.0, IMG, (10.0, PTS, BLEND)).label == 'neutral' and tg.base is None


def test_every_net_output_passes_bus_validate():
    net = FakeNet()
    for head in (toy_head(), None):
        tg = NetTagger(net, head)
        out = run(tg, net, [(3.2, 'Neutral'), (1.5, 'Happiness'), (1.5, 'Anger'), (1, None), (1.5, 'Surprise')])
        for _, e in out:
            assert e.label in EMOTIONS and set(e.scores) == set(EMOTIONS) and e.scores[e.label] == e.confidence
            validate(*decode(encode(tick_with(e))))


def test_worker_uses_the_net_off_thread_and_falls_back():
    net = FakeNet()
    w = EmotionWorker(net=net, head=toy_head())
    try:
        assert w.current() == RESTING
        t = 0.0
        for expr, secs in (('Neutral', 3.5), ('Happiness', 2.0)):
            net.expr = expr
            for _ in range(int(secs * 30)):
                w.submit(t, IMG, (t, PTS, BLEND))       # never blocks: the net runs on its own thread
                t += 1 / 30
                time.sleep(1 / 30 / 4)
        deadline = time.time() + 5
        while w.current().label != 'happy' and time.time() < deadline:
            w.submit(t, IMG, (t, PTS, BLEND))
            t += 1 / 30
            time.sleep(0.02)
        assert w.current().label == 'happy'
        assert net.calls < t * EmotionWorker.NET_HZ * 1.5     # at most NET_HZ frames a second reach the net
        net.logits = lambda *a: (_ for _ in ()).throw(RuntimeError('boom'))
        for _ in range(40):
            w.submit(t, IMG, (t, PTS, BLEND))
            t += 1 / 30
            time.sleep(0.02)
        assert w.current() == RESTING                    # repeated failures: neutral, not a jump to the rules
        w.submit(t, IMG, 'garbage')                      # never raises
    finally:
        w.close()


def test_worker_without_model_is_the_rule_tagger():
    w = EmotionWorker(net=None)
    assert w.net_tagger is None and w.current() == RESTING
    w.close()


def clip(cid, truth, labels, rest=False):
    return (cid, truth, rest, np.arange(len(labels)) / 25.0, None), list(labels)


def test_fit_scoring_tells_carry_over_from_new_flicker():
    rows = [clip('a', 'happy', ['happy'] * 10),
            clip('b', 'neutral', ['happy'] * 3 + ['neutral'] * 7),                 # carried over from 'a'
            clip('c', 'neutral', ['neutral'] * 5 + ['sad'] * 2 + ['neutral'] * 3),  # new wrong label
            clip('d', 'neutral', ['neutral'] * 10),
            clip('e', 'surprised', ['neutral'] * 4 + ['surprised'] * 6),
            clip('f', 'neutral', ['neutral'] * 10, rest=True)]
    clips, res = zip(*rows)
    acc, fresh, carry, rest = emotion_fit.score(list(clips), list(res))
    assert acc['happy'] == 1 and acc['surprised'] == 1 and acc['neutral'] == 1 and np.isnan(acc['sad'])
    assert fresh == pytest.approx(1 / 3) and carry == pytest.approx(1 / 3) and rest == 1


@pytest.mark.skipif(not ExpressionNet.available(), reason='lipread/setup.sh has not fetched the expression model')
def test_real_model_runs_on_a_face_box():
    net = ExpressionNet()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    z = net.logits(img, PTS)
    assert z.shape == (len(CLASSES),) and np.isfinite(z).all()
    t0 = time.perf_counter()
    for _ in range(10):
        net.logits(img, PTS)
    print(f'expression net: {(time.perf_counter() - t0) * 100:.1f} ms per face')
    assert MODEL_PATH.endswith('enet_b0_8_best_vgaf.onnx')


def test_worker_is_neutral_until_the_net_answers_and_when_stale():
    w = EmotionWorker(net=FakeNet(), head=toy_head())
    try:
        assert w.current() == RESTING
        happy = Emotion('happy', 1.0, {e: float(e == 'happy') for e in EMOTIONS})
        w._net = (happy, time.monotonic())
        assert w.current() == happy
        w._net = (happy, time.monotonic() - EmotionWorker.NET_STALE_S - 1)
        assert w.current() == RESTING                    # stale answer: neutral
    finally:
        w.close()


def test_setup_pins_the_same_checksum():
    setup = open(os.path.join(os.path.dirname(__file__), '..', '..', 'lipread', 'setup.sh')).read()
    assert f'HSE_SHA256={MODEL_SHA256}' in setup


def test_available_rejects_a_wrong_file(tmp_path):
    f = tmp_path / 'm.onnx'
    f.write_bytes(b'not the model')
    assert not ExpressionNet.available(str(f))
    pytest.importorskip('onnxruntime')
    assert ExpressionNet.available(str(f), sha256=None)                 # present, checksum not asked for
    assert not ExpressionNet.available(str(tmp_path / 'missing.onnx'), sha256=None)


def test_fit_head_survives_a_fold_without_a_label():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, N_FEATURES))
    y = np.array(['neutral'] * 30 + ['happy'] * 30)
    X[30:, 0] += 4
    h = emotion_fit.fit_head(X, y, C=1.0, cols=tuple(range(N_FEATURES)))
    p = h.predict(X)
    assert h.labels == emotion_fit.LABELS and p[:, 2].max() < 1e-6 and p[:, 3].max() < 1e-6
    assert (p[30:, 1] > p[30:, 0]).mean() > 0.9 and (p[:30, 0] > p[:30, 1]).mean() > 0.9


def test_fit_replay_is_the_live_tagger():
    """emotion_fit.replay drives NetTagger.step: a held smile switches after the hold, a face-less gap decays."""
    t = np.arange(0, 12, 1 / 30)
    p = np.tile(np.array([0.97, 0.01, 0.01, 0.01], np.float32), (len(t), 1))
    p[(t >= 4) & (t < 6)] = [0.02, 0.96, 0.01, 0.01]
    p[(t >= 6) & (t < 9)] = np.nan
    clips = [('x', 'happy', False, t, None)]
    labs = emotion_fit.replay(clips, [p], t_start=3.0, margin=0.5, hold_s=0.5, tau_s=0.3)[0]
    first_happy = t[labs.index('happy')]
    assert 4.5 <= first_happy <= 5.2 and labs[-1] == 'neutral'


def test_sync_worker_is_deterministic_and_frame_time_throttled():
    def go():
        net = FakeNet()
        w = EmotionWorker(net=net, head=toy_head(), sync=True)
        out = []
        for i in range(int(8 * 30)):
            t = i / 30
            net.expr = 'Happiness' if 4 <= t < 6 else 'Neutral'
            w.submit(t, IMG, (t, PTS, BLEND))
            out.append(w.current().label)
        w.close()
        return out, net.calls
    (a, calls), (b, _) = go(), go()
    assert a == b and 'happy' in a
    assert calls == pytest.approx(8 * EmotionWorker.NET_HZ, abs=2)      # NET_HZ per second of frame time


def test_hold_restarts_after_the_face_is_lost():
    net = FakeNet()
    tg = NetTagger(net, toy_head(hold_s=0.5))
    run(tg, net, [(3.2, 'Neutral')])
    out = run(tg, net, [(0.3, 'Happiness'), (0.4, None), (0.3, 'Happiness')], t0=3.2)
    assert {e.label for _, e in out} == {'neutral'}       # 0.6 s of smile in total, but never 0.5 s in one go


def test_reset_clears_the_previous_persons_emotion():
    net = FakeNet()
    w = EmotionWorker(net=net, head=toy_head(), sync=True)
    for i in range(int(6 * 30)):
        t = i / 30
        net.expr = 'Happiness' if t >= 3.5 else 'Neutral'
        w.submit(t, IMG, (t, PTS, BLEND))
    assert w.current().label == 'happy'
    w.reset_baseline()
    assert w.current() == RESTING
    w.close()


def test_close_joins_the_net_thread():
    w = EmotionWorker(net=FakeNet(), head=toy_head())
    w.close()
    assert not w._thread.is_alive()


def test_binary_fold_matches_sklearn_probabilities():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, N_FEATURES))
    y = np.array(['neutral'] * 40 + ['happy'] * 40)
    X[40:, 3] += 2
    h = emotion_fit.fit_head(X, y, C=1.0, cols=tuple(range(N_FEATURES)))
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(sc.transform(X), y)
    want = m.predict_proba(sc.transform(X))[:, list(m.classes_).index('happy')]
    assert np.allclose(h.predict(X)[:, 1], want, atol=1e-4)


# --- the running baseline and the logits-only head (what carries across sessions) -------------------------------
def feats_of(expr, offset=0.0):
    """60 features: the FakeNet logits for expr (an AffectNet class), plus a constant offset (light, posture)."""
    z = np.zeros(N_FEATURES, np.float32)
    z[CLASSES.index(expr)] = 6.0
    return z + offset


def test_an_expression_at_the_start_is_not_taken_for_neutral():
    """Session 2 opened on raised brows; the first seconds alone made that "neutral" for the whole call."""
    from lipread.emotion import RunningBaseline
    rb, t, base = RunningBaseline(), 0.0, None
    for _ in range(int(6 * 25)):                       # the call opens on a 6 s smile
        base = rb.push(t, feats_of('Happiness'))
        t += 0.04
    assert base is None and not rb.buf                 # the network alone reads it as a smile: kept out
    for _ in range(int(4 * 25)):                       # then a neutral face
        base = rb.push(t, feats_of('Neutral'))
        t += 0.04
    assert np.array_equal(base, feats_of('Neutral'))   # the median is the neutral face, not the smile


def test_running_baseline_follows_a_slow_change():
    from lipread.emotion import RunningBaseline
    rb, t = RunningBaseline(), 0.0
    for offset in (0.0, 1.0):                          # light changes: the whole face shifts
        for _ in range(int(70 * 25)):
            base = rb.push(t, feats_of('Neutral', offset))
            t += 0.04
        assert base[20] == offset


def test_running_baseline_after_a_long_gap_waits_for_new_frames():
    from lipread.emotion import RunningBaseline
    rb, t = RunningBaseline(), 0.0
    for _ in range(100):
        old = rb.push(t, feats_of('Neutral'))
        t += 0.04
    t += 200.0                                         # gone for longer than the 60 s window
    for _ in range(25):                                # 1 s back: one frame is not a baseline yet
        b = rb.push(t, feats_of('Neutral', 2.0))
        t += 0.04
    assert np.array_equal(b, old)
    for _ in range(3 * 25):
        b = rb.push(t, feats_of('Neutral', 2.0))
        t += 0.04
    assert np.isfinite(b).all() and b[20] == 2.0
    t += 200.0
    for _ in range(5 * 25):                            # back on a smile: it never replaces the neutral face
        b = rb.push(t, feats_of('Happiness'))
        t += 0.04
    assert b[20] == 2.0 and not rb.buf


def test_net_tagger_opening_on_a_smile_then_reads_neutral_and_the_next_smile():
    net = FakeNet()
    tg = NetTagger(net, toy_head())
    out = run(tg, net, [(6.0, 'Happiness'), (8.0, 'Neutral'), (3.0, 'Happiness'), (3.0, 'Neutral')])
    assert {e.label for ti, e in out if 7.0 <= ti < 14.0} == {'neutral'}
    assert 'happy' in labels_after([(ti, e) for ti, e in out if ti < 17.0], 14.0)
    assert tg.base is tg.baseline.base is not None


def test_running_baseline_ignores_no_face_frames_and_resets():
    from lipread.emotion import RunningBaseline
    rb = RunningBaseline()
    assert rb.push(0.0, None) is None and rb.push(5.0, None) is None
    for k in range(100):
        rb.push(10.0 + k * 0.04, np.ones(N_FEATURES, np.float32))
    rb.reset()
    assert rb.base is None and not rb.buf


def test_head_reads_only_its_columns_and_saves_them(tmp_path):
    from lipread.expression import LOGITS
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, N_FEATURES))
    y = np.array(['neutral', 'happy', 'sad', 'surprised'] * 20)
    X[y == 'happy', 4] += 3
    h = emotion_fit.fit_head(X, y)                     # defaults: the 8 logits, C 0.01
    assert list(h.cols) == list(LOGITS) and h.coef.shape == (4, 8)
    noisy = X.copy()
    noisy[:, 8:] = rng.normal(size=(80, N_FEATURES - 8)) * 100      # blendshapes are ignored
    assert np.allclose(h.predict(X), h.predict(noisy))
    h.save(tmp_path / 'h.npz')
    g = Head.load(tmp_path / 'h.npz')
    assert list(g.cols) == list(LOGITS) and np.allclose(g.predict(X), h.predict(X))


def test_head_rejects_a_mean_or_scale_that_does_not_match_its_columns():
    with pytest.raises(ValueError):
        Head(('neutral', 'happy'), np.zeros(N_FEATURES), np.ones(8), np.zeros((2, 8)), np.zeros(2), 0.5, 0.5, 0.3,
             cols=range(8))


def test_an_old_head_file_without_columns_reads_all_features(tmp_path, caplog):
    np.savez(tmp_path / 'old.npz', labels=np.array(['neutral', 'happy']), mean=np.zeros(N_FEATURES),
             scale=np.ones(N_FEATURES), coef=np.zeros((2, N_FEATURES)), intercept=np.zeros(2),
             smoothing=np.array([0.5, 0.5, 0.3]), info=np.array(''))
    assert len(Head.load(tmp_path / 'old.npz').cols) == N_FEATURES
    assert 'refit' in caplog.text


def test_session_median_without_neutral_frames_is_none():
    smile = np.tile(feats_of('Happiness'), (10, 1))
    clip = ('c', 'happy', False, np.arange(10) * 0.04, smile)
    assert emotion_fit.session_median([clip]) is None
    calm = np.tile(feats_of('Neutral', 1.0), (10, 1))
    assert np.array_equal(emotion_fit.session_median([clip, ('d', 'neutral', False, np.arange(10) * 0.04, calm)]),
                          feats_of('Neutral', 1.0))
