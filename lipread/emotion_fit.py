"""Fit the expression head on the speaker's recorded sessions and print held-out scores.

    venv-live/bin/python -m lipread.emotion_fit                                         # writes data/emotion/head.npz
    venv-live/bin/python -m lipread.emotion_fit --sessions session_01 --test session_02

Defaults (--features logits, C 0.01) are what carried across Austin's sessions 1 to 3: the blendshapes did not.
Training subtracts each session's median face, scoring replays the live RunningBaseline.
"""
import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter

import numpy as np

import lipread
from lipread.emotion import BASELINE_S
from lipread.expression import HEAD_PATH, LOGITS, MODEL_SHA256, N_FEATURES, Head

LABELS = ('neutral', 'happy', 'sad', 'surprised')
TRUTH = {'smile': 'happy', 'frown': 'sad', 'surprised': 'surprised'}
REST = {'Smile': 'happy', 'Raise your eyebrows': 'surprised'}
PROCESS = (1280, 720)
GRID = list(itertools.product((0.4, 0.5, 0.6, 0.7), (0.5, 0.8), (0.3, 0.5, 1.0)))   # margin, hold_s, tau_s
FEATURE_DIR = os.path.join(lipread.WORK, 'data', 'emotion')
CACHE_VERSION = 3
C_DEFAULT = 0.01     # strong: 10 clips per expression per session; best leave-one-session-out


def truth(row):
    if row.get('cue'):
        return REST.get(row['cue'], 'neutral')
    return TRUTH.get(row.get('expression') or 'neutral', 'neutral')


def session_rows(root, session):
    with open(os.path.join(root, session, 'manifest.jsonl')) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r['clip_t0'])


def extract(root, session, cache=FEATURE_DIR):
    """Per clip (id, truth, is_rest, t (T,), feats (T, 60) with NaN rows where no face), cached per session."""
    path = os.path.join(cache, f'features_{session}.npz')
    rows = session_rows(root, session)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        same = (int(z['version']) if 'version' in z else 0) == CACHE_VERSION and str(z['model']) == MODEL_SHA256 \
            and tuple(z['process']) == PROCESS and list(z['ids']) == [r['id'] for r in rows]
        if same:
            return [(r['id'], truth(r), bool(r.get('cue')), z[f't_{r["id"]}'], z[f'f_{r["id"]}']) for r in rows]
    import cv2
    from lipread.crop import BlendTracker
    from lipread.run import fit
    from lipread.types import Frame
    from lipread.expression import ExpressionNet
    net = ExpressionNet(threads=os.cpu_count() or 2)
    # one tracker per session, as live
    tracker = BlendTracker(os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task'))
    out, store, t0 = [], {}, time.time()
    for n, r in enumerate(rows):
        ts = np.load(os.path.join(root, session, r['video_ts'])).astype(np.float64).reshape(-1)
        cap = cv2.VideoCapture(os.path.join(root, session, r['video']))
        feats = []
        while len(feats) < len(ts):
            ok, img = cap.read()
            if not ok:
                break
            img = fit(Frame(t=ts[len(feats)], image=img, index=len(feats)), PROCESS).image    # as the crop thread
            obs = tracker.detect(img, int(round(ts[len(feats)] * 1000)))
            lg = None if obs is None or obs.blend is None else net.logits(img, obs.pts)
            feats.append(np.full(N_FEATURES, np.nan, np.float32) if lg is None else np.concatenate([lg, obs.blend]))
        cap.release()
        t, f = ts[:len(feats)], np.array(feats, np.float32).reshape(-1, N_FEATURES)
        out.append((r['id'], truth(r), bool(r.get('cue')), t, f))
        store[f't_{r["id"]}'], store[f'f_{r["id"]}'] = t, f
        if n % 25 == 24:
            print(f'  {session}: {n + 1}/{len(rows)} clips, {time.time() - t0:.0f} s', flush=True)
    tracker.close()
    os.makedirs(cache, exist_ok=True)
    np.savez(path, version=CACHE_VERSION, model=MODEL_SHA256, process=np.array(PROCESS),
             ids=np.array([r['id'] for r in rows]), **store)
    return out


def session_median(clips):
    """The median of the sentence frames RunningBaseline would keep (no labels read); None when there are none."""
    from lipread.emotion import RunningBaseline
    expressive = RunningBaseline().expressive
    f = [x for c in clips if not c[2] for x in c[4] if not np.isnan(x).any() and not expressive(x)]
    return np.median(f, 0) if f else None


def running_bases(clips):
    """Per clip, per frame: the live tagger's baseline at that moment, or None while it warms up."""
    from lipread.emotion import RunningBaseline
    rb, out = RunningBaseline(), []
    for _, _, _, t, f in clips:
        out.append([rb.push(ti, None if np.isnan(x[0]) else x) for ti, x in zip(t, f)])
    return out


def fit_head(X, y, C=C_DEFAULT, cols=LOGITS):
    """X: (N, N_FEATURES) baseline-subtracted frames; the head reads `cols` of them."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = X[:, list(cols)]
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=4000, class_weight='balanced').fit(sc.transform(X), y)
    classes = list(m.classes_)
    coef = np.zeros((len(LABELS), X.shape[1]), np.float32)
    intercept = np.full(len(LABELS), -20.0, np.float32)     # a label this fold never saw: never predicted
    for i, c in enumerate(LABELS):
        if c in classes:
            j = classes.index(c)
            if len(classes) > 2:
                coef[i], intercept[i] = m.coef_[j], m.intercept_[j]
            else:            # binary: sigmoid(w x + b) == softmax over +-(w x + b) / 2
                sign = 0.5 if j == 1 else -0.5
                coef[i], intercept[i] = sign * m.coef_[0], sign * m.intercept_[0]
    return Head(LABELS, sc.mean_, sc.scale_, coef, intercept, 0.5, 0.5, 0.3, cols=cols)


def training_frames(clips, base):
    """Frames after the first third of each clip, once the expression is held, minus the session's median face."""
    X, y = [], []
    for _, lab, _, t, f in clips:
        f = f[len(f) // 3:]
        f = f[~np.isnan(f).any(1)] - base
        X.append(f)
        y += [lab] * len(f)
    return np.vstack(X), np.array(y)


def replay(clips, probs, t_start, margin, hold_s, tau_s, hz=None):
    """Published labels per frame from the live NetTagger.step, fed at EmotionWorker.NET_HZ."""
    from lipread.emotion import EmotionWorker, NetTagger
    hz = hz or EmotionWorker.NET_HZ
    head = Head(LABELS, np.zeros(N_FEATURES), np.ones(N_FEATURES), np.zeros((len(LABELS), N_FEATURES)),
                np.zeros(len(LABELS)), margin, hold_s, tau_s)
    tagger = NetTagger(net=None, head=head)
    label, last, res = 'neutral', None, []
    for (_, _, _, t, _), p in zip(clips, probs):
        labs = []
        for ti, pi in zip(t, p):
            if ti > t_start and (last is None or ti - last >= 1.0 / hz - 1e-9):
                last = ti
                label = tagger.step(ti, None if np.isnan(pi[0]) else pi).label
            labs.append(label)
        res.append(labs)
    return res


def score(clips, res):
    """{label: accuracy on sentences}, fresh flicker share, carry-over share, rest-clip accuracy."""
    acc, fresh, carry, n_neutral, prev = {}, 0, 0, 0, 'neutral'
    for lab in LABELS:
        got = [Counter(r[len(r) // 2:]).most_common(1)[0][0] == lab for (_, tr, rest, _, _), r in zip(clips, res)
               if tr == lab and not rest and r]
        acc[lab] = float(np.mean(got)) if got else float('nan')
    for (_, tr, rest, _, _), r in zip(clips, res):
        if tr == 'neutral' and not rest and r:
            n_neutral += 1
            if any(x != 'neutral' for x in r):
                # carry-over: starts on the previous expression, then stays neutral
                held = r[0] != 'neutral' and prev != 'neutral' and 'neutral' in r
                if held and all(x == 'neutral' for x in r[r.index('neutral'):]):
                    carry += 1
                else:
                    fresh += 1
        prev = tr
    rest_ok = [Counter(r[len(r) // 2:]).most_common(1)[0][0] == tr for (_, tr, rest, _, _), r in zip(clips, res)
               if rest and r]
    return acc, fresh / max(1, n_neutral), carry / max(1, n_neutral), float(np.mean(rest_ok)) if rest_ok else 0.0


def held_out_probs(sessions, C=C_DEFAULT, cols=LOGITS):
    """Per session, per clip: (T, 4) probabilities from a head that never saw it (leave-one-session-out, or 5-fold
    by clip with one session), on the running baseline."""
    probs = [[None] * len(s['clips']) for s in sessions]
    if len(sessions) > 1:
        for k, s in enumerate(sessions):
            rest = [o for o in sessions if o is not s]
            X, Y = zip(*(training_frames(o['clips'], o['median']) for o in rest))
            head = fit_head(np.vstack(X), np.concatenate(Y), C, cols)
            probs[k] = [predict(head, c[4], b) for c, b in zip(s['clips'], s['bases'])]
        return probs
    from sklearn.model_selection import StratifiedGroupKFold
    s = sessions[0]
    y = np.array([c[1] for c in s['clips']])
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, groups=np.arange(len(y))):
        X, Y = training_frames([s['clips'][i] for i in train], s['median'])
        head = fit_head(X, Y, C, cols)
        for i in test:
            probs[0][i] = predict(head, s['clips'][i][4], s['bases'][i])
    return probs


def predict(head, feats, bases):
    """(T, 4) probabilities, NaN where there is no face or the running baseline is still warming up."""
    p = np.full((len(feats), len(LABELS)), np.nan, np.float32)
    ok = np.array([b is not None for b in bases], bool) & ~np.isnan(feats).any(1)
    if ok.any():
        p[ok] = head.predict(feats[ok] - np.stack([b for b, k in zip(bases, ok) if k]))
    return p


def load_session(root, name):
    clips = extract(root, name)
    bases = running_bases(clips)
    t_start = next((t for c, bs in zip(clips, bases) for t, b in zip(c[3], bs) if b is not None), None)
    if t_start is None:
        raise SystemExit(f'{name}: not {BASELINE_S:.0f} s of face anywhere')
    median = session_median(clips)
    if median is None:
        raise SystemExit(f'{name}: no neutral-looking face frames in its sentences')
    return dict(name=name, clips=clips, bases=bases, t_start=t_start, median=median)


def summary(acc, fresh, carry, rest):
    return (' '.join(f'{k} {v:.0%}' for k, v in acc.items()) +
            f'; neutral sentences: {fresh:.0%} new wrong label, {carry:.0%} carry-over; rest clips {rest:.0%}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=os.path.join(lipread.WORK, 'recordings'))
    ap.add_argument('--sessions', help='comma list (default: every session_0* under --root except --test)')
    ap.add_argument('--test', help='a session scored with the head fitted on --sessions, never fitted on')
    ap.add_argument('--out', default=HEAD_PATH)
    ap.add_argument('--features', choices=('logits', 'all'), default='logits',
                    help='logits: the HSEmotion scores alone (carry across sessions); all: plus the 52 blendshapes')
    ap.add_argument('--C', type=float, default=C_DEFAULT, help='logistic-regression regularisation (smaller: stronger)')
    args = ap.parse_args(argv)
    names = args.sessions.split(',') if args.sessions else sorted(
        d for d in os.listdir(args.root) if d.startswith('session_') and d != args.test and
        os.path.exists(os.path.join(args.root, d, 'manifest.jsonl')))
    if not names:
        sys.exit(f'no sessions with manifest.jsonl under {args.root}')
    sessions = []
    for name in names:
        s = load_session(args.root, name)
        sessions.append(s)
        print(f'{name}: {len(s["clips"])} clips, {Counter(c[1] for c in s["clips"] if not c[2])} sentences')
    missing = set(LABELS) - {c[1] for s in sessions for c in s['clips']}
    if missing:
        sys.exit(f'no {sorted(missing)} clips in {names}; the head needs every label in {LABELS}')
    cols = LOGITS if args.features == 'logits' else tuple(range(N_FEATURES))
    probs = held_out_probs(sessions, args.C, cols)
    best = None
    for margin, hold_s, tau_s in GRID:
        scored = [score(s['clips'], replay(s['clips'], p, s['t_start'], margin, hold_s, tau_s))
                  for s, p in zip(sessions, probs)]
        acc = {k: float(np.nanmean([sc[0][k] for sc in scored])) if any(not np.isnan(sc[0][k]) for sc in scored)
               else float('nan') for k in LABELS}
        fresh, carry, rest = (float(np.mean([sc[i] for sc in scored])) for i in (1, 2, 3))
        objective = float(np.nanmean(list(acc.values()))) - 2 * fresh    # labels with no sentences are left out
        if best is None or objective > best[0]:
            best = (objective, (margin, hold_s, tau_s), acc, fresh, carry, rest, scored)
    _, (margin, hold_s, tau_s), acc, fresh, carry, rest, scored = best
    how = 'each session scored by a head fitted on the others' if len(sessions) > 1 else 'held-out clips, 5-fold'
    held = f'{how} ({",".join(names)}): {summary(acc, fresh, carry, rest)}'
    print(f'smoothing margin {margin}, hold {hold_s} s, tau {tau_s} s\n{held}')
    for s, sc in zip(sessions, scored):
        print(f'  {s["name"]}: {summary(*sc)}')
    X, Y = zip(*(training_frames(s['clips'], s['median']) for s in sessions))
    head = fit_head(np.vstack(X), np.concatenate(Y), args.C, cols)
    head.margin, head.hold_s, head.tau_s = margin, hold_s, tau_s
    info = held
    if args.test:
        t = load_session(args.root, args.test)
        res = replay(t['clips'], [predict(head, c[4], b) for c, b in zip(t['clips'], t['bases'])], t['t_start'],
                     margin, hold_s, tau_s)
        test = f'{args.test} (never fitted on): {summary(*score(t["clips"], res))}'
        print(test)
        info += f' | {test}'
    head.info = info
    head.save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
