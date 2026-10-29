"""Run the fusion process offline on recorded sessions, fit the gate and the lock model, and test.

The camera side comes from a bus log of the camera reader replaying the session on the recorder's clock; the EMG side
from the session's own EMG files, sent as emg.raw messages in time order. Both go through fusion/core.py exactly as
live, so the numbers are the live system's on the recordings.

    # 1. the camera reads each session into a bus log (venv-live; recorder clock, so its times match the EMG's)
    python -m buslog.record --out logs/cam_s1 &
    venv-live/bin/python -m lipread.run --replay-session session_01 --realtime --recorder-clock --face-adapter A --no-face
    # 2. every pass's judge scores and lock features, words never locked (venv-svr)
    python -m fusion.offline passes session_01 logs/cam_s1 --svr checkpoints/S/model.pt --out data/fusion/passes_s1.pkl
    # 3. fit the gate on the final passes, re-run step 2 with it, then fit the lock model and choose tau0
    python -m fusion.offline fit-gate data/fusion/passes_s1.pkl data/fusion/passes_s2.pkl
    python -m fusion.offline fit-lock data/fusion/passes_s1_g.pkl data/fusion/passes_s2_g.pkl
    # 4. the test: the whole pipeline with locking, on a session nothing was fitted on
    python -m fusion.offline test session_03 logs/cam_s3 --svr checkpoints/W_s12_svr1_4ch/model.pt

Fitting data must be out of sample, as session 3 will be: each training session is read by a camera face adapter
and an SVR-1 fine-tuned on the other session only (cross-fitting); otherwise both readers look far surer on their
own training sentences than they will be live, and the gate and lock model would trust them too much.
"""
import argparse
import heapq
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PARAMS = os.path.join(WORK, 'data', 'fusion', 'params.json')
RECORDINGS = os.path.join(WORK, 'recordings')
CAM_LAG_S = 0.04          # capture -> crop thread -> tick loop, on top of the tick's own timing_ms.total
EMG_CHUNK = 10            # samples per emg.raw message (rig/publish.py: 100 messages/s)


# ------------------------------------------------------------------ data
def session_rows(session):
    folder = os.path.join(RECORDINGS, session)
    rows = []
    with open(os.path.join(folder, 'manifest.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            for k in ('emg', 'emg_ts', 'video_ts'):
                r[k] = os.path.join(folder, r[k])
            rows.append(r)
    return rows


def clip_windows(rows):
    """(start, end, row) on the recorder's clock, from each clip's frame times."""
    out = []
    for r in rows:
        if os.path.exists(r['video_ts']):
            ts = np.load(r['video_ts'])
            out.append((float(ts[0]), float(ts[-1]), r))
    return sorted(out, key=lambda x: x[0])


def clip_of(windows, t0, t1):
    """The clip that overlaps [t0, t1] the most."""
    best, ov = None, 0.0
    for a, b, r in windows:
        o = min(b, t1) - max(a, t0)
        if o > ov:
            best, ov = r, o
    return best


def camera_events(log):
    """(time, topic, msg) for every cam.tick and cam.final of a bus log, on the recorder's clock."""
    import buslog
    ticks, last = [], {}
    for t_recv, topic, msg in buslog.read(log, topics=['cam.tick', 'cam.final']):
        if topic == 'cam.tick':
            t = float(msg['frame_t'][-1]) + float(msg['timing_ms'].get('total', 0.0)) / 1000 + CAM_LAG_S
            last[int(msg['seg_id'])] = (t, t_recv)
            ticks.append((t, topic, msg))
        elif int(msg['seg_id']) in last:
            t_tick, recv_tick = last[int(msg['seg_id'])]
            ticks.append((t_tick + max(0.0, t_recv - recv_tick), topic, msg))
    return sorted(ticks, key=lambda x: x[0])


def emg_events(rows):
    """emg.raw messages of every clip, in time order, as (time of the last sample, topic, msg)."""
    for r in sorted(rows, key=lambda r: r['clip_t0']):
        x, ts = np.load(r['emg']), np.load(r['emg_ts'])
        for s in range(0, len(x), EMG_CHUNK):
            chunk = x[s:s + EMG_CHUNK]
            yield float(ts[min(s + EMG_CHUNK, len(x)) - 1]), 'emg.raw', {'t0': float(ts[s]), 'fs': 1000,
                                                                         'samples': chunk, 'counter': s}


def session_gain(live, rows):
    """The session's gain and effort reference from its own clips, as svr.finetune computes them (unsupervised)."""
    from svr.finetune import session_gains
    from svr.preprocess import rig_unscaled
    sess = rows[0]['session']
    unscaled = {r['id']: rig_unscaled(np.load(r['emg'])) for r in rows}
    gains = session_gains(unscaled, rows, live.target)
    live.gain = np.asarray(gains[sess], np.float32)


# ------------------------------------------------------------------ building the fusion process
def load_params(path=PARAMS):
    from fusion.lock import LockModel
    from fusion.search import Gate
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return Gate.from_dict(d['gate']), LockModel.from_dict(d['lock']), d
    return Gate(), LockModel(), {}


def build_search(lm_path=None, lookalike_vocab=50000):
    from bench.lookalikes import Lexicon
    from fusion.lm import DEFAULT_LM, ConversationLM
    from fusion.search import JointSearch
    lm = ConversationLM(lm_path or DEFAULT_LM)
    from svr.decoders import read_kenlm_vocabulary
    vocab = [w for w in read_kenlm_vocabulary(lm_path or DEFAULT_LM) if w.isalpha() or "'" in w]
    vocab = sorted(vocab, key=lambda w: -lm.model.score(w, bos=False, eos=False))[:lookalike_vocab]
    return JointSearch(lm, lexicon=Lexicon(vocab))


def run_session(session, cam_log, svr_path, gate, lock, use_emg=True, use_lm=True, record=True, no_lock=False,
                max_filled=0.05, limit=None, progress=True, emg_async=True):
    """Feed one recorded session through Fusion; returns (records, published messages, rows used)."""
    from fusion.core import Fusion
    from svr.finetune import usable_record
    rows = session_rows(session)
    windows = clip_windows(rows)
    ok_ids = {r['id'] for r in rows if usable_record(r, False, max_filled)}
    live = None
    if use_emg:
        from svr.live import LiveSVR1
        live = LiveSVR1(svr_path)
        session_gain(live, [r for r in rows if r['id'] in ok_ids])
    if no_lock:
        lock.tau0 = lock.tau_min = lock.tau_max = 2.0       # nothing locks: every word on every pass is an example
        lock.tau_end = lock.repeat_floor = None
    search = build_search()
    if not use_lm:   # camera alone: no LM-ranked short-word insertions either
        search.inserts = 0

    def oracle(u):
        frames_end = float(u.cam.frame_t[-1]) if u.cam is not None else u.seg_start + 1
        r = clip_of(windows, u.seg_start, frames_end)
        return r['prompt'] if r is not None and r.get('prompt') and r['id'] in ok_ids else None

    core = Fusion(search, gate, lock, svr=live, use_emg=use_emg, use_lm=use_lm, use_hints=False, idle_emg=False,
                  record=record, oracle=oracle, sentences=False, emg_async=emg_async)   # scored per recorded clip
    cam = camera_events(cam_log)
    if limit:
        segs = sorted({int(m['seg_id']) for _, tp, m in cam if tp == 'cam.tick'})[:limit]
        cam = [e for e in cam if int(e[2]['seg_id']) in set(segs)]
    used_rows = [r for r in rows if r['id'] in ok_ids]
    t_end = cam[-1][0] + 5 if cam else 0
    streams = [iter(cam)]
    if use_emg:
        streams.append(e for e in emg_events(used_rows) if e[0] <= t_end)
    out = []
    next_step = cam[0][0] if cam else 0
    n_ticks = sum(tp == 'cam.tick' for _, tp, _ in cam)
    done = 0
    for t, topic, msg in heapq.merge(*streams, key=lambda e: e[0]):
        while next_step <= t:
            out += [(next_step, tp, m) for tp, m in core.step(next_step)]
            next_step += 0.05
        if topic == 'cam.tick':
            done += 1
            if progress and done % 500 == 0:
                print(f'{session}: {done}/{n_ticks} passes', flush=True)
        out += [(t, tp, m) for tp, m in core.feed(topic, msg, t)]
    while next_step <= t_end:
        out += [(next_step, tp, m) for tp, m in core.step(next_step)]
        next_step += 0.05
    for rec in core.records or []:
        r = clip_of(windows, rec['seg_start'], rec['cam_frame_t'] or rec['seg_start'] + 1)
        rec['clip'] = r['id'] if r is not None else None
        rec['clip_ok'] = r is not None and r['id'] in ok_ids
        rec['prompt'] = r.get('prompt') if r is not None else None
    return core.records, out, windows


# ------------------------------------------------------------------ labels
def align(hyp, ref):
    """For each hypothesis word, the index of the reference word it matches in a minimum-edit alignment where
    skipping reference words at the start or end is free (a segment can read part of a sentence); else None."""
    n, m = len(hyp), len(ref)
    INF = 10 ** 9
    D = np.full((n + 1, m + 1), INF, np.int64)
    D[0, :] = 0                                   # free reference prefix
    for i in range(1, n + 1):
        D[i, 0] = i
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (hyp[i - 1] != ref[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    j = int(np.argmin(D[n]))                      # free reference suffix
    i, match = n, [None] * n
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (hyp[i - 1] != ref[j - 1]):
            if hyp[i - 1] == ref[j - 1]:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return match


def word_examples(records):
    """(X, y, groups) for the lock model: every word on every pass of a usable clip; y = right at its position and
    the same word stays there for the rest of the segment."""
    from fusion.search import norm_words
    by_utt = {}
    for r in records:
        by_utt.setdefault(r['utt_id'], []).append(r)
    X, y, info = [], [], []
    for utt, recs in by_utt.items():
        recs.sort(key=lambda r: r['t'])
        if not recs[0]['clip_ok'] or not recs[0]['prompt']:
            continue
        ref = norm_words(recs[0]['prompt'])
        bests = [r['best'] for r in recs]
        for k, r in enumerate(recs):
            match = align(list(r['best']), ref)
            for x in r['words']:
                i = x['i']
                right = match[i] is not None
                stays = all(len(b) > i and b[i] == x['w'] for b in bests[k + 1:])
                X.append(x['f'])
                y.append(int(right and stays))
                info.append((utt, k, i, right))
    return np.array(X), np.array(y), info


# ------------------------------------------------------------------ gate fitting
def word_errors(hyp, ref):
    """Word-level edit distance."""
    d = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (h != r))
    return d[-1]


def fit_gate(records, iters=300, l2=0.1):
    """Fit the gate on the final passes by minimum expected word errors: minimise sum_c softmax(S)_c * errors(c) over
    each pass's own candidates (errors against the true sentence, per word of it), so the weights are judged by the
    words they get right, on the candidates the search really has."""
    import torch
    from fusion.search import GATE_FEATURES, Gate
    rows = []
    for r in records:
        if not (r['final'] and r.get('truth') is not None and r['clip_ok']):
            continue
        sc = np.asarray(r['scores'], np.float64)
        sc = sc - sc.max(0, keepdims=True) * np.array([1, 1, 1, 0, 0, 1, 0])   # shift: softmax is unchanged
        err = np.array([word_errors(c, r['truth']) for c in r['cands']], np.float64) / max(len(r['truth']), 1)
        rows.append((torch.tensor(sc), torch.tensor(err), torch.tensor(np.asarray(r['gate'], np.float64))))
    if not rows:
        raise SystemExit('no final passes with a known sentence')
    K = max(len(sc) for sc, _, _ in rows)
    SC = torch.zeros(len(rows), K, 7, dtype=torch.float64)
    mask = torch.zeros(len(rows), K, dtype=torch.bool)
    for n, (sc, _, _) in enumerate(rows):
        SC[n, :len(sc)], mask[n, :len(sc)] = sc, True
    ERR = torch.zeros(len(rows), K, dtype=torch.float64)
    for n, (_, err, _) in enumerate(rows):
        ERR[n, :len(err)] = err
    F = torch.stack([f for _, _, f in rows])
    W = torch.zeros(3, len(GATE_FEATURES), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(np.log([0.6, 0.15, 0.25]), dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor(np.log(1.5), dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    g_final = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g_unseen = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    beta = 1.0      # Claude's hints: no offline replay of Claude, so not fitted here
    opt = torch.optim.Adam([W, b, log_scale, gamma, g_final, g_unseen], lr=0.05)
    for it in range(iters):
        opt.zero_grad()
        g = log_scale.exp() * torch.softmax(F @ W.T + b, 1)                     # (N, 3)
        S = (g[:, None, :] * SC[:, :, :3]).sum(2) + gamma * SC[:, :, 4] + g_final * SC[:, :, 5] + g_unseen * SC[:, :, 6]
        S = S.masked_fill(~mask, -1e9)
        loss = (torch.softmax(S, 1) * ERR).sum(1).mean() + l2 * (W ** 2).sum()
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            print(f'gate fit {it}: expected word errors {loss.item():.4f}', flush=True)
    t_cam, t_emg = (fit_temperature(SC[:, :, j], ERR, mask) for j in (0, 1))
    print(f'reader-only posteriors: t_cam {t_cam:.3f}, t_emg {t_emg:.4f}', flush=True)
    gate = Gate(W.detach().numpy(), b.detach().numpy(), float(log_scale.detach().exp()), beta, float(gamma.detach()),
                float(g_final.detach()), t_cam, t_emg, float(g_unseen.detach()))
    return gate, rows


def fit_temperature(score, err, mask, iters=200):
    """t such that softmax(t * score) puts its mass on each pass's best candidates (fewest word errors): one
    reader's scores read as a calibrated posterior."""
    import torch
    best = (err <= err.masked_fill(~mask, 1e9).min(1, keepdim=True).values + 1e-9) & mask
    log_t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        lp = torch.log_softmax((log_t.exp() * score).masked_fill(~mask, -1e9), 1)
        loss = -torch.logsumexp(lp.masked_fill(~best, -1e9), 1).mean()
        loss.backward()
        opt.step()
    return float(log_t.detach().exp())


def gate_wer(rows, gate):
    """Word error rate of the top candidate of each final pass, with these weights."""
    errs = 0.0
    for sc, err, f in rows:
        g = gate.weights(f.numpy())
        sc = sc.numpy()
        S = (g[0] * sc[:, 0] + g[1] * sc[:, 1] + g[2] * sc[:, 2] + gate.gamma * sc[:, 4] + gate.final * sc[:, 5]
             + gate.unseen * sc[:, 6])
        errs += float(err.numpy()[int(np.argmax(S))])
    return errs / len(rows)


# ------------------------------------------------------------------ lock simulation (for tau0)
from fusion.search import SHORT_WORDS  # noqa: E402


def simulate_locks(records, lock, tau, tau_end=None, repeat_floor=None):
    """Approximate locking on no-lock passes: word k locks on the first pass where it is the best candidate's word
    at k with p >= tau (and agreement), after word k-1. Returns per-utterance results."""
    from fusion.search import norm_words
    by_utt = {}
    for r in records:
        by_utt.setdefault(r['utt_id'], []).append(r)
    res = []
    for utt, recs in by_utt.items():
        recs.sort(key=lambda r: r['t'])
        if not recs[0]['clip_ok'] or not recs[0]['prompt']:
            continue
        ref = norm_words(recs[0]['prompt'])
        locked, probs, wrong, times = [], [], 0, []
        for r in recs:
            match = align(list(r['best']), ref)
            for x in r['words']:
                if x['i'] != len(locked):
                    continue
                p = float(lock.prob(x['f']))
                if not lock.may_lock(x['f'], p, min(tau, tau_end) if (r['final'] and tau_end is not None) else tau):
                    break
                locked.append(x['w'])
                probs.append(p)
                wrong += int(match[x['i']] is None)
                times.append((x['i'], r['t']))
        last = recs[-1]
        unlocked = len(last['best']) - len(locked)
        if repeat_floor is not None and unlocked > 0 and last['final']:   # the final pass speaks the rest, unless unreadable
            rest = [x for x in last['words'] if x['i'] >= len(locked)]
            if lock.settle(locked + [x['w'] for x in rest], probs + [float(lock.prob(x['f'])) for x in rest],
                           repeat_floor):
                match = align(list(last['best']), ref)
                for x in rest:
                    locked.append(x['w'])
                    wrong += int(match[x['i']] is None)
                    times.append((x['i'], last['t']))
                unlocked = 0
        res.append(dict(utt=utt, locked=locked, wrong=wrong, times=times, repeat=unlocked > 0,
                        ends=last.get('truth_ends'), ref=ref, best=last['best']))
    return res


def lock_summary(res):
    import jiwer
    n_locked = sum(len(r['locked']) for r in res)
    wrong = sum(r['wrong'] for r in res)
    delays = []
    for r in res:
        if r['ends']:
            m = align(r['locked'], r['ref'])
            for (i, t), j in zip(r['times'], m):
                if j is not None and j < len(r['ends']):
                    delays.append(t - r['ends'][j])
    refs = [' '.join(r['ref']) for r in res]
    hyps = [' '.join(r['locked']) for r in res]
    return dict(utterances=len(res), locked_words=n_locked, wrong_locks=wrong, wrong_lock_rate=wrong / max(n_locked, 1),
                repeat_rate=float(np.mean([r['repeat'] for r in res])) if res else 0.0,
                delay_median_s=float(np.median(delays)) if delays else None,
                delay_mean_s=float(np.mean(delays)) if delays else None,
                wer_locked=jiwer.wer(refs, hyps) if res else None)


# ------------------------------------------------------------------ commands
def cmd_passes(a):
    gate, lock, _ = load_params(a.params)
    recs, out, _ = run_session(a.session, a.cam_log, a.svr, gate, lock, use_emg=not a.no_emg, record=True,
                               no_lock=True, limit=a.limit)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump({'session': a.session, 'cam_log': a.cam_log, 'svr': a.svr, 'params': a.params,
                     'records': recs}, f)
    ms = [r['ms'] for r in recs]
    print(f'{a.out}: {len(recs)} passes, {len({r["utt_id"] for r in recs})} utterances; pass time median '
          f'{np.median(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms')


def load_records(paths):
    recs = []
    for k, p in enumerate(paths):
        with open(p, 'rb') as f:
            d = pickle.load(f)
        for r in d['records']:
            r['utt_id'] = (k, r['utt_id'])
            r['fold'] = k
        recs += d['records']
    return recs


def save_params(gate=None, lock=None, extra=None, path=PARAMS):
    _, _, d = load_params(path)
    if gate is not None:
        d['gate'] = gate.to_dict()
    if lock is not None:
        d['lock'] = lock.to_dict()
    d.update(extra or {})
    d.setdefault('gate', __import__('fusion.search', fromlist=['Gate']).Gate().to_dict())
    d.setdefault('lock', __import__('fusion.lock', fromlist=['LockModel']).LockModel().to_dict())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)
    print(f'saved {path}')


def cmd_fit_gate(a):
    from fusion.search import Gate
    recs = load_records(a.passes)
    gate, rows = fit_gate(recs)
    before, after = gate_wer(rows, Gate()), gate_wer(rows, gate)
    print(f'final pass WER (mean per sentence): default gate {100 * before:.1f}% -> fitted {100 * after:.1f}% '
          f'({len(rows)} final passes)')
    print('gate', json.dumps(gate.to_dict()))
    save_params(gate=gate, extra={'gate_fit': {'final_passes': len(rows), 'wer_default': before, 'wer_fitted': after,
                                               'from': a.passes}}, path=a.params)


def cmd_fit_lock(a):
    from fusion.lock import LockModel
    recs = load_records(a.passes)
    folds = sorted({r['fold'] for r in recs})
    taus = np.round(np.arange(0.5, 0.995, 0.02), 3)
    table = {float(t): [] for t in taus}
    for k in folds:   # each session scored by a lock model fitted on the other
        train = [r for r in recs if r['fold'] != k] if len(folds) > 1 else recs
        test = [r for r in recs if r['fold'] == k]
        X, y, _ = word_examples(train)
        lm = LockModel().fit(X, y)
        for t in taus:
            table[float(t)] += simulate_locks(test, lm, float(t))
    rows = []
    for t in taus:
        s = lock_summary(table[float(t)])
        rows.append((float(t), s))
        print(f"tau {t:.2f}: wrong locks {100 * s['wrong_lock_rate']:.1f}%  repeat {100 * s['repeat_rate']:.0f}%  "
              f"delay median {s['delay_median_s'] if s['delay_median_s'] is None else round(s['delay_median_s'], 2)} s"
              f"  WER locked {100 * s['wer_locked']:.1f}%", flush=True)
    # Every word left at sentence end is spoken (a wrong guess beats "Please repeat"), so an early lock only buys
    # time: 1. the sentence floor for "Please repeat", the highest that keeps it to at most max_repeat of sentences,
    # with no early locks; 2. tau0, the lowest whose WER stays within wer_slack of that; 3. tau_end the same way.
    models = {}
    for k in folds:
        train = [r for r in recs if r['fold'] != k] if len(folds) > 1 else recs
        X, y, _ = word_examples(train)
        models[k] = LockModel().fit(X, y)

    def sim(t0, te, fl):
        res = []
        for k in folds:
            res += simulate_locks([r for r in recs if r['fold'] == k], models[k], t0, te, fl)
        return lock_summary(res)

    top = float(taus[-1])
    floor, s_floor = 0.0, sim(top, None, 0.0)
    for fl in np.round(np.arange(0.0, 0.61, 0.02), 3):
        s = sim(top, None, float(fl))
        print(f"repeat floor {fl:.2f}: repeat {100 * s['repeat_rate']:.1f}%  WER {100 * s['wer_locked']:.1f}%", flush=True)
        if s['repeat_rate'] <= a.max_repeat:
            floor, s_floor = float(fl), s
    base = s_floor['wer_locked']
    t0, s0 = top, s_floor
    for t in taus[::-1]:
        s = sim(float(t), None, floor)
        print(f"tau0 {t:.2f}: WER {100 * s['wer_locked']:.1f}%  wrong locks {100 * s['wrong_lock_rate']:.1f}%  "
              f"repeat {100 * s['repeat_rate']:.1f}%  delay median {s['delay_median_s']} s", flush=True)
        if s['wer_locked'] <= base + a.wer_slack:
            t0, s0 = float(t), s
    t_end, s_end = t0, s0
    for te in taus[taus < t0][::-1]:
        s = sim(t0, float(te), floor)
        print(f"tau_end {te:.2f}: WER {100 * s['wer_locked']:.1f}%  delay median {s['delay_median_s']} s", flush=True)
        if s['wer_locked'] <= base + a.wer_slack:
            t_end, s_end = float(te), s
    s_floor = s_end
    X, y, _ = word_examples(recs)
    lock = LockModel().fit(X, y)
    lock.tau_end = t_end if t_end < t0 else None
    lock.repeat_floor = floor
    lock.tau0 = t0
    lock.tau_min, lock.tau_max = max(0.5, t0 - 0.15), min(0.99, t0 + 0.05)
    print(f'tau0 {t0}: {json.dumps(s0)}')
    print(f'tau_end {t_end}: {json.dumps(s_end)}')
    print(f'repeat floor {floor}: {json.dumps(s_floor)}')
    print('lock', json.dumps(lock.to_dict()))
    save_params(lock=lock, extra={'lock_fit': {'examples': int(len(y)), 'positive_share': float(y.mean()),
                                               'cross_fitted_at_tau0': s0, 'cross_fitted_at_tau_end': s_end,
                                               'cross_fitted_with_repeat_floor': s_floor, 'max_repeat': a.max_repeat,
                                               'max_wrong': a.max_wrong,
                                               'wrong_cost': a.wrong_cost,
                                               'curve': {f'{t:.2f}': s for t, s in rows},
                                               'from': a.passes}}, path=a.params)


def cmd_test(a):
    """The whole pipeline with locking on one session: camera alone, + EMG, + LM; WER, wrong locks, delay, repeats."""
    import jiwer
    from fusion.search import norm_words
    gate, lock, _ = load_params(a.params)
    results = {}
    configs = [('camera', False, False), ('camera+emg', True, False), ('camera+emg+lm', True, True)]
    for name, emg, lmon in configs:
        g, lk, _ = load_params(a.params)
        recs, out, windows = run_session(a.session, a.cam_log, a.svr, g, lk, use_emg=emg, use_lm=lmon, record=True,
                                         limit=a.limit, emg_async=not a.emg_sync)
        by_utt = {}
        for r in recs:
            by_utt.setdefault(r['utt_id'], []).append(r)
        captions = {}
        for t, tp, m in out:
            if tp == 'dec.hyp':
                captions[m['utt_id']] = m['locked']
        refs, hyps, wrong, n_locked, delays, repeats = [], [], 0, 0, [], 0
        repeat_utts = {m['utt_id'] for t, tp, m in out if tp == 'dec.repeat'}
        lock_t = {}
        for t, tp, m in out:
            if tp == 'dec.lock':
                for x in m['words']:
                    lock_t.setdefault(m['utt_id'], []).append(x['t'])
        for utt, rs in by_utt.items():
            rs.sort(key=lambda r: r['t'])
            if not rs[0]['clip_ok'] or not rs[0]['prompt']:
                continue
            ref = norm_words(rs[0]['prompt'])
            hyp = list(captions.get(utt, []))
            refs.append(' '.join(ref))
            hyps.append(' '.join(hyp))
            match = align(hyp, ref)
            n_locked += len(hyp)
            wrong += sum(mm is None for mm in match)
            repeats += utt in repeat_utts
            ends = next((r.get('truth_ends') for r in reversed(rs) if r.get('truth_ends')), None)
            if ends:
                for tt, j in zip(lock_t.get(utt, []), match):
                    if j is not None and j < len(ends):
                        delays.append(tt - ends[j])
        ms = [r['ms'] for r in recs]
        results[name] = dict(utterances=len(refs), wer=jiwer.wer(refs, hyps) if refs else None,
                             wrong_lock_rate=wrong / max(n_locked, 1), repeat_rate=repeats / max(len(refs), 1),
                             delay_median_s=float(np.median(delays)) if delays else None,
                             delay_p90_s=float(np.percentile(delays, 90)) if delays else None,
                             pass_ms_median=float(np.median(ms)), pass_ms_p95=float(np.percentile(ms, 95)))
        print(name, json.dumps(results[name]), flush=True)
    # reference: the camera's own final pass (cam.final's best) on the same sentences
    import buslog
    nb = {int(m['seg_id']): m['nbest'] for _, _, m in buslog.read(a.cam_log, topics=['cam.final'])}
    refs, hyps = [], []
    for utt, rs in by_utt.items():
        if rs[0]['clip_ok'] and rs[0]['prompt']:
            refs.append(' '.join(norm_words(rs[0]['prompt'])))
            h = nb.get(rs[0]['seg_id'])
            hyps.append(' '.join(norm_words(h[0]['text'])) if h else '')
    results['camera_final_pass_reference'] = dict(utterances=len(refs), wer=jiwer.wer(refs, hyps))
    print('camera final pass (reference)', json.dumps(results['camera_final_pass_reference']), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({'session': a.session, 'cam_log': a.cam_log, 'svr': a.svr, 'params': a.params,
                   'emg_sync': a.emg_sync, 'results': results}, f, indent=1)
    print(f'saved {a.out}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('passes')
    q.add_argument('session')
    q.add_argument('cam_log')
    q.add_argument('--svr', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--params', default=PARAMS)
    q.add_argument('--no-emg', action='store_true')
    q.add_argument('--limit', type=int, default=None, help='only the first N camera segments')
    q.set_defaults(fn=cmd_passes)
    q = sub.add_parser('fit-gate')
    q.add_argument('passes', nargs='+')
    q.add_argument('--params', default=PARAMS)
    q.set_defaults(fn=cmd_fit_gate)
    q = sub.add_parser('fit-lock')
    q.add_argument('passes', nargs='+')
    q.add_argument('--params', default=PARAMS)
    q.add_argument('--max-wrong', type=float, default=0.02)
    q.add_argument('--max-repeat', type=float, default=0.05, help='share of sentences that may end in Please repeat')
    q.add_argument('--wer-slack', type=float, default=0.03,
                   help='early locks may raise the WER by at most this over waiting for the sentence end')
    q.add_argument('--wrong-cost', type=float, default=3.0, help='when max-wrong is out of reach: a wrong lock '
                   'costs this many missing words')
    q.set_defaults(fn=cmd_fit_lock)
    q = sub.add_parser('test')
    q.add_argument('session')
    q.add_argument('cam_log')
    q.add_argument('--svr', required=True)
    q.add_argument('--params', default=PARAMS)
    q.add_argument('--out', default=os.path.join(WORK, 'benchmark_results', 'FUSION_TEST', 'summary.json'))
    q.add_argument('--limit', type=int, default=None)
    q.add_argument('--emg-sync', action='store_true', help='each pass waits for its own SVR-1 read (the old path)')
    q.set_defaults(fn=cmd_test)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
