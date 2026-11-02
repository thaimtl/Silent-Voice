"""Unit checks for the fusion process (fusion.search, fusion.lock, fusion.core) on synthetic camera readings and a
tiny hand-written n-gram (CPU; needs the camera's units file and SentencePiece model under RECLAIM_WORKDIR).

    venv-svr/bin/python -m tests.smoke_fusion_units
"""
import itertools
import os
import tempfile

import numpy as np

from fusion.core import Fusion
from fusion.lm import ConversationLM
from fusion.lock import FEATURES, LockModel
from fusion.lock import features as _lock_features
from fusion.offline import align
from fusion.search import BLANK, CamReading, Gate, JointSearch, Pieces, ctc_loglik, viterbi_word_ends

ARPA = r"""
\data\
ngram 1=10
ngram 2=5

\1-grams:
-6.0	<unk>	0
-99	<s>	-0.5
-0.8	</s>	0
-0.6	i	-0.3
-0.9	want	-0.3
-0.9	water	-0.3
-1.5	waiter	-0.3
-1.2	what	-0.3
-1.3	bill	-0.3
-1.3	mill	-0.3

\2-grams:
-0.2	<s> i
-0.3	i want
-0.4	want water
-0.2	water </s>
-1.4	want waiter

\end\
"""


def lockmod_features(joint, cam, emg, claude):
    """Lock features of a finished, stable word on a clear face and clean pads."""
    return _lock_features(joint, cam, emg, claude, True, 1.0, 1.0, 10, True, 1.0)


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


class StubLexicon:
    def __init__(self, table):
        self.table = table

    def lookalikes(self, word):
        return self.table.get(word, [])


def reading(pieces, words, seg_id=1, t0=100.0, gap=3, tail=4, noise=None, seed=0):
    """A peaky cam.tick: each piece on one frame (p 0.9), blanks between; noise {word: (alt, p_alt)} puts some of a
    word's mass on a look-alike's first piece."""
    rng = np.random.default_rng(seed)
    seq = []
    for w in words:
        ids = pieces.word_ids(w)
        alt = (noise or {}).get(w)
        for k, i in enumerate(ids):
            seq.append((i, pieces.word_ids(alt[0])[0] if alt and k == 0 else None, alt[1] if alt and k == 0 else 0))
    T = gap + len(seq) * (gap + 1) + tail
    ids = np.zeros((T, 32), np.int32)
    logp = np.zeros((T, 32), np.float32)
    blank = np.zeros(T, np.float32)
    filler = [i for i in range(2, 600) if i not in {s[0] for s in seq}][:40]
    hot = {gap + k * (gap + 1): s for k, s in enumerate(seq)}
    for t in range(T):
        top = {}
        if t in hot:
            tok, alt, pa = hot[t]
            top[tok] = 0.9 - pa
            if alt:
                top[alt] = pa
            pb = 0.05
        else:
            pb = 0.97
        rest = rng.permutation(filler)
        mass = 1.0 - pb - sum(top.values())
        for i in rest[:32 - len(top) - 1]:
            top[int(i)] = mass / 200
        top[BLANK] = pb
        items = sorted(top.items(), key=lambda kv: -kv[1])[:32]
        ids[t] = [i for i, _ in items]
        logp[t] = np.log([p for _, p in items])
        blank[t] = np.log(pb)
    frame_t = t0 + np.arange(T) * 0.04
    return {'seg_id': seg_id, 'seg_start': float(frame_t[0]), 'frame_t': frame_t, 'topk_ids': ids,
            'topk_logp': logp, 'blank_logp': blank, 'activity': np.full(T, 0.6, np.float32), 'face_ok': True}


def brute_ctc(cam, seq):
    """CTC forward over the extended label sequence, straight from the definition."""
    em = cam.emissions(sorted(set(seq)))
    col = {t: j for j, t in enumerate(sorted(set(seq)))}
    lab = [BLANK] + [x for t in seq for x in (t, BLANK)]
    S = len(lab)
    e = lambda t, s: cam.blank[t] if lab[s] == BLANK else em[t, col[lab[s]]]  # noqa: E731
    a = np.full(S, -np.inf)
    a[0], a[1] = e(0, 0), e(0, 1)
    for t in range(1, cam.T):
        b = np.full(S, -np.inf)
        for s in range(S):
            terms = [a[s]] + ([a[s - 1]] if s >= 1 else [])
            if s >= 2 and lab[s] != BLANK and lab[s] != lab[s - 2]:
                terms.append(a[s - 2])
            b[s] = np.logaddexp.reduce(terms) + e(t, s)
        a = b
    return np.logaddexp(a[-1], a[-2])


def main():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tiny.arpa')
        with open(path, 'w') as f:
            f.write(ARPA)
        lm = ConversationLM(path)
        pieces = Pieces()
        search = JointSearch(lm, pieces, StubLexicon({'water': ['waiter'], 'waiter': ['water']}))
        w = {'cam': 1.0, 'emg': 0.0, 'lm': 0.3, 'beta': 1.0, 'gamma': 0.5}

        cam = CamReading(reading(pieces, ['i', 'want', 'water']))
        seqs = [list(itertools.chain(*[pieces.word_ids(x) for x in s.split()])) for s in
                ('i want water', 'i want waiter', 'i want')]
        fast = ctc_loglik(cam, seqs)
        slow = np.array([brute_ctc(cam, s) for s in seqs])
        check('camera CTC log-likelihood matches the definition', np.allclose(fast, slow, atol=1e-3),
              f'{fast.round(2)} vs {slow.round(2)}')
        check('the true reading scores highest under the camera', fast[0] > fast[1] and fast[0] > fast[2])

        beam = search.camera_beam(cam, w, {})
        words, _ = pieces.split_words(beam[0][0])
        check('camera beam reads the sentence', words == ['i', 'want', 'water'], ' '.join(words))

        cands = search.candidates(cam, w, {}, final=True)
        texts = [' '.join(c.words) for c in cands]
        check('look-alike swaps join the candidates', 'i want waiter' in texts, f'{len(cands)} candidates')
        sc = search.judge_scores(cands, cam, None, {}, True)
        check('judge scores: one row per candidate, seven columns', sc.shape == (len(cands), 7))

        # the camera dropped a short word: 'i water' -> inserting 'want' isn't short, but 'i want water' -> 'i want the water'
        drop = CamReading(reading(pieces, ['want', 'water']))
        cands_d = search.candidates(drop, w, {}, final=True)
        ins = [c for c in cands_d if c.source == 'insert']
        check('short words the camera dropped are tried, unseen by the camera',
              ins and all(c.unseen() == 1 for c in ins) and any(c.words == ('i', 'want', 'water') for c in ins),
              str([' '.join(c.words) for c in ins][:5]))
        ph = search.candidates(drop, w, {'i': 0.3}, final=True, phrases=[['i', 'want', 'water']])
        check("Claude's phrase fills in the dropped word", any(c.source == 'phrase' and c.words == ('i', 'want', 'water')
                                                              for c in ph))
        sc_d = search.judge_scores(cands_d, drop, None, {}, True)
        k = next(j for j, c in enumerate(cands_d) if c.words == ('i', 'want', 'water'))
        k0 = next(j for j, c in enumerate(cands_d) if c.words == ('want', 'water'))
        check('an unseen word costs the camera nothing', abs(sc_d[k, 0] - sc_d[k0, 0]) < 1e-6 and sc_d[k, 6] == 1)
        lk2 = LockModel(repeat_floor=0.5)
        check('the final pass speaks a sentence unless it is unreadable; short words never count',
              lk2.settle(['i', 'want', 'water'], [0.1, 0.9, 0.8]) and not lk2.settle(['i', 'want', 'water'],
                                                                                      [0.9, 0.2, 0.3]))

        # EMG settles a look-alike: the camera is unsure between water and waiter, EMG strongly favours waiter
        cam2 = CamReading(reading(pieces, ['i', 'want', 'water'], noise={'water': ('waiter', 0.42)}))
        cands = search.candidates(cam2, w, {}, final=True)
        emg = lambda texts: np.array([0.0 if t.endswith('waiter') else -12.0 for t in texts])  # noqa: E731
        sc = search.judge_scores(cands, cam2, emg, {}, True)
        S0 = search.combine(sc, w)
        S1 = search.combine(sc, dict(w, emg=1.0))
        b0, b1 = cands[int(np.argmax(S0))].words, cands[int(np.argmax(S1))].words
        check('EMG can overturn the camera on a look-alike', b0[-1] == 'water' and b1[-1] == 'waiter', f'{b0} -> {b1}')
        hint = search.combine(search.judge_scores(cands, cam2, None, {'waiter': 30.0}, True), w)
        check("Claude's hints add to a word's score", cands[int(np.argmax(hint))].words[-1] == 'waiter')

        # Claude's hints at the live weight (fusion/run.py DEFAULT_BETA) against the camera, with the fitted gate's
        # camera, LM and length weights. "bill" and "mill" look the same on the lips, are one camera piece each, and
        # the test LM gives both the same probability, so only the camera and the hint decide.
        from fusion.run import DEFAULT_BETA
        live_w = {'cam': 4.9, 'emg': 0.0, 'lm': 1.9, 'beta': DEFAULT_BETA, 'gamma': 2.5, 'final': 0.0, 'unseen': -9.0}
        lk_search = JointSearch(lm, pieces, StubLexicon({'bill': ['mill'], 'mill': ['bill']}))

        def best_with(reading_msg, hints):
            c = CamReading(reading_msg)
            cs = lk_search.candidates(c, live_w, hints, final=True)
            return cs[int(np.argmax(lk_search.combine(lk_search.judge_scores(cs, c, None, hints, True), live_w)))].words, cs

        tie = reading(pieces, ['i', 'want', 'bill'], noise={'bill': ('mill', 0.42)})   # camera: bill 0.48, mill 0.42
        b0, _ = best_with(tie, {})
        b1, _ = best_with(tie, {'mill': 0.9})
        check("a predicted look-alike wins the camera's near-tie (beta %g, weight 0.9)" % DEFAULT_BETA,
              b0[-1] == 'bill' and b1[-1] == 'mill', f'{b0} -> {b1}')
        clear = reading(pieces, ['i', 'want', 'bill'])                                   # camera: clearly bill
        b2, _ = best_with(clear, {'mill': 0.9})
        check('clear camera evidence still beats a predicted look-alike', b2[-1] == 'bill', b2)
        b3, cs3 = best_with(clear, {'coffee': 1.0, 'mill': 1.0})
        check('a predicted word the camera clearly did not see does not appear',
              b3 == ('i', 'want', 'bill') and not any('coffee' in c.words for c in cs3), b3)

        locked = search.candidates(cam, w, {}, locked=['i', 'went'])
        check('every candidate keeps the locked words', all(c.words[:2] == ('i', 'went') for c in locked))

        ends = viterbi_word_ends(cam, [pieces.word_ids(x) for x in ('i', 'want', 'water')])
        check('word end times rise along the sentence', ends is not None and ends[0] < ends[1] < ends[2],
              str(np.round(ends, 2) if ends else None))

        check('align: exact', align(['i', 'want', 'water'], ['i', 'want', 'water']) == [0, 1, 2])
        check('align: a partial reading matches the start', align(['i', 'want'], ['i', 'want', 'water']) == [0, 1])
        check('align: a wrong word matches nothing', align(['i', 'went', 'water'], ['i', 'want', 'water']) == [0, None, 2])
        check('align: a later segment matches the end', align(['water'], ['i', 'want', 'water']) == [2])

        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, (400, len(FEATURES)))
        y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0.5).astype(int)
        lk = LockModel().fit(X, y)
        check('lock model learns from the joint posterior', lk.theta[0] > 1.0, f'theta_joint {lk.theta[0]:.2f}')
        lk.tau0, lk.kappa, lk.tau_min, lk.tau_max = 0.9, 0.2, 0.8, 0.95
        check('a predictable moment lowers the threshold, within its range',
              lk.threshold(0.0) == 0.9 and abs(lk.threshold(1.0) - 0.8) < 1e-9 and lk.threshold(0.25) < 0.9)
        f = np.zeros(len(FEATURES))
        check('no camera-EMG agreement, no lock', not lk.may_lock(f, 0.99, 0.9))

        # the core, camera only: ticks grow, words lock left to right, the voice gets them after the repair window
        lock = LockModel(tau0=0.5, tau_min=0.5, tau_max=0.5, agree_floor=0.0)
        core = Fusion(search, Gate(), lock, svr=None, use_hints=True, repair_s=0.3)
        full = reading(pieces, ['i', 'want', 'water'])
        out, t = [], 0.0
        T = len(full['frame_t'])
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        topics = [tp for tp, _ in out]
        check('dec.hyp on every pass', topics.count('dec.hyp') == len(range(8, T + 1, 3)))
        out2 = []
        for k in range(1, 40):
            out2 += core.step(t + 0.05 * k)
            if k == 12:
                out2 += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]},
                                  t + 0.05 * k)
            for tp, m in list(out2):
                if tp == 'dec.pause' and not m.get('answered'):     # as the Claude service would answer it
                    m['answered'] = True
                    out2 += core.feed('claude.end', {'utt_id': m['utt_id'], 'p_complete': 0.95}, t + 0.05 * k)
        out += out2
        pauses = [m for tp, m in out if tp == 'dec.pause']
        check('the lips stop: the sentence holds and asks Claude whether he finished',
              len(pauses) == 1 and pauses[0]['words'][:2] == ['i', 'want'], str([m['words'] for m in pauses]))
        closes = [m for tp, m in out if tp == 'dec.sentence' and m['event'] == 'close']
        check("Claude's 'finished' closes it, no sooner than the minimum hold",
              len(closes) == 1 and closes[0]['reason'] == 'Claude: finished' and closes[0]['held_s'] >= 0.5,
              str(closes))
        locks = [x['w'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        finals = [m for tp, m in out if tp == 'voice.say' and m['final']]
        check('the sentence locks', locks == ['i', 'want', 'water'], ' '.join(locks))
        check('the voice says the locked words once, in order', said == locks, ' '.join(said))
        check('the sentence ends with one final voice.say', len(finals) == 1)
        from bus.team import decode, encode
        try:
            back = [decode(encode(tp, m)) for tp, m in out]
            check('every message goes through the bus encoder', len(back) == len(out))
        except Exception as e:  # noqa: BLE001
            check('every message goes through the bus encoder', False, f'{type(e).__name__}: {e}')
        lock_t = [x['t'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        check('every lock carries its time, cam and alts', len(lock_t) == 3 and all(
            set(x) >= {'w', 'conf', 'alts', 'cam', 'emg', 't'} for tp, m in out if tp == 'dec.lock' for x in m['words']))

        # the camera's last tick of a segment comes ~0.55 s after the one before it (every sentence of session 3),
        # after end_wait_s has ended the sentence: it is still that sentence, so the voice never says it twice
        for late_at in (12, 30):                    # while waiting for cam.final, and after it
            core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
            out = []
            for n in range(8, T - 2, 3):
                msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
                t = float(msg['frame_t'][-1]) + 0.05
                out += core.feed('cam.tick', msg, t)
                out += core.step(t)
            for k in range(1, 60):
                out += core.step(t + 0.05 * k)
                if k == late_at:            # as the camera's: two more frames, the mouth still
                    late = {kk: (v[:n + 2] if isinstance(v, np.ndarray) else v) for kk, v in full.items()}
                    late['activity'] = np.full(n + 2, 0.02, np.float32)
                    out += core.feed('cam.tick', late, t + 0.05 * k)
                if k == 20:
                    out += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]},
                                      t + 0.05 * k)
            said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
            utts = {m['utt_id'] for tp, m in out if tp in ('voice.say', 'dec.hyp')}
            check(f'a late last tick (step {late_at}) is not a new sentence', said == ['i', 'want', 'water'] and
                  utts == {1}, f"{' '.join(said)}; utt_ids {sorted(utts)}")
        # a cam.final that beats end_wait_s (a short reply's beam) ends the sentence at once, with its n-best
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        out += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]}, t + 0.1)
        for k in range(1, 12):              # the repair window, well inside end_wait_s + final_wait_s
            out += core.step(t + 0.1 + 0.05 * k)
        check('an early cam.final is used, and ends the sentence at once',
              any(tp == 'voice.say' and m['final'] for tp, m in out) and core.utt is None and
              all(v.state == 'done' for v in core.recent.values()))

        # the camera caps a segment at 8 s and opens the next with its last 2 s of crops: those words are said once
        long = reading(pieces, ['i', 'want', 'water', 'i', 'want'])
        ft = long['frame_t']
        npc = [len(pieces.word_ids(x)) for x in ['i', 'want', 'water']]
        hot = lambda k: 3 + k * 4           # noqa: E731  reading(): gap 3, a piece every 4 frames
        cut = hot(sum(npc) - 1) + 3         # just after "water"'s last piece
        carry = cut - hot(sum(npc[:2])) + 2  # the carried frames start just before "water"'s first piece
        seg1 = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in long.items()}
        seg2 = {k: (v[cut - carry:] if isinstance(v, np.ndarray) else v) for k, v in long.items()}
        seg2.update(seg_id=2, seg_start=float(ft[cut - carry]))
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        for part in (seg1, seg2):
            n_all = len(part['frame_t'])
            for n in range(4, n_all + 1, 3):
                msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in part.items()}
                t = float(msg['frame_t'][-1]) + 0.05
                out += core.feed('cam.tick', msg, t)
                out += core.step(t)
        for k in range(1, 80):
            out += core.step(t + 0.05 * k)
            if k == 10:
                out += core.feed('cam.final', {'seg_id': 2, 'nbest': [{'text': 'WATER I WANT', 'score': 0.0}]},
                                  t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        check('words carried over a capped segment are said once', said == ['i', 'want', 'water', 'i', 'want'],
              ' '.join(said))

        # a failing end or final pass never stops fusion, and step() never raises: the sentence closes, its locked
        # words are said, and the next one is read
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        real = core.search.candidates

        def flaky(*a, **kw):     # every end and final pass fails (a CUDA error, say)
            if kw.get('final') or (len(a) > 5 and a[5]):
                raise RuntimeError('CUDA error (test)')
            return real(*a, **kw)
        core.search.candidates = flaky
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        raised = 0
        for k in range(1, 80):              # past end_wait_s and final_wait_s
            try:
                out += core.step(t + 0.05 * k)
            except Exception:  # noqa: BLE001
                raised += 1
        core.search.candidates = real
        locked = [x['w'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        said = [x for tp, m in out if tp == 'voice.say' and m['utt_id'] == 1 for x in m['words']]
        nxt2 = reading(pieces, ['i', 'want', 'water'], seg_id=2, t0=300.0)
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in nxt2.items()}
            out += core.feed('cam.tick', msg, float(msg['frame_t'][-1]) + 0.05)
        check('failed end and final passes: step() never raises, the locked words are said, the next sentence is read',
              raised == 0 and said == locked and core.utt is not None and core.utt.utt_id == 2 and
              any(tp == 'voice.say' and m['final'] and m['utt_id'] == 1 for tp, m in out),
              f'raised {raised}; locked {locked}; said {said}')

        # the idle EMG read fails (it runs once no sentence is open, as the last words are released): nothing is lost
        core = Fusion(search, Gate(), lock, svr=None, repair_s=2.0, sentences=False)    # a long window: words still wait

        def broken(now):
            raise RuntimeError('CUDA error in svr.read (test)')
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        out += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]}, t + 0.1)
        # the final pass is done and no sentence is open: from now on every step also runs the idle EMG read
        core.use_emg = core.idle_emg = True
        core._idle_emg = broken
        for k in range(1, 80):
            out += core.step(t + 0.1 + 0.05 * k)
        locked = [x['w'] for tp, m in out if tp == 'dec.lock' for x in m['words']]
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        check('a failing idle EMG read loses no word and no final', said == locked == ['i', 'want', 'water'] and
              any(tp == 'voice.say' and m['final'] for tp, m in out), f'said {said}')

        # the camera stalls 0.4 s mid-sentence (end_wait_s ends it): it reopens, every word is said once
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        ns = list(range(8, T + 1, 3))
        half = len(ns) // 2
        t_shift = 0.0
        for j, n in enumerate(ns):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            if j == half:                   # the stall: no tick for 0.4 s, with steps going on
                t_stall = t
                for k in range(1, 9):
                    out += core.step(t_stall + 0.05 * k)
                t_shift = 0.4
            t = float(msg['frame_t'][-1]) + 0.05 + t_shift
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        for k in range(1, 80):
            out += core.step(t + 0.05 * k)
            if k == 10:
                out += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]},
                                  t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        utts = {m['utt_id'] for tp, m in out if tp in ('voice.say', 'dec.hyp')}
        check('a mid-sentence stall reopens the sentence: every word said once', said == ['i', 'want', 'water']
              and utts == {1}, f"{' '.join(said)}; utt_ids {sorted(utts)}")

        # a stall long enough that the sentence already finished, then Austin goes on mouthing in the same segment:
        # only the new frames are read, every word said once
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        for n in range(4, cut + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in long.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        for k in range(1, 70):              # past end_wait_s and final_wait_s: done
            out += core.step(t + 0.05 * k)
        t0 = t + 3.5
        for n in range(cut + 3, len(long['frame_t']) + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in long.items()}
            t = t0 + 0.04 * (n - cut)
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        for k in range(1, 80):
            out += core.step(t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        check('words after a finished stall are read from the new frames only, said once',
              said == ['i', 'want', 'water', 'i', 'want'], ' '.join(said))

        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        core.feed('cam.tick', reading(pieces, ['i', 'want', 'water'], seg_id=1, t0=100.0), 101.0)
        for k in range(1, 80):
            core.step(101.0 + 0.05 * k)
        out = core.feed('cam.tick', reading(pieces, ['i', 'want', 'water'], seg_id=1, t0=500.0), 501.0)
        check('a restarted camera reusing seg_id 1 still starts a new sentence',
              any(tp == 'dec.hyp' and m['utt_id'] == 2 for tp, m in out))
        # no repair window (the default): every word still said once, in order, with one final (no Claude here to say
        # Austin has finished, so the sentence closes after the no-Claude hold)
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.0, claude_end=False)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
            out += core.step(t)
        for k in range(1, 40):
            out += core.step(t + 0.05 * k)
            if k == 10:
                out += core.feed('cam.final', {'seg_id': 1, 'nbest': [{'text': 'I WANT WATER', 'score': 0.0}]},
                                  t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        check('no repair window: words said once, in order, one final', said == ['i', 'want', 'water'] and
              sum(1 for tp, m in out if tp == 'voice.say' and m['final']) == 1, ' '.join(said))

        # a repair changes the captions only: the voice says what locked, even inside the repair window
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        lk_msgs = [m for tp, m in out if tp == 'dec.lock']
        idx = sum(len(m['words']) for m in lk_msgs) - 1
        last = [x for m in lk_msgs for x in m['words']][-1]['w']
        out += core.feed('claude.repair', {'utt_id': 1, 'edits': [{'index': idx, 'from': last, 'to': 'what'}],
                                           'please_repeat': False}, t)
        for k in range(1, 60):
            out += core.step(t + 0.05 * k)
        said = [x for tp, m in out if tp == 'voice.say' for x in m['words']]
        caps = [m for tp, m in out if tp == 'dec.hyp'][-1]['locked']
        check('a repair, even inside the repair window, changes the captions, never the voice',
              said[idx] == last and caps[idx] == 'what', f"said {' '.join(said)}; captions {' '.join(caps)}")

        # Claude's please_repeat: ignored once every word of the sentence was spoken, kept while some are not
        for repair_s, name in ((0.0, 'all spoken'), (30.0, 'words still unspoken')):
            core = Fusion(search, Gate(), lock, svr=None, repair_s=repair_s, sentences=False)
            out = []
            for n in range(8, T + 1, 3):
                msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
                t = float(msg['frame_t'][-1]) + 0.05
                out += core.feed('cam.tick', msg, t)
                out += core.step(t)
            rep = core.feed('claude.repair', {'utt_id': 1, 'edits': [], 'please_repeat': True}, t + 0.01)
            asked = any(tp == 'dec.repeat' for tp, _ in rep)
            check(f"Claude's please repeat: {name}", asked == (name != 'all spoken'),
                  f"spoken {[x for tp, m in out if tp == 'voice.say' for x in m['words']]}, dec.repeat {asked}")

        # a word below the threshold at the end: Please repeat, and the voice doesn't say it
        lock = LockModel(tau0=0.999, tau_min=0.999, tau_max=0.999, agree_floor=0.0, repeat_floor=0.9999)
        core = Fusion(search, Gate(), lock, svr=None, repair_s=0.3, sentences=False)
        out = []
        for n in range(8, T + 1, 3):
            msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
            t = float(msg['frame_t'][-1]) + 0.05
            out += core.feed('cam.tick', msg, t)
        for k in range(1, 80):
            out += core.step(t + 0.05 * k)
        rep = [m for tp, m in out if tp == 'dec.repeat']
        check('an unclear sentence ends in dec.repeat', len(rep) == 1 and rep[0]['reason']['words'],
              str(rep[0]['reason'] if rep else None))
        check('the final voice.say still closes the sentence',
              any(tp == 'voice.say' and m['final'] for tp, m in out))

        # sentences (live): a pause ends the camera's segment but only holds the sentence; Claude says whether
        # Austin finished; the judge talking, Claude's silence (fail open) and the backstop close it too
        lock = LockModel(tau0=0.5, tau_min=0.5, tau_max=0.5, agree_floor=0.0)

        def mouth(core, words, seg, t0, out):
            full = reading(pieces, words, seg_id=seg, t0=t0)
            t = t0
            for n in range(8, len(full['frame_t']) + 1, 3):
                msg = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in full.items()}
                t = float(msg['frame_t'][-1]) + 0.05
                out += core.feed('cam.tick', msg, t)
                out += core.step(t)
            return t

        def wait(core, t, secs, out, answer=None, judge_at=None):
            """secs of still lips; answer: the P(finished) Claude gives each dec.pause; judge_at: the judge speaks."""
            for k in range(1, int(round(secs / 0.05)) + 1):
                now = t + 0.05 * k
                out += core.step(now)
                for tp, m in list(out):
                    if answer is not None and tp == 'dec.pause' and not m.get('answered'):
                        m['answered'] = True
                        out += core.feed('claude.end', {'utt_id': m['utt_id'], 'p_complete': answer}, now)
                if judge_at is not None and k == int(round(judge_at / 0.05)):
                    out += core.feed('partner.text', {'text': 'so what did you build', 'is_final': False}, now)
            return t + secs

        def events(out):
            return [(m['event'], m['reason']) for tp, m in out if tp == 'dec.sentence']

        core = Fusion(search, Gate(), lock, svr=None, use_hints=True, repair_s=0.3)
        out = []
        t = mouth(core, ['i', 'want'], 1, 100.0, out)
        t = wait(core, t, 1.0, out, answer=0.1)             # a pause; Claude: not finished
        t = mouth(core, ['water'], 2, t + 0.2, out)
        t = wait(core, t, 3.0, out, answer=0.9)             # Claude: finished
        ids = {m['utt_id'] for tp, m in out if tp in ('dec.hyp', 'dec.lock', 'voice.say')}
        hyps = [m for tp, m in out if tp == 'dec.hyp']
        said = [(m['offset'], m['words']) for tp, m in out if tp == 'voice.say' and m['words']]
        flat = [w for _, ws in said for w in ws]
        offsets_ok = all(off == sum(len(ws) for _, ws in said[:k]) for k, (off, ws) in enumerate(said))
        finals = [m for tp, m in out if tp == 'voice.say' and m['final']]
        check('a pause, then more words: one sentence, one utt_id on the bus', ids == {1}, str(sorted(ids)))
        check('its captions and voice go on from the first segment',
              hyps[-1]['locked'] == ['i', 'want', 'water'] and flat == ['i', 'want', 'water'] and offsets_ok,
              f"captions {hyps[-1]['locked']}; voice {said}")
        check('one final, after Claude said finished', len(finals) == 1 and events(out)[-1] ==
              ('close', 'Claude: finished'), str(events(out)))
        check('every hold and continuation is logged', [e for e, _ in events(out)] ==
              ['hold', 'continue', 'hold', 'close'], str(events(out)))
        out += core.feed('claude.repair', {'utt_id': 1, 'edits': [{'index': 2, 'from': 'water', 'to': 'waiter'}],
                                           'please_repeat': False}, t)
        check("Claude's repair finds the word by the sentence's index",
              core.sentences[1].utts[1].locked[0]['w'] == 'waiter')

        core = Fusion(search, Gate(), lock, svr=None, use_hints=True, repair_s=0.3)
        out = []
        t = mouth(core, ['i', 'want', 'water'], 1, 100.0, out)
        t = wait(core, t, 2.0, out, judge_at=0.6)           # no answer yet; the judge starts talking
        t = mouth(core, ['water'], 2, t + 0.5, out)
        t = wait(core, t, 4.0, out)                         # Claude never answers
        ev = events(out)
        check('the judge talking closes the held sentence', ('close', 'judge spoke') in ev, str(ev))
        check('the next words are a new sentence', {m['utt_id'] for tp, m in out if tp == 'dec.hyp'} == {1, 2})
        check('no answer from Claude: it closes anyway (fail open)', ev[-1] == ('close', 'Claude did not answer'),
              str(ev))

        core = Fusion(search, Gate(), lock, svr=None, use_hints=True, repair_s=0.3)
        out = []
        t = mouth(core, ['i', 'want'], 1, 100.0, out)
        t = wait(core, t, 9.0, out, answer=0.1)             # Claude keeps saying not finished
        pauses = [m for tp, m in out if tp == 'dec.pause']
        check("'not finished' holds and asks again as the silence grows, up to the backstop",
              len(pauses) >= 3 and pauses[-1]['silence_s'] > pauses[0]['silence_s'] and
              events(out)[-1] == ('close', 'backstop'), f"{len(pauses)} asks; {events(out)}")

        core = Fusion(search, Gate(), lock, svr=None, use_hints=False, repair_s=0.3)
        out = []
        t = mouth(core, ['i', 'want', 'water'], 1, 100.0, out)
        t = wait(core, t, 2.0, out)
        check('without Claude: a short hold, then it closes', events(out)[-1] == ('close', 'no Claude') and
              not any(tp == 'dec.pause' for tp, _ in out), str(events(out)))

        core = Fusion(search, Gate(), lock, svr=None, use_hints=False, repair_s=0.3, sentences=False)
        out = []
        t = mouth(core, ['i', 'want'], 1, 100.0, out)
        t = wait(core, t, 0.5, out)
        t = mouth(core, ['water'], 2, t + 0.2, out)
        t = wait(core, t, 3.0, out)
        check('offline (sentences off): one utt_id per segment, as before',
              {m['utt_id'] for tp, m in out if tp == 'dec.hyp'} == {1, 2} and
              len([m for tp, m in out if tp == 'voice.say' and m['final']]) == 2)
    from fusion.run import drain

    class Q:
        def __init__(self, items):
            self.items = list(items)

        def recv(self, timeout_ms=None):
            return self.items.pop(0) if self.items else (None, None)

    tick = lambda seg, k: ('cam.tick', {'seg_id': seg, 'k': k})  # noqa: E731
    got = drain(Q([tick(1, 2), ('emg.raw', {}), tick(1, 3), tick(2, 1)]), tick(1, 1))
    check('drain skips stale ticks of a segment, keeps the rest in order',
          [(t, m.get('k')) for t, m in got] == [('emg.raw', None), ('cam.tick', 3), ('cam.tick', 1)],
          str([(t, m.get('seg_id'), m.get('k')) for t, m in got]))
    # Claude alone can never lock a word (CLAUDE.md): a word only Claude supports fails the camera-EMG agreement rule,
    # with the default lock model and with the fitted one, however strongly Claude predicts it
    from fusion.offline import load_params
    _, fitted, meta = load_params()
    for name, lk in (('default', LockModel()), ('fitted', fitted if meta else LockModel())):
        f_claude = lockmod_features(joint=1.0, cam=0.0, emg=0.0, claude=1.0)
        check(f'Claude alone can never lock a word ({name} lock model)', not lk.may_lock(f_claude, 1.0, 0.0),
              f'agree {f_claude[FEATURES.index("agree")]}, claude feature {f_claude[FEATURES.index("claude")]}')
    print('FUSION UNITS PASS')


if __name__ == '__main__':
    main()
