"""The joint search (docs/SVR1_SPEC.md section 6): camera + EMG + language model + Claude's hints, every 100 ms.

One pass over the current segment:
  1. Camera prefix beam: a CTC prefix beam search (beam 16) over the camera's word pieces (cam.tick's top 32 per
     frame), with the conversational LM, Claude's hints and a length bonus added each time a word completes.
  2. Candidates: the beam's hypotheses, each with its words swapped one at a time for their lip look-alikes
     (same viseme sequence, bench/lookalikes.py), SVR-1's own greedy reading, and at segment end the camera's
     attention n-best (cam.final). Words already locked are kept as every candidate's first words.
  3. Rescoring: every candidate gets one score per judge, each computed on its own:
        cam    CTC log-likelihood of its word pieces under the camera's frames
        emg    S_emg, SVR-1's letter + phoneme CTC + verifier score (svr/live.py verify)
        lm     the conversational LM with the call cache, over its finished words (all of them at segment end)
        hint   the sum of Claude's prediction weights of its words
        final  at segment end only: the camera's own final-pass score for the candidates cam.final lists
     S = g_cam cam + g_emg emg + g_lm lm + beta hint + gamma |words| + g_final final, (g_cam, g_emg, g_lm) from the
     gate.
  4. Posteriors: softmax(S) over the candidates; a word's posterior at position i is the mass of the candidates
     with that word there. The camera-only and EMG-only posteriors (softmax of cam or emg alone) are the lock
     model's agreement features (fusion/lock.py).

As built, EMG enters by rescoring the beam's candidates and their look-alikes (one batched SVR-1 call per pass)
rather than inside every beam step: SVR-1 re-reads the whole segment every pass anyway, and grading the 16 to 64
finished candidates costs one GPU call instead of thousands of tiny ones.

The gate is a softmax over (camera, EMG, LM) from a linear layer on six signals, times one learned scale; it is
fitted on sessions 1 and 2 by making the true sentence win among the candidates (python -m fusion.offline fit).
"""
import functools
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
UNITS = os.path.join(WORK, 'third_party', 'lipflow', 'lipflow', 'unigram5000_units.txt')
SPM = os.path.join(WORK, 'third_party', 'auto_avsr', 'spm', 'unigram', 'unigram5000.model')
if not os.path.exists(SPM):   # lipread/setup.sh puts the same file here
    SPM = os.path.join(WORK, 'third_party', 'lipflow', 'models', 'lm', 'unigram5000.model')
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from svr.text import clean_text  # noqa: E402

BLANK, VOCAB = 0, 5049            # bus/schema.md: blank 0, ids 1..5047 = unigram5000_units.txt lines, <eos> 5048
NEG = -1e9
PRUNE = math.log(1e-4)            # a frame's non-blank piece below this is never a beam extension
FLOOR_MIN = -30.0                 # log-prob of a piece outside a frame's top 32, at least
GATE_FEATURES = ('cam_conf', 'face', 'activity', 'speak', 'emg_quality', 'rate')
# short words the camera often drops (they barely show on the lips): the search may insert them, unseen by the camera,
# and an unclear one never makes a sentence "Please repeat"
SHORT_WORDS = ('a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'for', 'and', 'is', 'are', 'was', 'it', 'i', 'you',
               'that', 'me', 'my', 'so', 'be', 'do', 'got', 'had', 'have', 'can', 'will', 'just', 'not', 'with', 'up')
JUDGES = ('cam', 'emg', 'lm')


def norm_words(text):
    return clean_text(text).split()


class Pieces:
    """The camera's word pieces: model ids <-> pieces ('▁' starts a word), and words -> ids via SentencePiece."""

    def __init__(self, units=UNITS, spm=SPM):
        self.piece = ['<blank>'] + [''] * (VOCAB - 1)
        self.id = {}
        with open(units, encoding='utf-8') as f:
            for line in f:
                p, i = line.split()
                self.piece[int(i)] = p
                self.id[p] = int(i)
        self.piece[VOCAB - 1] = '<eos>'
        self.starts = np.array([p.startswith('▁') for p in self.piece])
        self.text_piece = np.array([not (p.startswith('<') and p.endswith('>')) and p != '' for p in self.piece])
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(model_file=spm)
        self._cache = {}

    def word_ids(self, word):
        """One lowercase word -> its model ids (the first starts with '▁')."""
        ids = self._cache.get(word)
        if ids is None:
            ids = tuple(self.id.get(p, 1) for p in self.sp.encode(word.upper(), out_type=str))
            self._cache[word] = ids
        return ids

    def split_words(self, ids):
        """Model ids -> (words, pieces per word), lowercase, special pieces dropped."""
        words, spans = [], []
        for i in ids:
            if not self.text_piece[i]:
                continue
            p = self.piece[i]
            if p.startswith('▁') or not words:
                words.append(p.lstrip('▁'))
                spans.append([i])
            else:
                words[-1] += p
                spans[-1].append(i)
        keep = [k for k, w in enumerate(words) if norm_words(w)]
        return [' '.join(norm_words(words[k])) for k in keep], [tuple(spans[k]) for k in keep]


class CamReading:
    """One cam.tick: per-frame log-probs of any piece (top 32 as sent; the rest share what's left of the mass)."""

    def __init__(self, msg):
        self.ids = np.asarray(msg['topk_ids'], np.int64)
        self.logp = np.asarray(msg['topk_logp'], np.float64)
        self.blank = np.asarray(msg['blank_logp'], np.float64)
        self.frame_t = np.asarray(msg['frame_t'], np.float64)
        self.activity = np.asarray(msg.get('activity', np.zeros(len(self.frame_t))), np.float64)
        self.face_ok = bool(msg.get('face_ok', True))
        g = msg.get('greedy')                                   # the camera's own reading ('': no word seen; None:
        self.greedy = None if g is None else str(g).strip()     # not sent)
        self.seg_id, self.seg_start = int(msg['seg_id']), float(msg['seg_start'])
        T = len(self.ids)
        p = np.exp(self.logp)
        in_top = (self.ids == BLANK).any(1)
        rest = 1.0 - p.sum(1) - np.where(in_top, 0.0, np.exp(self.blank))
        self.floor = np.maximum(np.log(np.maximum(rest, 1e-30) / (VOCAB - 33)), FLOOR_MIN)
        self.floor = np.minimum(self.floor, self.logp[:, -1])  # never above the 32nd piece
        self.T = T

    def emissions(self, labels):
        """(T, len(labels)) log-probs of the given model ids at every frame."""
        labels = np.asarray(labels, np.int64)
        hit = self.ids[:, :, None] == labels[None, None, :]
        val = np.where(hit, self.logp[:, :, None], -np.inf).max(1)
        return np.where(hit.any(1), val, self.floor[:, None])

    def confidence(self, frames=12):
        """Mean top probability (blank included) over the last `frames` (0.5 s at 25 fps)."""
        top = np.maximum(self.logp[-frames:, 0], self.blank[-frames:])
        return float(np.exp(top).mean()) if len(top) else 0.0


def ctc_loglik(cam, token_seqs):
    """CTC log-likelihood of each token sequence (model ids) under the camera's frames; -inf where none fits."""
    import torch
    import torch.nn.functional as F
    K = len(token_seqs)
    out = np.full(K, -np.inf)
    rows = [k for k in range(K) if len(token_seqs[k])]
    for k in range(K):
        if not len(token_seqs[k]):
            out[k] = float(cam.blank.sum())
    if not rows:
        return out
    labels = sorted({t for k in rows for t in token_seqs[k]})
    col = {t: j + 1 for j, t in enumerate(labels)}
    em = cam.emissions(labels)                                   # (T, U)
    # each candidate gets its own small alphabet (blank + its distinct pieces), so CTC's repeat rule stays right
    width = 1 + max(len(set(token_seqs[k])) for k in rows)
    lp = np.full((cam.T, len(rows), width), -1e4, np.float32)
    lp[:, :, 0] = cam.blank[:, None]
    targets, lens = [], []
    for r, k in enumerate(rows):
        local = {}
        for t in token_seqs[k]:
            if t not in local:
                local[t] = len(local) + 1
                lp[:, r, local[t]] = em[:, col[t] - 1]
        targets += [local[t] for t in token_seqs[k]]
        lens.append(len(token_seqs[k]))
    nll = F.ctc_loss(torch.from_numpy(lp), torch.tensor(targets), torch.full((len(rows),), cam.T, dtype=torch.long),
                     torch.tensor(lens), blank=0, reduction='none', zero_infinity=False)
    out[rows] = -nll.double().numpy()
    return out


def viterbi_word_ends(cam, word_tokens):
    """Frame time at which each word's last piece ends in the camera's best alignment of these words (CTC Viterbi);
    None if they don't fit."""
    seq = [t for w in word_tokens for t in w]
    if not seq or len(seq) > cam.T:
        return None
    em = cam.emissions(sorted(set(seq)))
    col = {t: j for j, t in enumerate(sorted(set(seq)))}
    S = 2 * len(seq) + 1
    lab = [BLANK if s % 2 == 0 else seq[s // 2] for s in range(S)]
    e = np.array([[cam.blank[t] if lab[s] == BLANK else em[t, col[lab[s]]] for s in range(S)] for t in range(cam.T)])
    dp = np.full((cam.T, S), -np.inf)
    bp = np.zeros((cam.T, S), np.int64)
    dp[0, 0], dp[0, 1] = e[0, 0], e[0, 1]
    for t in range(1, cam.T):
        for s in range(S):
            best, arg = dp[t - 1, s], s
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if s >= 2 and lab[s] != BLANK and lab[s] != lab[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], s - 2
            dp[t, s], bp[t, s] = best + e[t, s], arg
    s = S - 1 if dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    if not np.isfinite(dp[-1, s]):
        return None
    last_frame = {}
    for t in range(cam.T - 1, -1, -1):
        if lab[s] != BLANK:
            last_frame.setdefault(s // 2, t)
        s = bp[t, s]
    ends, k = [], -1
    for w in word_tokens:
        k += len(w)
        ends.append(float(cam.frame_t[last_frame[k]]) + 0.02)
    return ends


class Gate:
    """(g_cam, g_emg, g_lm) = scale * softmax(W f + b), plus the hint weight beta, the length bonus gamma and the
    weight of the camera's own final-pass score (cam.final, end of segment only). t_cam and t_emg calibrate the
    camera-only and EMG-only posteriors (softmax(t * judge score)): SVR-1's raw sentence scores are far too sure of
    themselves to be read as probabilities. `unseen` is added once per word inserted without camera evidence."""

    def __init__(self, W=None, b=None, scale=1.5, beta=1.0, gamma=0.5, final=1.0, t_cam=1.0, t_emg=0.05,
                 unseen=-3.0):
        self.W = np.zeros((3, len(GATE_FEATURES))) if W is None else np.asarray(W, np.float64)
        self.b = np.log([0.6, 0.15, 0.25]) if b is None else np.asarray(b, np.float64)
        self.scale, self.beta, self.gamma, self.final = float(scale), float(beta), float(gamma), float(final)
        self.t_cam, self.t_emg = float(t_cam), float(t_emg)   # reader-only posteriors: softmax(t * that judge)
        self.unseen = float(unseen)   # per inserted word the camera didn't see

    def weights(self, f):
        z = self.W @ np.asarray(f, np.float64) + self.b
        z = np.exp(z - z.max())
        return self.scale * z / z.sum()

    def to_dict(self):
        return {'W': self.W.tolist(), 'b': self.b.tolist(), 'scale': self.scale, 'beta': self.beta,
                'gamma': self.gamma, 'final': self.final, 't_cam': self.t_cam, 't_emg': self.t_emg,
                'unseen': self.unseen, 'features': list(GATE_FEATURES)}

    @classmethod
    def from_dict(cls, d):
        return cls(d['W'], d['b'], d['scale'], d['beta'], d['gamma'], d.get('final', 1.0), d.get('t_cam', 1.0),
                   d.get('t_emg', 0.05), d.get('unseen', -3.0))


def gate_features(cam, emg_read, emg_quality):
    """The gate's six signals for this pass: camera confidence (last 0.5 s), face visible, mouth activity, EMG
    speaking probability, EMG quality (1 clean .. 0 railed or flat), rate (SVR-1's z-score)."""
    if cam is not None:
        conf, face, act = cam.confidence(), float(cam.face_ok), float(cam.activity[-12:].mean())
    else:
        conf, face, act = 0.0, 0.0, 0.0
    if emg_read is not None:
        speak, rate = float(np.mean(emg_read['speak_prob'][-43:])), float(emg_read['rate_z'])
    else:
        speak, rate = 0.0, 0.0
    return np.array([conf, face, act, speak, emg_quality, rate])


# Austin's known lines (fusion/script.txt): where the camera's reading of a span looks like one on the lips (sharing
# a word the camera read exactly), that span of the line races too, with a bonus. Claude can't know his exact words;
# the script does, for the lines he says on every demo. Free speech never lip-matches a line this closely.
SCRIPT_LIP = 0.6       # lip similarity (viseme sequences, 0 to 1) a span needs; random word pairs reach it 6% of the time
SCRIPT_MIN_WORDS = 3   # ... over at least this many words
SCRIPT_BONUS = 3.0     # added to a matching candidate's hint score (times beta)
SCRIPT_CANDS = 6       # spans per pass


# letters -> rough lip shapes, for words no lexicon knows (the camera's own inventions, "burinadoes")
LETTER_VISEME = {**{c: 'P' for c in 'bmp'}, **{c: 'F' for c in 'fv'}, **{c: 'T' for c in 'tdnl'},
                 **{c: 'S' for c in 'szcx'}, **{c: 'K' for c in 'kgq'}, 'j': 'SH', 'r': 'R', 'w': 'W', 'y': 'Y',
                 'h': 'H', 'a': 'A', 'e': 'E', 'i': 'E', 'o': 'O', 'u': 'O'}


@functools.lru_cache(maxsize=50000)
def visemes(word):
    """A word's lip shapes: its pronunciation (CMUdict + our supplement) as viseme classes, else from its letters
    (doubled letters once)."""
    from bench.lookalikes import VISEME
    from svr.phonemes import cmudict
    p = cmudict().get(word)
    if p:   # "h" is just breath on the lips, nothing like the k/g group bench.lookalikes files it under
        return tuple('H' if x == 'HH' else VISEME.get(x, x) for x in p)
    return tuple(v for i, ch in enumerate(word)
                 if (v := LETTER_VISEME.get(ch)) and not (i and word[i - 1] == ch))


@functools.lru_cache(maxsize=200000)
def lip_similarity(a, b):
    """How alike two word sequences (tuples) look on the lips: 1 - viseme edit distance / the longer (1 = identical)."""
    x = [v for w in a for v in visemes(w)]
    y = [v for w in b for v in visemes(w)]
    if not x or not y:
        return 0.0
    d = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        prev, d[0] = d[0], i
        for j, yj in enumerate(y, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (xi != yj))
    return 1.0 - d[-1] / max(len(x), len(y))


def load_script(path=None):
    """fusion/script.txt (one line per sentence, '#' comments) -> [word tuples]."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script.txt')
    try:
        with open(path, encoding='utf-8') as f:
            lines = [norm_words(x) for x in f if x.strip() and not x.lstrip().startswith('#')]
    except OSError:
        return []
    return [tuple(x) for x in lines if x]


class Cand:
    __slots__ = ('words', 'tokens', 'source')

    def __init__(self, words, tokens, source):
        self.words, self.tokens, self.source = tuple(words), tuple(tuple(t) for t in tokens), source

    def flat(self):
        """The camera's pieces; a word with no pieces was inserted without the camera seeing it."""
        return [t for w in self.tokens for t in w]

    def unseen(self):
        return sum(not t for t in self.tokens)


class JointSearch:
    def __init__(self, lm, pieces=None, lexicon=None, beam=16, lookalikes=6, swap_top=4, max_cands=80,
                 max_new=8, inserts=12, phrase_cands=8):
        """lm: fusion.lm.ConversationLM; lexicon: bench.lookalikes.Lexicon (None: no look-alike swaps)."""
        self.lm, self.pieces, self.lexicon = lm, pieces or Pieces(), lexicon
        self.beam, self.n_lookalikes, self.swap_top, self.max_cands, self.max_new = (beam, lookalikes, swap_top,
                                                                                     max_cands, max_new)
        self.inserts, self.phrase_cands = inserts, phrase_cands
        self._lk = {}
        self.script = []            # Austin's known lines (load_script); empty offline, so the fitted runs stay as they were
        self.script_hits = set()    # this pass's candidates that are a span of one

    # ------------------------------------------------------------------ language side
    def word_bonus(self, state, word, w, hints):
        """g_lm * ln P_LM(word | history) + beta * hint(word) + gamma: what finishing `word` adds; and the next state."""
        lp, nxt = self.lm.score(state, word)
        return w['lm'] * lp + w['beta'] * hints.get(word, 0.0) + w['gamma'], nxt

    # ------------------------------------------------------------------ 1. camera prefix beam
    def camera_beam(self, cam, w, hints):
        """CTC prefix beam over the camera's pieces; the LM, hints and length bonus join as each word completes.
        Returns [(ids tuple, beam score)] best first."""
        p = self.pieces
        start = self.lm.start()   # the beam reads the whole segment, locked words included; candidates() fixes them
        # prefix -> [log p_blank, log p_nonblank]; ext[prefix] = (language score, lm state, partial word pieces)
        beam = {(): [0.0, NEG]}
        ext = {(): (0.0, start, '')}
        top = cam.logp
        for t in range(cam.T):
            ids, lps, blank = cam.ids[t], top[t], cam.blank[t]
            frame = dict(zip(ids.tolist(), lps.tolist()))
            new = [(int(i), float(v)) for i, v in zip(ids, lps) if i != BLANK and v >= PRUNE and p.text_piece[i]]
            new = new[:self.max_new]
            nb = {}
            for pre, (pb, pn) in beam.items():
                tot = np.logaddexp(pb, pn)
                cur = nb.setdefault(pre, [NEG, NEG])
                cur[0] = np.logaddexp(cur[0], tot + blank)
                if pre:
                    last = pre[-1]
                    cur[1] = np.logaddexp(cur[1], pn + frame.get(last, cam.floor[t]))
                else:
                    last = None
                for tok, lp in new:
                    q = pre + (tok,)
                    if q not in ext:
                        lang, st, part = ext[pre]
                        piece = p.piece[tok]
                        if p.starts[tok]:   # a new word starts: the previous one is finished
                            words = norm_words(part)
                            if words:
                                add, st = self.word_bonus(st, words[0], w, hints)
                                lang += add
                            part = piece[1:]
                        else:
                            part += piece
                        ext[q] = (lang, st, part)
                    c = nb.setdefault(q, [NEG, NEG])
                    c[1] = np.logaddexp(c[1], (pb if tok == last else tot) + lp)
            ranked = sorted(nb.items(), key=lambda kv: w['cam'] * np.logaddexp(*kv[1]) + ext[kv[0]][0], reverse=True)
            beam = dict(ranked[:self.beam])
        return [(pre, w['cam'] * np.logaddexp(*v) + ext[pre][0]) for pre, v in
                sorted(beam.items(), key=lambda kv: w['cam'] * np.logaddexp(*kv[1]) + ext[kv[0]][0], reverse=True)]

    # ------------------------------------------------------------------ 2. candidates
    def lookalikes(self, word):
        if self.lexicon is None:
            return []
        out = self._lk.get(word)
        if out is None:
            alts = self.lexicon.lookalikes(word)
            if alts:
                st = self.lm.start()
                alts = sorted(alts, key=lambda a: -self.lm.score(st, a)[0])
            out = self._lk[word] = alts[:self.n_lookalikes]
        return out

    def candidates(self, cam, w, hints, locked=(), extra_texts=(), final=False, phrases=(), locked_tokens=None):
        """Candidates for this pass: the camera beam, extra texts (cam.final's n-best, SVR-1's greedy reading), the
        beam's best reading with short words from Claude's predicted phrases filled in, then short words inserted
        where the LM gains most, then look-alike swaps of the beam's best few and of the first extra. Inserted words
        have no camera pieces (unseen). Every candidate starts with the locked words (locked_tokens: their pieces,
        () for one that was inserted)."""
        p = self.pieces
        locked = tuple(locked)
        lock_tok = list(locked_tokens) if locked_tokens is not None else [p.word_ids(x) for x in locked]
        out, seen = [], set()

        def add(words, tokens, source):
            words, tokens = tuple(words), list(tokens)
            if words[:len(locked)] != locked:   # keep the locked words, then the candidate's own reading after them
                k = len(locked)
                words, tokens = locked + words[k:], lock_tok + tokens[k:]
            if words not in seen and len(out) < self.max_cands:
                seen.add(words)
                out.append(Cand(words, tokens, source))

        beam = self.camera_beam(cam, w, hints) if cam is not None else []
        for ids, _ in beam:
            words, spans = p.split_words(ids)
            add(words, spans, 'camera')
        for text in extra_texts:
            words = norm_words(text)
            if words:
                add(words, [p.word_ids(x) for x in words], 'extra')
        base = [c for c in out if c.source == 'camera'][:2]
        if base:
            for words, tokens in self.phrase_fill(base[0], phrases, len(locked))[:self.phrase_cands]:
                add(words, tokens, 'phrase')
        self.script_hits = set()
        if base and self.script:
            for span in self.script_spans(base[0].words, len(locked)):
                words = locked + span
                self.script_hits.add(words)
                add(words, lock_tok + [p.word_ids(x) for x in span], 'script')
        for words, tokens in self.short_inserts(base, len(locked), final)[:self.inserts]:
            add(words, tokens, 'insert')
        firsts = [c for c in out if c.source == 'camera'][:self.swap_top]
        firsts += [c for c in out if c.source == 'extra'][:1] if final else []   # cam.final's best
        for c in firsts:
            for i in range(len(locked), len(c.words)):   # the newest word too: its look-alikes race before it locks
                for alt in self.lookalikes(c.words[i]):
                    add(c.words[:i] + (alt,) + c.words[i + 1:],
                        c.tokens[:i] + (p.word_ids(alt),) + c.tokens[i + 1:], 'lookalike')
        if not out:
            add(locked, lock_tok, 'locked')
        return out

    def script_spans(self, best, k):
        """Spans of Austin's known lines that look like the camera's best reading after the k locked words: anchored
        where a line has a word the camera read exactly (len > 3), one word shorter, as long, or one word longer."""
        read = tuple(best[k:])
        if len(read) < SCRIPT_MIN_WORDS:
            return []
        out = {}
        anchors = {w: j for j, w in enumerate(read) if len(w) > 3}
        for line in self.script:
            for i, w in enumerate(line):
                j = anchors.get(w)
                if j is None:
                    continue
                s = i - j
                for n in (len(read) - 1, len(read), len(read) + 1):
                    span = line[max(0, s):max(0, s) + n]
                    if len(span) >= SCRIPT_MIN_WORDS and span not in out:   # the exact reading too: its bonus
                        lip = lip_similarity(span, read)
                        if lip >= SCRIPT_LIP:
                            out[span] = lip
        return sorted(out, key=lambda x: -out[x])[:SCRIPT_CANDS]

    def short_inserts(self, base, start, final):
        """(words, tokens) of each base reading with one short word inserted (unseen), best LM gain first."""
        out = []
        for c in base:
            ref = self.lm_score(c.words, final)
            last = len(c.words) if final else len(c.words) - 1   # mid-sentence, never after the unfinished word
            for g in range(start, last + 1):
                for sw in SHORT_WORDS:
                    if (g > 0 and c.words[g - 1] == sw) or (g < len(c.words) and c.words[g] == sw):
                        continue
                    words = c.words[:g] + (sw,) + c.words[g:]
                    out.append((self.lm_score(words, final) - ref, words, c.tokens[:g] + ((),) + c.tokens[g:]))
        out.sort(key=lambda x: -x[0])
        return [(words, tokens) for _, words, tokens in out]

    @staticmethod
    def phrase_fill(c, phrases, start):
        """Claude predicted a phrase ("a little nervous") and the camera read part of it ("little nervous"): the
        reading with the phrase's missing short words put back (unseen). At most two words filled per phrase."""
        out = []
        words = list(c.words)
        for ph in phrases:
            if len(ph) < 2:
                continue
            for s0 in range(start, len(words)):
                if words[s0] not in ph:
                    continue
                k0 = ph.index(words[s0])
                new, toks = words[:s0], list(c.tokens[:s0])
                fill = [x for x in ph[:k0]]
                ok = all(x in SHORT_WORDS for x in fill)
                new += fill
                toks += [()] * len(fill)
                i, k = s0, k0
                while ok and k < len(ph):
                    if i < len(words) and words[i] == ph[k]:
                        new.append(words[i])
                        toks.append(c.tokens[i])
                        i += 1
                    elif ph[k] in SHORT_WORDS:
                        new.append(ph[k])
                        toks.append(())
                        fill.append(ph[k])
                    else:
                        break
                    k += 1
                if ok and 0 < len(fill) <= 2 and k > k0 + 1:
                    out.append((tuple(new + words[i:]), toks + list(c.tokens[i:])))
                break
        return out

    def extend(self, cands, texts, locked=()):
        """Add texts (SVR-1's greedy reading) to a candidate list, locked words first, without duplicates."""
        locked, seen = tuple(locked), {c.words for c in cands}
        for text in texts:
            words = tuple(norm_words(text))
            words = locked + words[len(locked):]
            if words and words not in seen and len(cands) < self.max_cands + len(texts):
                seen.add(words)
                cands.append(Cand(words, [self.pieces.word_ids(x) for x in words], 'emg'))
        return cands

    # ------------------------------------------------------------------ 3. rescoring
    def lm_score(self, words, final):
        """LM log-prob of the finished words (all of them + end of sentence when final)."""
        st, total = self.lm.start(), 0.0
        n = len(words) if final else len(words) - 1
        for x in words[:max(n, 0)]:
            lp, st = self.lm.score(st, x)
            total += lp
        return total + (self.lm.end(st) if final and words else 0.0)

    def judge_scores(self, cands, cam, emg_fn, hints, final, nbest=()):
        """(K, 7) array per candidate: cam, emg, lm, hint, number of words, the camera's final-pass score
        (cam.final's n-best score, its worst minus 5 for a candidate it doesn't list; 0 before the segment ends), and
        the number of words inserted without camera pieces.
        emg_fn(texts) -> S_emg or None; nbest: cam.final's [{'text', 'score'}]."""
        K = len(cands)
        cam_s = ctc_loglik(cam, [c.flat() for c in cands]) if cam is not None else np.zeros(K)
        emg_s = None
        if emg_fn is not None:
            emg_s = emg_fn([' '.join(c.words) for c in cands])
        emg_s = np.zeros(K) if emg_s is None else np.asarray(emg_s, np.float64)
        lm_s = np.array([self.lm_score(c.words, final) for c in cands])
        hint_s = np.array([sum(hints.get(x, 0.0) for x in set(c.words)) + SCRIPT_BONUS * (c.words in self.script_hits)
                           for c in cands])
        n = np.array([len(c.words) for c in cands], np.float64)
        fin = np.zeros(K)
        listed = {}
        for h in nbest:
            words = tuple(norm_words(h['text']))
            listed[words] = max(listed.get(words, -1e9), float(h['score']))
        if listed:
            floor = min(listed.values()) - 5.0
            fin = np.array([listed.get(c.words, floor) for c in cands])
        unseen = np.array([c.unseen() for c in cands], np.float64)
        return fix_infeasible(np.stack([cam_s, emg_s, lm_s, hint_s, n, fin, unseen], 1))

    @staticmethod
    def combine(scores, w):
        """S per candidate from the judge scores and this pass's weights."""
        return (w['cam'] * scores[:, 0] + w['emg'] * scores[:, 1] + w['lm'] * scores[:, 2] + w['beta'] * scores[:, 3]
                + w['gamma'] * scores[:, 4] + w.get('final', 0.0) * scores[:, 5] + w.get('unseen', 0.0) * scores[:, 6])


def fix_infeasible(scores):
    """A candidate one judge can't fit (CTC has no path: too many pieces for the frames) gets that judge's worst
    feasible score minus 20 nats, instead of -inf / -1e4, so one judge never vetoes alone."""
    s = scores.copy()
    for j in (0, 1):
        col = s[:, j]
        bad = ~np.isfinite(col) | (col <= -1e3)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = col[~bad].min() - 20.0
    return s


def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def word_posteriors(cands, probs, start=0):
    """For each position i >= start of the best candidate: {word: posterior mass at i}."""
    best = cands[int(np.argmax(probs))]
    out = []
    for i in range(start, len(best.words)):
        d = {}
        for c, q in zip(cands, probs):
            if i < len(c.words):
                d[c.words[i]] = d.get(c.words[i], 0.0) + q
        out.append(d)
    return best, out
