"""How the camera reads while a sentence is still being mouthed, not only once it is finished.

lipread/adapt.py scores whole clips, which is the last tick only. Live, the joint search locks words from every 100 ms
tick, each a read of an unfinished segment. This replays every recorded sentence tick by tick (the run.py due points,
the front-end cache, the same LiveModel) and scores each reading against the true words that were already mouthed:

    word ends     CTC forced alignment of the true sentence on a whole-clip read by an aligner model that never saw
                  the session (lipread/adapt.py's cross-session adapters), so every model is timed against the same
                  word ends
    seen WER      over every tick: errors among the true words already finished at that tick (substituted or missing),
                  so a word still on the lips never counts, averaged over ticks
    stable lag    per true word: time from its end on the lips to the first tick from which the reading has it right on
                  every later tick; words wrong on the last tick count as never
    flips         changes of the word read at a finished true word's place, per finished word

Readings scored: 'greedy' (the CTC best path, as cam.tick's text) and 'rescore' (a CTC prefix beam over the tick's
top-k, its n-best rescored by the attention decoder in one batched teacher-forced pass: the two-pass decoding of U2,
https://arxiv.org/abs/2012.05481), at several CTC weights, chosen afterwards from the stored scores.

    venv-live/bin/python -m lipread.streaming_eval --sessions session_02_EMG-dropped-frames \\
        --adapter data/face_adapter/austin_s1only.pth --aligner data/face_adapter/austin_s1only.pth
Never on session 3 (the test session).
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

import lipread
from bus import bus_text
from lipread.adapt import crops, load_alignments, norm, save_alignments, sentence_clips
from lipread.rescore import LOG0, attention_scores_multi, combined, ctc_prefix_beam, lm_scores
from lipread.run import due_after
from lipread.types import FPS

CTC_WEIGHTS = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)


def tokenize(reader, text):
    """True sentence -> model token ids, as lipflow/train_vsr.py's targets."""
    from lipflow.train_vsr import _targets
    return _targets(reader, text)


def token_words(reader, ids):
    """Token ids -> per token, the index of the word it belongs to ('▁' starts a word); and the word texts."""
    word_of, words, cur = [], [], ''
    for i in ids:
        piece = reader.token_list[i]
        if piece.startswith('▁') or not words:
            words.append('')
        words[-1] += piece.lstrip('▁')
        word_of.append(len(words) - 1)
    return word_of, [norm(w) for w in words]


def forced_align(logp, ids):
    """CTC Viterbi alignment of ids on logp (T, V) -> per frame the token position (0..U-1) or -1 for blank."""
    T, U = len(logp), len(ids)
    S = 2 * U + 1
    lab = np.zeros(S, np.int64)
    lab[1::2] = ids
    emit = logp[:, lab]                                     # (T, S)
    dp = np.full((T, S), LOG0)
    bp = np.zeros((T, S), np.int8)
    dp[0, 0] = emit[0, 0]
    if S > 1:
        dp[0, 1] = emit[0, 1]
    skip = np.zeros(S, bool)
    skip[3::2] = lab[3::2] != lab[1:-2:2]                    # a label may skip the blank before it unless repeated
    for t in range(1, T):
        prev = dp[t - 1]
        stay, step = prev, np.concatenate([[LOG0], prev[:-1]])
        jump = np.where(skip, np.concatenate([[LOG0, LOG0], prev[:-2]]), LOG0)
        cand = np.stack([stay, step, jump])
        bp[t] = cand.argmax(0)
        dp[t] = cand.max(0) + emit[t]
    s = S - 1 if S == 1 or dp[-1, S - 1] >= dp[-1, S - 2] else S - 2
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    return np.where(path % 2 == 1, path // 2, -1)


def word_spans(path, word_of, n_words):
    """Per word (first frame, last frame + 1) from a token-position path; a word with no frame gets its neighbour's."""
    first, last = [None] * n_words, [None] * n_words
    for t, u in enumerate(path):
        if u >= 0:
            w = word_of[u]
            first[w] = t if first[w] is None else first[w]
            last[w] = t + 1
    for w in range(n_words):
        if first[w] is None:
            first[w] = last[w - 1] if w and last[w - 1] is not None else 0
            last[w] = first[w] + 1
    return first, last


@torch.inference_mode()
def attention_scores(model, enc, hyps):
    """Attention-decoder log-prob of each hypothesis (token tuples, eos appended) given enc (T, D): one batched
    teacher-forced pass of the decoder."""
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    m = model.reader.model
    dev = next(m.decoder.parameters()).device
    eos = m.eos
    L = max(len(h) for h in hyps) + 1
    ys_in = torch.full((len(hyps), L), eos, dtype=torch.long)
    ys_out = torch.full((len(hyps), L), -1, dtype=torch.long)
    for j, h in enumerate(hyps):
        ys_in[j, 1:len(h) + 1] = torch.tensor(h, dtype=torch.long) if h else ys_in[j, 1:1]
        ys_out[j, :len(h)] = torch.tensor(h, dtype=torch.long) if h else ys_out[j, :0]
        ys_out[j, len(h)] = eos
    ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
    mem = enc.float().unsqueeze(0).expand(len(hyps), -1, -1).to(dev)
    mask = subsequent_mask(L, device=dev).unsqueeze(0)
    pred, _ = m.decoder(ys_in, mask, mem, None)
    logp = pred.float().log_softmax(-1)
    valid = ys_out >= 0
    tok = logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok * valid).sum(-1).cpu().numpy()


def align_words(ref, hyp):
    """Levenshtein alignment -> for each ref word, the hyp word it aligns to (None when deleted)."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), np.int32)
    d[:, 0], d[0, :] = np.arange(R + 1), np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [None] * R, R, H
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = hyp[j - 1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def edit_errors(ref, hyp):
    """Word edit distance."""
    R, H = len(ref), len(hyp)
    prev = list(range(H + 1))
    for i in range(1, R + 1):
        cur = [i] + [0] * H
        for j in range(1, H + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[H]


def score_clip(words, starts, ends, ticks):
    """ticks: [(frames read, hyp words)] in order. Returns per-clip sums for the streaming metrics."""
    n = len(words)
    seen_err = seen_n = 0
    right = []                                              # per tick, per word: right (True/False) or None (unfinished)
    for p, hyp in ticks:
        visible = [w for w in range(n) if starts[w] < p]
        aligned = align_words([words[w] for w in visible], hyp)
        row = [None] * n
        for k, w in enumerate(visible):
            if ends[w] <= p:
                row[w] = aligned[k] == words[w]
                seen_err += not row[w]
                seen_n += 1
        right.append((p, row, {w: aligned[k] for k, w in enumerate(visible) if ends[w] <= p}))
    lags, never, flips, fin = [], 0, 0, 0
    for w in range(n):
        rows = [(p, r[w], a.get(w)) for p, r, a in right if r[w] is not None]
        if not rows:
            never += 1
            continue
        fin += 1
        flips += sum(rows[k][2] != rows[k - 1][2] for k in range(1, len(rows)))
        if not rows[-1][1]:
            never += 1
            continue
        k = len(rows) - 1
        while k > 0 and rows[k - 1][1]:
            k -= 1
        lags.append((rows[k][0] - ends[w]) / FPS)
    final_err = edit_errors(words, ticks[-1][1])
    return dict(seen_err=seen_err, seen_n=seen_n, lags=lags, never=never, flips=flips, fin=fin, n=n,
                final_err=final_err)


def summarize(parts):
    lags = [x for p in parts for x in p['lags']]
    n = sum(p['n'] for p in parts)
    return dict(final_wer=sum(p['final_err'] for p in parts) / n,
                seen_wer=sum(p['seen_err'] for p in parts) / max(1, sum(p['seen_n'] for p in parts)),
                lag_p50=statistics.median(lags) if lags else math.nan,
                lag_p90=float(np.percentile(lags, 90)) if lags else math.nan,
                never=sum(p['never'] for p in parts) / n,
                flips_per_word=sum(p['flips'] for p in parts) / max(1, sum(p['fin'] for p in parts)))


def words_of(reader, tokens):
    return bus_text(''.join(reader.token_list[i] for i in tokens)).split()


def tick_points(T, tick_ms=100):
    """The frame counts run.py reads a T-frame segment at: its due points, then the whole segment."""
    points, done = [], 0
    while True:
        p = min(due_after(done, tick_ms), T)
        points.append(p)
        if p >= T:
            return points
        done = p


def truth(reader, aligner, clips, cache_path=None):
    """Per clip (words, starts, ends): the true words and their frame spans from the aligner's whole-clip read.
    cache_path: a JSON file keyed by clip id (the aligner is part of the caller's file name)."""
    cached = load_alignments(cache_path)
    out = []
    for c in clips:
        ids = tokenize(reader, c['text'])
        word_of, words = token_words(reader, ids)
        hit = cached.get(c['id'])
        if hit is None or hit['words'] != words:
            with torch.inference_mode():
                enc = aligner.encode(c['rois'])
                logp = aligner.ctc.log_softmax(enc.float().unsqueeze(0))[0].cpu().numpy()
            starts, ends = word_spans(forced_align(logp, ids), word_of, len(words))
            hit = cached[c['id']] = dict(words=words, starts=[int(x) for x in starts], ends=[int(x) for x in ends])
        out.append((hit['words'], hit['starts'], hit['ends']))
    if cache_path:
        save_alignments(cache_path, cached)
    return out


def rescored(reader, ctc, att, hyps, ctc_weights):
    return {w: words_of(reader, hyps[int(np.argmax(w * ctc + (1 - w) * att))]) for w in ctc_weights}


def readings_per_tick(model, clips, tick_ms=100, beam=8, per_frame=8, rescore=True, ctc_weights=CTC_WEIGHTS):
    """As live: one read per tick with the front-end cache. -> per clip [(p, greedy words, {w: rescored words})],
    and per-tick timings."""
    from lipread.model import FrontendCache
    reader = model.reader
    out, t_read, t_beam, t_att = [], [], [], []
    for c in clips:
        cache, ticks = FrontendCache(), []
        for p in tick_points(len(c['rois']), tick_ms):
            t0 = time.perf_counter()
            r = model.read([x for x in c['rois'][:p]], keep_enc=rescore, cache=cache)
            t_read.append((time.perf_counter() - t0) * 1000)
            res = {}
            if rescore:
                t0 = time.perf_counter()
                nbest = ctc_prefix_beam(r.topk_ids, r.topk_logp, r.blank_logp, beam, per_frame)
                t1 = time.perf_counter()
                hyps = [h for h, _ in nbest]
                att = attention_scores(model, r.enc, hyps)
                t_att.append((time.perf_counter() - t1) * 1000)
                t_beam.append((t1 - t0) * 1000)
                res = rescored(reader, np.array([s for _, s in nbest]), att, hyps, ctc_weights)
            ticks.append((p, r.greedy.split(), res))
        out.append(ticks)
    timing = dict(read_ms_p50=statistics.median(t_read), read_ms_p95=float(np.percentile(t_read, 95)))
    if t_beam:
        timing.update(prefix_beam_ms_p50=statistics.median(t_beam), prefix_beam_ms_p95=float(np.percentile(t_beam, 95)),
                      attention_ms_p50=statistics.median(t_att), attention_ms_p95=float(np.percentile(t_att, 95)))
    return out, timing


@torch.inference_mode()
def read_same_length(model, rois_list):
    """Whole-segment reads of B clips of the same length in one batch (no padding, so each equals its own read up
    to fp16 rounding): -> (topk ids, topk logp, blank logp, greedy text, enc) per clip, as LiveModel.read."""
    from bus import BLANK, TOPK
    e = model.encoder
    x = torch.stack([model.reader.to_tensor(np.asarray(r)) for r in rois_list]).to(model.device, model.dtype)
    h = e.embed(e.frontend(x))
    h, _ = e.encoders(h, None)
    if isinstance(h, tuple):
        h = h[0]
    if e.normalize_before:
        h = e.after_norm(h)
    logp = model.ctc.log_softmax(h.float())
    top = logp.topk(TOPK, dim=-1)
    ids = top.indices.int().cpu().numpy()
    vals = top.values.cpu().numpy()
    blank = logp[..., BLANK].cpu().numpy()
    return [(ids[b], vals[b], blank[b], model.text(ids[b, :, 0]), h[b]) for b in range(len(rois_list))]


def readings_batched(model, clips, tick_ms=100, beam=8, per_frame=8, rescore=True, lm=None, max_frames=1500):
    """The same readings as readings_per_tick, with every read of the same length (across clips) in one batch of at
    most max_frames frames: the whole prefix is encoded, which FrontendCache matches (tests/lipread/test_run.py).
    -> per clip [(p, greedy words, n-best)], n-best = (hyps, ctc, att, lm or None) for selection afterwards."""
    jobs = {}
    for ci, c in enumerate(clips):
        for k, p in enumerate(tick_points(len(c['rois']), tick_ms)):
            jobs.setdefault(p, []).append((ci, k))
    out = [dict() for _ in clips]
    t0 = time.perf_counter()
    for p, items in sorted(jobs.items()):
        step = max(1, max_frames // p)
        for i in range(0, len(items), step):
            chunk = items[i:i + step]
            reads = read_same_length(model, [clips[ci]['rois'][:p] for ci, _ in chunk])
            res = [None] * len(chunk)
            if rescore:
                nbests = [ctc_prefix_beam(ids, lp, bl, beam, per_frame) for ids, lp, bl, _, _ in reads]
                hyps = [[h for h, _ in nb] for nb in nbests]
                atts = attention_scores_multi(model, torch.stack([r[4] for r in reads]), hyps)
                lms = lm_scores(lm, hyps, model.reader.model.eos) if lm is not None else [None] * len(chunk)
                res = [(hs, np.array([s for _, s in nb]), att, l) for hs, nb, att, l in zip(hyps, nbests, atts, lms)]
            for (ci, k), r, rs in zip(chunk, reads, res):
                out[ci][k] = (p, r[3].split(), rs)
    seconds = time.perf_counter() - t0
    return [[d[k] for k in sorted(d)] for d in out], dict(batched_seconds=seconds, reads=sum(map(len, jobs.values())))


def pick(reader, nbest, w_ctc, w_lm=0.0, bonus=0.0):
    """Best words of a tick's n-best: w_ctc x CTC + (1 - w_ctc) x attention + w_lm x LM + bonus x tokens."""
    hyps = nbest[0]
    return words_of(reader, hyps[int(np.argmax(combined(*nbest, w_ctc, w_lm, bonus)))])


def evaluate(model, aligner, clips, tick_ms=100, beam=8, per_frame=8, rescore=True, configs=None, lm=None,
             batched=True, align_cache=None):
    """clips: [{'rois', 'text', 'id'}]. configs: {name: (w_ctc, w_lm, bonus)} to score from the same n-best (default:
    the CTC weights alone). Returns ({reading: summary}, timing dict, per-clip readings)."""
    reader = model.reader
    spans = truth(reader, aligner, clips, align_cache)
    configs = configs or {f'rescore_ctc{w}': (w, 0.0, 0.0) for w in CTC_WEIGHTS}
    t0 = time.perf_counter()
    if batched:
        readings, timing = readings_batched(model, clips, tick_ms, beam, per_frame, rescore, lm)
    else:
        readings, timing = readings_per_tick(model, clips, tick_ms, beam, per_frame, rescore, CTC_WEIGHTS)
    timing['readings_seconds'] = time.perf_counter() - t0
    parts = {'greedy': [score_clip(w, s, e, [(p, g) for p, g, _ in r]) for (w, s, e), r in zip(spans, readings)]}
    if rescore:
        for name, (wc, wl, b) in configs.items():
            if batched:
                get = lambda nb: pick(reader, nb, wc, wl, b)
            else:
                get = lambda rs: rs[wc]
            parts[name] = [score_clip(w, s, e, [(p, get(nb)) for p, _, nb in r]) for (w, s, e), r in zip(spans, readings)]
    return {k: summarize(v) for k, v in parts.items()}, timing, readings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sessions', nargs='+', required=True)
    ap.add_argument('--adapter', default=None, help='face adapter to score (none: base Auto-AVSR)')
    ap.add_argument('--aligner', required=True, help='adapter that times the true words (never trained on --sessions)')
    ap.add_argument('--clips', type=int, default=None, help='only the first N clips')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--no-rescore', action='store_true')
    ap.add_argument('--beam', type=int, default=8)
    ap.add_argument('--lm', action='store_true', help="also score the n-best with Auto-AVSR's LM, over a weight grid")
    ap.add_argument('--per-tick', action='store_true', help='one read per tick as live (slow) instead of batched')
    ap.add_argument('--out', default=None, help='JSON of the results')
    args = ap.parse_args(argv)
    if any('session_03' in s for s in args.sessions):
        ap.error('session 3 is the test session: never score development runs on it')
    from lipread.model import LiveModel
    face = os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task')
    rows = sentence_clips(args.sessions)[:args.clips]
    clips = [dict(rois=crops(r, face), text=r['prompt'], id=r['id']) for r in rows]
    aligner = LiveModel(warmup=(), face_adapter=args.aligner)
    # the batched path never times single reads, so the 200-length warm-up (cuDNN / kernel first-call cost only) is waste
    model = LiveModel(half=not args.fp32, face_adapter=args.adapter, warmup=() if not args.per_tick else range(1, 201))
    t0 = time.time()
    stem = os.path.splitext(os.path.basename(args.aligner))[0]
    cache = os.path.join(lipread.WORK, 'data', 'streaming_eval', f'align_{stem}.json')
    lm, configs = None, None
    if args.lm:
        from lipread.final import load_lm
        lm = load_lm(len(model.token_list)).to(next(model.reader.model.decoder.parameters()).device)
        configs = {f'ctc{wc}_lm{wl}_b{b}': (wc, wl, b) for wc in (0.1, 0.3, 0.5) for wl in (0.0, 0.1, 0.2, 0.3, 0.5)
                   for b in (0.0, 0.5, 1.0)}
    res, timing, _ = evaluate(model, aligner, clips, beam=args.beam, rescore=not args.no_rescore, configs=configs,
                              lm=lm, batched=not args.per_tick, align_cache=cache)
    out = dict(sessions=args.sessions, adapter=args.adapter, aligner=args.aligner, clips=len(clips),
               fp16=not args.fp32, results=res, timing=timing, seconds=time.time() - t0)
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
