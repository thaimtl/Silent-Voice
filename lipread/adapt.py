"""Austin's face adapter: Lipflow's speaker adaptation (lipflow/train_vsr.py) on the team's recordings.

Only the visual front end, the embed and Conformer block 0 learn (Lipflow's default scope, 6 epochs, lr 1e-4, its
augmentation); the rest of Auto-AVSR, the CTC head, the decoder and the LM keep their weights, so a read costs the
same (lipread/MEASURED.md). Crops are made as live: the recording shrunk to --process-size, then LiveCropper.

    venv-live/bin/python -m lipread.adapt --sessions session_01 --test-every 5    # 4/5 train, 1/5 held out, report
    venv-live/bin/python -m lipread.adapt --sessions session_01 --test-sessions session_02   # held out: a new session
    venv-live/bin/python -m lipread.adapt --sessions session_01 session_02         # train on everything, save
    venv-live/bin/python -m lipread.run --cam 0 --face-adapter data/face_adapter/austin.pth

The default recipe is C (--recipe c: --cut-p 0.5 --ctc-weight 0.5 --train-ctc --extra-blocks 2, seen WER 0.275 vs
0.314 for Lipflow's, 3 seeds, data/avsr_experiments/HANDOFF.md); --recipe lipflow is Lipflow's own, and any of the four
flags overrides its part of the recipe. The cut augmentation needs word ends: a forced alignment on the base model,
cached per clip in data/streaming_eval/align_base.json (lipread/streaming_eval.py's format).

With --test-every or --test-sessions, the held-out clips are scored before and after: the streaming seen WER
(lipread/streaming_eval.py, greedy, tick by tick as live) gates saving, and the final pass (beam 5 + LM) is reported
beside it. The adapter is saved only if the seen WER is not worse (--force saves anyway). Without a held-out set
neither the Beam nor the LM is built. Rest clips are skipped. A session is a folder under recordings/ (a renamed folder works: clips are read from the
folder given, not the manifest's session field). Crops are cached per clip in data/face_adapter/crops/. The adapter
goes to data/ in RECLAIM_WORKDIR (outside git); never train on session 3, the test session.
"""
import argparse
import json
import os
import re
import sys
import time

import jiwer
import numpy as np
import torch

import lipread
from bus import bus_text
from lipread.model import FACE_ADAPTER_PREFIXES
from lipread.source import RECORDINGS, ReplaySource

RECIPES = {
    'c': dict(cut_p=0.5, ctc_weight=0.5, train_ctc=True, extra_blocks=2),
    'lipflow': dict(cut_p=0.0, ctc_weight=0.1, train_ctc=False, extra_blocks=0),
}
ALIGN_CACHE = os.path.join(lipread.WORK, 'data', 'streaming_eval', 'align_base.json')
DEFAULT_OUT = os.path.join(lipread.WORK, 'data', 'face_adapter', 'austin.pth')
CROP_CACHE = os.path.join(lipread.WORK, 'data', 'face_adapter', 'crops')


def norm(text):
    """Text as WER compares it: uppercase, A-Z 0-9 and ', hyphens split words."""
    return ' '.join(re.sub(r"[^A-Z0-9' ]", ' ', text.upper().replace('-', ' ')).split())


def sentence_clips(sessions, root=RECORDINGS):
    """Manifest rows with a prompt (rest clips out), in id order, across the sessions. row['dir'] is the folder."""
    rows = []
    for s in sessions:
        with open(os.path.join(root, s, 'manifest.jsonl')) as f:
            rows += [dict(json.loads(line), dir=s) for line in f if line.strip()]
    return sorted((r for r in rows if r.get('prompt') and '_rest' not in r['id']), key=lambda r: r['id'])


def split(rows, test_every):
    """(train, test): every test_every-th clip held out (0: all train). Spreads the held-out clips over the
    session, so every pace and delivery is in both."""
    if not test_every:
        return rows, []
    return ([r for i, r in enumerate(rows) if (i + 1) % test_every],
            [r for i, r in enumerate(rows) if not (i + 1) % test_every])


def crops(row, face_model, process_size=(1280, 720), root=RECORDINGS, cache=CROP_CACHE):
    """(T, 96, 96) uint8 mouth crops at 25 fps of one recording, as the live reader makes them (cached in cache/,
    None: no cache)."""
    from lipread.crop import LiveCropper
    from lipread.run import fit
    path = cache and os.path.join(cache, f"{row['id']}_{process_size[0]}x{process_size[1]}.npy")
    if path and os.path.exists(path):
        return np.load(path)
    src = ReplaySource(row['id'], realtime=False, root=os.path.join(root, row.get('dir', row['session'])))
    cropper = LiveCropper(face_model)
    try:
        out = [c.roi96 for f in src.frames() for c in cropper.push(fit(f, process_size))]
        out += [c.roi96 for c in cropper.flush()]
    finally:
        src.close()
        cropper.close()
    out = np.stack(out)
    if path:
        os.makedirs(cache, exist_ok=True)
        np.save(path + '.tmp.npy', out)
        os.replace(path + '.tmp.npy', path)
    return out


def score(reader, beam, clips):
    """(greedy WER, beam WER, exact greedy, exact beam) over clips [{'rois', 'text'}]."""
    ref, greedy, final = [], [], []
    for c in clips:
        with torch.inference_mode():
            enc = reader.encode(c['rois'])
            best = reader.model.ctc.argmax(enc.unsqueeze(0))[0].tolist()
        pieces = [reader.token_list[i] for j, i in enumerate(best) if j == 0 or i != best[j - 1]]
        ref.append(norm(c['text']))
        greedy.append(norm(bus_text(''.join(pieces))))
        final.append(norm(beam(enc=enc)[0][0]))
    return (jiwer.wer(ref, greedy), jiwer.wer(ref, final), sum(a == b for a, b in zip(ref, greedy)),
            sum(a == b for a, b in zip(ref, final)))


def streaming_seen(clips, aligner, adapter=None, cache=ALIGN_CACHE):
    """Streaming metrics (lipread/streaming_eval.py, greedy reading tick by tick, fp16 as live) of the base model, or
    of the model with this adapter file, on clips [{'rois', 'text', 'id'}]: dict(seen_wer, final_wer, lag_p50, ...).
    aligner: a LiveModel that times the true words. The model is freed on return."""
    from lipread.model import LiveModel
    from lipread.streaming_eval import evaluate
    model = LiveModel(half=True, warmup=(), face_adapter=adapter)
    try:
        res, _, _ = evaluate(model, aligner, clips, rescore=False, align_cache=cache)
    finally:
        del model
        torch.cuda.empty_cache()
    return res['greedy']


def resolve_recipe(recipe, cut_p=None, ctc_weight=None, train_ctc=None, extra_blocks=None):
    """RECIPES[recipe] with any flag that was given (not None) on top."""
    r = dict(RECIPES[recipe])
    r.update({k: v for k, v in dict(cut_p=cut_p, ctc_weight=ctc_weight, train_ctc=train_ctc,
                                    extra_blocks=extra_blocks).items() if v is not None})
    return r


def scope_params(reader, extra_blocks=0, train_ctc=False):
    """Lipflow's scope (front end, embed, Conformer block 0), plus the last extra_blocks Conformer blocks and the CTC
    head's projection: (parameters, state-dict prefixes). Everything else is frozen."""
    from lipflow.train_vsr import DEFAULT_SCOPE, trainable
    params = trainable(reader, DEFAULT_SCOPE)
    prefixes = list(FACE_ADAPTER_PREFIXES)
    m = reader.model
    n = len(m.encoder.encoders)
    for i in range(max(1, n - extra_blocks), n):
        params += list(m.encoder.encoders[i].parameters())
        prefixes.append(f'encoder.encoders.{i}.')
    if train_ctc:
        params += list(m.ctc.ctc_lo.parameters())
        prefixes.append('ctc.ctc_lo.')
    for p in params:
        p.requires_grad_(True)
    return params, tuple(prefixes)


def load_alignments(path):
    """{clip id: dict(words, starts, ends)} from a JSON file (lipread/streaming_eval.py's alignment cache); {} if none."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_alignments(path, cache):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(cache, f)
    os.replace(path + '.tmp', path)


def word_ends(reader, rois, ids, clip_id=None, cache=None):
    """(word_of, ends): per token its word, per word the frame after its last frame, by CTC forced alignment on a
    whole-clip read (lipread.streaming_eval). cache: a dict from load_alignments, filled in place; a hit needs the
    same clip id and the same words."""
    from lipread.streaming_eval import forced_align, token_words, word_spans
    word_of, words = token_words(reader, ids)
    hit = cache.get(clip_id) if cache is not None and clip_id is not None else None
    if hit is not None and hit['words'] == words:
        return word_of, hit['ends']
    with torch.inference_mode():
        x = reader.to_tensor(rois).unsqueeze(0).to(reader.enc_device)
        enc, _ = reader.model.encoder(x, None)
        logp = reader.model.ctc.log_softmax(enc)[0].float().cpu().numpy()
    starts, ends = word_spans(forced_align(logp, ids), word_of, len(words))
    ends = [int(x) for x in ends]
    if cache is not None and clip_id is not None:
        cache[clip_id] = dict(words=words, starts=[int(x) for x in starts], ends=ends)
    return word_of, ends


def cut(rois, ids, word_of, ends, rng, min_frac=0.4):
    """A random unfinished read of the clip, as a tick sees it: the first p frames (p from min_frac of the clip up), and
    as target the words already finished by frame p. None when no word is finished yet."""
    T = len(rois)
    p = int(rng.integers(max(1, int(min_frac * T)), T + 1))
    done = [u for u, w in enumerate(word_of) if ends[w] <= p]
    return (rois[:p], [ids[u] for u in done]) if done else None


def speed(rois, rng, spread):
    """Time-stretched clip (frames repeated or dropped), rate in 1 +- spread."""
    if not spread:
        return rois
    rate = rng.uniform(1 - spread, 1 + spread)
    idx = np.round(np.arange(0, len(rois) - 1e-6, rate)).astype(int)
    return rois[np.clip(idx, 0, len(rois) - 1)]


def clip_loss(reader, x, ys, ctc_weight):
    """Auto-AVSR's own loss, ctc_weight x CTC + (1 - ctc_weight) x attention (label-smoothed), per token: lipflow's
    clip_loss with the weight free (0.1 there)."""
    import torch.nn.functional as F
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    m, dev = reader.model, reader.enc_device
    hs, _ = m.encoder(x.unsqueeze(0).to(dev), None)
    logp = m.ctc.ctc_lo(hs).log_softmax(-1).transpose(0, 1)
    y = torch.tensor(ys, dtype=torch.long, device=dev)
    ctc = F.ctc_loss(logp.float(), y.unsqueeze(0), torch.tensor([logp.shape[0]]), torch.tensor([len(ys)]), blank=0,
                     reduction='sum', zero_infinity=True)
    ys_in = torch.tensor([[m.eos] + ys], dtype=torch.long, device=dev)
    ys_out = torch.tensor([ys + [m.eos]], dtype=torch.long, device=dev)
    pred, _ = m.decoder(ys_in, subsequent_mask(ys_in.shape[1], device=dev).unsqueeze(0), hs, None)
    return (ctc_weight * ctc + (1 - ctc_weight) * m.criterion(pred, ys_out)) / len(ys)


def finetune(clips, reader, epochs=6, lr=1e-4, ctc_weight=0.1, cut_p=0.0, speed_spread=0.0, extra_blocks=0,
             train_ctc=False, seed=0, align_cache=None, log=print):
    """Lipflow's speaker adaptation (train_vsr.finetune: AdamW, clip 5, BatchNorm frozen, its augmentation), and
    optionally: cut_p of the steps train on an unfinished read (cut()), a free CTC weight, time-stretch, a wider scope.
    The defaults are Lipflow's recipe (main() picks recipe C). With cut_p, clips need an 'id' to use align_cache (a
    JSON path, see word_ends). Returns the state-dict prefixes it trained (what save() keeps)."""
    import random
    from lipflow.train_vsr import _augment, _targets
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = reader.model
    m.decoder.to(reader.enc_device)
    m.ctc.to(reader.enc_device)
    params, prefixes = scope_params(reader, extra_blocks, train_ctc)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    data = []
    aligned = load_alignments(align_cache) if cut_p else {}
    n_aligned = len(aligned)
    for c in clips:
        ids = _targets(reader, c['text'])
        word_of, ends = word_ends(reader, c['rois'], ids, c.get('id'), aligned) if cut_p else (None, None)
        data.append((c['rois'], ids, word_of, ends))
    if align_cache and len(aligned) > n_aligned:
        save_alignments(align_cache, aligned)
        log(f'  aligned {len(aligned) - n_aligned} new clips ({len(aligned)} in {align_cache})')
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        for mod in m.modules():
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                mod.eval()
        random.shuffle(data)
        total = 0.0
        for rois, ids, word_of, ends in data:
            ys = ids
            if cut_p and rng.random() < cut_p:
                part = cut(rois, ids, word_of, ends, rng)
                if part is not None:
                    rois, ys = part
            loss = clip_loss(reader, _augment(speed(rois, rng, speed_spread)), ys, ctc_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss.detach())
        log(f'  epoch {ep + 1}/{epochs}: loss {total / len(data):.3f}  ({time.time() - t0:.0f}s)')
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return prefixes


def save(reader, path, prefixes=FACE_ADAPTER_PREFIXES):
    state = {k: v.detach().cpu() for k, v in reader.model.state_dict().items() if k.startswith(tuple(prefixes))}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)
    return len(state)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sessions', nargs='+', default=['session_01'], help='folders under recordings/')
    ap.add_argument('--test-every', type=int, default=0, help='hold out every Nth clip and report (0: train on all)')
    ap.add_argument('--test-sessions', nargs='+', default=[], help='hold out these whole sessions and report')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--recipe', choices=sorted(RECIPES), default='c',
                    help="c: --cut-p 0.5 --ctc-weight 0.5 --train-ctc --extra-blocks 2; lipflow: Lipflow's own "
                         "(0, 0.1, off, 0). The four flags below override their part")
    ap.add_argument('--ctc-weight', type=float, default=None, help="CTC's share of the loss")
    ap.add_argument('--cut-p', type=float, default=None,
                    help='share of steps on an unfinished read: a random prefix, its finished words as target')
    ap.add_argument('--speed', type=float, default=0.0, help='time-stretch augmentation, rate in 1 +- this')
    ap.add_argument('--extra-blocks', type=int, default=None, help='also train the last N Conformer blocks')
    ap.add_argument('--train-ctc', action=argparse.BooleanOptionalAction, default=None,
                    help="also train the CTC head's projection")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--force', action='store_true', help='save even if the held-out seen WER got worse')
    args = ap.parse_args(argv)
    if any('session_03' in s for s in args.sessions):
        ap.error('session 3 is the test session: never train on it')
    if set(args.sessions) & set(args.test_sessions):
        ap.error('a session cannot be both trained on and held out')

    from lipflow.train_vsr import DEFAULT_SCOPE
    from lipflow.vsr import LipReader
    recipe = resolve_recipe(args.recipe, args.cut_p, args.ctc_weight, args.train_ctc, args.extra_blocks)
    face = os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task')
    train_rows, test_rows = split(sentence_clips(args.sessions), args.test_every)
    test_rows += sentence_clips(args.test_sessions)
    t0 = time.time()
    train = [dict(rois=crops(r, face), text=r['prompt'], id=r['id']) for r in train_rows]
    test = [dict(rois=crops(r, face), text=r['prompt'], id=r['id']) for r in test_rows]
    print(f'{len(train)} train clips from {", ".join(args.sessions)} / {len(test)} held out '
          f'({", ".join(args.test_sessions) or (f"every {args.test_every}th" if args.test_every else "none")}), crops in '
          f'{time.time() - t0:.0f} s; scope {DEFAULT_SCOPE}, {args.epochs} epochs, lr {args.lr}, recipe {args.recipe} '
          f'{recipe}', flush=True)

    reader = LipReader(personal=False)
    beam = aligner = None
    if test:
        from lipread.final import Beam
        from lipread.model import LiveModel
        beam = Beam(reader)
        beam.encode = None
        aligner = LiveModel(warmup=(), face_adapter=None)
        before = score(reader, beam, test)
        seen_before = streaming_seen(test, aligner)
        print(f'base:    seen WER {seen_before["seen_wer"]:.3f} | tick WER {before[0]:.3f} (exact {before[2]}) | '
              f'final WER {before[1]:.3f} (exact {before[3]})', flush=True)
    t0 = time.time()
    prefixes = finetune(train, reader, epochs=args.epochs, lr=args.lr, speed_spread=args.speed, seed=args.seed,
                        align_cache=ALIGN_CACHE, log=lambda s: print(s, flush=True), **recipe)
    print(f'trained in {time.time() - t0:.0f} s', flush=True)
    if test:
        after = score(reader, beam, test)
        candidate = args.out + '.candidate'
        save(reader, candidate, prefixes)
        try:
            seen_after = streaming_seen(test, aligner, candidate)
        finally:
            os.remove(candidate)
        print(f'adapted: seen WER {seen_after["seen_wer"]:.3f} | tick WER {after[0]:.3f} (exact {after[2]}) | '
              f'final WER {after[1]:.3f} (exact {after[3]})', flush=True)
        if seen_after['seen_wer'] > seen_before['seen_wer'] and not args.force:
            print('held-out seen WER got worse: not saved (--force to save anyway)')
            return 1
    n = save(reader, args.out, prefixes)
    print(f'saved {n} tensors to {args.out}  (lipread.run --face-adapter {args.out})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
