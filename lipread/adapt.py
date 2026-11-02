"""Austin's face adapter: Lipflow's speaker adaptation (lipflow/train_vsr.py) on the team's recordings.

Only the visual front end, the embed and Conformer block 0 learn (Lipflow's default scope, 6 epochs, lr 1e-4, its
augmentation); the rest of Auto-AVSR, the CTC head, the decoder and the LM keep their weights, so a read costs the
same (lipread/MEASURED.md). Crops are made as live: the recording shrunk to --process-size, then LiveCropper.

    venv-live/bin/python -m lipread.adapt --sessions session_01 --test-every 5    # 4/5 train, 1/5 held out, report
    venv-live/bin/python -m lipread.adapt --sessions session_01 --test-sessions session_02   # held out: a new session
    venv-live/bin/python -m lipread.adapt --sessions session_01 session_02         # train on everything, save
    venv-live/bin/python -m lipread.run --cam 0 --face-adapter data/face_adapter/austin.pth

With --test-every or --test-sessions, the held-out clips are scored before and after (greedy = the tick reading,
beam 5 + LM = the final pass) and the adapter is saved only if the final pass is not worse (--force saves anyway).
Rest clips are skipped. A session is a folder under recordings/ (a renamed folder works: clips are read from the
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
from lipread.source import RECORDINGS, ReplaySource, fit

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


def save(reader, path):
    state = {k: v.detach().cpu() for k, v in reader.model.state_dict().items() if k.startswith(FACE_ADAPTER_PREFIXES)}
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
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--force', action='store_true', help='save even if the held-out final pass got worse')
    args = ap.parse_args(argv)
    if any('session_03' in s for s in args.sessions):
        ap.error('session 3 is the test session: never train on it')
    if set(args.sessions) & set(args.test_sessions):
        ap.error('a session cannot be both trained on and held out')

    from lipflow.train_vsr import DEFAULT_SCOPE, finetune
    from lipflow.vsr import LipReader
    from lipread.final import Beam
    face = os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task')
    train_rows, test_rows = split(sentence_clips(args.sessions), args.test_every)
    test_rows += sentence_clips(args.test_sessions)
    t0 = time.time()
    train = [dict(rois=crops(r, face), text=r['prompt']) for r in train_rows]
    test = [dict(rois=crops(r, face), text=r['prompt']) for r in test_rows]
    print(f'{len(train)} train clips from {", ".join(args.sessions)} / {len(test)} held out '
          f'({", ".join(args.test_sessions) or (f"every {args.test_every}th" if args.test_every else "none")}), crops in '
          f'{time.time() - t0:.0f} s; scope {DEFAULT_SCOPE}, {args.epochs} epochs, lr {args.lr}', flush=True)

    reader = LipReader(personal=False)
    beam = Beam(reader)
    beam.encode = None
    if test:
        before = score(reader, beam, test)
        print(f'base:    tick WER {before[0]:.3f} (exact {before[2]}) | final WER {before[1]:.3f} (exact {before[3]})',
              flush=True)
    t0 = time.time()
    finetune(train, epochs=args.epochs, lr=args.lr, reader=reader, log=lambda s: print(s, flush=True))
    print(f'trained in {time.time() - t0:.0f} s', flush=True)
    if test:
        after = score(reader, beam, test)
        print(f'adapted: tick WER {after[0]:.3f} (exact {after[2]}) | final WER {after[1]:.3f} (exact {after[3]})',
              flush=True)
        if after[1] > before[1] and not args.force:
            print('held-out final pass got worse: not saved (--force to save anyway)')
            return 1
    n = save(reader, args.out)
    print(f'saved {n} tensors to {args.out}  (lipread.run --face-adapter {args.out})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
