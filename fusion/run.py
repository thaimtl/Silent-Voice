"""The fusion process on the bus: camera + EMG + Claude -> dec.hyp, dec.lock, dec.repeat, voice.say, timing.dec.

    venv-svr/bin/python -m fusion.run                       # SVR-1 W_s12_svr1_4ch, fitted gate and lock model
    venv-svr/bin/python -m fusion.run --calibrate 60        # first 60 s of emg.raw (Austin mouthing) refit the gains
    venv-svr/bin/python -m fusion.run --no-emg              # camera + LM + Claude only (pads off)

Without --calibrate, the last calibration's gains are reused (data/fusion/calibration.json, saved by each
--calibrate run): right while the pads stay where they were; after re-padding, calibrate again.

Connects to 'cam' (cam.tick, cam.final), 'emg' (emg.raw), 'claude' (claude.predict, claude.repair) and 'call'
(partner.text); binds 'dec' (bus/team.py). The logic is fusion/core.py; this file only moves messages. Messages are
drained before each pass, and a cam.tick with a newer one of the same segment already waiting is skipped, so a slow
pass never builds a backlog: the next pass reads the newest frames.

Needs, under RECLAIM_WORKDIR: checkpoints/W_s12_svr1_4ch/model.pt, data/lm/conversational.binary,
data/lexicon/cmudict.dict (look-alikes), data/fusion/params.json (python -m fusion.offline fit-gate / fit-lock;
without it the unfitted defaults run), and the camera's units file and SentencePiece model in third_party/.
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

import bus
from bus.team import Pub, Sub
from fusion.offline import PARAMS, WORK, build_search, load_params

DEFAULT_SVR = os.path.join(WORK, 'checkpoints', 'W_s12_svr1_4ch', 'model.pt')
CALIBRATION = os.path.join(WORK, 'data', 'fusion', 'calibration.json')
TOPICS = ['cam.tick', 'cam.final', 'emg.raw', 'claude.predict', 'claude.repair', 'claude.end', 'partner.text']
STEP_S = 0.02
# Claude's hints, live: a hint adds DEFAULT_BETA x its weight (0 to 1) to a candidate's score. 1.0, as in
# params.json (never fitted: fusion/offline.py runs without Claude); --beta raises it.
DEFAULT_BETA = 1.0


def drain(sub, first):
    """`first` plus everything already queued, with stale cam.ticks (a newer one of the same segment waiting) out."""
    msgs = [first]
    while True:
        topic, msg = sub.recv(timeout_ms=0)
        if topic is None:
            break
        msgs.append((topic, msg))
    newest = {}
    for k, (topic, msg) in enumerate(msgs):
        if topic == 'cam.tick':
            newest[int(msg['seg_id'])] = k
    return [(t, m) for k, (t, m) in enumerate(msgs) if t != 'cam.tick' or newest[int(m['seg_id'])] == k]


def calibrate(live, sub, seconds):
    print(f'fusion: calibrating on {seconds:.0f} s of emg.raw: mouth a few sentences now', file=sys.stderr)
    raw, t_end = [], bus.now() + seconds
    while bus.now() < t_end:
        topic, msg = sub.recv(timeout_ms=100)
        if topic == 'emg.raw':
            raw.append(np.asarray(msg['samples']).reshape(-1, 4))
    if not raw:
        sys.exit('fusion: no emg.raw arrived: is rig/publish.py running?')
    fit = live.calibrate(np.concatenate(raw))
    print(f'fusion: calibration {fit}', file=sys.stderr)
    os.makedirs(os.path.dirname(CALIBRATION), exist_ok=True)
    with open(CALIBRATION, 'w') as f:
        json.dump({**fit, 'saved': time.strftime('%Y-%m-%d %H:%M')}, f)


def last_calibration(live, path=CALIBRATION):
    """The last --calibrate run's gains and references; without one, the fine-tuning session's stay."""
    try:
        with open(path) as f:
            fit = json.load(f)
    except (OSError, ValueError):
        print(f"fusion: no saved calibration ({path}): the fine-tuning session's gains", file=sys.stderr)
        return
    live.gain = np.asarray(fit['gain'], np.float32)
    live.effort_ref, live.pace_ref = fit['effort_ref'], fit['pace_ref']
    print(f"fusion: reusing the last calibration, from {fit.get('saved', '?')}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--svr', default=DEFAULT_SVR, help='SVR-1 checkpoint')
    ap.add_argument('--params', default=PARAMS, help='fitted gate and lock model (fusion.offline)')
    ap.add_argument('--repair-s', type=float, default=0.0,
                    help="wait before a locked word is spoken (default 0: Claude's repairs fix the captions only, so "
                         "waiting buys nothing; each 0.1 s here is 0.1 s on every word)")
    ap.add_argument('--beta', type=float, default=DEFAULT_BETA,
                    help="weight of Claude's hints in the joint search (a hint adds beta x its weight, 0 to 1)")
    ap.add_argument('--tau0', type=float, default=None, help="lock threshold at predictability 0 (default: fitted)")
    ap.add_argument('--tau-end', type=float, default=None, help="end-of-sentence lock threshold (before cam.final)")
    ap.add_argument('--repeat-floor', type=float, default=None,
                    help='Please repeat only when the sentence confidence is below this (0: never; default: fitted)')
    ap.add_argument('--no-emg', action='store_true', help='camera + LM + Claude only')
    ap.add_argument('--no-claude', action='store_true', help="ignore Claude's predictions and repairs")
    ap.add_argument('--script', default=None, help="Austin's known lines (default fusion/script.txt)")
    ap.add_argument('--no-script', action='store_true', help='no known lines: free speech only')
    ap.add_argument('--calibrate', type=float, default=0.0, metavar='S',
                    help='refit the EMG gains on the first S s and save them (default: reuse the last saved ones)')
    ap.add_argument('--sub', action='append', default=[], help='endpoints to connect to (default cam emg claude call)')
    ap.add_argument('--pub', default='dec')
    args = ap.parse_args()

    from fusion.core import Fusion
    gate, lock, meta = load_params(args.params)
    gate.beta = args.beta   # set here, so the launcher needs no flag
    if args.tau0 is not None:   # keep the fitted range's shape around the new threshold
        lock.tau_min, lock.tau_max = lock.tau_min + args.tau0 - lock.tau0, min(0.99, lock.tau_max + args.tau0 - lock.tau0)
        lock.tau0 = args.tau0
    if args.tau_end is not None:
        lock.tau_end = args.tau_end
    if args.repeat_floor is not None:
        lock.repeat_floor = args.repeat_floor
    if not meta:
        print(f'fusion: {args.params} missing: unfitted gate and lock model', file=sys.stderr)
    live = None
    if not args.no_emg:
        from svr.live import LiveSVR1
        live = LiveSVR1(args.svr)
    search = build_search()
    if not args.no_script:
        from fusion.search import load_script
        search.script = load_script(args.script)
        print(f'fusion: {len(search.script)} known lines from {args.script or "fusion/script.txt"}', file=sys.stderr)
    topics = [t for t in TOPICS if not (args.no_claude and t.startswith('claude.'))]
    sub = Sub(args.sub or ['cam', 'emg', 'claude', 'call'], topics, rcvhwm=2000)
    pub = Pub(args.pub)
    if live is not None and args.calibrate > 0:
        calibrate(live, sub, args.calibrate)
    elif live is not None:
        last_calibration(live)
    core = Fusion(search, gate, lock, svr=live, repair_s=args.repair_s, use_hints=not args.no_claude)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    print(f'fusion: running (EMG {"off" if live is None else os.path.basename(os.path.dirname(args.svr))}, '
          f'repair window {args.repair_s} s)', file=sys.stderr)
    next_step = bus.now()
    try:
        while True:
            topic, msg = sub.recv(timeout_ms=int(STEP_S * 1000))
            out = []
            if topic is not None:
                for t, m in drain(sub, (topic, msg)):
                    try:
                        out += core.feed(t, m, bus.now())
                    except Exception as e:  # noqa: BLE001  one bad message never stops the captions
                        print(f'fusion: {t}: {type(e).__name__}: {e}', file=sys.stderr)
            now = bus.now()
            if now >= next_step:
                next_step = now + STEP_S
                try:
                    out += core.step(now)
                except Exception as e:  # noqa: BLE001  a failed end, final or EMG pass never stops the captions
                    print(f'fusion: step: {type(e).__name__}: {e}', file=sys.stderr)
            for t, m in out:
                pub.send(t, m)
                if t == 'dec.lock':
                    print('locked:', ' '.join(x['w'] for x in m['words']), f"[sentence {m['utt_id']}]", file=sys.stderr)
                elif t == 'dec.repeat':
                    print('please repeat:', m['reason'].get('reader'), file=sys.stderr)
                elif t == 'dec.sentence':
                    p = '-' if m['p'] is None else f"{m['p']:.2f}"
                    print(f"sentence {m['utt_id']} {m['event']}" + (f" ({m['reason']})" if m['reason'] else '')
                          + f": held {m['held_s']:.1f} s, Claude p {p} ({m['asks']} asks), {m['segments']} segments,"
                          f" {m['words']} words", file=sys.stderr)
                elif t == 'dec.pause':
                    print(f"sentence {m['utt_id']} asks Claude #{m['ask']} after {m['silence_s']:.1f} s still:",
                          ' '.join(m['words']), file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        time.sleep(0.1)


if __name__ == '__main__':
    main()
