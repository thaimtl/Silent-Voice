"""The camera reader process (E2): webcam or replay -> mouth crops -> re-read the open segment every 100 ms
(--tick-ms) -> cam.tick on the team bus.

Threads: the source's own capture thread (CameraSource) -> crop thread (owns the source and the cropper: MediaPipe +
align, never waits for the GPU) -> queue -> tick loop on the main thread (frame times, segments, encoder, publish).

Tick schedule: due points at ceil(k * tick_ms * 25 / 1000) frames from the segment start (100 ms: 3, 5, 8, 10, 13,
...; 200 ms: 5, 10, 15, ...); a tick reads the whole segment once its length reaches the next due point. In replay
without --realtime every crop is handled on its own (deterministic). Live, the loop drains whatever crops queued up
while it was busy, pushes them all, then ticks once on the newest state (backlog depth and lag go in the log).

Segments come from lipread/endpoint.py's Segmenter (E4: activity rises -> start with 300 ms pre-roll, 0.5 s still ->
end, 8 s cap with 2 s carried over). --segments stream (Segmenter(never_end=True)) keeps one segment always open
instead (debugging, replays of continuous speech). Live, no crop for still_s (camera stall) ends the open segment in
either mode.

Each crop gets its model-frame time once, when it reaches the tick loop: its capture time, or the previous frame's
time + 40 ms if the source repeated a frame (strictly increasing), so a frame carried over at a cap has the same
frame_t in both segments. Replay times are rebased onto now() at the first frame, so cam.tick times are
on the laptop clock as the bus requires (and lag is real in --realtime). --recorder-clock instead keeps a recorded
clip's own capture times (its video_ts, the recorder's bus.now()), so offline replays line up with the clip's EMG
(same clock) for fitting and testing the joint search.

The crop thread's queue carries (crop, crop ms) items and then one End: End(None) for a clean end of the source,
End(err) when the thread died.

Failures: a tick that raises is logged and skipped (MAX_FAILURES in a row: exit 1); a crop thread that dies ends the
run with exit 1 after the last read. Ctrl+C: last read + 'end' of the open segment, then the publisher closes.

    venv-live/bin/python -m lipread.run --replay third_party/lipflow/samples/2016-03-12.mov --start 20.4 --end 28.1
    venv-live/bin/python -m lipread.run --replay-session session_01 --realtime --recorder-clock   # offline, on the EMG's clock
    venv-live/bin/python -m lipread.run --cam 0                          # webcam, Ctrl+C stops

Capture size: the webcam opens at --size (default 1920x1080); the crop thread shrinks any larger frame (camera or a
1080p recording) to fit --process-size (default 1280x720, INTER_AREA, aspect kept; 0 disables) before the cropper
and the emotion worker, so both see the same 720p frame. Measured: 1080p MJPEG decode 4.8 ms + resize 0.5-1.3 ms +
720p crop 5.2 ms, about 11 ms per frame. Frames that already fit pass through untouched.
    venv-live/bin/python -m bus.listen                           # in another terminal
"""
import argparse
import dataclasses
import json
import logging
import os
import queue
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

import lipread
from bus import CAM_PUB, Publisher, now, face_msg
from lipread.crop import LiveCropper
from lipread.emotion import RESTING, EmotionWorker
from lipread.endpoint import START, Segmenter
from bus import CamTick
from lipread.types import FPS, Frame

log = logging.getLogger('lipread.run')
TICK_MS = 100
MAX_FAILURES = 10
BACKLOG_WARN = 5
JOIN_S = 10.0


@dataclass(frozen=True)
class TickStat:
    """One tick's costs; timing_ms() is the bus's cam.tick timing_ms, from the same numbers."""
    seg: int
    T: int
    mediapipe_ms: float
    frontend_ms: float
    conformer_ms: float
    enc_ms: float
    ctc_ms: float
    total_ms: float
    lag_ms: float
    greedy: str
    backlog: int = 0

    def timing_ms(self):
        return {'T': self.T, 'mediapipe': self.mediapipe_ms, 'frontend': self.frontend_ms,
                'conformer': self.conformer_ms, 'ctc': self.ctc_ms, 'total': self.total_ms}


@dataclass(frozen=True)
class End:
    """The crop thread's last queue item: err is None for a clean end of the source, else what killed the thread."""
    err: BaseException = None


def due_after(n, tick_ms=TICK_MS):
    """Smallest due point ceil(k * tick_ms * FPS / 1000), k >= 1, that is > n frames."""
    per = tick_ms * FPS
    k = n * 1000 // per + 1
    return -(-k * per // 1000)


class Runner:
    def __init__(self, model, publish, segmenter, emotion=None, tick_ms=TICK_MS, final=None, fe_cache=False):
        self.model, self.publish = model, publish
        self.fe_cache = fe_cache
        self.cache = None
        if fe_cache:
            from lipread.model import FrontendCache
            self._new_cache = FrontendCache
        self.final = final
        self.last_enc = (None, 0, None)
        self.emotion = emotion or (lambda: RESTING)
        self.segmenter = segmenter
        self.tick_ms = int(tick_ms)
        self.read = (None, 0)
        self.last_t = -np.inf
        self.capture_t = -np.inf
        self.crop_ms = []
        self.stats = []

    def unread(self, seg):
        return len(seg.crops) - (self.read[1] if self.read[0] == seg.seg_id else 0)

    def stamp(self, crop):
        """The crop with its model-frame time: capture time, or previous + 40 ms for a repeated frame. Once per crop."""
        t = crop.t if crop.t > self.last_t + 1e-6 else self.last_t + 1.0 / FPS
        self.last_t, self.capture_t = t, crop.t
        return crop if t == crop.t else dataclasses.replace(crop, t=t)

    def tick_if_due(self, backlog=0):
        seg = self.segmenter.segment
        if seg is None:
            self.crop_ms.clear()
            return []
        done = len(seg.crops) - self.unread(seg)
        return [self.tick(seg, backlog)] if len(seg.crops) >= due_after(done, self.tick_ms) else []

    def on_crops(self, items):
        """(crop, crop_ms) items (one in replay, a drained backlog live): push them all, then at most one scheduled
        tick on the newest state. Returns the TickStats of the ticks it sent (a cap also reads the closed segment)."""
        events = []
        for crop, ms in items:
            if ms is not None:
                self.crop_ms.append(float(ms))
            events += self.segmenter.push(self.stamp(crop))
        return self.closed(events) + self.tick_if_due(backlog=len(items) - 1)

    def closed(self, events):
        """'cap' / 'end': one last read of the whole closed segment, then the final beam pass (E6) in the background.
        A failed last read still sends the segment to the final pass (which then re-encodes), then raises."""
        out, err = [], None
        for ev, seg in events:
            if ev == START:
                continue
            if self.unread(seg) > 0:
                try:
                    out.append(self.tick(seg))
                except Exception as e:  # noqa: BLE001
                    err = err or e
            if self.final is not None and seg.crops:
                sid, T, enc = self.last_enc
                self.final.submit(seg.seg_id, np.stack([c.roi96 for c in seg.crops]),
                                  enc if (sid, T) == (seg.seg_id, len(seg.crops)) else None)
        if err is not None:
            raise err
        return out

    def finish(self):
        """End of the stream (or stall / Ctrl+C): close the open segment (with its last read)."""
        return self.closed(self.segmenter.flush())

    def tick(self, seg, backlog=0):
        t0 = time.perf_counter()
        crops = seg.crops
        keep_enc = self.final is not None
        if self.fe_cache:
            if self.cache is None or self.cache.seg_id != seg.seg_id:
                self.cache = self._new_cache(seg.seg_id)
            r = self.model.read([c.roi96 for c in crops], keep_enc=keep_enc, cache=self.cache)
        else:
            r = self.model.read(np.stack([c.roi96 for c in crops]), keep_enc=keep_enc)
        self.last_enc = (seg.seg_id, len(crops), r.enc)
        ft = np.array([c.t for c in crops], np.float64)
        emotion = self.emotion()
        stat = TickStat(seg.seg_id, len(crops), float(np.mean(self.crop_ms)) if self.crop_ms else 0.0,
                        float(r.frontend_ms), float(r.conformer_ms), float(r.enc_ms), float(r.ctc_ms),
                        (time.perf_counter() - t0) * 1000, (now() - self.capture_t) * 1000, r.greedy,
                        backlog)
        self.publish(CamTick(seg.seg_id, float(ft[0]), ft, r.topk_ids, r.topk_logp, r.blank_logp, r.greedy,
                             np.asarray(seg.activity, np.float32), bool(crops[-1].face_ok), emotion,
                             stat.timing_ms()))
        self.read = (seg.seg_id, len(crops))
        self.crop_ms = []
        self.stats.append(stat)
        return stat


def rebased(frames):
    """Replay frames with their video times moved onto now() (offset fixed at the first frame)."""
    off = None
    for f in frames:
        if off is None:
            off = now() - f.t
        yield Frame(t=f.t + off, image=f.image, index=f.index)


def shifted(frames, off):
    """Replay frames with their video times moved by a fixed offset (--recorder-clock: the source's t_offset)."""
    for f in frames:
        yield Frame(t=f.t + off, image=f.image, index=f.index)


def size_arg(s):
    """'WxH' -> (w, h)."""
    try:
        w, h = (int(v) for v in s.lower().split('x'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'{s!r}: WxH expected, e.g. 1280x720') from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f'{s!r}: positive sizes expected')
    return w, h


def fit(frame, size):
    """The frame shrunk (INTER_AREA, aspect kept) to fit size = (w, h); unchanged if it fits or size is None."""
    h, w = frame.image.shape[:2]
    if size is None or (w <= size[0] and h <= size[1]):
        return frame
    k = min(size[0] / w, size[1] / h)
    img = cv2.resize(frame.image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return dataclasses.replace(frame, image=img)


def crop_thread(source, cropper, q, stop, emotion=None, rebase=False, process_size=None, face_pub=None, face_every=2,
                clock_offset=None, preview=None):
    """Frames -> (crop, crop ms of the frame's push) on q, until the source ends or stop is set. This thread owns the
    source and the cropper and closes both itself (cap.release() from another thread while this one is inside
    cap.read() segfaults). Each frame is fit() to process_size first. The last item is End.
    face_pub(CamFace): every face_every-th frame's face for the dashboard (cam.face), sent without blocking.
    clock_offset: shift frame times by it instead of rebasing (--recorder-clock).
    preview (lipread.preview.PreviewWorker): every frame and its face handed over for cam.preview; never waits."""
    err, emotion_errors, face_errors, n = None, 0, 0, 0
    try:
        frames = source.frames()
        if clock_offset is not None:
            frames = shifted(frames, clock_offset)
        elif rebase:
            frames = rebased(frames)
        for frame in frames:
            if stop.is_set():
                break
            frame = fit(frame, process_size)
            t0 = time.perf_counter()
            out = cropper.push(frame)
            ms = (time.perf_counter() - t0) * 1000
            for c in out:
                q.put((c, ms))
            if emotion is not None:
                try:
                    emotion.submit(frame.t, frame.image, cropper.last_face)
                except Exception:  # noqa: BLE001  (emotion is optional: it never stops the lip reader)
                    emotion_errors += 1
                    if emotion_errors == 1:
                        log.exception('emotion submit failed (later failures are only counted)')
            n += 1
            if preview is not None:
                preview.submit(frame.t, frame.image, cropper.last_face, getattr(cropper, 'last_z', None))
            if face_pub is not None and n % face_every == 0:
                try:
                    h, w = frame.image.shape[:2]
                    face_pub(face_msg(frame.t, cropper.last_face, getattr(cropper, 'last_pose', None), w, h))
                except Exception:  # noqa: BLE001  (the dashboard's face is optional: it never stops the lip reader)
                    face_errors += 1
                    if face_errors == 1:
                        log.exception('cam.face publish failed (later failures are only counted)')
        if not stop.is_set():
            for c in cropper.flush():
                q.put((c, None))
    except Exception as e:  # noqa: BLE001
        log.exception('crop thread died')
        err = e
    finally:
        if emotion_errors:
            log.warning('emotion submit failed on %d frames', emotion_errors)
        if face_errors:
            log.warning('cam.face publish failed on %d frames', face_errors)
        for close in (source.close, cropper.close):
            try:
                close()
            except Exception:  # noqa: BLE001
                log.exception('closing %s failed', close)
        q.put(End(err))


class TooManyFailures(RuntimeError):
    pass


def drain(q, first):
    """first plus every item already queued behind it, up to the End: (items, End or None)."""
    items = [first]
    while True:
        try:
            nxt = q.get_nowait()
        except queue.Empty:
            return items, None
        if isinstance(nxt, End):
            return items, nxt
        items.append(nxt)


def serve(runner, q, live_mode, report=lambda stats: None, stall_s=None, max_failures=MAX_FAILURES):
    """The tick loop: crops from q until the crop thread's End. Returns the exit code (0 clean, 1 crop thread died or
    MAX_FAILURES ticks in a row failed). Replay: one crop per tick check. Live: drains backlogs, and ends the open
    segment after stall_s without a crop (default: the segmenter's still_s)."""
    if stall_s is None:
        stall_s = runner.segmenter.n_still / FPS
    failures = 0

    def guarded(fn, *a):
        nonlocal failures
        try:
            stats = fn(*a)
        except Exception:  # noqa: BLE001
            failures += 1
            log.exception('tick failed (%d in a row)', failures)
            if failures >= max_failures:
                raise TooManyFailures(f'{failures} ticks in a row failed')
            return
        failures = 0
        report(stats)

    try:
        end = None
        while end is None:
            try:
                item = q.get(timeout=stall_s) if live_mode else q.get()
            except queue.Empty:
                seg = runner.segmenter.segment
                if seg is not None:
                    log.warning('no crop for %.2f s (camera stall): ending segment %d at %d frames', stall_s,
                                seg.seg_id, len(seg.crops))
                    guarded(runner.finish)
                continue
            if isinstance(item, End):
                end = item
                break
            items, end = drain(q, item) if live_mode else ([item], None)
            if len(items) > 1:
                lag = (now() - items[-1][0].t) * 1000
                (log.warning if len(items) - 1 >= BACKLOG_WARN else log.debug)(
                    'backlog: %d crops drained, newest captured %.0f ms ago', len(items) - 1, lag)
            guarded(runner.on_crops, items)
        guarded(runner.finish)
    except TooManyFailures as e:
        log.error('%s: exiting 1', e)
        return 1
    if end.err is not None:
        log.error('crop thread died (%r): exiting 1', end.err)
        return 1
    return 0


def summary(stats):
    if not stats:
        return 'no ticks'
    def pct(key):
        v = [getattr(s, key) for s in stats]
        return f'p50 {statistics.median(v):.1f} p95 {np.percentile(v, 95):.1f} max {max(v):.1f}'
    mean = {k: np.mean([getattr(s, k) for s in stats]) for k in ('mediapipe_ms', 'frontend_ms', 'conformer_ms',
                                                                  'ctc_ms')}
    return (f'{len(stats)} ticks, segments {len({s.seg for s in stats})}, T max {max(s.T for s in stats)}, '
            f'backlog max {max(s.backlog for s in stats)}\n'
            f'  tick total ms {pct("total_ms")}; encoder {pct("enc_ms")}; lag ms {pct("lag_ms")}\n'
            f'  mean ms: mediapipe/frame {mean["mediapipe_ms"]:.1f}, frontend {mean["frontend_ms"]:.1f}, '
            f'conformer {mean["conformer_ms"]:.1f}, ctc {mean["ctc_ms"]:.1f}')


def positive(s):
    v = float(s)
    if not v > 0:
        raise argparse.ArgumentTypeError(f'{s}: a number above 0 expected')
    return v


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--replay', default=None, help='video file instead of the webcam')
    ap.add_argument('--replay-session', default=None, metavar='SESSION',
                    help='every clip of a recording session (e.g. session_01) in order, 1.5 s apart (SessionSource)')
    ap.add_argument('--clips', type=int, default=None, help='--replay-session: only the first N clips')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=None)
    ap.add_argument('--realtime', action='store_true', help='replay at real speed (default: as fast as possible)')
    ap.add_argument('--recorder-clock', action='store_true',
                    help="replay: keep the recording's capture times (video_ts) instead of moving them onto now()")
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--camera-settings', action='store_true',
                    help="open the webcam driver's settings dialog at start (Windows: lock exposure and focus there)")
    ap.add_argument('--exposure', type=float, default=None,
                    help='manual webcam exposure (DirectShow: log2 seconds, auto settles near -6; default auto)')
    ap.add_argument('--size', type=size_arg, default=(1920, 1080), metavar='WxH', help='webcam capture size')
    ap.add_argument('--process-size', type=lambda s: None if s == '0' else size_arg(s), default=(1280, 720),
                    metavar='WxH',
                    help='shrink larger frames to fit this before cropping (0: never)')
    ap.add_argument('--fp32', action='store_true', help='encoder in fp32 (default fp16 via .half(), parity-checked)')
    ap.add_argument('--face-adapter', default=None, metavar='PTH',
                    help="Lipflow face adapter over the base weights (lipread/adapt.py; same cost per read)")
    ap.add_argument('--segments', choices=('auto', 'stream'), default='auto',
                    help='auto: E4 activity segments; stream: one segment always open (8 s cap)')
    ap.add_argument('--tick-ms', type=int, choices=(100, 200), default=TICK_MS, help='tick period')
    ap.add_argument('--pub', default=CAM_PUB)
    ap.add_argument('--log', default=None, help='JSONL of per-tick timings')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--no-emotion', action='store_true', help='publish neutral instead of running E5')
    ap.add_argument('--no-face', action='store_true', help='no cam.face (the dashboard\'s face mesh, ~12.5 Hz)')
    ap.add_argument('--no-preview', action='store_true', help='no cam.preview (the dashboard\'s mirror of the webcam)')
    ap.add_argument('--preview-hz', type=positive, default=30, help='cam.preview frames per second at most')
    ap.add_argument('--no-pose', action='store_true',
                    help='no head pose in cam.face (+0.02 ms/frame at 1280x720 on the dev box, so on by default)')
    ap.add_argument('--no-final', action='store_true', help='no cam.final (E6 beam + LM at segment end)')
    ap.add_argument('--no-fe-cache', action='store_true', help='re-run the front end on the whole segment every tick')
    return ap.parse_args(argv)


def make_source(args):
    from lipread.source import CameraSource, ReplaySource, SessionSource
    if args.replay_session:
        return SessionSource(args.replay_session, realtime=args.realtime, limit=args.clips)
    if args.replay:
        return ReplaySource(args.replay, args.start, args.end, realtime=args.realtime)
    w, h = args.size
    return CameraSource(args.cam, width=w, height=h, exposure=args.exposure, settings=args.camera_settings)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

    from lipread.model import LiveModel
    import torch

    pub = Publisher(args.pub)
    replay = bool(args.replay or args.replay_session)
    live_mode = not replay or args.realtime
    final = emotion = th = logf = preview = None
    runner = Runner(None, pub.send, None)
    stop = threading.Event()
    code = 1
    try:
        box = {}
        if not args.no_final:
            from lipread.final import FinalPass
            starter = threading.Thread(target=lambda: box.update(final=FinalPass(None, pub.send)), name='final-start',
                                       daemon=True)
            starter.start()
        model = LiveModel(half=not args.fp32, face_adapter=args.face_adapter)
        log.info('model ready (warm-up %.1f s%s)', model.warmup_s,
                 f', face adapter {args.face_adapter}' if args.face_adapter else '')
        if not args.no_final:
            starter.join()
            final = box.get('final')
            if final is None:
                log.error('final pass failed to start: running without cam.final')
            else:
                final.beam.model = model
                log.info('final pass ready (beam 5 + LM, background process)')
        source = make_source(args)
        cropper = LiveCropper(os.path.join(lipread.LIPFLOW_MODELS, 'face_landmarker.task'), pose=not args.no_pose)
        segmenter = Segmenter() if args.segments == 'auto' else Segmenter.stream()
        if not args.no_emotion:
            emotion = EmotionWorker(sync=not live_mode)     # a fast replay: the same emotion every run
        runner = Runner(model, pub.send, segmenter, emotion.current if emotion else None, tick_ms=args.tick_ms,
                        final=final, fe_cache=not args.no_fe_cache)
        q = queue.Queue()
        face_pub = None if args.no_face else (lambda m: pub.send(m, block=False))
        offset = getattr(source, 't_offset', 0.0) if replay and args.recorder_clock else None
        if not args.no_preview:
            from lipread.preview import PreviewWorker
            preview = PreviewWorker(lambda m: pub.send(m, block=False), hz=args.preview_hz)
        th = threading.Thread(target=crop_thread, args=(source, cropper, q, stop, emotion, replay, args.process_size),
                              kwargs=dict(face_pub=face_pub, preview=preview, clock_offset=offset), name='crop',
                              daemon=True)
        logf = open(args.log, 'w') if args.log else None

        def report(stats):
            for s in stats:
                if logf:
                    logf.write(json.dumps(asdict(s)) + '\n')
                if not args.quiet:
                    print(f'seg {s.seg:3d} T {s.T:3d} fe {s.frontend_ms:5.1f} cf {s.conformer_ms:5.1f} '
                          f'total {s.total_ms:5.1f} ms' + (f' lag {s.lag_ms:5.0f} ms' if live_mode else '')
                          + (f' backlog {s.backlog}' if s.backlog else '') + f'  {s.greedy}', flush=True)

        th.start()
        log.info('publishing cam.tick on %s every %d ms', pub.addr, args.tick_ms)
        try:
            code = serve(runner, q, live_mode, report)
        except KeyboardInterrupt:
            log.info('Ctrl+C: last read of the open segment, then stop')
            stop.set()
            try:
                report(runner.finish())
                code = 0
            except Exception:  # noqa: BLE001
                log.exception('last read failed')
    except KeyboardInterrupt:
        log.info('Ctrl+C during start-up')
    finally:
        stop.set()
        if th is not None:
            th.join(timeout=JOIN_S)
            if th.is_alive():
                log.error('crop thread still running after %.0f s (source stuck?): exiting without closing it', JOIN_S)
                code = code or 1
        if emotion is not None:
            emotion.close()
        if preview is not None:
            preview.close()
        if final is not None:
            final.close(timeout=3.0)
            log.info('final pass: %d beams, %d dropped, %d errors', len(final.stats), len(final.dropped),
                     len(final.errors))
        pub.close()
        if logf:
            logf.close()
    print(summary(runner.stats))
    if torch.cuda.is_available():
        print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB')
    return code


if __name__ == '__main__':
    sys.exit(main())
