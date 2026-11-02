"""Frame sources for the lip reader: a video file replayed at 25 fps, or the webcam (MJPEG).

Both yield lipread.types.Frame from `frames()` and release the device or file on `close()`.

ReplaySource: the video's own times (frame i -> start + i / fps, as Lipflow's load_clip), resampled to a steady 25 fps
with LipReader.resample's rule (first source frame at or after each 1/25 s grid point), paced to the wall clock or as
fast as possible. A recording with per-frame capture times (<id>_video_ts.npy, <id>.video_t.npy or .csv next to the
video, or a manifest's video_ts) is replayed on those times instead; --replay also takes a recording id (looked up in
recordings/*/manifest.jsonl or recordings/labels.csv). See ReplaySource.

CameraSource: MJPEG asked for at open, buffer 1, DirectShow on Windows and V4L2 on Linux. 50 or 60 fps only where V4L2 lists them
(every other frame kept at 50), else 30, resampled to 25 by grab timestamp (only kept frames are decoded). The first
second of grabs is measured: a driver that reports a rate it doesn't deliver is caught there, and under 28 fps logs a
warning (auto exposure slowing the camera in dim light). Asked-for size or MJPG not granted: logged as an error.
Linux: manual exposure and autofocus off (v4l2-ctl). Windows and macOS: auto exposure on (lock_controls);
settings=True opens the driver's own dialog (Windows) to lock exposure and focus by hand. Grab times from bus.now(); a capture thread hands frames over through a size-1
drop-oldest queue. A stalled or unplugged camera is reopened a few times, then the stream ends with `error` set.

    venv-live/bin/python -m lipread.source --replay third_party/lipflow/samples/2016-03-12.mov --start 20.4 --end 28.1
    venv-live/bin/python -m lipread.source --probe --device 0 --fps 25 30 50 60 --seconds 5
"""
import argparse
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import replace

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bus import now  # noqa: E402
from lipread.types import FPS, Frame  # noqa: E402

log = logging.getLogger(__name__)

GRID = 1.0 / FPS
MAX_REPEAT = 3
MEASURE_GRABS = 31
SLOW_FPS = 28
# Windows DirectShow (OpenCV 5): grab() returns at once and retrieve() waits for the frame, so a grab-only loop spins
# and stamps frames before they arrive. There every frame is read, and stamped once it is in.
READ_EVERY_FRAME = sys.platform == 'win32'


def fit(frame, size):
    h, w = frame.image.shape[:2]
    if size is None or (w <= size[0] and h <= size[1]):
        return frame
    k = min(size[0] / w, size[1] / h)
    img = cv2.resize(frame.image, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return replace(frame, image=img)


class Resampler:
    """Streaming LipReader.resample: for each 1/25 s grid point from the first frame on, the first frame at or after it.

    On a finished clip this picks exactly LipReader.resample(ts, n) (the grid ends at the last frame).
    """

    def __init__(self, max_repeat=None):
        self.next_t = None
        self.max_repeat = max_repeat

    def push(self, t):
        """Number of grid points this frame (time t) fills: 0 = skip it, 2+ = repeat it."""
        if self.next_t is None:
            self.next_t = t
        n = 0
        while self.next_t <= t + 1e-9:
            n += 1
            self.next_t += GRID
        if self.max_repeat is not None and n > self.max_repeat:
            self.next_t = t + GRID
            n = 1
        return n


TS_SUFFIXES = ('_video_ts', '.video_t', '_video_t', '.video_ts')
MANIFEST, VIDEO_MANIFEST = 'manifest.jsonl', 'manifest_video.jsonl'
VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm')
RECORDINGS = os.path.join(os.environ.get('RECLAIM_WORKDIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'recordings')


def ts_name(stem, part=False):
    """File name of the per-frame times recorder/record.py writes for <stem>.mp4 (part: while the take is open)."""
    return f'{stem}{TS_SUFFIXES[0]}{".part" if part else ""}.npy'


def load_ts(path):
    """Per-frame times (s) from a .npy or .csv file, float64 (n,)."""
    if path.endswith('.npy'):
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    rows = []
    with open(path) as f:
        for line in f:
            cell = line.replace(';', ',').split(',')[0].strip()
            try:
                rows.append(float(cell))
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def find_ts(video):
    """The timestamp file next to a video (TS_SUFFIXES, .npy then .csv), or None."""
    stem = os.path.splitext(video)[0]
    for suffix in TS_SUFFIXES:
        for ext in ('.npy', '.csv'):
            if os.path.exists(stem + suffix + ext):
                return stem + suffix + ext
    return None


def _manifest_rows(root):
    """(base dir, row) of every recording listed under root: <session>/manifest*.jsonl (svr/recordings.py, recorder/record.py)
    and labels.csv (RECORDING.md: id, session, ..., video), at root or in a session folder."""
    import csv
    import json
    dirs = [root] + sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for d in dirs:
        for name in (MANIFEST, VIDEO_MANIFEST):
            p = os.path.join(d, name)
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            yield d, json.loads(line)
        p = os.path.join(d, 'labels.csv')
        if os.path.exists(p):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    yield d, {k.strip(): (v or '').strip() for k, v in r.items() if k}


def resolve_recording(spec, root=None):
    """(video path, timestamp path or None) for a video file, a path without extension, or a recording id.

    An id is looked up in the manifests under root (default recordings/): its `video` and `video_ts` fields, relative to
    the manifest's folder (or to root / root/<session> for labels.csv), else <id>.<ext> in a session folder. A video
    without a `video_ts` field gets the sibling timestamp file find_ts sees, if any."""
    if os.path.isfile(spec):
        return spec, find_ts(spec)
    for ext in VIDEO_EXTS:
        if os.path.isfile(spec + ext):
            return spec + ext, find_ts(spec + ext)
    root = root or RECORDINGS
    if not os.path.isdir(root):
        raise FileNotFoundError(f'{spec}: no such video, and no recordings folder {root} to look the id up in')
    for base, r in _manifest_rows(root):
        if r.get('id') != spec or not r.get('video'):
            continue
        for d in (base, os.path.join(root, str(r.get('session', ''))), root):
            video = os.path.normpath(os.path.join(d, r['video']))  # labels.csv paths use /
            if os.path.isfile(video):
                ts = os.path.normpath(os.path.join(d, r['video_ts'])) if r.get('video_ts') else None
                return video, ts if ts and os.path.isfile(ts) else find_ts(video)
    for d in sorted(os.listdir(root)):
        for ext in VIDEO_EXTS:
            video = os.path.join(root, d, spec + ext)
            if os.path.isfile(video):
                return video, find_ts(video)
    raise FileNotFoundError(f'{spec}: not a video file and no recording with that id under {root}')


class ReplaySource:
    """A video file (or a recording id, see resolve_recording) fed as if it were the webcam.

    ts='auto' (default): use the recording's per-frame capture times when a timestamp file sits next to the video
    (find_ts) or the manifest names one; a path forces that file; None ignores them. With times, frame i's t is
    ts[i] - ts[0] (seconds from the clip's first frame; t_offset = ts[0] gives the recorder's clock back), so pacing
    follows the recording, dropped frames and a variable rate included, and start / end are seconds from the first
    frame. The 25 fps grid then follows CameraSource's rule (a gap over MAX_REPEAT grid steps re-anchors). Without
    times, frame i -> start + i / fps as before.

    close() may be called from another thread while frames() runs: frames() then ends after at most one read.
    """

    def __init__(self, path, start=0.0, end=None, realtime=True, ts='auto', root=None):
        self.start, self.end, self.realtime = start, end, realtime
        self.path, found = resolve_recording(path, root)
        self.ts_path = found if ts == 'auto' else ts
        self.ts = load_ts(self.ts_path) if self.ts_path else None
        if self.ts is not None and (self.ts.ndim != 1 or len(self.ts) == 0):
            raise ValueError(f'{self.ts_path or "ts"}: expected a non-empty 1-D array of frame times')
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f'could not open {self.path}')
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.t_offset = 0.0
        self.ts_gaps = 0
        if self.ts is not None:
            self.t_offset = float(self.ts[0])
            d = np.diff(self.ts)
            if len(d) and np.any(d <= 0):
                log.warning('%s: %d non-increasing frame times', self.ts_path, int(np.sum(d <= 0)))
            if len(d):
                span = float(self.ts[-1] - self.ts[0])
                self.src_fps = len(d) / span if span > 0 else self.src_fps
                self.ts_gaps = int(np.sum(d > 1.5 * float(np.median(d))))
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n and n != len(self.ts):
                log.warning('%s: %d frames but %d timestamps; replaying the first %d', self.path, n, len(self.ts),
                            min(n, len(self.ts)))
        elif start:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        self.dropped_frames = 0

    def _read(self):
        with self._lock:
            if self.cap is None:
                return False, None
            return self.cap.read()

    def frames(self):
        rs = Resampler(max_repeat=MAX_REPEAT if self.ts is not None else None)
        wall0 = t0 = None
        i = index = 0
        while not self._closed.is_set():
            if self.ts is not None and i >= len(self.ts):
                break
            ok, image = self._read()
            if not ok:
                break
            t = self.ts[i] - self.t_offset if self.ts is not None else self.start + i / self.src_fps
            i += 1
            if t < self.start - 1e-9:
                continue
            if self.end is not None and t > self.end:
                break
            for _ in range(rs.push(t)):
                if self.realtime:
                    if wall0 is None:
                        wall0, t0 = now(), t
                    if self._closed.wait(max(0.0, wall0 + (t - t0) - now())):
                        return
                yield Frame(t=t, image=image, index=index)
                index += 1

    def close(self):
        self._closed.set()
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


def session_clips(session, root=None):
    """Video paths of a session folder (or its name under root, default recordings/) in recording order (clip_t0)."""
    root = root or RECORDINGS
    d = session if os.path.isdir(session) else os.path.join(root, session)
    if not os.path.isdir(d):
        raise FileNotFoundError(f'{session}: no session folder {d}')
    rows = [r for base, r in _manifest_rows(d) if base == d and r.get('video')]
    if not rows:
        raise FileNotFoundError(f'{session}: no manifest with videos in {d}')
    return [os.path.join(d, r['video']) for r in sorted(rows, key=lambda r: r.get('clip_t0') or 0.0)]


class SessionSource:
    """Every clip of a recording session, in recording order, as one replay (lipread.run --replay-session).

    Each clip is a ReplaySource on its own capture times; t is the recorder's clock minus the first clip's first
    frame (t_offset gives the recorder's clock back, as in ReplaySource), so the real gaps between clips stay in t.
    Wall time between clips is gap_s instead: with realtime, the camera reader's stall rule (no crop for 0.5 s) ends
    the open segment there, so a segment never spans two clips. Clips without per-frame times are skipped, since
    nothing places them on the shared clock."""

    def __init__(self, session, realtime=True, gap_s=1.5, limit=None, root=None):
        self.paths = []
        self.skipped = []
        for path in session_clips(session, root)[:limit]:
            (self.paths if find_ts(path) else self.skipped).append(path)
        for path in self.skipped:
            log.warning('%s: no frame times, skipped', path)
        if not self.paths:
            raise FileNotFoundError(f'{session}: no clip with frame times')
        self.t_offset = float(load_ts(find_ts(self.paths[0]))[0])
        self.realtime, self.gap_s = realtime, gap_s
        self._closed = threading.Event()
        self._cur = None

    def frames(self):
        index = 0
        for path in self.paths:
            if self._closed.is_set():
                return
            src = self._cur = ReplaySource(path, realtime=self.realtime)
            shift = src.t_offset - self.t_offset
            for f in src.frames():
                yield Frame(t=f.t + shift, image=f.image, index=index)
                index += 1
            src.close()
            if self.realtime and self._closed.wait(self.gap_s):
                return

    def close(self):
        self._closed.set()
        if self._cur is not None:
            self._cur.close()


def _v4l2_dev(device):
    return device if isinstance(device, str) else f'/dev/video{device}'


def v4l2_fps(device, width, height, fourcc='MJPG'):
    """Frame rates the driver lists for this format and size (v4l2-ctl), [] when unknown."""
    if not shutil.which('v4l2-ctl'):
        return []
    try:
        out = subprocess.run(['v4l2-ctl', '-d', _v4l2_dev(device), '--list-formats-ext'], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rates, fmt, size = [], None, None
    for line in out.splitlines():
        m = re.search(r"'(\w{4})'", line)
        if m:
            fmt = m.group(1)
        m = re.search(r'Size: Discrete (\d+)x(\d+)', line)
        if m:
            size = (int(m.group(1)), int(m.group(2)))
        m = re.search(r'\(([\d.]+) fps\)', line)
        if m and fmt == fourcc and size == (width, height):
            rates.append(float(m.group(1)))
    return sorted(set(rates))


def v4l2_controls(device):
    """{name: current value} from v4l2-ctl -l, {} when v4l2-ctl is missing."""
    if not shutil.which('v4l2-ctl'):
        return {}
    try:
        out = subprocess.run(['v4l2-ctl', '-d', _v4l2_dev(device), '-l'], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return {m.group(1): int(m.group(2)) for m in re.finditer(r'^\s*(\w+)\s+0x[0-9a-f]+ \(\w+\)\s*:.*?value=(-?\d+)',
                                                             out, re.M)}


LOCKS = [('auto_exposure', 1), ('exposure_auto', 1), ('exposure_dynamic_framerate', 0), ('exposure_auto_priority', 0),
         ('focus_automatic_continuous', 0), ('focus_auto', 0)]


def lock_controls(device, cap=None, exposure=None):
    """Linux (v4l2-ctl): manual exposure, autofocus off. Elsewhere through OpenCV: auto exposure on, unless an exposure
    is given, since a manual exposure another program left in the camera can make it open nearly black. Returns what
    was set."""
    have = v4l2_controls(device)
    sets = [(k, v) for k, v in LOCKS if k in have]
    if exposure is not None:
        sets += [(k, int(exposure)) for k in ('exposure_time_absolute', 'exposure_absolute') if k in have]
    if sets:
        for k, v in sets:
            subprocess.run(['v4l2-ctl', '-d', _v4l2_dev(device), '-c', f'{k}={v}'], capture_output=True, timeout=5)
        after = v4l2_controls(device)
        return {k: after.get(k) for k, _ in sets}
    done = {}
    if cap is not None:
        if exposure is None:
            done['auto_exposure'] = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            done['auto_exposure'] = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            done['exposure'] = cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return done


BACKENDS = {'linux': cv2.CAP_V4L2, 'win32': cv2.CAP_DSHOW, 'darwin': cv2.CAP_AVFOUNDATION}


def keep_every(cam_fps):
    """2 at 50 fps (every other frame is the 25 fps grid), else 1 (resampled)."""
    return 2 if abs(cam_fps - 2 * FPS) < 2 else 1


def resampled(cam_fps, keep):
    """Kept frames go through the 25 fps resampler unless they already arrive at 25 fps; 27 fps passed straight
    through would stretch time by 8%."""
    return abs(cam_fps / keep - FPS) > 0.5


class CameraSource:
    """The webcam as a 25 fps Frame stream (module docstring).

    A stall or unplug: after stall_s without a successful grab (grab() failing, or a retrieve failing), the device is
    reopened up to reopen_tries times, waiting backoff_s, 2 backoff_s, 4 backoff_s... before each; when none works the
    capture thread ends, frames() returns and `error` says why (None while healthy or after a clean close()).
    close() may be called from any thread, also while frames() runs: only the capture thread touches the device, and it
    releases it on the way out.
    """

    def __init__(self, device=0, width=1920, height=1080, fps=None, exposure=None, lock=True, stall_s=2.0,
                 reopen_tries=3, backoff_s=0.5, settings=False):
        self.device, self.width, self.height = device, width, height
        self.exposure, self.lock = exposure, lock
        self.stall_s, self.reopen_tries, self.backoff_s = stall_s, reopen_tries, backoff_s
        self.cap = self._open()
        if self.cap is None:
            raise RuntimeError(f'could not open camera {device}')
        self.offered = v4l2_fps(device, width, height)
        self.cam_fps = self._pick_fps(fps)
        self.keep_every = keep_every(self.cam_fps)
        self.measured_fps = None
        self.fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.size != (width, height) or self.fourcc != 'MJPG':
            log.error('camera %s: asked for %dx%d MJPG, the driver gave %dx%d %s', device, width, height, *self.size,
                      self.fourcc)
        self.locked = lock_controls(device, self.cap, exposure) if lock else {}
        if settings:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        self.dropped_frames = 0
        self.grab_failures = 0
        self.reopens = 0
        self.error = None
        self.grab_ts = deque(maxlen=600)
        self.emit_ts = deque(maxlen=600)
        self._q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='live-camera', daemon=True)
        self._thread.start()

    def _open(self):
        """A configured VideoCapture (MJPEG, size, buffer 1), or None when the device does not open. MJPEG and the
        size are asked for at open: set afterwards, DirectShow keeps the uncompressed mode (5 fps at 1080p on a C920)."""
        backend = BACKENDS.get(sys.platform, cv2.CAP_ANY)
        params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'), cv2.CAP_PROP_FRAME_WIDTH, self.width,
                  cv2.CAP_PROP_FRAME_HEIGHT, self.height, cv2.CAP_PROP_FPS, 30]
        cap = cv2.VideoCapture(self.device, backend, params)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.device, backend)
            if not cap.isOpened():
                cap.release()
                return None
            for prop, value in zip(params[::2], params[1::2]):
                cap.set(prop, value)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _pick_fps(self, fps):
        """Requested rate, or 50 / 60 when V4L2 lists them and the driver accepts them, else 30. Without a listing
        (Windows, macOS) only 30: some drivers echo back any rate they are asked for. Unless asked for, a rate is not set
        again when the camera already runs at it or doesn't say (DirectShow reports -1 after an open with 30 fps asked
        for): setting it on DirectShow drops the C920 back to 640x480 YUY2."""
        tries = [fps] if fps else [r for r in (50, 60) if r in self.offered] + [30]
        for r in tries:
            cur = self.cap.get(cv2.CAP_PROP_FPS)
            if fps or (cur > 0 and abs(cur - r) >= 1):
                self.cap.set(cv2.CAP_PROP_FPS, r)
            got = self.cap.get(cv2.CAP_PROP_FPS)
            if fps or abs(got - r) < 1 or r == 30:
                return got if got > 0 else r
        return 30.0

    def _reopen(self):
        """Release the stalled device and open it again with the same settings; False when every try failed."""
        log.warning('camera %s: no frame for %.1f s (%d failed grabs), reopening', self.device, self.stall_s,
                    self.grab_failures)
        self.cap.release()
        self.cap = None
        for k in range(self.reopen_tries):
            if self._stop.wait(self.backoff_s * 2 ** k):
                return False
            cap = self._open()
            if cap is not None:
                cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
                if self.lock:
                    lock_controls(self.device, cap, self.exposure)
                self.cap = cap
                self.reopens += 1
                log.warning('camera %s: reopened (try %d)', self.device, k + 1)
                return True
            log.warning('camera %s: reopen try %d/%d failed', self.device, k + 1, self.reopen_tries)
        self.error = f'camera {self.device} lost: no frame for {self.stall_s:g} s and {self.reopen_tries} reopens failed'
        log.error(self.error)
        return False

    def _run(self):
        try:
            self._capture()
        except Exception as e:
            self.error = f'camera {self.device} capture failed: {e!r}'
            log.exception(self.error)
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def _capture(self):
        rs = Resampler(max_repeat=MAX_REPEAT)
        n = index = fails = 0
        last_ok = now()
        while not self._stop.is_set():
            if READ_EVERY_FRAME:
                ok, image = self.cap.read()
            else:
                ok, image = self.cap.grab(), None
            t = now()
            if ok:
                self.grab_ts.append(t)
                n += 1
                if n == MEASURE_GRABS:
                    self._measure()
                if (n - 1) % self.keep_every:
                    last_ok, fails = t, 0
                    continue
                reps = rs.push(t) if resampled(self.cam_fps, self.keep_every) else 1
                if not reps:
                    last_ok, fails = t, 0
                    continue
                if image is None:
                    ok, image = self.cap.retrieve()
            if not ok:
                fails += 1
                self.grab_failures += 1
                if fails == 1 or fails % 100 == 0:
                    log.warning('camera %s: %d consecutive failed grabs', self.device, fails)
                if t - last_ok >= self.stall_s:
                    if self._stop.is_set() or not self._reopen():
                        return
                    last_ok, fails = now(), 0
                else:
                    self._stop.wait(0.005)
                continue
            last_ok, fails = t, 0
            self._put((Frame(t=t, image=image, index=index), reps))
            self.emit_ts.extend([t] * reps)
            index += reps

    def _measure(self):
        """The rate the camera really delivers, from the first MEASURE_GRABS grabs; trusted over the driver's."""
        ts = list(self.grab_ts)[-MEASURE_GRABS:]
        rate = self.measured_fps = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        if abs(rate - self.cam_fps) > 1.5:
            log.warning('camera %s: driver says %g fps but delivers %.1f: using %.1f', self.device, self.cam_fps, rate,
                        rate)
            self.cam_fps, self.keep_every = rate, keep_every(rate)
        if rate < SLOW_FPS:
            log.warning('camera %s: only %.1f fps, auto exposure is probably slowing it in dim light: add light or lock '
                        'exposure (--camera-settings on Windows)', self.device, rate)

    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self.dropped_frames += self._q.get_nowait()[1]
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def frames(self):
        """25 fps frames; a frame that fills several grid points comes that many times (index counting up), with the
        same t (lipread/run.py stamps the repeats 40 ms apart)."""
        while not self._stop.is_set() or not self._q.empty():
            try:
                frame, reps = self._q.get(timeout=0.2)
                for k in range(reps):
                    yield frame if k == 0 else Frame(t=frame.t, image=frame.image, index=frame.index + k)
            except queue.Empty:
                if not self._thread.is_alive() and self._q.empty():
                    return

    def close(self):
        """Stop the capture thread; it releases the device. Safe from any thread and more than once."""
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                log.warning('camera %s: capture thread still in a driver call after close()', self.device)


def interval_stats(ts):
    """fps and frame-interval stats (ms) of a list of timestamps."""
    d = np.diff(np.asarray(ts, dtype=np.float64)) * 1000
    if len(d) == 0:
        return {'n': len(ts), 'fps': 0.0}
    return {'n': len(ts), 'fps': round(1000 / d.mean(), 2), 'mean_ms': round(d.mean(), 2),
            'std_ms': round(d.std(), 2), 'p95_ms': round(float(np.percentile(d, 95)), 2), 'max_ms': round(d.max(), 2)}


def probe(device, fps, seconds, width, height, settings=False):
    """Open the camera at one rate, consume the 25 fps stream for `seconds`, print raw and output stats."""
    src = CameraSource(device, width, height, fps=fps, settings=settings)
    print(f'requested {fps} fps: driver {src.cam_fps:g} fps, {src.fourcc} {src.size[0]}x{src.size[1]}, '
          f'keep every {src.keep_every}, offered {src.offered or "?"}, locked {src.locked}')
    t_end, got = now() + 1.0 + seconds, []
    for f in src.frames():
        t = now()
        if t > t_end:
            break
        if t > t_end - seconds:
            got.append(f)
    raw = [t for t in src.grab_ts if t > t_end - seconds]
    src.close()
    gaps = sum(b.index - a.index - 1 for a, b in zip(got, got[1:]))
    print(f'  raw grabs   {interval_stats(raw)}')
    print(f'  25fps out   {interval_stats([f.t for f in got])}  consumer gaps {gaps}, dropped {src.dropped_frames}')
    return interval_stats([f.t for f in got])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--replay')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float)
    p.add_argument('--fast', action='store_true', help='replay as fast as possible')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--device', default='0')
    p.add_argument('--fps', type=int, nargs='*', default=[0])
    p.add_argument('--seconds', type=float, default=5.0)
    p.add_argument('--size', default='1920x1080')
    p.add_argument('--settings', action='store_true', help="open the driver's settings dialog (Windows)")
    a = p.parse_args()
    if a.replay:
        src = ReplaySource(a.replay, a.start, a.end, realtime=not a.fast)
        w0, ts = now(), [f.t for f in src.frames()]
        src.close()
        print(f'{a.replay}: source {src.src_fps:g} fps -> {interval_stats(ts)}, wall {now() - w0:.2f} s')
    if a.probe:
        device = int(a.device) if a.device.isdigit() else a.device
        w, h = map(int, a.size.split('x'))
        for r in a.fps:
            try:
                probe(device, r or None, a.seconds, w, h, a.settings)
            except Exception as e:
                print(f'requested {r or "auto"} fps: FAIL {e}')


if __name__ == '__main__':
    main()
