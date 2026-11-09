# Team bus: `cam.tick`, `cam.final`, `cam.face`

The fields are those of `docs/TEAM_TASKS_AND_INTERFACES.md`.
This page adds the wire details.
Schema version **v = 2**.
Code: `bus/__init__.py`; stub: `bus/stub.py`; listener: `bus/listen.py`.

## Transport

- ZeroMQ PUB/SUB over TCP on the laptop's loopback, the same on Windows, Linux and macOS.
  The camera reader **binds** `tcp://127.0.0.1:5556`, which the env `RECLAIM_CAM_PUB` overrides.
  Subscribers **connect**; they may connect before the publisher exists.
  A second publisher on the same address fails with `EADDRINUSE`.
- Each message is multipart `[topic, payload]`, with the topic `b'cam.tick'`, `b'cam.final'` or `b'cam.face'`.
  Subscribe to `b'cam.tick'` and `b'cam.final'` (the `Subscriber` default).
  Only the dashboard should add `b'cam.face'` (~12.5 per second).
- The payload is msgpack: `packb(obj, use_bin_type=True)` / `unpackb(b, raw=False)`.
- Clock: every time on the bus is `bus.now()`, which is `time.perf_counter()`, in seconds.
  On Windows with Python before 3.13, `time.monotonic()` only ticks every 15.6 ms, while `perf_counter` is precise
  and shared by all processes. On Linux and macOS the two are the same clock.
- Queues: ZeroMQ holds up to 20 messages at the publisher (`SNDHWM`) and 100 at each subscriber (`RCVHWM`).
  Past that it drops the **newest** messages, not the oldest.
  A subscriber that does slow work between messages (a Claude call) must read with `Subscriber.drain()`, which
  takes everything waiting at once: every `cam.final`, and only the newest `cam.tick` per segment.
  A gap in `seq` shows a drop.
  If `seq` goes back down, the camera process restarted and counts from 1 again, so treat it as a restart, not a gap.
- `Subscriber.recv()` and `drain()` validate every message; one that fails is logged once, counted in `.bad` and
  skipped, never raised.
- `Publisher.close()` waits up to 0.5 s so the last `cam.final` still goes out; a `send()` after `close()` (the
  final-pass thread during shutdown) is dropped and returns 0.
- `bus.validate` rejects NaN, booleans where numbers belong, duplicate ids within a row and unsorted rows.
  It allows `-inf` log-probs (log 0). `decode` raises `ValueError` on any malformed message.

## numpy arrays

Each array is a msgpack map:

```
{'nd': 1, 'dtype': '<f4', 'shape': [T, 32], 'data': <bin: the array's bytes, C order>}
```

Decode with `np.frombuffer(d['data'], np.dtype(d['dtype'])).reshape(d['shape'])`. This is a read-only view with no
copy. Any msgpack library can read it without an extension hook.

## `cam.tick`

Sent every 100 ms (2.5 new 25 fps frames on average: 3, 2, 3, 2, ...) while a segment is open, plus one last read
when the segment closes or is capped. **Each tick re-reads the whole segment.** Past
frames' probabilities change between ticks (the Conformer looks both ways), so keep only the newest tick per `seg_id`.

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | int | schema version, 2 |
| `seq` | int | counts up per topic from 1; a gap means a dropped message |
| `t_pub` | float | `bus.now()` at send, for latency logs |
| `seg_id` | int ≥ 1 | the segment; a new id starts a new segment |
| `seg_start` | float | capture time of the segment's first frame |
| `frame_t` | `<f8` (T,) | capture time of each 25 fps model frame, increasing |
| `topk_ids` | `<i4` (T, 32) | token ids, most likely first; blank (0) can be among them |
| `topk_logp` | `<f4` (T, 32) | CTC log-probs of `topk_ids`, sorted from highest to lowest |
| `blank_logp` | `<f4` (T,) | CTC log-prob of blank |
| `greedy` | str | CTC best path, UPPERCASE; for debugging only |
| `activity` | `<f4` (T,) | mouthing activity, 0 to 1 |
| `face_ok` | bool | face found on the newest frame |
| `timing_ms` | map | cost of this pass, required: `T` (int, == len(`frame_t`)), `mediapipe`, `frontend`, `conformer`, `ctc`, `total` (ms, finite, ≥ 0; a CUDA-graph read reports its whole replay as `conformer` and `frontend` 0) |
| `emotion` | map | `label` (one of neutral, happy, sad, angry, surprised: Cartesia's names; with the expression head, `angry` stays 0), `confidence` 0 to 1, `scores` {label: 0 to 1} |

**Token ids** are the model's ids: blank is 0, and ids 1..5047 (`<unk>` = 1) are the line numbers of
`third_party/lipflow/lipflow/unigram5000_units.txt` (the same as `auto_avsr/spm/unigram/unigram5000_units.txt`), and
`<eos>` is 5048, so the vocabulary is 5049. **These are not the SentencePiece model's own ids** (`▁WATER` is 4899 here
but 334 in `unigram5000.model`). Map ids to pieces with the units file. `▁` marks the start of a word.

**Segments:** a segment starts with 300 ms of pre-roll and ends after 0.5 s without mouthing. At 8 s (T = 200)
the segment is capped: a new `seg_id` starts, and its first 2 s (50 frames) repeat the end of the previous segment.
The new `seg_start` is the time of the first repeated frame.

Size: about 55 KB per tick at T = 200 (53.5 KB max measured with the stub, timing_ms adds ~70 B), so about 550 KB/s
at 10 ticks/s: twice the v1 message rate and bandwidth, still trivial on loopback.

## `cam.final`

Sent once per segment, with the segment's closing read: the camera reader rescores it on the GPU on the tick thread
(`lipread.run --final gpu`, about 50 ms on the 5090). With `--final cpu` it comes about 1.1 s or more after the segment
ends (beam 5 + LM in a background process). It is also sent for a segment closed by the 8 s cap.

| Field | Type | Meaning |
| --- | --- | --- |
| `v`, `seq`, `t_pub`, `seg_id` | | as in `cam.tick` |
| `nbest` | list of {`text`, `score`} | up to 5, best first; text UPPERCASE; score = 0.9 attention + 0.1 CTC + 0.3 LM log-prob, where higher is better |

## `cam.face`

The newest face for the dashboard's mesh (Frank), sent from the crop thread on every 2nd frame (~12.5 Hz at 25 fps)
with `Publisher.send(msg, block=False)`: if the tick or final-pass thread is sending, the face is dropped, never
waited for (`dropped_busy` counts them; dropped faces use no `seq`). `run.py --no-face` turns it off. About 2.2 KB
with a face, 72 bytes without.

| Field | Type | Meaning |
| --- | --- | --- |
| `v`, `seq`, `t_pub` | | as in `cam.tick` (no `seg_id`: faces are not tied to segments) |
| `t` | float | capture time of the frame |
| `w`, `h` | int | the processed frame size the points are normalised by (1280 x 720 by default, `--process-size`) |
| `face_ok` | bool | a face was found; when false, `pts`, `blend` and `pose` are nil |
| `pts` | `<f2` (N, 2) | MediaPipe landmarks, x / w and y / h, N = 478 (468 mesh + 10 iris); may sit slightly outside 0..1 at the frame edge, validate allows -1..2 |
| `blend` | `<f2` (52,) or nil | MediaPipe blendshape scores 0..1, in the order of `lipread.crop.BLEND_NAMES` |
| `pose` | `<f4` (3,) or nil | head yaw, pitch, roll in radians from MediaPipe's facial transformation matrix; nil with `--no-pose` |

Head pose costs +0.02 ms per frame at 1280 x 720 (dev box), so it is on by default, and the landmarks are
bit-identical with it on or off (`tests/lipread/test_cam_face.py`).

## Subscriber

```python
from bus import Subscriber

sub = Subscriber()                      # tcp://127.0.0.1:5556, cam.tick + cam.final
while True:
    if sub.sock.poll(100):
        finals, ticks = sub.drain()     # validated; newest tick per segment, every final
        for seg_id, tick in ticks.items():
            print(seg_id, tick['greedy'])
        for f in finals:
            print(f['seg_id'], f['nbest'][0]['text'])
```

Without this package, any ZeroMQ + msgpack client works: connect a SUB socket to the address, subscribe to
`b'cam.tick'` and `b'cam.final'` (not `b'cam.'`, which also matches `cam.face`), and decode arrays as above.

To try it without a camera: run `python -m bus.stub`, then `python -m bus.listen` (which validates each
message and prints one line per message).

## Changes since v1 (for Austin)

- `v` is 2; `validate` refuses v1 messages.
- Address unchanged: `tcp://127.0.0.1:5556` (a short-lived `ipc://` default was dropped: no ipc on Windows).
- Times are `bus.now()` (`time.perf_counter()`), not `time.monotonic()`.
- `cam.tick` every 100 ms instead of 200 ms: 3 or 2 new frames per tick, about twice the messages.
- New required field `timing_ms` on `cam.tick` (see the table). A tick without it is invalid; there is no
  `None` fallback.
- `cam.final` is unchanged apart from `v`.
