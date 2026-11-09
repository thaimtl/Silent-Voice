# Lip reader (Evan)

The webcam goes through MediaPipe for the mouth crop and the blendshapes.
Every 100 ms, Auto-AVSR (as is) re-reads the current segment, and the result goes out as `cam.tick`: per-frame CTC top-32, blank, greedy text, activity, emotion and `timing_ms`.
At each segment end, GPU rescoring with the attention decoder and the language model sends `cam.final` (5-best) with the closing read (`--final cpu`: a background beam search instead).
Everything travels on the team bus (`bus/`, ZeroMQ + msgpack on `tcp://127.0.0.1:5556`), whose contract is [bus/schema.md](../bus/schema.md).
The team interfaces are in `docs/TEAM_TASKS_AND_INTERFACES.md`, and the milestones in `docs/EVAN_REALTIME_LIP_READING.md`.
Data (`third_party/`, `recordings/`) is found through `RECLAIM_WORKDIR` (README, "Code and data").

## Setup (once, from the repo root)

On Windows, run these in Git Bash; the venv's Python is `venv-live/Scripts/python.exe` instead of `venv-live/bin/python`.

```bash
bash lipread/setup.sh                                   # venv-live, Lipflow pinned, weights, sample
venv-live/bin/python -m pytest tests/lipread tests/bus -q
venv-live/bin/python -m lipread.bench.laptop_check      # on Austin's laptop: driver, CUDA, webcam, power, encoder
```

## Run

```bash
venv-live/bin/python -m lipread.run --cam 0                       # webcam
venv-live/bin/python -m lipread.run --cam 0 --camera-settings     # Windows: lock exposure and focus in the dialog
venv-live/bin/python -m lipread.run --replay third_party/lipflow/samples/2016-03-12.mov --start 15 --end 37 --realtime
venv-live/bin/python -m lipread.run --replay s1_001               # a recording id (recordings/ labels.csv)
venv-live/bin/python -m lipread.run --replay-session session_01 --realtime --recorder-clock   # a whole session
venv-live/bin/python -m bus.listen                                # second terminal: validates and prints every message
venv-live/bin/python -m bus.stub                                  # fake cam.tick / cam.final, no camera or GPU
venv-live/bin/python -m lipread.bench.tick_bench                  # tick p50/p95 at 2/4/8 s, fp32 vs fp16
```

Useful flags: `--size 1920x1080` (camera capture, default), `--process-size 1280x720` (frames shrunk before MediaPipe, default), `--fp32` (encoder in fp32; fp16 is the default, parity-checked), `--face-adapter data/face_adapter/austin.pth` (Austin's face adapter, below), `--no-fe-cache`, `--no-emotion`, `--no-final`, `--final cpu`, `--segments stream`, `--tick-ms 200`, `--log ticks.jsonl`.

## Files

| File | What |
| --- | --- |
| `run.py` | the process: crop thread, then the tick loop (segments, re-read, publish), then the final pass |
| `source.py` | webcam (DirectShow on Windows, V4L2 on Linux) and replay sources |
| `crop.py`, `crop_view.py` | MediaPipe landmarks + blendshapes, Auto-AVSR-aligned 96×96 crops, debug view |
| `endpoint.py` | mouthing activity and segments (300 ms pre-roll, 0.5 s still ends, 8 s cap with 2 s carried over) |
| `model.py` | Auto-AVSR encoder + CTC top-k per read, front end / Conformer timing |
| `final.py` | the final pass that publishes `cam.final`: GPU rescoring on the tick thread (default), or beam 5 + LM in a background process (`--final cpu`) |
| `emotion.py` | emotion tag for the voice: the expression network when its model is there (on its own thread), else the blendshape rules |
| `expression.py` | HSEmotion expression network on the face crop + the head fitted on the speaker's sessions (reads the 8 HSEmotion scores) |
| `emotion_fit.py` | fits that head on `recordings/session_0*` and prints held-out scores (`--test session_02` scores a session it never saw) |
| `adapt.py` | Austin's face adapter: Lipflow's speaker adaptation on the recordings (`python -m lipread.adapt`) |
| `types.py` | `Frame` and `Crop`; the message types live in `bus/` |
| `bench/` | `laptop_check.py` (the laptop), `tick_bench.py` (tick speed) |

Recording uses the team's `recorder/`, and `--replay` reads its sessions.

## Face adapter

Lipflow's speaker adaptation (`lipflow/train_vsr.py`): only the front end, the embed and Conformer block 0 learn Austin's
face from his recorded sessions; the rest of Auto-AVSR, the decoder and the language model are unchanged, so a read
costs the same. The file (about 100 MB) lives in `data/face_adapter/`, outside git.

```bash
venv-live/bin/python -m lipread.adapt --sessions session_01 --test-every 5    # train on 4/5, report on the held-out 1/5
venv-live/bin/python -m lipread.adapt --sessions session_01 --test-sessions session_02   # report on a session it never saw
venv-live/bin/python -m lipread.adapt --sessions session_01 session_02         # train on everything -> data/face_adapter/austin.pth
venv-live/bin/python -m lipread.run --cam 0 --face-adapter data/face_adapter/austin.pth
```

Never train on session 3: it is the test session.

Measured numbers and decisions: [MEASURED.md](MEASURED.md).
