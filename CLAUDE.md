# Silent Voice

Cal Hacks 13.0 project, October 23 to 25, 2026: silent speech for people who lost their voice box.
Read README.md, HARDWARE.md and docs/TEAM_TASKS_AND_INTERFACES.md before answering; docs/REALTIME_PIPELINE.md explains how the system works.

Decisions already made, so don't re-litigate them without new evidence:
- One mode: free speech in real time, open mouth, silent. Camera + EMG + Claude together; no separate camera or EMG mode, no closed-mouth speech, no quick phrases, no Jev.
- The audience is laryngectomy patients; the demo speaker is Austin, who is healthy. Performance comes first.
- Camera: Auto-AVSR as is, made real time by re-reading the current segment every 100 ms (docs/EVAN_REALTIME_LIP_READING.md). No recording or fine-tuning for it.
- EMG: our model SVR-1, the only thing we train, on 4 BioAmp EXG Pill channels at Gaddy sites 3, 2, 6, 1 (pins A0 to A3) into an Arduino Uno; noisy channels hurt more than missing ones, so 4, not 6. All EMG, including the Berkeley pre-training data, goes through `rig/filters.py` in order, never `filtfilt`. It proposes words, settles lip look-alikes inside the joint search, detects speech start, word ends and sentence end, carries captions when the face is hidden, gives the voice its pace and effort, and tells Claude which words are uncertain.
- Claude follows the whole conversation (Anthropic track), through the Claude API: `claude-haiku-5-5` (thinking off, low effort, about 2 to 3 s a call) predicts likely words before Austin mouths them, for speed; `claude-opus-5-5` repairs look-alikes in the captions and, when a sentence ends, rewrites its caption as what Austin most likely said, for accuracy (captions only, never the voice); no local LLM. It never adds ideas.
- A word locks (goes to captions and voice) when one learned confidence score from camera, EMG and Claude crosses a threshold that moves with the conversation; Claude alone can never lock a word. Unclear words end in "Please repeat".
- The demo is a Twilio phone call: the judge speaks, Austin mouths, his cloned voice goes into the call. Nothing is trained on the pitch.
- Everything is logged at all times, with a replay tool.
- Cartesia Sonic 3.6 speaks in Austin's cloned voice while he mouths, without waiting for the sentence to end, with emotion from his face. Cartesia Ink transcribes the judge from the call into the captions and into Claude's context.
- No pad on the throat, no AD8232, no ESP32 in the rig, and each board has its own REF pad.
- Everything runs and is tested on Austin's laptop (RTX 5060), on battery whenever the pads are on skin: no charger while the Uno is connected (no USB isolator).
- Parts talk over ZeroMQ on the laptop, with the message formats in docs/TEAM_TASKS_AND_INTERFACES.md.
- Recording: three sessions of varied sentences with camera + EMG, recorded with `recorder/record.py`, following RECORDING.md (prompts in docs/recording/). Run the signal check (.claude/skills/signal-check) every time pads go on.

All code lives in this repo, never only in a working directory: datasets, checkpoints and environments stay outside git, and code finds them through `RECLAIM_WORKDIR` (README, "Code and data").
One exception: the cluster can't reach GitHub, so its copy of the code comes from a snapshot in the bucket (`repo/`). Upload it only with `cluster/upload_snapshot.sh`, from a clean checkout of a pushed commit; it stamps the commit into `cluster/VERSION`.
Never commit API keys or sponsor codes.
Diagrams are Mermaid in Markdown.
Read `sponsors/*.md` before any pitch, demo, or sponsor-facing work.
Every change goes through a PR. Pushing straight to main is blocked.
