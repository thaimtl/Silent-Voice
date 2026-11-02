"""The live call as the dashboard shows it: bus messages in, one JSON-able snapshot out (Thai's Figma v5).

Pure: no sockets, no clock of its own. feed(topic, msg, now) for every bus message, snapshot(now) for the page.
`now` and every time here are bus.now() seconds (perf_counter on the laptop), the clock every producer stamps.

What each part of the screen comes from (docs/TEAM_TASKS_AND_INTERFACES.md):
    header           call.state (start, end), claude.repair (fixed by Claude), dec.repeat (asked to repeat),
                     lips to voice: timing.voice sonic_first_audio's t_pub minus the utterance's first dec.lock word t
    judge            partner.text (Cartesia Ink): partial while speaking, final into the transcript
    Austin           dec.hyp (locked words and who settled each, unlocked words), claude.repair (struck word)
    candidates       dec.hyp `cands` + `threshold`, when the decoder sends them (optional; hidden otherwise)
    please repeat    dec.repeat
    video chips      cam.tick emotion and face_ok; emg.raw per-pad level and contact
    heard on call    voice.say (what the voice was given), the emotion it went out with
    latency          cam.tick timing_ms.total, timing.dec, timing.claude, timing.voice; "--" when nothing arrived
    status           how recently each part published
    end of call      everything above, kept per sentence; reader disagreement from dec.lock `cam_top` / `emg_top`
                     (each reader's own pick, when the decoder sends it), never a made-up error rate
"""
import statistics
from collections import deque

import numpy as np

from dashboard.emg_level import PadLevels

PADS = (('A0', 'Tongue'), ('A1', 'Jaw'), ('A2', 'Lower lip'), ('A3', 'Upper lip'))   # HARDWARE.md pad sites
CHIP = {'neutral': 'Neutral face', 'happy': 'Smiling', 'sad': 'Frowning', 'surprised': 'Surprised',
        'angry': 'Tense'}
TONE = {'neutral': 'Sounds calm', 'happy': 'Sounds warm', 'sad': 'Sounds concerned', 'surprised': 'Sounds surprised',
        'angry': 'Sounds firm'}
TONE_ORDER = ('warm', 'concerned', 'calm', 'surprised', 'firm')    # the end screen's donut, in this colour order
REUSE_S = 10.0              # a finished sentence's utt_id seen again after this long: the decoder restarted
SPEAKING_MAX_S = 8.0        # no 'played' mark this long after the first audio: the voice has stopped anyway
STALE_S = {'cam': 1.0, 'emg': 1.0, 'claude': 60.0, 'voice': 120.0, 'log': 5.0}
EMG_RAIL = (8, 1015)        # a pad off the skin pins its channel to a rail of the 10-bit range
EMG_FLAT = 1.0              # standard deviation below this (raw units): no contact


def pretty(words, first=True):
    """The lip reader's UPPERCASE words as written: lowercase, 'I' capitalised, and the first word if first."""
    out = []
    for i, w in enumerate(words):
        w = str(w).lower()
        if w == 'i' or w.startswith("i'"):
            w = 'I' + w[1:]
        out.append(w[:1].upper() + w[1:] if first and i == 0 else w)
    return ' '.join(out)


def _bare(w):
    return str(w).strip().strip('.,!?;:"“”')


def _median(xs):
    return round(statistics.median(xs)) if xs else None


class CallState:
    def __init__(self, number='', benchmark=None, channel='call'):
        self.number = number
        self.channel = channel               # how the judge talks: 'call' (Ink hears them) or 'sms' (they text)
        self.switchable = False              # the page's Call / Text switch works (the call server answers)
        self.benchmark = benchmark           # word error per reader on a scored session (end screen), or None
        self.seen = {}                       # producer -> last bus.now() a message came from it
        self.emg = [dict(level=0.0, contact=False) for _ in PADS]
        self.levels = PadLevels(len(PADS))   # filtered as rig/filters.py, measured above each pad's own floor
        self.emotion = ('neutral', 0.0)
        self.face_ok, self.face_hidden_since = True, None
        self.face_pts, self.face_size = None, None
        self.cam_frames = deque(maxlen=30)   # median frame spacing per cam.tick, for the camera's fps
        self.reset_call()

    def reset_call(self, t=None):
        self.call = dict(state='waiting' if t is None else 'connected', t_start=t, t_end=None, call_sid=None, ink=None)
        if self.face_hidden_since is not None and t is not None:
            self.face_hidden_since = t       # hidden when the call starts: count from the start, not before
        self.judge_live = ''
        self.transcript = []                 # {who, text, t, seq, ...}: judge lines and Austin's sentences, in order
        self.seq = 0                         # each line's own number: the page keys its bubbles on it (utt_id can repeat)
        self.utts = {}                       # utt_id -> Austin's sentence as it builds
        self.cur_utt = None
        self.repairs, self.repeats = [], []
        self.captions_fixed = 0     # sentences whose caption Opus rewrote (claude.caption)
        self.last_repeat = None
        self.lat = dict(read=deque(maxlen=200), lock=deque(maxlen=200), claude=deque(maxlen=200),
                        window=deque(maxlen=200), sonic=deque(maxlen=200), lips_to_voice=deque(maxlen=200))
        self.claude_error = None
        self.face_hidden_s = 0.0
        self._summary = None
        self.emg_words = 0                   # words locked while the face was hidden (muscles carried them)

    def _utt(self, utt_id, t, current=False):
        """Austin's sentence utt_id. Only the decoder's messages (current=True) make it the live one, so a late
        voice.say or claude.repair for an earlier sentence never pulls the caption back."""
        old = self.utts.get(utt_id)
        restarted = current and old is not None and old['final'] and t - old['t_final'] > REUSE_S
        if restarted:                       # the decoder restarted and counts from 1 again: keep the old sentence
            self.utts[('old', utt_id, old['t'])] = self.utts.pop(utt_id)
        if utt_id not in self.utts:
            self.utts[utt_id] = dict(utt_id=utt_id, t=t, locked=[], unlocked=[], source=[], conf=[], repaired={},
                                     spoken=[], said=[], final=False, t_final=None, emotion=None, lock_t=[],
                                     cam_top=[],
                                     emg_top=[], cands=None, threshold=None, repeat=None, face_hidden=False,
                                     speaking=None, voiced=False)
        if current and (restarted or self.cur_utt is None or utt_id >= self.cur_utt):
            self.cur_utt = utt_id
        return self.utts[utt_id]

    # --- messages in -------------------------------------------------------------------------------------------
    def feed(self, topic, msg, now):
        handler = HANDLERS.get(topic)
        if handler is None and topic.startswith('timing.'):
            handler = CallState._timing
        if handler is not None:
            handler(self, topic, msg, now)

    def set_channel(self, channel, switchable):
        if channel in ('call', 'sms'):
            self.channel = channel
        self.switchable = bool(switchable)

    def _call_state(self, topic, msg, now):
        self.seen['voice'] = now
        if msg.get('channel') in ('call', 'sms'):       # on connect and on every switch (a replay carries it too)
            self.channel = msg['channel']
        if msg.get('state') == 'connected':      # call time on our clock: right for a replay at any speed too
            self.reset_call(now)
            self.call.update(call_sid=msg.get('call_sid'), ink=msg.get('ink'))
        elif msg.get('state') == 'ended' and self.call['state'] == 'connected':
            t_end = now
            if self.face_hidden_since is not None:          # close the open interval at the end of the call
                self.face_hidden_s += max(0.0, t_end - self.face_hidden_since)
                self.face_hidden_since = t_end
            self.call.update(state='ended', t_end=t_end)
            self._close_open_sentences(t_end)

    def _partner(self, topic, msg, now):
        self.seen['voice'] = now
        text = str(msg.get('text', '')).strip()
        if msg.get('is_final'):
            if text:
                self._line(dict(who='judge', text=text, t=now))     # our clock, like Austin's (a replay too)
            self.judge_live = ''
        else:
            self.judge_live = text

    def _hyp(self, topic, msg, now):
        self.seen['dec'] = now
        u = self._utt(msg['utt_id'], now, current=True)
        u['locked'] = [str(w) for w in msg.get('locked') or []]
        u['unlocked'] = [str(w) for w in msg.get('unlocked') or []]
        u['source'] = [str(s) for s in msg.get('source') or []]
        u['conf'] = [float(c) for c in msg.get('confidence') or []]
        cands = msg.get('cands')            # the newest word's candidates: gone when the decoder stops sending them
        u['cands'] = [dict(w=str(c['w']), cam=float(c.get('cam', 0)), emg=float(c.get('emg', 0)),
                           claude=float(c.get('claude', 0))) for c in cands][:3] if cands else None
        u['threshold'] = msg.get('threshold') if cands else None

    def _lock(self, topic, msg, now):
        self.seen['dec'] = now
        u = self._utt(msg['utt_id'], now, current=True)
        for w in msg.get('words') or []:
            u['lock_t'].append(w.get('t', now))
            u['cam_top'].append(w.get('cam_top'))
            u['emg_top'].append(w.get('emg_top'))
            if (not self.face_ok or now - self.seen.get('cam', -1e9) >= STALE_S['cam']) and self.muscles(now):
                self.emg_words += 1
                u['face_hidden'] = True

    def _repeat(self, topic, msg, now):
        self.seen['dec'] = now
        u = self._utt(msg['utt_id'], now, current=True)
        reason = msg.get('reason') or {}
        words = reason.get('words') or []
        u['repeat'] = dict(reader=reason.get('reader'), words=[str(w.get('w', '')) for w in words],
                           alts=[[str(a) for a in w.get('alts') or []] for w in words])
        self.repeats.append(dict(utt_id=u['utt_id'], t=now, words=u['repeat']['words']))
        self.last_repeat = u['utt_id']

    def logging(self, now):
        """The bus logger is writing (the server checks its newest file); the plan logs every call."""
        self.seen['log'] = now

    def _repair(self, topic, msg, now):
        self.seen['claude'] = now
        u = self._utt(msg['utt_id'], now)
        for e in msg.get('edits') or []:
            frm, to = _bare(e['from']), _bare(e['to'])
            u['repaired'][int(e['index'])] = (frm, to)
            self.repairs.append(dict(utt_id=u['utt_id'], index=int(e['index']), frm=pretty([frm], False),
                                     to=pretty([to], False)))

    def _caption(self, topic, msg, now):
        """claude.caption: Opus's corrected sentence replaces the caption (the voice already said what locked)."""
        self.seen['claude'] = now
        words = [str(w) for w in msg.get('words') or [] if str(w).strip()]
        if not words or not msg.get('changed', True):
            return
        u = self._utt(msg['utt_id'], now)
        u['caption'] = words
        for line in reversed(self.transcript):
            if line.get('who') == 'austin' and line.get('utt_id') == u['utt_id']:
                line.setdefault('original', line['text'])
                line['text'] = pretty(words)
                line['caption_fixed'] = True
                self.captions_fixed += 1
                break

    def _say(self, topic, msg, now):
        u = self._utt(msg['utt_id'], now)
        words = [str(w) for w in msg.get('words') or []]
        off = msg.get('offset')
        if off is not None:
            words = words[max(0, len(u['spoken']) - int(off)):]
        u['spoken'] += words
        if u['emotion'] is None and words:
            u['emotion'] = self.emotion[0]
        if msg.get('final'):
            self._close(u, now)

    @staticmethod
    def _said(u):
        """What the call heard: the voice's own phrases once reported, else the words sent to it."""
        return ' '.join(u['said']) if u['said'] else pretty(u['spoken'])

    def _close(self, u, t):
        if u['final']:
            return
        u['final'], u['t_final'] = True, t
        text = pretty(u['caption']) if u.get('caption') else (self._said(u) or pretty(u['locked']))
        if text:
            emotion = u['emotion'] or self.emotion[0]
            self._line(dict(who='austin', text=text, t=t, utt_id=u['utt_id'], emotion=emotion,
                                        tone=TONE.get(emotion, '').replace('Sounds', 'sounded'),
                                        repairs=[dict(index=i, frm=pretty([f], False), to=pretty([to], False))
                                                 for i, (f, to) in u['repaired'].items()],
                                        repeat=u['repeat'], face_hidden=u['face_hidden']))

    def _line(self, line):
        self.seq += 1
        self.transcript.append(dict(line, seq=self.seq))

    def _close_open_sentences(self, t):
        for u in self.utts.values():
            if not u['final'] and (u['spoken'] or u['locked']):
                self._close(u, t)

    def _tick(self, topic, msg, now):
        self.seen['cam'] = now
        e = msg.get('emotion') or {}
        if e.get('label'):
            self.emotion = (str(e['label']), float(e.get('confidence', 0)))
        ft = msg.get('frame_t')
        if ft is not None and len(ft) > 1:
            self.cam_frames.append(float(np.median(np.diff(np.asarray(ft, np.float64)))))
        total = (msg.get('timing_ms') or {}).get('total')
        if total is not None:
            self.lat['read'].append(float(total))
        self._face(bool(msg.get('face_ok', True)), now)

    def _face_msg(self, topic, msg, now):
        self.seen['cam'] = now
        self._face(bool(msg.get('face_ok', True)), now)
        pts = msg.get('pts')
        if msg.get('face_ok') and pts is not None and len(pts):
            self.face_pts = np.asarray(pts, np.float32).reshape(-1, 2)
            self.face_size = (msg.get('w'), msg.get('h'))

    def _face(self, ok, now):
        if self.face_ok and not ok:
            self.face_hidden_since = now
        elif ok and not self.face_ok and self.face_hidden_since is not None:
            if self.call['state'] == 'connected':
                start = self.call['t_start'] if self.call['t_start'] is not None else now
                self.face_hidden_s += max(0.0, now - max(self.face_hidden_since, start))
            self.face_hidden_since = None
        self.face_ok = ok

    def _emg(self, topic, msg, now):
        self.seen['emg'] = now
        x = np.asarray(msg.get('samples'), np.float32).reshape(-1, len(PADS))
        if not len(x):
            return
        for i, pad in enumerate(self.emg):
            col = x[:, i]
            pad['contact'] = bool(col.std() >= EMG_FLAT and EMG_RAIL[0] < col.mean() < EMG_RAIL[1])
        levels = self.levels.push(x, np.array([p['contact'] for p in self.emg]))
        for pad, level in zip(self.emg, levels):
            pad['level'] = float(level)

    def _timing(self, topic, msg, now):
        stage, ms = msg.get('stage'), msg.get('ms')
        if topic == 'timing.claude':
            if ms is None:              # the service reporting a bad bus message, not a call to Claude
                return
            self.seen['claude'] = now
            self.claude_error = msg.get('error')
            if not msg.get('error'):
                self.lat['claude'].append(float(ms))
        elif topic == 'timing.voice':
            self.seen['voice'] = now
            u = self.utts.get(msg.get('utt_id'))
            if stage == 'voice_phrase' and u and msg.get('text'):
                u['said'].append(str(msg['text']))
                for entry in reversed(self.transcript if u['final'] else []):   # open: its entry doesn't exist yet
                    if entry['who'] == 'austin' and entry.get('utt_id') == u['utt_id']:
                        entry['text'] = self._said(u)
                        break
            if stage == 'sonic_first_audio' and ms is not None:
                self.lat['sonic'].append(float(ms))
                if u and not u['voiced']:        # a sentence's first audio only (a fallback replay isn't a new sample)
                    u['voiced'] = True
                    if u['lock_t']:
                        self.lat['lips_to_voice'].append(1000 * (msg.get('t_pub', now) - u['lock_t'][0]))
                for other in self.utts.values():   # one voice at a time
                    other['speaking'] = None
                if u:
                    u['speaking'] = now
            elif stage == 'played_to_judge' and u:
                u['speaking'] = None
        elif topic == 'timing.dec' and ms is not None and stage in ('lock', 'window'):
            self.lat[stage].append(float(ms))

    # --- snapshot out ------------------------------------------------------------------------------------------
    def status(self, now):
        fresh = {k: now - self.seen.get(k, -1e9) < s for k, s in STALE_S.items()}
        spacing = [d for d in self.cam_frames if d > 0]
        fps = round(1 / statistics.median(spacing)) if spacing else None
        pads = sum(p['contact'] for p in self.emg) if fresh['emg'] else 0
        problems = []
        if not fresh['cam']:
            problems.append('Camera: no signal.')
        elif not self.face_ok:
            problems.append('Face out of view: the muscle sensors are carrying the caption.')
        if not fresh['emg'] or pads == 0:
            problems.append('Muscle sensors: no signal. The camera and Claude are carrying the caption.')
        elif pads < len(PADS):
            problems.append(f'Muscle sensors: {pads} of {len(PADS)} pads in contact.')
        if self.claude_error:
            problems.append('Claude: not answering, captions go on without it.')
        if not fresh['log']:
            problems.append('Not logging: start python -m buslog.record.')
        connected = self.call['state'] == 'connected'
        linked = (['Claude'] if fresh['claude'] and not self.claude_error else []) + (['Cartesia', 'Twilio'] if connected
                                                                                   else [])
        names = ', '.join(linked[:-1]) + ' and ' + linked[-1] if len(linked) > 1 else ''.join(linked)
        tail = f'{names} connected.' if linked else ''
        if problems:
            return dict(ok=False, text=' '.join(problems + [tail]).strip())
        parts = [f'camera {fps} fps' if fps else 'camera on', f'{pads} of {len(PADS)} pads in contact']
        parts.append(f'{names} connected' if linked else 'waiting for the call')
        parts.append('logging for replay')
        return dict(ok=True, text='All systems normal: ' + ', '.join(parts) + '.')

    def latency(self, muscles=True, face=True):
        lips = 'Muscles only' if not face and muscles else 'Lips and muscles' if muscles else 'Lips only'
        return [dict(key='read', label=lips, ms=_median(self.lat['read'])),
                dict(key='lock', label='Locking', ms=_median(self.lat['lock'])),
                self.claude_stage(),
                dict(key='sonic', label='Cartesia Sonic', ms=_median(self.lat['sonic']))]

    def claude_stage(self):
        """What Claude adds to a word's trip to the voice: the decoder's repair window (a word waits at most that
        long for Claude), measured from timing.dec. With no window (the default: repairs fix the captions only) or
        none measured, Claude's own reply time, labelled as such: it runs beside the voice, never in its way."""
        window = _median(self.lat['window'])
        if window:
            return dict(key='claude', label="Claude's window", ms=window)
        return dict(key='claude', label='Claude reply', ms=_median(self.lat['claude']))

    def _sentence(self, u, now):
        if u is None:
            return None
        words = []
        for i, w in enumerate(u['locked']):
            src = u['source'][i] if i < len(u['source']) else 'camera'
            rep = u['repaired'].get(i)
            if rep and _bare(w) == rep[0]:     # the decoder hasn't applied the repair yet: show it, punctuation kept
                w = rep[1] + w[len(w.rstrip('.,!?;:')):]
            conf = u['conf'][i] if i < len(u['conf']) else None
            words.append(dict(w=pretty([w], i == 0), src='claude' if rep else src, locked=True,
                              was=pretty([rep[0]], i == 0) if rep else None, conf=conf))
        n = len(u['locked'])
        words += [dict(w=pretty([w], n + j == 0), src=None, locked=False, was=None,
                       conf=u['conf'][n + j] if n + j < len(u['conf']) else None) for j, w in enumerate(u['unlocked'])]
        cands = None
        if u['cands']:
            cands = dict(rows=u['cands'], threshold=u['threshold'])
        return dict(utt_id=u['utt_id'], t=u['t'], words=words, cands=cands, repeat=u['repeat'],
                    emotion=u['emotion'], tone=TONE.get(u['emotion'] or self.emotion[0], ''),
                    speaking=u['speaking'] is not None and now - u['speaking'] < SPEAKING_MAX_S,
                    repairs=[dict(index=i, frm=pretty([f], False), to=pretty([t], False))
                             for i, (f, t) in u['repaired'].items()],
                    spoken=self._said(u), final=u['final'])

    def snapshot(self, now):
        c = self.call
        t_end = c['t_end'] if c['t_end'] is not None else now
        elapsed = round(t_end - c['t_start']) if c['t_start'] is not None else 0
        cur = self.utts.get(self.cur_utt)
        muscles = self.muscles(now)
        snap = dict(
            mode={'waiting': 'waiting', 'connected': 'live', 'ended': 'end'}[c['state']],
            number=self.number, channel=self.channel, switchable=self.switchable, elapsed=elapsed,
            metrics=dict(lips_to_voice=_median(self.lat['lips_to_voice']), fixed=len(self.repairs) + self.captions_fixed,
                         repeats=len(self.repeats)),
            judge_live=self.judge_live, transcript=self._recent(),
            austin=self._sentence(cur, now),
            emotion=dict(label=self.emotion[0], chip=CHIP.get(self.emotion[0], self.emotion[0]),
                         tone=TONE.get(self.emotion[0], '')),
            face_ok=self.face_ok,
            pads=[dict(pin=pin, name=name, level=round(p['level'], 3), contact=p['contact'])
                  for (pin, name), p in zip(PADS, self.emg)],
            muscles=muscles, latency=self.latency(muscles, self.face_ok), status=self.status(now),
            face=self.face_payload(),
        )
        if snap['mode'] == 'end':
            if self._summary is None:       # the call is over: nothing changes, so build it once
                self._summary = self.summary(elapsed, self.face_hidden_s)
            snap['summary'] = self._summary
        return snap

    def _recent(self, n=12, most=40):
        """The last n lines, reaching back to the judge's last line (the page shows it big) when Austin said more
        than n sentences since, up to `most` lines."""
        start = max(0, len(self.transcript) - n)
        judge = next((i for i in range(len(self.transcript) - 1, -1, -1) if self.transcript[i]['who'] == 'judge'), None)
        if judge is not None and judge < start:     # the judge's line, then at most `most` of Austin's after it
            return [self.transcript[judge]] + self.transcript[max(judge + 1, len(self.transcript) - most):]
        return self.transcript[start:]

    def muscles(self, now):
        """The muscle sensors are on: emg.raw is arriving and at least one pad touches the skin."""
        return now - self.seen.get('emg', -1e9) < STALE_S['emg'] and any(p['contact'] for p in self.emg)

    def face_payload(self):
        if self.face_pts is None:
            return None
        return dict(pts=np.round(self.face_pts, 4).tolist(), size=self.face_size)

    def summary(self, elapsed, hidden):
        austin = [s for s in self.transcript if s['who'] == 'austin']
        n_words = sum(len(s['text'].split()) for s in austin)
        locked = [(u, i) for u in self.utts.values() for i in range(len(u['locked']))]
        disagree = {}
        for reader, key in (('lips', 'cam_top'), ('muscles', 'emg_top')):
            pairs = [(u[key][i], u['locked'][i]) for u, i in locked if i < len(u[key]) and u[key][i] is not None]
            same = [_bare(a).lower() == _bare(b).lower() for a, b in pairs]   # Auto-AVSR has no case or punctuation
            disagree[reader] = round(100 * same.count(False) / len(same)) if same else None
        lat = {k: _median(self.lat[k]) for k in ('read', 'lock', 'sonic', 'lips_to_voice')}
        claude = self.claude_stage()
        lat['claude'], lat['claude_label'] = claude['ms'], claude['label']
        said = [TONE.get(s.get('emotion'), '').replace('Sounds ', '') for s in austin]
        tones = {t: said.count(t) for t in TONE_ORDER if said.count(t)}
        return dict(seconds=elapsed, words=n_words, tones=tones, benchmark=self.benchmark,
                    wpm=round(n_words / (elapsed / 60)) if elapsed >= 10 else None,
                    repairs=self.repairs[:], repeats=self.repeats[:], face_hidden_s=round(hidden, 1),
                    emg_words=self.emg_words, disagree=disagree, latency=lat, transcript=self.transcript[:])


HANDLERS = {
    'call.state': CallState._call_state, 'partner.text': CallState._partner,
    'dec.hyp': CallState._hyp, 'dec.lock': CallState._lock, 'dec.repeat': CallState._repeat,
    'claude.repair': CallState._repair, 'claude.caption': CallState._caption, 'voice.say': CallState._say,
    'cam.tick': CallState._tick, 'cam.face': CallState._face_msg, 'emg.raw': CallState._emg,
}
