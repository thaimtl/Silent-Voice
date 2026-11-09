"""The fusion process's logic, without sockets or a clock of its own: bus messages in, bus messages out.

    core = Fusion(JointSearch(lm, lexicon=...), Gate.from_dict(...), LockModel.from_dict(...), svr=LiveSVR1(...))
    out = core.feed(topic, msg, now)      # every cam.tick, cam.final, emg.raw, claude.*, partner.text
    out += core.step(now)                 # every ~20 ms: ends sentences, releases words from the repair window
    for topic, msg in out: pub.send(topic, msg)

`now` is bus.now() live; offline (fusion/offline.py) it is the recording's clock, which the camera's frame times
and the EMG's sample times share (lipread.run --recorder-clock).

One utterance (utt_id) is one camera segment. Every cam.tick is one pass of the joint search (fusion/search.py) over
the whole segment so far; the lock model (fusion/lock.py) then locks words left to right. Locked words go on
dec.lock at once (Claude's repair lane) and on voice.say after the repair window (0 s by default: Claude's repairs
change the captions only, never the voice, so waiting for them buys nothing). When the segment's ticks stop (end_wait_s), or its cam.final comes first, the end
pass reads it once more as a finished sentence; when cam.final's n-best arrives (or final_wait_s passes) the final
pass adds it to the candidates, and any word still below the threshold ends the sentence in "Please repeat"
(dec.repeat) instead of being spoken. A tick that comes after end_wait_s ended its sentence (a camera stall, or the
last tick of a replayed clip, ~0.55 s late) reopens that sentence; once it is done, only frames with the mouth moving
after its reading go on, as a new utterance that reads just those. When the camera caps a segment (8 s) and opens the
next with its last 2 s of crops, the next utterance drops those carried frames (and their EMG and n-best), so no word
is read or spoken twice; a word cut in half at the cap may come out wrong. step() never raises: each stage is guarded,
so a failure loses none of what the others built.

A sentence (Sentence: one utt_id on the bus, so one caption line and one voice context) is every camera segment Austin
mouths until he has finished, not one segment: a pause makes the camera end its segment, but the sentence only holds.
On the hold it asks Claude whether he has finished (dec.pause; claude.end answers P(finished)), and asks again as the
silence grows while the answer is no. It closes when the judge starts talking, when Claude says finished (once the hold
is HOLD_MIN_S old; the bar falls from END_P to END_P_LATE as the silence grows), when Claude doesn't answer (fail open),
at a pause once it has MAX_WORDS words, or at the backstop; mouthing again first goes on with it:
its words are numbered on from the earlier segments' (dec.lock, dec.hyp, voice.say offset, dec.repeat, claude.repair),
and the voice's final waits for the close. A capped segment's continuation is the same sentence too. The earlier
segment's final pass runs when the next one joins (without cam.final's n-best if that hasn't come), so its words are
counted first. Every hold, continuation and close goes on dec.sentence with its reason. Offline fitting
(sentences=False) keeps one sentence per segment, as before.

EMG alone never writes captions: SVR-1 transcribes about 96% of words wrong on its own (dev, svr/dev_compare.py), so
speech the muscles detect while the camera has no face ends in "Please repeat" (reader: camera), not in guesses.
"""
import sys
import time
from collections import deque

import numpy as np

from fusion import lock as lockmod
from fusion.search import CamReading, gate_features, softmax, viterbi_word_ends, word_posteriors

EMG_RAIL = (8, 1015)       # dashboard/state.py: a pad off the skin pins its channel to a rail of the 10-bit range
EMG_FLAT = 1.0             # raw-unit standard deviation below this: no contact
MIN_NEW = 5                # frames (0.2 s, the shortest word) a finished segment's late tick must add to go on,
STILL_THR = 0.12           # ... with the mouth moving in them (lipread/endpoint.py's STILL_THR)
# a word Claude's hint carried over the camera's own pick locks only if the camera gives it at least this share of its
# top word's support (a near tie: Claude breaks it); otherwise the camera's word is the one said
CAM_AGREE = 0.5
PHRASE_WEIGHT = 0.3        # hint weight of a word that only appears in one of Claude's predicted phrases
# When a sentence ends (Sentence). The hold starts once the camera ended its segment (0.5 s of still lips) and the end
# pass ran (end_wait_s): LIPS_STILL_S after the lips stopped. Only the backstop and the fallbacks are fixed times.
LIPS_STILL_S = 0.8
HOLD_MIN_S = 0.5           # a held sentence never closes sooner (~1.3 s of still lips, Larpy's 1.2 s minimum hold)
HOLD_BACKSTOP_S = 8.0      # still held: he gave up mid-sentence
END_P = 0.6                # Claude's P(finished) at which a held sentence closes, as the hold starts; the bar falls
END_P_LATE = 0.3           # ... to this over END_P_FALL_S, so a long silence and a lukewarm Claude close it (on the
END_P_FALL_S = 3.0         # live run of 25/10 a constant 0.6 glued Austin's next sentences on: 85 words, 9 segments)
MAX_WORDS = 30             # a sentence this long closes at its next pause
END_RECHECK_S = 1.5        # Claude said not finished: ask again, with the longer silence, this often
END_WAIT_S = 3.0           # no answer from Claude this long after asking: close (fail open: never hold forever)
NO_CLAUDE_HOLD_S = 1.0     # without Claude (--no-claude, or nothing locked to ask about): hold this long
JUDGE_WORDS = 2            # the judge's live transcript closes a held sentence once it has this many words


class Utterance:
    def __init__(self, utt_id, seg_id, seg_start):
        self.utt_id, self.seg_id, self.seg_start = utt_id, seg_id, seg_start
        self.cam = None
        self.last_tick = None          # `now` of the newest cam.tick
        self.locked = []               # dec.lock word dicts (w, conf, alts, cam, emg, t, cam_top, emg_top)
        self.unlocked, self.unlocked_conf, self.unlocked_src = [], [], []
        self.stable = {}               # position -> (word, passes it has been the best candidate's word there)
        self.queue = deque()           # (release time, index) of locked words waiting out the repair window
        self.spoken = 0                # words sent on voice.say
        self.state = 'open'            # open -> ending (waiting for cam.final) -> done
        self.final_deadline = None
        self.nbest = None
        self.final_sent = False
        self.repeat_sent = False
        self.carried_from = None       # the utterance whose segment the camera capped (8 s) to open this one
        self.sent = None               # its Sentence


class Sentence:
    """Austin's sentence: one utt_id on the bus, made of the camera segments he mouths until he has finished.
    open (a segment is being read) -> holding (his lips stopped) -> closed."""

    def __init__(self, sid):
        self.sid = sid
        self.utts = []                 # its segments' utterances, in order
        self.state = 'open'
        self.hold_t = None             # `now` the current hold began
        self.asked_t = None            # `now` of the last end check sent to Claude
        self.asks = 0                  # end checks sent this hold
        self.p = None                  # Claude's newest P(finished) this hold
        self.answered_t = None
        self.claude = False            # this hold asks Claude
        self.reason = None             # why it closed

    def before(self, u):
        """The locked words of its segments before u's: u's word i is the sentence's word len(before) + i."""
        out = []
        for v in self.utts:
            if v is u:
                break
            out += v.locked
        return out

    def words(self):
        return [x['w'] for v in self.utts for x in v.locked] + (list(self.utts[-1].unlocked) if self.utts else [])


class Fusion:
    def __init__(self, search, gate, lock, svr=None, repair_s=0.3, end_wait_s=0.3, final_wait_s=2.5,
                 use_emg=True, use_lm=True, use_hints=True, idle_emg=True, record=False, oracle=None, sentences=True,
                 claude_end=None):
        """idle_emg: between camera segments, keep reading SVR-1 to catch speech without a face (live only);
        record: keep every pass's judge scores and lock features in self.records (fitting, evaluation);
        oracle(utterance) -> true sentence or None, scored on the final pass when recording (gate fitting);
        sentences: a sentence goes on across pauses (module docstring), else one per segment (offline fitting);
        claude_end: ask Claude whether Austin has finished (default: use_hints)."""
        self.search, self.gate, self.lock, self.svr = search, gate, lock, svr
        self.repair_s, self.end_wait_s, self.final_wait_s = repair_s, end_wait_s, final_wait_s
        self.use_emg, self.use_lm, self.use_hints = use_emg and svr is not None, use_lm, use_hints
        self.records = [] if record else None
        self.oracle = oracle
        self.idle_emg = idle_emg
        self.hold = sentences
        self.claude_end = use_hints if claude_end is None else claude_end
        self.utt = None                # the open utterance
        self.closing = []              # utterances still speaking or waiting for cam.final
        self.recent = {}               # utt_id -> Utterance (its segment's, internal)
        self.sent = None               # the newest Sentence
        self.sentences = {}            # sentence utt_id (on the bus) -> Sentence, for late Claude repairs and answers
        self.next_id = 1
        self.hints, self.hints_until, self.predictability = {}, -1e18, 0.0
        self.phrases = []              # Claude's predicted phrases, as word lists
        self.raw = deque(maxlen=500)   # the last 0.5 s of raw EMG, for the pads' quality
        self.emg_quality = 1.0
        self.pace, self.effort = 1.0, 1.0
        self.last_emg_read = None
        self.idle_read_t = -1e18
        self.emg_only = None           # time an EMG-detected segment started with no camera segment
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(1, thread_name_prefix='svr')

    # ------------------------------------------------------------------ input
    def feed(self, topic, msg, now):
        out = []
        if topic == 'emg.raw':
            x = np.asarray(msg['samples']).reshape(-1, 4)
            self.raw.extend(x)
            if self.svr is not None:
                self.svr.feed(x, msg.get('t0'))
        elif topic == 'cam.tick':
            out += self._tick(msg, now)
        elif topic == 'cam.final':
            if self.utt is not None and self.utt.seg_id == int(msg['seg_id']):
                out += self._end(self.utt, now)     # a short reply's beam can beat end_wait_s: end it now
            for u in self.closing:
                if u.seg_id == int(msg['seg_id']) and u.state == 'ending':
                    # a capped continuation's n-best also reads the carried words, already spoken: not a candidate
                    u.nbest = [] if u.carried_from is not None else [
                        dict(text=str(h['text']), score=float(h['score'])) for h in msg.get('nbest') or []]
                    out += self._final(u, now)
        elif topic == 'claude.predict' and self.use_hints:
            h = {}
            for p in msg.get('phrases') or []:
                for word in self.search_words(p):
                    h[word] = max(h.get(word, 0.0), PHRASE_WEIGHT)
            for item in msg.get('words') or []:
                for word in self.search_words(str(item.get('w', ''))):
                    h[word] = max(h.get(word, 0.0), float(item.get('weight', 0.0)))
            self.hints = h
            self.phrases = [self.search_words(str(p)) for p in msg.get('phrases') or []]
            self.hints_until = now + float(msg.get('ttl_s', 10.0))
            self.predictability = float(msg.get('predictability', 0.0))
        elif topic == 'claude.repair':
            out += self._repair(msg, now)
        elif topic == 'claude.end':
            s = self.sentences.get(msg.get('utt_id'))
            if s is not None and s.state == 'holding':
                s.p, s.answered_t = float(msg.get('p_complete', 0.0)), now
        elif topic == 'partner.text':
            text = str(msg.get('text', ''))
            if msg.get('is_final'):
                self.search.lm.add_text(text)
            if len(text.split()) >= JUDGE_WORDS:
                out += self._close_held('judge spoke', now)
        return out

    @staticmethod
    def search_words(text):
        from fusion.search import norm_words
        return norm_words(text)

    def step(self, now):
        """Never raises: each stage is guarded, so one failing (a CUDA error in the idle EMG read, say) never loses
        what the others built, like voice.say words _release already counted as spoken."""
        out = []
        u = self.utt
        if u is not None and u.last_tick is not None and now - u.last_tick >= self.end_wait_s:
            out += self._guard('end pass', self._end, u, now)
        for v in list(self.closing):
            if v.state == 'ending' and now >= v.final_deadline:
                out += self._guard('final pass', self._final, v, now)
        out += self._guard('sentence end', self._held, now)
        for v in ([self.utt] if self.utt else []) + self.closing:
            out += self._guard('release', self._release, v, now)
        # done: its sentence's last segment once it sent the final, an earlier one once its words are out
        self.closing = [v for v in self.closing if not (v.state == 'done' and (
            v.final_sent or (not v.queue and v.sent is not None and v.sent.utts[-1] is not v)))]
        if self.utt is None and self.use_emg and self.idle_emg and now - self.idle_read_t >= 0.1:
            out += self._guard('idle EMG read', self._idle_emg, now)
        return out

    @staticmethod
    def _guard(what, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001  logged; the captions and the voice go on
            print(f'fusion: {what}: {type(e).__name__}: {e}', file=sys.stderr)
            return []

    # ------------------------------------------------------------------ passes
    def _tick(self, msg, now):
        out = []
        seg, start = int(msg['seg_id']), float(msg['seg_start'])
        current = self.utt is not None and self.utt.seg_id == seg and self.utt.seg_start == start
        ended = None if current else self._ended(seg, start)
        if ended is not None:
            # A tick of a segment whose sentence end_wait_s already ended: the camera stalled (its last tick of a
            # replayed clip comes ~0.55 s late) or a pass ran long. It is still that sentence, never a new one.
            if self.utt is not None:        # a newer segment is open: too late to go on, only the reading counts
                if ended.state == 'ending':
                    ended.cam = self._cam(msg, ended) or ended.cam
                return out
            if ended.state == 'ending':     # reopen it: the cam.final that follows still ends it at once
                self.closing.remove(ended)
                ended.state, ended.final_deadline = 'open', None
                self.utt = ended
                s = ended.sent
                if s is not None and s.state == 'holding' and s.utts[-1] is ended:
                    out += self._sentence_event(s, 'continue', now)
                    s.state, s.hold_t = 'open', None
            elif self._goes_on(msg, ended):
                out += self._open(seg, start, now, carried_from=ended)  # done: only frames after its reading are new
            else:                           # a late last tick's couple of trailing frames hold no word
                return out
        prev = None
        if self.utt is not None and not (self.utt.seg_id == seg and self.utt.seg_start == start):
            prev = self.utt
            out += self._end(self.utt, now, hold=False)     # the next segment opens at once: no hold between
        if self.utt is None:
            capped = prev is not None and prev.cam is not None and start <= prev.cam.frame_t[-1]
            out += self._open(seg, start, now, carried_from=prev if capped else None)   # capped: carries the last 2 s
        u = self.utt
        cam = self._cam(msg, u)
        if cam is None:                     # only carried frames so far
            return out
        u.cam = cam
        t0 = time.perf_counter()
        out += self._pass(u, now, final=False)
        u.last_tick = now + (time.perf_counter() - t0)  # end_wait_s counts from the end of the pass: a slow pass
        return out                                      # is never mistaken for the end of the sentence

    @staticmethod
    def _goes_on(msg, ended):
        """Whether a tick of a finished sentence's segment holds more speech: MIN_NEW new frames, the mouth moving."""
        if ended.cam is None:
            return False
        new = np.asarray(msg['frame_t'], np.float64) > ended.cam.frame_t[-1]
        act = np.asarray(msg.get('activity', np.ones(len(new))), np.float64)[new]
        return int(new.sum()) >= MIN_NEW and bool((act >= STILL_THR).any())

    def _open(self, seg, start, now, carried_from=None):
        """A new segment's utterance: it goes on with the sentence unless that closed (or sentences are off)."""
        out = []
        s = self.sent
        if self.hold and s is not None and s.state != 'closed':
            for v in s.utts:        # count the earlier segments' words first: read what still waits for cam.final now
                if v.state == 'ending':
                    out += self._final(v, now)      # a "Please repeat" there closes the sentence
        uid = self.next_id
        self.next_id += 1
        if not self.hold or s is None or s.state == 'closed':
            s = self.sent = Sentence(uid)           # its first segment's id: offline, utt_id = segment as before
            self.sentences[uid] = s
            for k in [k for k in self.sentences if k < uid - 20]:
                del self.sentences[k]
        else:
            out += self._sentence_event(s, 'continue', now)
            s.state, s.hold_t = 'open', None
        self.utt = Utterance(uid, seg, start)
        self.utt.carried_from = carried_from
        self.utt.sent = s
        s.utts.append(self.utt)
        self.recent[uid] = self.utt
        for k in [k for k in self.recent if k < uid - 20]:
            del self.recent[k]
        self.emg_only = None
        return out

    @staticmethod
    def _cam(msg, u):
        """The tick's reading, without the frames a capped continuation carries over: the capped segment read them
        (and the voice said them), so they're never read twice. None when nothing is left."""
        if u.carried_from is None or u.carried_from.cam is None:
            return CamReading(msg)
        keep = np.asarray(msg['frame_t'], np.float64) > u.carried_from.cam.frame_t[-1]
        if not keep.any():
            return None
        m = dict(msg)
        for k in ('frame_t', 'topk_ids', 'topk_logp', 'blank_logp', 'activity'):
            if k in m:
                m[k] = np.asarray(m[k])[keep]
        return CamReading(m)

    def _ended(self, seg, seg_start):
        """The ended utterance of this camera segment, if any (seg_start too: a restarted camera reuses seg_ids)."""
        found = [v for v in list(self.closing) + list(self.recent.values())
                 if v is not self.utt and v.state != 'open' and v.seg_id == seg and v.seg_start == seg_start]
        return max(found, key=lambda v: v.utt_id) if found else None   # the newest: a stall can split it again

    def _end(self, u, now, hold=True):
        """The segment's ticks stopped: read it once more as a finished sentence, then wait for cam.final; its
        sentence holds (hold=False: the next segment is already here)."""
        out = self._guard('end pass', self._pass, u, now, True) if u.cam is not None else []
        u.state, u.final_deadline = 'ending', now + self.final_wait_s   # even after a failed pass: never retried
        self.closing.append(u)
        if self.utt is u:
            self.utt = None
        s = u.sent
        if s is not None and s.state == 'open' and s.utts[-1] is u:
            if not self.hold:
                s.state, s.reason = 'closed', 'segment ended'   # offline: one sentence per segment
            elif hold:
                s.state, s.hold_t, s.p, s.answered_t, s.asks = 'holding', now, None, None, 0
                s.claude = self.claude_end and bool(s.words())
                out += self._sentence_event(s, 'hold', now)
                if s.claude:
                    out += self._ask_end(s, now)
        return out

    # ------------------------------------------------------------------ the end of a sentence
    def _held(self, now):
        """A held sentence: close it, ask Claude again, or keep holding (module docstring)."""
        s = self.sent
        if s is None or s.state != 'holding':
            return []
        held = now - s.hold_t
        bar = END_P - (END_P - END_P_LATE) * min(1.0, held / END_P_FALL_S)
        why = None
        if held >= HOLD_BACKSTOP_S:
            why = 'backstop'
        elif held >= HOLD_MIN_S and sum(len(v.locked) for v in s.utts) >= MAX_WORDS:
            why = 'long sentence'
        elif not s.claude:
            if held >= NO_CLAUDE_HOLD_S:
                why = 'no Claude'
        elif held >= HOLD_MIN_S:
            if s.p is not None and s.p >= bar:
                why = 'Claude: finished'
            elif s.p is None and now - s.asked_t >= END_WAIT_S:
                why = 'Claude did not answer'
        if why:
            return self._close(s, why, now)
        if s.claude and s.p is not None and s.p < bar and now - max(s.answered_t, s.asked_t) >= END_RECHECK_S:
            return self._ask_end(s, now)    # the answer was no: ask again with the longer silence
        return []

    def _ask_end(self, s, now):
        s.asks += 1
        s.asked_t = now
        words = s.words()
        return [('dec.pause', {'utt_id': s.sid, 'words': words, 'n_words': len(words), 'ask': s.asks,
                               'silence_s': round(now - s.hold_t + LIPS_STILL_S, 2)})]

    def _close(self, s, why, now):
        """Close a sentence: its last segment sends the voice's final once its words are out (_release)."""
        out = self._sentence_event(s, 'close', now, reason=why)
        s.state, s.reason = 'closed', why
        return out

    def _close_held(self, why, now):
        s = self.sent
        return self._close(s, why, now) if s is not None and s.state == 'holding' else []

    @staticmethod
    def _sentence_event(s, event, now, reason=None):
        """dec.sentence: every hold, continuation and close, with why (the logger keeps them; fusion.run prints)."""
        held = now - s.hold_t if s.hold_t is not None else 0.0
        return [('dec.sentence', {'utt_id': s.sid, 'event': event, 'reason': reason, 'held_s': round(held, 2),
                                  'p': s.p, 'asks': s.asks, 'segments': len(s.utts),
                                  'words': sum(len(v.locked) for v in s.utts)})]

    def _final(self, u, now):
        """The final pass, with the camera's attention n-best: every word left is spoken, unless the sentence as a
        whole is unreadable (LockModel.settle); then the words left end in "Please repeat"."""
        out = self._guard('final pass', lambda: self._pass(u, now, final=True, nbest=u.nbest or (), settle=True)) \
            if u.cam is not None else []
        u.state = 'done'                    # even after a failed pass: its locked words are said, the rest repeated
        rest = list(range(len(u.locked), len(u.locked) + len(u.unlocked)))
        if rest and not u.repeat_sent:
            out.append(('dec.repeat', self._repeat_msg(u, rest)))
            u.repeat_sent = True
            if u.sent is not None and u.sent.state != 'closed':
                out += self._close(u.sent, 'please repeat', now)   # what follows is a new sentence
        words = ' '.join(w['w'] for w in u.locked)
        if words:
            self.search.lm.add_text(words)      # Austin's own words join the call cache
        return out

    def _weights(self, cam, emg_read):
        f = gate_features(cam, emg_read, self.emg_quality)
        g = self.gate.weights(f)
        return f, {'cam': g[0], 'emg': g[1] if emg_read is not None else 0.0, 'lm': g[2] if self.use_lm else 0.0,
                   'beta': self.gate.beta if self.use_hints else 0.0, 'gamma': self.gate.gamma,
                   'final': self.gate.final, 'unseen': self.gate.unseen}

    def _pass(self, u, now, final, nbest=(), settle=False):
        t_start = time.perf_counter()
        self._quality()
        # SVR-1 re-reads the segment on the GPU while the camera beam runs here; the gate uses the newest EMG read
        # it has (the previous pass's, 100 ms old)
        start = float(u.cam.frame_t[0]) if u.carried_from is not None else u.seg_start   # carried EMG: read already
        job = self._pool.submit(self.svr.read, True, start) if self.use_emg else None
        gf, w = self._weights(u.cam, self.last_emg_read if self.use_emg else None)
        hints = self.hints if (self.use_hints and now < self.hints_until) else {}
        locked = [x['w'] for x in u.locked]
        cands = self.search.candidates(u.cam, w, hints, locked, [h['text'] for h in nbest], final,
                                       phrases=self.phrases if hints else (),
                                       locked_tokens=[x['tokens'] for x in u.locked])
        emg_read = job.result() if job is not None else None
        if emg_read is not None:
            self.last_emg_read = emg_read
            self.pace = emg_read['pace'] if np.isfinite(emg_read['pace']) else 1.0
            if emg_read['effort'] is not None and np.isfinite(emg_read['effort']):
                self.effort = emg_read['effort']
            if w['emg'] == 0.0:             # this pass's weights came before any EMG read: weigh it in now
                gf, w = self._weights(u.cam, emg_read)
            from svr.decoders import greedy_decode
            self.search.extend(cands, [greedy_decode(emg_read['letter_logp'])], locked)
        elif self.use_emg:
            w['emg'] = 0.0
        emg_fn = self.svr.verify if emg_read is not None else None
        scores = self.search.judge_scores(cands, u.cam, emg_fn, hints, final, nbest)
        P = softmax(self.search.combine(scores, w))
        Pc = softmax(self.gate.t_cam * scores[:, 0])
        Pe = softmax(self.gate.t_emg * scores[:, 1]) if emg_read is not None else np.full(len(cands), 1.0 / len(cands))
        best, post = word_posteriors(cands, P, start=len(locked))
        tau = self.lock.threshold(self.predictability if hints else 0.0, final)
        pause = 1.0 - float(u.cam.activity[-8:].mean()) if u.cam is not None and len(u.cam.activity) else 1.0
        quiet = 1.0 - float(np.mean(emg_read['speak_prob'][-26:])) if emg_read is not None else 0.0
        words = []
        n_best = len(best.words)
        for k, d in enumerate(post):
            i = len(locked) + k
            word = best.words[i]
            mc = {c.words[i]: 0.0 for c in cands if i < len(c.words)}
            me = dict(mc)
            for c, qc, qe in zip(cands, Pc, Pe):
                if i < len(c.words):
                    mc[c.words[i]] += qc
                    me[c.words[i]] += qe
            prev = u.stable.get(i)
            stable = prev[1] + 1 if prev and prev[0] == word else 0
            u.stable[i] = (word, stable)
            f = lockmod.features(d.get(word, 0.0), mc.get(word, 0.0), me.get(word, 0.0) if emg_read is not None
                                 else mc.get(word, 0.0), hints.get(word, 0.0), final or i < n_best - 1, pause, quiet,
                                 stable, u.cam.face_ok if u.cam is not None else False, self.emg_quality)
            p = float(self.lock.prob(f))
            alts = self._alts(word, i, best, cands, d)
            words.append(dict(i=i, w=word, p=p, f=f, alts=alts, cam=mc.get(word, 0.0), tokens=best.tokens[i],
                              emg=me.get(word, 0.0) if emg_read is not None else None,
                              cam_top=max(mc, key=mc.get) if mc else word,
                              emg_top=max(me, key=me.get) if (me and emg_read is not None) else None,
                              claude=hints.get(word, 0.0), cands=sorted(d.items(), key=lambda kv: -kv[1])[:3],
                              mc=mc, me=me))
        prior = u.sent.before(u) if u.sent is not None else []     # the sentence's earlier segments' words
        sid = u.sent.sid if u.sent is not None else u.utt_id
        new = []
        if settle and self.lock.settle([x['w'] for x in prior] + locked + [x['w'] for x in words],
                                       [x['conf'] for x in prior + u.locked] + [x['p'] for x in words]):
            # a wrong guess beats "Please repeat"; but where only Claude's hint carried a word, the camera's own
            # reading goes to the voice (Claude predicts, it never decides a word the camera doesn't nearly agree on)
            new = [dict(x, w=x['cam_top'], claude=0.0) if self._claude_only(x) else x for x in words]
        else:
            for x in words:   # left to right: a word locks only after every word before it
                if self._claude_only(x) or not self.lock.may_lock(x['f'], x['p'], tau):
                    break
                new.append(x)
        for x in new:
            u.locked.append(dict(w=x['w'], conf=x['p'], alts=x['alts'], cam=x['cam'], emg=x['emg'], t=now,
                                 cam_top=x['cam_top'], emg_top=x['emg_top'], tokens=x['tokens']))
            u.queue.append((now + self.repair_s, len(u.locked) - 1))
        rest = words[len(new):]
        u.last_words = rest
        # the camera reads no word in this segment and the pads hear no speech: the search's guesses ('s', 'ok') are
        # not a word in his mouth, so none is shown (the voice reads an empty unlocked as his pause)
        if u.cam is not None and u.cam.greedy == '' and (emg_read is None or quiet >= 0.5):
            rest = []
        u.unlocked = [x['w'] for x in rest]
        u.unlocked_conf = [x['p'] for x in rest]
        u.unlocked_src = [self._source(x) for x in rest]
        ms = (time.perf_counter() - t_start) * 1000
        if self.records is not None:
            rec = dict(utt_id=u.utt_id, sentence=sid, seg_id=u.seg_id, seg_start=u.seg_start, t=now, final=final,
                       with_nbest=bool(nbest), nbest=list(nbest), gate=gf, weights=dict(w), scores=scores,
                       cands=[c.words for c in cands], sources=[c.source for c in cands], best=best.words,
                       n_locked_before=len(locked), locked_now=[x['w'] for x in new],
                       words=[dict(i=x['i'], w=x['w'], p=x['p'], f=x['f'], cam_top=x['cam_top'],
                                   emg_top=x['emg_top']) for x in words], tau=tau, ms=ms,
                       cam_frame_t=float(u.cam.frame_t[-1]) if u.cam is not None else None)
            if final and self.oracle is not None:
                truth = self.oracle(u)
                if truth:
                    from fusion.search import Cand, norm_words
                    tw = norm_words(truth)
                    tc = [Cand(tw, [self.search.pieces.word_ids(x) for x in tw], 'truth')]
                    rec['truth'] = tuple(tw)
                    rec['truth_scores'] = self.search.judge_scores(tc, u.cam, emg_fn, hints, True, nbest)[0]
                    rec['truth_in_cands'] = tuple(tw) in set(rec['cands'])
                    if u.cam is not None:   # when each true word ends on the lips: the lock delay's reference
                        rec['truth_ends'] = viterbi_word_ends(u.cam, tc[0].tokens)
            self.records.append(rec)
        out = []
        if new:
            out.append(('dec.lock', {'utt_id': sid, 'words': [dict(w=x['w'], conf=x['p'], alts=x['alts'],
                                                                  cam=x['cam'], emg=x['emg'], t=now,
                                                                  cam_top=x['cam_top'], emg_top=x['emg_top'])
                                                             for x in new]}))
        newest = words[-1] if words else None
        # his mouth is moving on a word now (the camera's activity over the last 0.3 s, or the pads hear speech): the
        # voice ends a phrase when this goes False. Not "unlocked words": those lag 1-3 s behind his lips
        lips = u.cam is not None and u.cam.greedy != '' and 1.0 - pause >= STILL_THR
        mouthing = not final and (lips or (emg_read is not None and quiet < 0.5))
        hyp = {'utt_id': sid, 'mouthing': mouthing, 'locked': [x['w'] for x in prior + u.locked], 'unlocked': u.unlocked,
               'confidence': [x['conf'] for x in prior + u.locked] + u.unlocked_conf,
               'source': [self._source(x) for x in prior + u.locked] + u.unlocked_src, 'threshold': tau}
        if newest is not None:
            hyp['cands'] = [dict(w=c, cam=newest['mc'].get(c, 0.0), emg=newest['me'].get(c, 0.0) if
                                 emg_read is not None else 0.0, claude=hints.get(c, 0.0)) for c, _ in newest['cands']]
        out.append(('dec.hyp', hyp))
        out.append(('timing.dec', {'stage': 'lock', 'ms': ms, 'utt_id': sid, 'final': final,
                                   'candidates': len(cands)}))
        return out

    def _alts(self, word, i, best, cands, d):
        """What word i could be instead (Claude may only swap it for one of these): words that candidates identical
        to the best one everywhere else put there, then its lip look-alikes; real words only. Position i of a
        candidate with a word dropped or added is a neighbour shifted in (43% of locked words had one on session 3)."""
        pron = getattr(getattr(self.search, 'lexicon', None), 'pron', None)
        n = len(best.words)
        swaps = {c.words[i] for c in cands if len(c.words) == n and c.words[:i] == best.words[:i]
                 and c.words[i + 1:] == best.words[i + 1:]}
        out = [x for x, _ in sorted(d.items(), key=lambda kv: -kv[1]) if x in swaps]
        out += [x for x in self.search.lookalikes(word) if x not in out]
        near = set(best.words[max(0, i - 2):i] + best.words[i + 1:i + 3])
        return [x for x in out if x != word and x not in near and (pron is None or x in pron)][:4]

    # ------------------------------------------------------------------ voice, repairs, repeats
    @staticmethod
    def _claude_only(x):
        """Claude's hint carried this word over the camera's and EMG's own picks, and the camera gives it less than
        CAM_AGREE of its top word's support: a hint alone never locks a word."""
        if not x.get('claude') or x['w'] in (x.get('cam_top'), x.get('emg_top')):
            return False
        return x['mc'].get(x['w'], 0.0) < CAM_AGREE * x['mc'].get(x['cam_top'], 0.0)

    @staticmethod
    def _source(x):
        """Who settled the word: the camera when its own pick, else EMG when EMG's pick, else Claude if predicted."""
        if x.get('cam_top') == x['w']:
            return 'camera'
        if x.get('emg_top') == x['w']:
            return 'EMG'
        if x.get('claude'):
            return 'Claude'
        return 'camera'

    def _release(self, u, now):
        out = []
        words = []
        while u.queue and u.queue[0][0] <= now:
            _, i = u.queue.popleft()
            words.append(u.locked[i].get('voice', u.locked[i]['w']))
        s = u.sent
        # the voice's final: the sentence closed and this, its last segment, has said everything
        last = u.state == 'done' and not u.queue and (s is None or (s.state == 'closed' and s.utts[-1] is u))
        sid, base = (s.sid, len(s.before(u))) if s is not None else (u.utt_id, 0)
        if words or (last and not u.final_sent):
            out.append(('voice.say', {'utt_id': sid, 'words': words, 'offset': base + u.spoken, 'final': last,
                                      'pace': float(self.pace), 'effort': float(self.effort)}))
            if words:
                out.append(('timing.dec', {'stage': 'window', 'ms': self.repair_s * 1000, 'utt_id': sid}))
            u.spoken += len(words)
            u.final_sent = u.final_sent or last
        return out

    def _repair(self, msg, now):
        """Claude's repairs and please-repeats, by the sentence's word index (its segments' words numbered on)."""
        s = self.sentences.get(msg.get('utt_id'))
        out = []
        if s is None:
            return out
        at = [(v, j) for v in s.utts for j in range(len(v.locked))]     # sentence index -> (segment, its index)
        for e in msg.get('edits') or []:
            i = int(e.get('index', -1))
            words = self.search_words(str(e.get('to', '')))
            frm = self.search_words(str(e.get('from', '')))
            if 0 <= i < len(at) and len(words) == 1:
                v, j = at[i]
                if frm[:1] == [v.locked[j]['w']]:
                    v.locked[j].setdefault('voice', v.locked[j]['w'])
                    v.locked[j]['w'] = words[0]     # the captions only: the voice says what locked, window or not
        # Claude's "please repeat" only while something of the sentence is still unspoken: once the voice has said it
        # all, asking to repeat a sentence the judge already heard would only confuse the screen
        unspoken = any(j >= v.spoken for v, j in at) or any(v.unlocked for v in s.utts)
        if msg.get('please_repeat') and unspoken and not any(v.repeat_sent for v in s.utts):
            out.append(('dec.repeat', {'utt_id': s.sid, 'reason': {
                'reader': 'Claude', 'words': [dict(index=i, w=v.locked[j]['w'], alts=v.locked[j]['alts'])
                                              for i, (v, j) in enumerate(at) if j >= v.spoken]}}))
            for v in s.utts:
                v.repeat_sent = True
        return out

    def _repeat_msg(self, u, indices):
        rest = {x['i']: x for x in getattr(u, 'last_words', [])}
        base = len(u.sent.before(u)) if u.sent is not None else 0
        words, cam_low = [], 0
        for i in indices:
            x = rest.get(i)
            if x is None:
                continue
            words.append(dict(index=base + i, w=x['w'], alts=x['alts'], p=x['p'], cam=x['cam'], emg=x['emg']))
            cam_low += (x['emg'] is not None and x['cam'] < x['emg'])
        face = u.cam.face_ok if u.cam is not None else False
        has_emg = any(x['emg'] is not None for x in words)
        reader = 'camera' if (not face or not has_emg or cam_low * 2 > len(words)) else 'EMG'
        sid = u.sent.sid if u.sent is not None else u.utt_id
        return {'utt_id': sid, 'reason': {'reader': reader, 'words': words, 'threshold': self.lock.threshold(
            self.predictability)}}

    def _quality(self):
        """1 - the worst pad's share of railed samples over the last 0.5 s; 0 for a flat (unconnected) pad."""
        if len(self.raw) < 50:
            return
        x = np.asarray(self.raw, np.float64)
        railed = ((x <= EMG_RAIL[0]) | (x >= EMG_RAIL[1])).mean(0)
        flat = x.std(0) < EMG_FLAT
        self.emg_quality = float(1.0 - max(railed.max(), float(flat.any())))

    def _idle_emg(self, now):
        """No camera segment: keep SVR-1's own segmentation going; speech it hears without a face -> Please repeat."""
        self.idle_read_t = now
        r = self.svr.read(camera_active=False)
        if r is None:
            return []
        if r['segment_started']:
            self.emg_only = now
        if r['segment_ended'] and self.emg_only is not None:
            self.emg_only = None
            uid = self.next_id
            self.next_id += 1
            return [('dec.repeat', {'utt_id': uid, 'reason': {'reader': 'camera', 'words': []}})]
        return []
