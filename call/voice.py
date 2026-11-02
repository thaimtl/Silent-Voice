"""Austin's voice: voice.say words -> Cartesia Sonic over one long-lived websocket -> mu-law audio for the call.

One Sonic context per utterance (utt_id), continue=True ("more to come"), the final message closes it, so speech starts
while Austin is still mouthing. Every chunk carries a leading space after the first and the last one closing
punctuation, so Sonic joins them as one sentence, in one continuous recording.

Phrases (the default, CARTESIA_PACED=1): Sonic sounds natural only with a few words at once, so locked words go as
phrases, each its own flush, cut at his pauses (dec.hyp), never while he keeps mouthing unless PHRASE_MAX words wait.
Sonic's padding is trimmed (call/audio.py) and each phrase plays as soon as it arrives.

Unpaced (CARTESIA_PACED=0): audio is played as it arrives, with the server buffer CARTESIA_MAX_BUFFER_MS, and
optionally words grouped client-side (CARTESIA_PHRASE_WORDS=3 with CARTESIA_MAX_BUFFER_MS=0).

Audio order: each utterance's audio goes into its own queue and one player task plays utterances in the order they
started, so two sentences never interleave on the phone, and on_done fires only after an utterance's last audio is
queued (the server puts a Twilio mark behind it).

Reconnect: the SDK's connection manager caches its first connection, so its own reconnect hands back the dead socket;
on a failed push we open a fresh connection ourselves and resend that phrase. Phrases of open utterances whose audio
had not arrived go to the fallback with the rest of their sentence (paced; unpaced, they are lost and logged).

Fallback (docs/REALTIME_PIPELINE.md: "If streaming voice glitches, the voice falls back to speaking whole sentences at
the end"): words whose streaming push still fails after the reconnect are kept, and when the utterance ends they are
spoken in one non-streaming request (tts.generate), after whatever did stream.

voice.say may carry only the new words or the whole utterance so far; Utterance.new_words() handles both, and an
explicit `offset` (index of the first word in the message) removes the guesswork when the decoder sends it.
Emotion comes from the newest cam.tick (Evan's camera reader); pace and effort from voice.say. All three go through
generation_config: emotion name, speed 0.6-1.5, volume 0.5-2.0, sent with every push. Phrases: a face held
FACE_HOLD_S mid-sentence starts a new recording (Sonic keeps one recording's mood); a clench is an inline <volume>.
"""
import asyncio
import logging
import re
import time

import numpy as np

from call.audio import RATE, StreamTrim, to_mulaw

log = logging.getLogger(__name__)

OUTPUT = {'container': 'raw', 'encoding': 'pcm_mulaw', 'sample_rate': 8000}
PCM = {'container': 'raw', 'encoding': 'pcm_s16le', 'sample_rate': RATE}     # phrases: trimmed here, then mu-law
# camera label (live/types.py EMOTIONS) -> Sonic emotion; all five exist in Sonic's list
# "happy" is weak in Sonic; "excited" carries
EMOTION = {'neutral': 'neutral', 'happy': 'excited', 'sad': 'sad', 'angry': 'angry', 'surprised': 'surprised'}
EXCLAIM = {'excited', 'surprised', 'angry'}    # Sonic emotions whose sentences end in '!'
MIN_CONFIDENCE = 0.5        # below this the face label is a guess: speak neutral
FACE_HOLD_S = 0.8           # a new face must hold this long to split a sentence into a fresh recording
SPEED_BIAS = 1.1            # speech plays about as fast as Austin mouths; 10% faster keeps the voice up with him
FIRST_WORDS, NEXT_WORDS, MAX_WORDS = 2, 3, 5
PAUSE_S = 0.5               # Austin stopped mouthing for this long: speak what is waiting
# phrases go at his pauses only: a 3- then 4-word cut while he kept mouthing held 56% of the words, the first word of a
# sentence 1.4 s (logs/20261025-072828-demo); a long stretch without a pause still goes at PHRASE_MAX words
PHRASE_MAX = 4
PHRASE_PAUSE_S = 0.05       # phrases: his decoder shows nothing being mouthed for this long: he paused, send
PHRASE_IDLE_S = 0.6         # phrases: no new word and no word in his mouth for this long: send (no dec.hyp either)
PHRASE_TAIL_S = 0.15        # the pause kept after each phrase (Sonic pads it with ~0.5 s)
CLENCH = 1.4                # phrases: a word mouthed with at least this effort is spoken louder ...
CLENCH_VOLUME = 1.8         # ... at least this loud (Sonic: small volume changes are hard to hear), the rest at 1.0
CONTEXT_TIMEOUT_S = 20.0    # a context that hears nothing from Sonic for this long is given up (dead socket)
REUSE_S = 10.0              # a message for a finished utt_id older than this starts a new utterance (decoder restart)
STALE_S = 3.0               # an open utterance silent this long, with a newer one waiting to play, is closed (lost final)
FUNCTION_WORDS = {'a', 'an', 'the', 'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'with', 'my', 'your',
                  'his', 'her', 'our', 'their', 'is', 'are', 'was', 'were', 'i', 'you', 'we', 'they', 'it', 'that',
                  'this', 'very', 'so', 'if', 'than', 'from', 'by', 'as', 'be', 'have', 'has', 'had', 'will',
                  'would', 'can'}
QUESTION_WORDS = {'what', 'who', 'where', 'when', 'why', 'how', 'which'}
# these start a question only with a subject next ("can you", "is it"), not "have a great day"
QUESTION_AUX = {'do', 'does', 'did', 'is', 'are', 'can', 'could', 'would', 'will', 'should', 'have', 'has', 'was',
                'were', 'am'}
SUBJECTS = {'i', 'you', 'we', 'they', 'he', 'she', 'it', 'there', 'that', 'this', 'your', 'my', 'the'}


def speakable(words, first):
    """Lip-reader words (UPPERCASE, no punctuation) -> text Sonic reads naturally. All caps can be read as letters,
    so lowercase, then capitalise the sentence start and 'I' / "I'm"-style words."""
    out = []
    for i, w in enumerate(words):
        w = w.lower()
        if w == 'i' or w.startswith("i'"):
            w = 'I' + w[1:]
        if first and i == 0:
            w = w[:1].upper() + w[1:]
        out.append(w)
    return ' '.join(out)


def is_question(sent):
    """The sentence so far starts like a question: a question word, or an auxiliary then its subject."""
    first = sent[0].lower() if sent else ''
    return first in QUESTION_WORDS or (first in QUESTION_AUX and len(sent) > 1 and sent[1].lower() in SUBJECTS)


def end_mark(sent, emotion=None):
    """Closing punctuation (Sonic waits for it): '?' for a question, '!' for an excited face, else '.'."""
    if is_question(sent):
        return '?'
    return '!' if emotion in EXCLAIM else '.'


def generation_config(emotion=None, confidence=1.0, pace=None, effort=None):
    """Map the face label, pace and effort onto Sonic's generation_config. Pace and effort are relative to Austin's
    normal (1.0 = normal; svr/live.py: effort = mouthing RMS / his median speaking RMS): speed = pace x SPEED_BIAS,
    volume = effort, each clamped to Sonic's range."""
    cfg = {'emotion': EMOTION.get(emotion, 'neutral') if confidence >= MIN_CONFIDENCE else 'neutral',
           'speed': round(min(1.5, max(0.6, SPEED_BIAS * (1.0 if pace is None else float(pace)))), 2)}
    if effort is not None:
        cfg['volume'] = round(min(2.0, max(0.5, float(effort))), 2)
    return cfg


def ready(pending, spoken, phrase_words=1):
    """How many of the pending words to send now (0 = wait for more). phrase_words=1, the repo plan: every word as soon
    as it arrives. phrase_words>1, the opt-in experiment: 2 words first, then phrase_words, never ending on a function
    word, at most MAX_WORDS."""
    n = len(pending)
    if phrase_words <= 1:
        return n
    if n >= MAX_WORDS:
        return MAX_WORDS
    need = FIRST_WORDS if not spoken else phrase_words
    if n >= need and pending[-1].lower() not in FUNCTION_WORDS:
        return n
    return 0


def phrase_ready(pending, spoken=1):
    """Phrases: how many pending words to send now without waiting for a pause (0: wait). Only PHRASE_MAX words: he
    kept mouthing that long without a pause."""
    n = len(pending)
    return n if n >= PHRASE_MAX else 0


class Flush:
    """One phrase sent to Sonic: its words, the settings it was sent with, and its audio as it streams in."""
    __slots__ = ('words', 'gen', 'text', 'trim', 'carry', 'got', 'samples')

    def __init__(self, words, gen, text=''):
        self.words, self.gen = words, gen
        self.text = re.sub(r'\s+', ' ', re.sub(r'<[^>]*>', '', text)).strip()   # as spoken, without the tags
        self.trim = StreamTrim(post_s=PHRASE_TAIL_S)
        self.carry = b''                    # an odd byte left over between chunks
        self.got = False                    # any audio came
        self.samples = 0                    # samples passed on to the player


class Phrase:
    """A piece of a phrase's speech for the player (int16, trimmed); done: its last piece (then it is logged)."""
    __slots__ = ('words', 'text', 'pcm', 'gen', 'done', 'samples')

    def __init__(self, f, pcm, done=False):
        self.words, self.text, self.gen = ' '.join(f.words), f.text, f.gen
        self.pcm, self.done, self.samples = pcm, done, f.samples


class Utterance:
    """One utt_id: words taken (never spoken twice), words waiting to form a phrase, its Sonic context, its audio."""

    def __init__(self, utt_id):
        self.utt_id, self.sent, self.final = utt_id, [], False
        self.pending, self.spoken = [], 0
        self.pending_gen = []               # phrases: per pending word, his face, pace and clench when it locked
        self.flushes = {}                   # phrases: this context's flushes, 1, 2, ... -> Flush
        self.n_flush = 0                    # phrases: flushes sent on this context
        self.flush_base = None              # phrases: Sonic's flush_id for our flush 1 (0 measured, its docs say 1)
        self.gen, self.timer, self.seq = None, None, 0
        self.pause_armed = False            # phrases: the short pause timer runs (dec.hyp repeats every 100 ms)
        self.continued = False              # phrases: a face change moved the rest of the sentence to a new part
        self.finishing = False              # _finish is closing it (a reconnect then leaves the ending to _finish)
        self.said_gen = None                # the speed / volume / emotion Sonic is speaking with in this context
        self.ctx, self.reader = None, None
        self.audio = asyncio.Queue()        # bytes, then None when the utterance's audio is complete
        self.queued = self.ended = False
        self.lost = []                      # words streaming couldn't take: spoken whole at the end (fallback)
        self.broken = False                 # streaming failed twice: every later word goes to `lost`, keeping order
        self.pace = self.effort = None      # the last pace and effort the decoder sent (a message may leave them out)
        self.t_first_text = self.t_first_audio = self.t_final = None
        self.t_last = time.perf_counter()   # last voice.say for this utt_id

    def new_words(self, words, offset=None):
        words = [w for w in words if w]
        if offset is not None:              # position of words[0] in the sentence: resent words are skipped
            new = words[max(0, len(self.sent) - int(offset)):]
        else:                               # no offset: only the new words (the decoder's voice.say), never a guess
            new = words
        self.sent += new
        return new

    def end_audio(self):
        if not self.ended:
            self.ended = True
            self.audio.put_nowait(None)


class Voice:
    """Owns the Sonic websocket. say() is called per voice.say message; audio goes to `sink(bytes)` in order."""

    def __init__(self, cfg, sink, on_timing=None, client=None, on_done=None):
        self.cfg, self.sink, self.on_timing, self.on_done = cfg, sink, on_timing, on_done
        self.client = client
        self.conn = self._cm = None
        self.utts = {}              # utt_id -> Utterance
        self.open = set()           # utterances (and parts of one) with a Sonic context open
        self.order = asyncio.Queue()    # utterances in the order their audio started
        self.emotion = ('neutral', 1.0)
        self.emotion_since = float('-inf')     # when the face's Sonic emotion last changed
        self._lock = asyncio.Lock()
        self._tasks = set()
        self._player = None

    # --- connection ---------------------------------------------------------------------------------------------
    async def start(self):
        if self.client is None:
            from cartesia import AsyncCartesia
            self.client = AsyncCartesia(api_key=self.cfg.cartesia_key)
        await self._connect()
        self._player = self._spawn(self._play())

    async def _connect(self):
        self._cm = self.client.tts.websocket_connect()
        self.conn = await self._cm.__aenter__()
        log.info('Sonic websocket open (model %s)', self.cfg.tts_model)

    def _alive(self):
        try:
            from websockets.protocol import State
            return self.conn._connection.state not in (State.CLOSED, State.CLOSING)   # SDK internals, best effort
        except Exception:  # noqa: BLE001  (fakes / SDK changes: assume alive, a failed push still reconnects)
            return True

    async def _reconnect(self):
        log.warning('Sonic websocket lost: opening a new one')
        try:
            await self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        await self._connect()
        for u in list(self.open):           # contexts on the old socket are gone (parts of a sentence included)
            self.open.discard(u)
            if u.reader:
                u.reader.cancel()
            u.ctx = u.reader = None
            self._salvage(u)                # paced: what came plays, words Sonic never answered go to the fallback
            if u.final and not u.finishing:     # nothing more will be pushed (_finish ends its own)
                if u.lost:
                    self._spawn(self._lost_then_end(u))
                else:
                    log.warning('utterance %s lost its remaining audio in the reconnect', u.utt_id)
                    u.end_audio()

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self._tasks.add(t)

        def done(task):
            self._tasks.discard(task)
            if not task.cancelled() and task.exception():
                log.error('voice task failed', exc_info=task.exception())
        t.add_done_callback(done)
        return t

    def set_emotion(self, label, confidence):
        if generation_config(label, float(confidence))['emotion'] != generation_config(*self.emotion)['emotion']:
            self.emotion_since = time.perf_counter()
        self.emotion = (label, float(confidence))

    # --- words in -------------------------------------------------------------------------------------------------
    async def say(self, utt_id, words, final=False, pace=None, effort=None, offset=None):
        async with self._lock:      # messages stay in order
            u = self.utts.get(utt_id)
            if u is not None and u.final:
                if time.perf_counter() - u.t_final < REUSE_S:
                    log.debug('late message for finished utterance %s ignored', utt_id)
                    return
                log.warning('utt_id %s reused after it finished (decoder restart?): new utterance', utt_id)
                u = None
            if u is None:
                u = self.utts[utt_id] = Utterance(utt_id)
                self._prune()
            u.seq += 1
            u.t_last = time.perf_counter()
            if u.t_first_text is None and words:
                u.t_first_text = time.perf_counter()
            u.pace = u.pace if pace is None else pace
            u.effort = u.effort if effort is None else effort
            u.gen = generation_config(*self.emotion, pace=u.pace, effort=u.effort)
            if self.cfg.paced and u.said_gen and u.gen['emotion'] != u.said_gen['emotion'] and words:
                if time.perf_counter() - self.emotion_since >= FACE_HOLD_S:
                    u = await self._new_part(u)     # his face changed: the rest in a fresh recording
                else:
                    u.gen = dict(u.gen, emotion=u.said_gen['emotion'])     # a passing look keeps the mood
            new = u.new_words(words, offset)
            u.pending += new
            u.pending_gen += [u.gen] * len(new)
            if final:
                await self._finish(u)
                return
            if self.cfg.paced:              # a phrase waits for his pause, up to PHRASE_MAX words
                n = phrase_ready(u.pending, u.spoken)
            else:
                n = ready(u.pending, u.spoken, self.cfg.phrase_words)
            if n:
                await self._send(u, n)
            self._arm(u)

    async def _new_part(self, u):
        """Phrases: his face changed mid-sentence. Sonic keeps one recording's mood, so the rest gets a new part."""
        gen, u.gen = u.gen, u.said_gen      # the words still waiting were mouthed with the old face
        u.pending_gen = [dict(g, emotion=u.said_gen['emotion']) for g in u.pending_gen]
        u.continued = True                  # the sentence goes on in the next part: one on_done, at its end
        await self._finish(u, end=bool(u.pending))
        v = self.utts[u.utt_id] = Utterance(u.utt_id)
        v.sent, v.pace, v.effort, v.gen = u.sent, u.pace, u.effort, gen
        v.t_first_text = u.t_first_text
        return v

    async def _finish(self, u, end=True):
        """The utterance is over (end: close the sentence with its punctuation), or this part of it is."""
        u.final, u.t_final, u.finishing = True, time.perf_counter(), True
        if u.timer:
            u.timer.cancel()
        try:
            await self._send(u, len(u.pending), end=end)
        finally:
            if u.ctx is not None and u.broken and u.lost:   # streaming failed on this context: don't wait on it
                if u.reader:
                    u.reader.cancel()
                self._salvage(u)
                u.ctx = u.reader = None
                self.open.discard(u)
            if u.ctx is not None:
                try:
                    await u.ctx.no_more_inputs()
                except Exception:  # noqa: BLE001
                    log.exception('closing utterance %s failed', u.utt_id)
                    u.end_audio()
            elif u.lost:            # nothing in flight, words left over: the slow fallback runs off the lock,
                self._ensure_run(u)     # its place in the play order kept
                self._spawn(self._lost_then_end(u))
            else:
                u.end_audio()
            u.finishing = False     # from here a reconnect ends it as usual

    async def _send(self, u, n, end=False, pause=False):
        """Send the first n pending words as one chunk; they leave `pending` only once Sonic accepted them. Sonic joins
        continued chunks exactly as written: a leading space after the first chunk, closing punctuation on the last.
        pause: he paused after these words."""
        words, gens = u.pending[:n], u.pending_gen[:n]
        if self.cfg.paced and words:        # the phrase's face and pace; a loud word only where he clenched
            loud = [g.get('volume') or 1.0 for g in gens]
            gen = dict(gens[0], volume=1.0)
            said = [speakable([w], not u.spoken and i == 0) for i, w in enumerate(words)]
            parts = [f'<volume ratio="{max(CLENCH_VOLUME, vol):.2f}"/> {w} <volume ratio="1.00"/>' if vol >= CLENCH
                     else w for w, vol in zip(said, loud)]
            text = ('' if not u.spoken else ' ') + ' '.join(parts)
        else:
            gen = u.gen
            text = (('' if not u.spoken else ' ') + speakable(words, not u.spoken)) if words else ''
        if self.cfg.paced and not words:    # a mark alone can't change speech already made: just close
            return
        if end and (u.spoken or words):
            text += end_mark(u.sent, (gen or {}).get('emotion'))
        elif self.cfg.paced and is_question(u.sent):
            text += '?' if pause else ''    # a question rises where he pauses (most often its end)
        elif self.cfg.paced and (gen or {}).get('emotion') in EXCLAIM and (pause or len(words) >= PHRASE_MAX):
            text += '!'                     # an excited face exclaims at his pauses and full phrases (decided now)
        elif pause and self.cfg.paced:
            text += ','                     # his pause: a comma, so the voice breathes where he did
        if not text or (not words and u.ctx is None):   # only the end mark, and no context left to close
            return
        if u.broken:                # streaming already failed twice: the rest goes to the fallback, in order
            u.lost += words
            u.pending, u.pending_gen, u.spoken = u.pending[n:], u.pending_gen[n:], u.spoken + len(words)
            return
        for attempt in (1, 2):
            try:
                if not self._alive():
                    await self._reconnect()
                if u.broken:            # the reconnect lost earlier words to the fallback: these follow them
                    u.lost += words
                    break
                ctx = u.ctx or self._open(u, gen)
                # each push carries its own generation_config, or the SDK resends the context's first one
                if self.cfg.paced:      # its own flush: Sonic tags its audio with this flush_id, so it can be trimmed
                    fid = u.n_flush = u.n_flush + 1
                    u.flushes[fid] = Flush(words, dict(gen or {}), text)
                    try:
                        await ctx.push(transcript=text, flush=True, generation_config=gen)
                    except BaseException:
                        u.flushes.pop(fid, None)
                        u.n_flush -= 1
                        raise
                else:
                    await ctx.push(transcript=text, generation_config=gen)
                u.said_gen = gen
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    log.exception('Sonic streaming failed twice: "%s" and the rest are spoken whole at the end', text)
                    u.lost += words
                    u.broken = True
                    break
                try:
                    await self._reconnect()
                except Exception:  # noqa: BLE001  (no network at all: the fallback will try at the end)
                    log.exception('Sonic reconnect failed')
        u.pending, u.pending_gen = u.pending[n:], u.pending_gen[n:]
        u.spoken += len(words)

    def _arm(self, u, delay=None):
        """(Re)start the pause timer: if no new words arrive for the delay, speak what is waiting."""
        if u.timer:
            u.timer.cancel()
        u.pause_armed = delay == PHRASE_PAUSE_S
        if u.pending:
            seq = u.seq
            delay = delay if delay is not None else PHRASE_IDLE_S if self.cfg.paced else PAUSE_S
            u.timer = asyncio.get_running_loop().call_later(delay, lambda: self._spawn(self._on_pause(u, seq)))

    def mouthing(self, utt_id, active):
        """Phrases: dec.hyp. Nothing mouthed: send after PHRASE_PAUSE_S. A word in his mouth: keep waiting."""
        u = self.utts.get(utt_id)
        if self.cfg.paced and u is not None and not u.final and u.pending and (active or not u.pause_armed):
            self._arm(u, PHRASE_IDLE_S if active else PHRASE_PAUSE_S)

    async def _on_pause(self, u, seq):
        async with self._lock:
            if seq == u.seq and not u.final and u.pending:  # a newer message re-armed or closed it: do nothing
                await self._send(u, len(u.pending), pause=True)

    # --- audio out ------------------------------------------------------------------------------------------------
    def _ensure_run(self, u):
        """Queue u's audio for the player. An utterance whose audio already ended (a context that timed out, or words
        left for the fallback) gets a fresh run, queued as its own item, so the player never waits on a finished one."""
        if u.ended:
            u.audio, u.ended, u.queued = asyncio.Queue(), False, False
        if not u.queued:
            u.queued = True
            self.order.put_nowait((u, u.audio))

    def _open(self, u, gen=None):
        """A Sonic context for u, speaking with gen (default: u's current face, pace and clench)."""
        self._ensure_run(u)
        u.said_gen = gen = gen or u.gen
        paced = self.cfg.paced
        u.flushes, u.n_flush, u.flush_base = {}, 0, None    # a new context numbers its flushes afresh
        u.ctx = self.conn.context(f'utt-{u.utt_id}-{int(time.perf_counter() * 1000)}', timeout=CONTEXT_TIMEOUT_S,
                                  model_id=self.cfg.tts_model, voice={'id': self.cfg.voice_id},
                                  output_format=PCM if paced else OUTPUT, generation_config=gen,
                                  max_buffer_delay_ms=0 if paced else self.cfg.max_buffer_delay_ms)
        u.reader = self._spawn(self._read(u, u.ctx))
        self.open.add(u)
        return u.ctx

    async def _read(self, u, ctx):
        """One context's events -> the utterance's audio queue. Ends the queue on done, error or timeout; a cancel
        (reconnect) leaves it open for the replacement context."""
        try:
            async for ev in ctx.receive():
                kind = getattr(ev, 'type', None)
                if kind == 'chunk' and ev.data:
                    if u.t_first_audio is None:
                        u.t_first_audio = time.perf_counter()
                        if self.on_timing:
                            await self.on_timing({'stage': 'sonic_first_audio', 'utt_id': u.utt_id,
                                                  'ms': (u.t_first_audio - u.t_first_text) * 1000})
                    if self.cfg.paced:
                        self._phrase_audio(u, self._flush_of(u, ev), ev.audio)
                    else:
                        u.audio.put_nowait(ev.audio)
                elif kind == 'flush_done' and self.cfg.paced:
                    self._phrase_done(u, self._flush_no(u, ev))
                elif kind == 'error':
                    log.error('Sonic error on utt %s: %s', u.utt_id, getattr(ev, 'error', ev))
                    break
                if getattr(ev, 'done', False):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  timeout or a dead socket
            log.exception('Sonic reader for utt %s stopped', u.utt_id)
        if u.ctx is ctx:
            self._salvage(u)
            u.ctx = u.reader = None
            self.open.discard(u)
        if u.final:
            await self._speak_lost(u)
        u.end_audio()

    def _salvage(self, u):
        """Phrases, a context ended early: play the audio that came, send unanswered words to the fallback."""
        for fid in sorted(u.flushes):
            if u.flushes[fid].got:
                self._phrase_done(u, fid)
            else:
                u.lost += u.flushes.pop(fid).words
                u.broken = True             # the rest of the sentence follows them to the fallback, in order
        u.flushes = {}

    @staticmethod
    def _flush_no(u, ev):
        """Our flush number (1, 2, ...) for an event: the base is learned from Sonic's first id; no id: the oldest."""
        fid = getattr(ev, 'flush_id', None)
        if fid is None:
            return min(u.flushes, default=None)
        if u.flush_base is None:
            u.flush_base = fid - 1
        return fid - u.flush_base

    def _flush_of(self, u, ev):
        """The flush a chunk belongs to (one we never sent, e.g. the closing one: kept apart, played as is)."""
        return u.flushes.setdefault(self._flush_no(u, ev) or 0, Flush([], {}))

    @staticmethod
    def _phrase_audio(u, f, audio):
        """Phrases: a chunk of a flush's audio -> its trimmed speech so far, to the player at once."""
        data, f.carry = f.carry + audio, b''
        if len(data) % 2:
            data, f.carry = data[:-1], data[-1:]
        f.got = True
        pcm = f.trim.push(np.frombuffer(data, np.int16))
        if len(pcm):
            f.samples += len(pcm)
            u.audio.put_nowait(Phrase(f, pcm))

    def _phrase_done(self, u, fid):
        """Phrases: one flush's audio is complete -> the rest of its speech, with a natural pause after it."""
        if fid not in u.flushes:            # e.g. the closing flush, or nothing was sent: no audio of ours
            return
        f = u.flushes.pop(fid)
        tail = f.trim.end()
        f.samples += len(tail)
        if f.samples:
            u.audio.put_nowait(Phrase(f, tail, done=True))

    async def _lost_then_end(self, u):
        await self._speak_lost(u)
        u.end_audio()

    async def _speak_lost(self, u):
        """The fallback: the words streaming couldn't take, as one whole-sentence request, queued after the rest."""
        if not u.lost:
            return
        words, u.lost = u.lost, []
        text = speakable(words, words[0] == u.sent[0]) + end_mark(u.sent, (u.gen or {}).get('emotion'))
        try:
            resp = await self.client.tts.generate(model_id=self.cfg.tts_model, transcript=text,
                                                  voice={'id': self.cfg.voice_id}, output_format=OUTPUT,
                                                  generation_config=u.gen or generation_config())
            audio = await resp.read()
        except Exception:  # noqa: BLE001
            log.exception('fallback voice failed too: "%s" not spoken', text)
            return
        self._ensure_run(u)
        u.audio.put_nowait(audio)
        log.warning('spoke "%s" with the whole-sentence fallback', text)
        if self.on_timing:                  # the dashboard shows what the call heard, fallback included
            await self.on_timing({'stage': 'voice_phrase', 'utt_id': u.utt_id, 'words': ' '.join(words), 'text': text,
                                  'fallback': True, **(u.gen or {})})

    async def _play(self):
        """Utterances one after another, each to its end, so audio never interleaves."""
        while True:
            u, run = await self.order.get()
            heard = False
            while True:
                try:
                    b = await asyncio.wait_for(run.get(), 0.5)
                except TimeoutError:
                    self._close_if_stale(u)
                    continue
                if b is None:
                    break
                if isinstance(b, Phrase):
                    await self._play_phrase(u, b)
                else:
                    await self.sink(b)
                heard = True
            if heard and self.on_done and not u.continued:
                await self.on_done(u.utt_id, u.t_first_text)

    async def _play_phrase(self, u, p):
        if len(p.pcm):
            await self.sink(to_mulaw(p.pcm))
        if p.done and self.on_timing:
            await self.on_timing({'stage': 'voice_phrase', 'utt_id': u.utt_id, 'words': p.words, 'text': p.text,
                                  'ms': round(p.samples / RATE * 1000), **p.gen})

    def _close_if_stale(self, u):
        """A lost final (decoder crash, dropped message) would hold every later sentence behind this one."""
        if not u.final and not self.order.empty() and time.perf_counter() - u.t_last > STALE_S:
            log.warning('utterance %s got no final for %.0f s while another waits: closing it', u.utt_id, STALE_S)
            u.final, u.t_final = True, time.perf_counter()     # once; a late message for it is ignored as usual
            self._spawn(self._close_stale(u))

    async def _close_stale(self, u):
        async with self._lock:
            await self._finish(u)

    def _prune(self):
        """Forget utterances whose audio finished a while ago."""
        now = time.perf_counter()
        for k in [k for k, u in self.utts.items() if u.final and u.ended and now - u.t_final > 3 * REUSE_S]:
            del self.utts[k]

    async def close(self):
        for u in self.utts.values():
            if u.timer:
                u.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        if self._cm is not None:
            try:
                await self.conn.close()
            except Exception:  # noqa: BLE001
                pass
