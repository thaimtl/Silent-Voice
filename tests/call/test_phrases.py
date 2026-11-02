"""The voice in phrases (call/voice.py, call/audio.py), with a fake Sonic that numbers flushes from 0 and pads them.

    venv-call/bin/python -m pytest tests/call -q
"""
import asyncio
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from call import audio  # noqa: E402
from call import voice as cv  # noqa: E402
from call.audio import RATE, to_mulaw, trim  # noqa: E402
from call.ink import MULAW  # noqa: E402
from call.voice import PCM, Voice, phrase_ready  # noqa: E402

from .test_call import FakeTTS, cfg  # noqa: E402


def tone(seconds, hz=180.0, amp=8000):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def padded(seconds, pad=0.25):
    """Speech the way Sonic sends a flush: between ~0.25 s of silence on each side."""
    z = np.zeros(int(pad * RATE), np.int16)
    return np.concatenate([z, tone(seconds), z])


# --- audio -------------------------------------------------------------------------------------------------------
def test_mulaw_encoder_matches_the_reference_codec_on_every_sample():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            import audioop                  # CPython's reference G.711 (gone in 3.13: then the next test still runs)
        except ImportError:
            return
    x = np.arange(-32768, 32768).astype('<i2')
    assert to_mulaw(x) == audioop.lin2ulaw(x.tobytes(), 2)


def test_mulaw_encoder_inverts_the_decoder_for_every_code():
    codes = np.arange(256, dtype=np.uint8)
    back = np.frombuffer(to_mulaw(MULAW[codes].astype(np.int16)), np.uint8)
    assert (MULAW[back] == MULAW[codes]).all()          # 0x7F and 0xFF both decode to 0: either byte is right


def test_trim_keeps_the_speech_and_a_natural_pause():
    w = trim(padded(0.8), post_s=0.15)
    assert abs(len(w) / RATE - (0.8 + audio.PRE_S + 0.15)) < 0.02
    assert len(trim(np.zeros(4000, np.int16))) == 0               # silence only: nothing to play
    assert len(trim(np.zeros(10, np.int16))) == 0


# --- the phrase policy -----------------------------------------------------------------------------------------------
def test_phrase_policy():
    assert phrase_ready(['THEY', 'LOST']) == 0                    # wait for his pause
    assert phrase_ready(['I', 'GOT', 'HOME'], spoken=0) == 0     # no cut while he keeps mouthing ...
    assert phrase_ready(['THEY', 'LOST', 'MY', 'BAG']) == 4       # ... but never hold more than 4
    assert phrase_ready(['SO', 'I', 'HAD', 'TO', 'THE']) == 5


async def pause(v, utt_id):
    """His decoder shows nothing in his mouth: he paused, the phrase goes."""
    v.mouthing(utt_id, False)
    await asyncio.sleep(cv.PHRASE_PAUSE_S + 0.05)


# --- the voice with a fake Sonic -------------------------------------------------------------------------------------
class PhraseCtx:
    """Answers each push with padded speech and a flush_done (ids from 0); no_more_inputs gets a bare flush_done."""

    def __init__(self, cid, latency=0.03, **defaults):
        self.cid, self.defaults, self.latency = cid, defaults, latency
        self.pushed, self.gens, self.closed = [], [], False
        self.q = asyncio.Queue()

    async def push(self, transcript, generation_config=None, flush=False):
        self.pushed.append((transcript, flush))
        self.gens.append(generation_config)
        fid = len(self.pushed) - 1

        async def later():
            await asyncio.sleep(self.latency)
            pcm = padded(0.3).tobytes()
            for i in range(0, len(pcm), 1600):
                await self.q.put(SimpleNamespace(type='chunk', data='x', audio=pcm[i:i + 1600], done=False,
                                                 flush_id=fid))
            await self.q.put(SimpleNamespace(type='flush_done', done=False, flush_id=fid))
        asyncio.create_task(later())

    async def no_more_inputs(self):
        self.closed = True
        fid = len(self.pushed)

        async def later():
            await asyncio.sleep(self.latency * 2)
            await self.q.put(SimpleNamespace(type='flush_done', done=False, flush_id=fid))
            await self.q.put(SimpleNamespace(type='done', data=None, done=True))
        asyncio.create_task(later())

    async def receive(self):
        while True:
            ev = await self.q.get()
            yield ev
            if ev.done:
                return


class PhraseConn:
    def __init__(self, ctx_class=PhraseCtx):
        self.contexts, self.ctx_class = [], ctx_class

    def context(self, cid, **kw):
        c = self.ctx_class(cid, **kw)
        self.contexts.append(c)
        return c


async def make(conn=None, **kw):
    heard, timing, fallback = [], [], []

    async def sink(b):
        heard.append(b)

    async def on_timing(row):
        timing.append(row)

    async def generate(**kw):
        fallback.append(kw['transcript'])
        return SimpleNamespace(read=lambda: asyncio.sleep(0, b'\xff' * 800))
    conn = conn or PhraseConn()
    tts = FakeTTS(conn)
    tts.generate = generate
    v = Voice(cfg(paced=True, **kw), sink, on_timing=on_timing, client=SimpleNamespace(tts=tts))
    await v.start()
    return v, conn, heard, timing, fallback


def phrases(timing):
    """The phrases that streamed (the whole-sentence fallback reports its text too, marked fallback)."""
    return [r['words'] for r in timing if r['stage'] == 'voice_phrase' and not r.get('fallback')]


async def test_words_go_as_phrases_at_his_pauses():
    v, conn, heard, timing, _ = await make()
    for w in ['YES']:
        await v.say(1, [w])
    await pause(v, 1)
    for w in ['THEY', 'LOST', 'MY']:                              # he keeps mouthing: no cut ...
        await v.say(1, [w])
        v.mouthing(1, True)
    await v.say(1, ['AT'])                                       # ... until PHRASE_MAX words wait
    await v.say(1, ['THE', 'AIRPORT'], final=True)
    await asyncio.sleep(0.3)
    ctx = conn.contexts[0]
    assert ctx.defaults['output_format'] == PCM and ctx.defaults['max_buffer_delay_ms'] == 0
    assert ctx.pushed == [('Yes,', True), (' they lost my at', True), (' the airport.', True)]   # his pause: ','
    assert phrases(timing) == ['YES', 'THEY LOST MY AT', 'THE AIRPORT']
    for r in timing:                                             # each phrase: its speech and a short pause, not 0.5 s
        if r['stage'] == 'voice_phrase':
            assert abs(r['ms'] / 1000 - (0.3 + audio.PRE_S + cv.PHRASE_TAIL_S)) < 0.02
    assert abs(sum(map(len, heard)) / RATE - 3 * (0.3 + audio.PRE_S + cv.PHRASE_TAIL_S)) < 0.06
    await v.close()


async def test_a_word_in_his_mouth_keeps_the_phrase_open():
    v, conn, _, _, _ = await make()
    await v.say(2, ['I', 'NEED'])
    await v.say(2, ['MY', 'PILLS'][:1])
    for _ in range(8):                                           # dec.hyp every 100 ms: he is mouthing the next word,
        v.mouthing(2, True)                                      # longer than PHRASE_IDLE_S
        await asyncio.sleep(0.1)
    assert conn.contexts == []                                   # nothing while he is mid-phrase
    for _ in range(3):                                           # then he pauses; dec.hyp keeps saying so
        v.mouthing(2, False)
        await asyncio.sleep(0.1)
    assert conn.contexts[0].pushed == [('I need my,', True)]     # repeats: pause not pushed back
    await v.close()


async def test_a_long_stretch_without_a_pause_goes_at_four_words():
    v, conn, _, _, _ = await make()
    for w in 'I THINK WE SHOULD GET LUNCH AFTER'.split():
        await v.say(3, [w])
        v.mouthing(3, True)
    assert [p for p, _ in conn.contexts[0].pushed] == ['I think we should']
    await v.close()


async def test_without_dec_hyp_a_quiet_spell_still_sends_the_phrase():
    v, conn, _, _, _ = await make()
    await v.say(3, ['HELLO', 'THERE'])
    await asyncio.sleep(cv.PHRASE_IDLE_S + 0.1)
    assert conn.contexts[0].pushed == [('Hello there,', True)]
    await v.close()


async def test_a_typed_sentence_goes_as_one_phrase():
    v, conn, heard, _, _ = await make()
    await v.say(4, ['I', 'GOT', 'THE', 'JOB'], final=True)       # call/say.py: one message, final
    await asyncio.sleep(0.2)
    assert conn.contexts[0].pushed == [('I got the job.', True)] and heard
    await v.close()


async def test_a_flickering_face_never_splits_a_sentence():
    v, conn, _, _, _ = await make()
    words = 'SO I THINK WE SHOULD GET LUNCH AFTER THE DEMO'.split()
    for i, w in enumerate(words):
        v.set_emotion('happy' if i % 2 else 'neutral', 0.6)
        await v.say(7, [w], final=i == len(words) - 1)
    await asyncio.sleep(0.3)
    assert len(conn.contexts) == 1
    assert {g['emotion'] for g in conn.contexts[0].gens} == {'neutral'}
    await v.close()


async def test_a_face_held_mid_sentence_splits_it(monkeypatch):
    monkeypatch.setattr(cv, 'FACE_HOLD_S', 0.1)
    v, conn, _, _, _ = await make()
    v.set_emotion('happy', 0.9)
    await v.say(5, ['I', 'GOT', 'THE', 'JOB'])
    await pause(v, 5)
    v.set_emotion('sad', 0.9)
    await v.say(5, ['BUT'])                                      # too soon: still the happy recording
    await asyncio.sleep(0.15)
    v.set_emotion('sad', 0.9)
    await v.say(5, ['I', 'MISS', 'YOU'], final=True)
    await asyncio.sleep(0.3)
    assert [c.defaults['generation_config']['emotion'] for c in conn.contexts] == ['excited', 'sad']
    assert [p for p, _ in conn.contexts[1].pushed] == ['I miss you.']
    await v.close()


async def test_a_change_of_face_mid_sentence_starts_a_fresh_recording(monkeypatch):
    monkeypatch.setattr(cv, 'FACE_HOLD_S', 0.0)
    v, conn, heard, timing, _ = await make()
    v.set_emotion('happy', 0.9)
    await v.say(5, ['I', 'GOT', 'THE', 'JOB'])
    await pause(v, 5)
    await v.say(5, ['AND', 'I'])                                 # still happy: waits for more words
    v.set_emotion('sad', 0.9)
    await v.say(5, ['WILL', 'MISS', 'YOU'], final=True)
    await asyncio.sleep(0.3)
    first, second = conn.contexts
    assert first.defaults['generation_config']['emotion'] == 'excited'                 # Sonic's stronger happy
    assert [p for p, _ in first.pushed] == ['I got the job!', ' and I!'] and first.closed  # excited: '!'
    assert second.defaults['generation_config']['emotion'] == 'sad'
    assert [p for p, _ in second.pushed] == ['Will miss you.']                  # starts like a sentence
    assert phrases(timing) == ['I GOT THE JOB', 'AND I', 'WILL MISS YOU'] and heard
    assert [r.get('emotion') for r in timing if r['stage'] == 'voice_phrase'] == ['excited', 'excited', 'sad']
    await v.close()


async def test_a_clench_inside_a_phrase_is_loud_on_that_word_only():
    v, conn, _, _, _ = await make()
    await v.say(8, ['I', 'TOLD'], effort=1.0)
    await v.say(8, ['THEM'], effort=1.0)
    await pause(v, 8)
    await v.say(8, ['TODAY'], effort=1.9)                         # he clenches on "today"
    await v.say(8, ['NOT', 'TOMORROW'], effort=1.0, final=True)
    await asyncio.sleep(0.2)
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert pushed == ['I told them,', ' <volume ratio="1.90"/> today <volume ratio="1.00"/> not tomorrow.']
    await v.close()


async def test_a_light_clench_is_not_tagged_and_a_strong_one_is_clearly_louder():
    v, conn, _, _, _ = await make()
    await v.say(9, ['WE', 'WON'], effort=1.2)                     # a little more effort: no tag, the flow stays clean
    await v.say(9, ['AGAIN'], effort=1.5, final=True)            # a clench: at least CLENCH_VOLUME, to be heard
    await asyncio.sleep(0.2)
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert pushed == ['We won <volume ratio="1.80"/> again <volume ratio="1.00"/>.']
    await v.close()


async def test_a_sonic_error_mid_sentence_never_shifts_words_onto_the_wrong_audio():
    """An error before the second phrase's audio sends it and the rest of the sentence to the fallback, in order."""
    class ErrCtx(PhraseCtx):
        async def push(self, transcript, generation_config=None, flush=False):
            if not self.pushed:
                return await super().push(transcript, generation_config, flush)
            self.pushed.append((transcript, flush))

            async def later():
                await asyncio.sleep(self.latency * 3)
                await self.q.put(SimpleNamespace(type='error', done=True, error='boom'))
            asyncio.create_task(later())

    class ErrConn(PhraseConn):
        def context(self, cid, **kw):
            c = (ErrCtx if not self.contexts else PhraseCtx)(cid, **kw)
            self.contexts.append(c)
            return c
    v, conn, _, timing, fallback = await make(ErrConn())
    await v.say(6, ['THEY', 'LOST', 'MY', 'BAG'])                # the first phrase: sent whole at his pause
    await pause(v, 6)
    await v.say(6, ['AT', 'THE', 'BIG', 'AIRPORT'])
    await pause(v, 6)
    await asyncio.sleep(0.3)
    await v.say(6, ['SO', 'I', 'WAITED'], final=True)
    await asyncio.sleep(0.5)
    assert phrases(timing) == ['THEY LOST MY BAG']             # streamed before the error
    assert fallback == ['at the big airport so I waited.']        # the rest, in order, never ahead of the lost words
    await v.close()


async def test_a_reconnect_mid_sentence_sends_unanswered_phrases_to_the_fallback():
    v, conn, _, timing, fallback = await make()
    await v.say(7, ['I', 'NEED', 'MY', 'PILLS', 'RIGHT', 'NOW'])  # six words: sent; no audio yet when the socket drops
    dead = iter([False])
    v._alive = lambda: next(dead, True)
    await v.say(7, ['PLEASE'], final=True)                       # reconnects first
    await asyncio.sleep(0.5)
    assert phrases(timing) == []                                 # "please" never plays before "I need my pills"
    assert fallback == ['I need my pills right now please.']     # not lost, in order
    await v.close()


async def test_his_face_and_pauses_write_the_punctuation():
    """An excited face ends phrases in '!', a pause gives ',', a question word gives '?'."""
    v, conn, _, _, _ = await make()
    v.set_emotion('happy', 0.9)
    await v.say(1, ['I', 'GOT', 'THE', 'JOB'])
    await pause(v, 1)
    await v.say(1, ['I', 'START'])
    await pause(v, 1)
    await v.say(1, ['MONDAY'], final=True)
    v.set_emotion('neutral', 0.9)
    await v.say(2, ['WHERE', 'IS', 'IT'], final=True)
    await asyncio.sleep(0.2)
    assert [p for p, _ in conn.contexts[0].pushed] == ['I got the job!', ' I start!', ' monday!']
    assert [p for p, _ in conn.contexts[1].pushed] == ['Where is it?']
    await v.close()


async def test_a_question_waits_for_his_pause_so_it_can_rise():
    v, conn, _, _, _ = await make()
    for w in ['WHERE', 'IS', 'HE']:                              # no early cut for a question
        await v.say(1, [w])
    assert conn.contexts == []
    v.mouthing(1, False)
    await asyncio.sleep(cv.PHRASE_PAUSE_S + 0.05)
    await v.say(1, [], final=True)                               # an empty final: no bare '?' pushed after
    await asyncio.sleep(0.1)
    assert [p for p, _ in conn.contexts[0].pushed] == ['Where is he?']
    await v.close()


async def test_a_sentence_split_by_a_face_change_is_reported_played_once(monkeypatch):
    monkeypatch.setattr(cv, 'FACE_HOLD_S', 0.0)
    done = []

    async def on_done(utt_id, t):
        done.append(utt_id)
    v, conn, _, _, _ = await make()
    v.on_done = on_done
    v.set_emotion('happy', 0.9)
    await v.say(5, ['I', 'GOT', 'THE', 'JOB'])
    await pause(v, 5)
    v.set_emotion('sad', 0.9)
    await v.say(5, ['BUT', 'I', 'MISS', 'YOU'], final=True)
    await asyncio.sleep(0.4)
    assert len(conn.contexts) == 2 and done == [5]
    await v.close()


def test_stream_trim_plays_from_the_first_sound_and_trims_only_the_tail():
    from call.audio import StreamTrim
    st, x = StreamTrim(post_s=0.15), padded(0.6, pad=0.5).tobytes()     # Sonic's real padding: ~0.5 s
    out = [st.push(np.frombuffer(x[i:i + 1600], np.int16)) for i in range(0, len(x), 1600)]
    first = next(i for i, o in enumerate(out) if len(o))
    assert first <= 6                                            # plays from the chunk its first sound is in
    total = sum(map(len, out)) + len(st.end())
    assert abs(total / RATE - (0.6 + audio.PRE_S + 0.15)) < 0.02    # the same length trim() gives the whole
    assert len(StreamTrim().end()) == 0                          # silence only: nothing


async def test_a_long_question_is_still_capped_and_rises_only_at_its_end():
    v, conn, _, _, _ = await make()
    words = 'WHAT DO YOU THINK ABOUT THE NEW PLAN FOR THE DEMO TOMORROW'.split()
    for w in words[:-1]:
        await v.say(1, [w])
        v.mouthing(1, True)
    await v.say(1, [words[-1]], final=True)
    await asyncio.sleep(0.2)
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert len(pushed) >= 2 and pushed[-1].endswith('?')     # not held to the end; '?' only at the end
    assert not any(p.endswith('?') for p in pushed[:-1])
    await v.close()


async def test_have_a_great_day_is_not_a_question():
    v, conn, _, _, _ = await make()
    v.set_emotion('happy', 0.9)
    for w in ['HAVE', 'A', 'GREAT', 'DAY']:
        await v.say(1, [w])
    await v.say(1, ['FRIEND'], final=True)
    await asyncio.sleep(0.2)
    pushed = [p for p, _ in conn.contexts[0].pushed]
    assert pushed == ['Have a great day!', ' friend!']         # exclamations, not a question
    await v.close()


def test_stream_trim_ignores_faint_noise_in_the_padding():
    from call.audio import StreamTrim
    rng = np.random.default_rng(0)
    z = (rng.normal(0, 0.005 * 32768, int(0.5 * RATE))).astype(np.int16)        # Sonic padding with a noise floor
    x = np.concatenate([z, tone(0.6), z]).tobytes()
    st = StreamTrim(post_s=0.15)
    total = sum(len(st.push(np.frombuffer(x[i:i + 1600], np.int16))) for i in range(0, len(x), 1600)) + len(st.end())
    assert total / RATE < 0.6 + 0.5 + 0.15                       # the trailing noise is not taken for speech


async def test_a_reconnect_after_a_sentence_closed_still_ends_it():
    """_finish's guard against a reconnect ending the sentence twice must not outlive _finish."""
    v, conn, _, timing, fallback = await make()
    await v.say(1, ['I', 'GOT', 'HOME'], final=True)              # closed; its audio still on the way
    dead = iter([False])
    v._alive = lambda: next(dead, True)
    await v.say(2, ['THEN', 'I', 'SLEPT'], final=True)            # reconnects while 1 is still streaming
    await asyncio.sleep(0.5)
    assert v.utts[1].ended
    assert all(r.get('words') for r in timing if r['stage'] == 'voice_phrase')
    await v.close()
