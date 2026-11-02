"""The call service: Twilio media websocket <-> Ink and Sonic, and the team bus.

    POST /twiml               TwiML for an inbound call to our Twilio number (connects its audio to /media/<token>)
    POST /sms                 a text to our Twilio number (SMS mode): the judge's words, answered with no reply
    GET|POST /channel         on 127.0.0.1:CALL_CONTROL_PORT only (never through the tunnel): read or switch the mode,
                              {"channel": "call" | "sms"}; the dashboard's Call/Text switch uses it
    GET  /media/<token>       Twilio Media Streams websocket (only Twilio learns the token, from our TwiML)
    GET  /health              ok

Bus: reads voice.say (Austin's decoder) and cam.tick (emotion); publishes partner.text, timing.voice, call.state.

Two channels for the judge's side (CALL_CHANNEL at start, switchable any time, even mid-call): 'call' (default) they
speak and Cartesia Ink transcribes them; 'sms' (a loud room) they text our number from the phone on the call and hear
Austin's voice on that call. Ink stays off then, and only texts from the phone on the live call count. Austin's voice
goes into the call the same way in both.

    venv-call/bin/python -m call.server                       # inbound: the judge calls our number
    venv-call/bin/python -m call.server --dial +14155550142   # outbound: Austin calls the judge
Needs PUBLIC_HOST (ngrok / cloudflared to PORT) and the keys in call/keys.env (see call/.env.example).
"""
import argparse
import asyncio
import logging
import secrets
import time

from aiohttp import WSMsgType, web

from call import bus, twilio_media
from call.config import Config
from call.ink import Ink
from call.voice import Voice

log = logging.getLogger('call.server')
INK_START_S = 5.0           # Ink handshake budget; past it the call goes on without captions
CHANNELS = ('call', 'sms')
SMS_MAX_CHARS = 1600        # Twilio's own limit for one (joined) message


class Call:
    """One live phone call: the Twilio socket, its stream id, and audio out in order (marks time each utterance)."""

    def __init__(self, ws, sid, peer=None):
        self.ws, self.sid = ws, sid
        self.peer = peer            # the other phone's number (E.164), from our TwiML; texts must come from it
        self.ink = None             # Cartesia Ink on this call (call mode only)
        self.ink_lock = asyncio.Lock()      # one start or stop at a time (a mode switch can land mid-start)
        self.out = asyncio.Queue()
        self.sent_frames = 0

    def start(self):
        self.task = asyncio.create_task(self.pump())
        self.task.add_done_callback(self._pump_done)
        return self.task

    def _pump_done(self, task):
        if not task.cancelled() and task.exception():
            log.error('audio to Twilio stopped: %s', task.exception())
            asyncio.ensure_future(self.ws.close())      # end the call cleanly instead of queueing audio forever

    async def pump(self):
        """Send queued audio to Twilio as 20 ms frames. Twilio plays them back to back, so we don't pace."""
        while True:
            item = await self.out.get()
            if isinstance(item, str):           # a mark name
                await self.ws.send_str(twilio_media.mark_msg(self.sid, item))
                continue
            for f in twilio_media.frames(item):
                await self.ws.send_str(twilio_media.media_msg(self.sid, f))
                self.sent_frames += 1


class Service:
    def __init__(self, cfg, voice=None, ink_factory=None, pub=None, sub=None, control=False):
        self.cfg = cfg
        self.channel = cfg.channel                  # 'call' or 'sms'; the dashboard can switch it
        self.control = control                      # serve /channel on 127.0.0.1:cfg.control_port
        self.control_runner = None
        self.pub = pub or bus.Publisher(cfg.call_pub)
        self.sub = sub or bus.Subscriber([cfg.dec_pub, cfg.cam_pub], [bus.VOICE_SAY, bus.CAM_TICK, bus.DEC_HYP])
        self.voice = voice or Voice(cfg, self._audio, on_timing=self._timing, on_done=self._voice_done)
        self.ink_factory = ink_factory or (lambda on_text: Ink(cfg, on_text, gate=cfg.noise_gate,
                                                                 record=cfg.record_inbound))
        self.call = None
        self.marks = {}             # mark name -> (utt_id, time sent)
        self.runs = 0               # each played run gets its own mark name (an utterance can play twice: fallback)
        self.says = asyncio.Queue()     # voice.say for the voice worker: the bus reader never waits on Sonic
        self.tasks = []

    # --- audio and events -------------------------------------------------------------------------------------
    async def _audio(self, mulaw):
        if self.call is not None:
            await self.call.out.put(mulaw)

    async def _timing(self, row):
        await self.pub.send(bus.TIMING_VOICE, row)

    async def _partner(self, text, is_final, t, source='ink'):
        await self.pub.send(bus.PARTNER_TEXT, {'text': text, 'is_final': bool(is_final), 't': float(t),
                                               'source': source})

    async def _voice_done(self, utt_id, t_first_text):
        """Sonic finished an utterance: queue a mark behind its audio; Twilio echoes it once the judge heard it all."""
        if self.call is not None:
            self.runs += 1
            name = f'utt-{utt_id}-{self.runs}'
            self.marks[name] = (utt_id, t_first_text or time.perf_counter())
            await self.call.out.put(name)

    async def _state(self, state, **extra):
        await self.pub.send(bus.CALL_STATE, {'state': state, 't': time.perf_counter(), **extra})

    async def bus_loop(self):
        while True:
            topic, msg = await self.sub.recv()
            try:
                if topic == bus.CAM_TICK:
                    e = msg.get('emotion') or {}
                    if 'label' in e:
                        self.voice.set_emotion(e['label'], e.get('confidence', 1.0))
                elif topic == bus.VOICE_SAY:
                    self.says.put_nowait(msg)       # in order; a slow Sonic (reconnect) doesn't hold up emotion
                elif topic == bus.DEC_HYP:          # his decoder: a word in his mouth now? (phrases end on pauses)
                    # `mouthing`: his lips (or the pads) now; an older decoder only had its unlocked words, which lag
                    self.voice.mouthing(msg['utt_id'], bool(msg['mouthing'] if 'mouthing' in msg else msg.get('unlocked')))
            except Exception:  # noqa: BLE001  one bad message must not end the call
                log.exception('bad %s message', topic)

    async def say_loop(self):
        while True:
            msg = await self.says.get()
            try:
                await self.voice.say(msg['utt_id'], list(msg.get('words', [])), bool(msg.get('final', False)),
                                     msg.get('pace'), msg.get('effort'), msg.get('offset'))
            except Exception:  # noqa: BLE001  one bad message must not end the call
                log.exception('bad voice.say message')

    # --- Twilio ---------------------------------------------------------------------------------------------------
    async def _signed_params(self, request):
        """The webhook's form fields, if Twilio really sent it (X-Twilio-Signature); 403 otherwise (fail closed)."""
        params = dict(await request.post())
        url = f'https://{self.cfg.public_host}{request.path_qs}'
        if not twilio_media.signed(self.cfg, url, params, request.headers.get('X-Twilio-Signature', '')):
            log.warning('refused %s without a valid Twilio signature for %s (set the number\'s webhook to exactly '
                        'this URL)', request.path, url)
            raise web.HTTPForbidden()
        return params

    async def twiml(self, request):
        """Only Twilio may fetch this (it contains the media token)."""
        params = await self._signed_params(request)
        return web.Response(text=twilio_media.twiml_stream(self.cfg.public_host, self.cfg.media_token,
                                                           params.get('From')), content_type='text/xml')

    async def sms(self, request):
        """A text to our number. In SMS mode, a text from the phone on the live call is the judge's next line
        (partner.text, final). Always answered with empty TwiML: we never text back."""
        params = await self._signed_params(request)
        text = ' '.join(str(params.get('Body', '')).split())[:SMS_MAX_CHARS]
        sender = params.get('From', '')
        call = self.call
        if self.channel != 'sms':
            log.info('a text arrived but this is call mode: ignored')
        elif call is None:
            log.warning('a text arrived with no call live: ignored (call the number first, then text it)')
        elif sender != call.peer:                      # an unknown peer matches nobody: fail closed
            log.warning('a text from %s, not the phone on the call (%s): ignored', _tail(sender), _tail(call.peer))
        elif text:
            log.info('judge texted %d characters', len(text))
            await self._partner(text, True, time.perf_counter(), source='sms')
        return web.Response(text=twilio_media.EMPTY_TWIML, content_type='text/xml')

    async def media(self, request):
        if not secrets.compare_digest(request.match_info.get('token', ''), self.cfg.media_token):
            raise web.HTTPNotFound()
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        mine = None                 # this socket's call
        pump = None
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                ev, msg = twilio_media.parse(m.data)
                if ev == 'start':
                    if self.call is not None:
                        log.warning('second call while one is live: refusing it')
                        break
                    start = msg['start']
                    self.call = Call(ws, start['streamSid'], (start.get('customParameters') or {}).get('peer'))
                    pump = self.call.start()
                    mine = self.call
                    if not str(mine.peer or '').startswith('+'):
                        log.warning('the phone on this call hides its number, so its texts can\'t be matched and will '
                                    'be ignored in SMS mode: turn off caller ID blocking and call again')
                    await self._sync_ink(mine)
                    await self._state('connected', call_sid=start.get('callSid'), ink=mine.ink is not None,
                                      channel=self.channel)
                elif ev == 'media' and mine is not None and mine.ink is not None \
                        and msg['media'].get('track', 'inbound') == 'inbound':
                    mine.ink.feed(msg['audio'])
                elif ev == 'mark':
                    name = msg['mark']['name']
                    if name in self.marks:
                        utt, t0 = self.marks.pop(name)
                        await self._timing({'stage': 'played_to_judge', 'utt_id': utt,
                                            'ms': (time.perf_counter() - t0) * 1000})   # first word in -> heard
                elif ev == 'stop':
                    break
        finally:
            ours = self.call is not None and self.call.ws is ws
            if ours:                    # free the line first, so a failure below can't block the next call
                self.call = None
                self.marks.clear()
            if pump:
                pump.cancel()
            if mine is not None:
                async with mine.ink_lock:
                    ink, mine.ink = mine.ink, None
                if ink:
                    try:
                        await ink.close()
                    except Exception:  # noqa: BLE001
                        log.exception('closing Ink failed')
            if ours:
                await self._state('ended')
        return ws

    async def _sync_ink(self, call):
        """Ink on in call mode, off in SMS mode, for this call. Called at its start and on every mode switch."""
        async with call.ink_lock:
            if self.channel == 'call' and call.ink is None and self.call is call:
                ink = self.ink_factory(self._partner)
                try:
                    await asyncio.wait_for(ink.start(), INK_START_S)
                    call.ink = ink
                except Exception:  # noqa: BLE001  the voice still works without the judge's captions
                    log.exception('Ink did not start: no judge captions on this call')
                    try:
                        await ink.close()          # a half-open Ink websocket would stay open (and billed)
                    except Exception:  # noqa: BLE001
                        log.exception('closing the half-started Ink failed')
            elif self.channel == 'sms' and call.ink is not None:
                ink, call.ink = call.ink, None
                try:
                    await ink.close()
                except Exception:  # noqa: BLE001
                    log.exception('closing Ink failed')

    async def set_channel(self, channel):
        """Switch between call mode and SMS mode, also in the middle of a call."""
        if channel not in CHANNELS:
            raise ValueError(channel)
        if channel != self.channel:
            self.channel = channel
            log.info('switched to %s mode', channel)
        call = self.call
        if call is not None:
            await self._sync_ink(call)
        await self._state('channel', channel=channel, ink=call is not None and call.ink is not None)

    async def channel_get(self, request):
        return web.json_response({'channel': self.channel})

    async def channel_post(self, request):
        """Local only. The custom header makes a browser preflight it, so no web page can switch it behind our back."""
        if request.headers.get('X-Silent-Voice') != '1':
            raise web.HTTPForbidden()
        try:
            channel = (await request.json()).get('channel')
            await self.set_channel(channel)
        except (ValueError, AttributeError, TypeError):
            raise web.HTTPBadRequest(text='channel must be "call" or "sms"') from None
        return web.json_response({'channel': self.channel})

    def control_app(self):
        a = web.Application()
        a.add_routes([web.get('/channel', self.channel_get), web.post('/channel', self.channel_post)])
        return a

    async def health(self, request):
        return web.Response(text='ok')

    def app(self):
        a = web.Application()
        a.add_routes([web.post('/twiml', self.twiml), web.post('/sms', self.sms),
                      web.get('/media/{token}', self.media), web.get('/health', self.health)])
        a.on_startup.append(self._startup)
        a.on_cleanup.append(self._cleanup)
        return a

    async def _startup(self, app):
        if self.control:            # its own site on the loopback: the tunnel only ever reaches the main port
            self.control_runner = web.AppRunner(self.control_app(), access_log=None)
            await self.control_runner.setup()
            await web.TCPSite(self.control_runner, '127.0.0.1', self.cfg.control_port).start()
            log.info('mode switch on http://127.0.0.1:%d/channel', self.cfg.control_port)
        await self.voice.start()
        self.tasks.append(asyncio.create_task(self.bus_loop()))
        self.tasks.append(asyncio.create_task(self.say_loop()))

    async def _cleanup(self, app):
        for t in self.tasks:
            t.cancel()
        if self.control_runner is not None:
            await self.control_runner.cleanup()
        await self.voice.close()
        self.sub.close()
        self.pub.close()


def _tail(number):
    """The last 4 digits of a phone number, for logs (the full number is personal)."""
    return '...' + str(number)[-4:] if number else 'an unknown number'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dial', help='call this number (E.164, e.g. +14155550142) once the server is up')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cfg = Config()
    if cfg.channel not in CHANNELS:
        raise SystemExit(f'CALL_CHANNEL must be one of {", ".join(CHANNELS)}, not {cfg.channel!r}')
    need = cfg.missing('cartesia_key', 'voice_id', 'public_host', 'twilio_token')   # the token signs every /twiml
    if args.dial:
        need += cfg.missing('twilio_sid', 'twilio_number')
    if need:
        raise SystemExit(f'missing settings in call/keys.env: {", ".join(n.upper() for n in need)}')
    service = Service(cfg, control=True)
    app = service.app()

    async def dial_after_start(app):
        if args.dial:
            sid = await asyncio.to_thread(twilio_media.dial, cfg, args.dial)
            log.info('calling %s (call %s)', args.dial, sid)
    app.on_startup.append(dial_after_start)
    log.info('listening on :%d (%s mode), media at wss://%s/media/<token>', cfg.port, cfg.channel, cfg.public_host)
    web.run_app(app, port=cfg.port, print=None, access_log=None)    # access log would write the media token


if __name__ == '__main__':
    main()
