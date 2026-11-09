"""The live dashboard: every team-bus topic -> CallState -> the page (Thai's Figma v5), over one websocket.

    venv-call/bin/python -m dashboard.server [--port 8790] [--number "(628) 888-6598"]
    then open http://127.0.0.1:8790 full screen (F11). Replay a logged call with buslog.replay, or rehearse with
    python -m dashboard.demo_feed; the dashboard can't tell them from a live call.

It only listens (SUB to every endpoint in bus/team.py), so it can start and stop at any time without touching
the call. Snapshots go out SNAPSHOT_HZ times a second as JSON text frames. Each cam.preview JPEG goes out on arrival as
a binary frame with that frame's face points (pack_frame); a slow browser gets the newest frame, never a backlog.
"""
import argparse
import asyncio
import json
import logging
import os
import struct
import time
from collections import deque

import numpy as np
import zmq
import zmq.asyncio
from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

import bus
from bus.team import ENDPOINTS, decode
from buslog import log_root
from dashboard.state import CallState

log = logging.getLogger('dashboard')
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
TOPICS = ('call.state', 'partner.text', 'dec.', 'claude.repair', 'voice.say', 'cam.tick', 'cam.face',
          'cam.preview', 'emg.raw', 'timing.')
SNAPSHOT_HZ = 15
FRAME_MAGIC = b'RCF1'   # binary frame: magic, 4-byte big-endian header length, JSON header, JPEG
FACE_MATCH_S = 0.06     # max gap between a preview and its cam.face
LOG_FRESH_S = 5.0   # the logger flushes every 0.2 s: older than this, it isn't running
CONTROL_EVERY_S = 3.0   # how often the page's mode switch re-reads the call server's mode


SHIPPED_BENCHMARK = os.path.join(os.path.dirname(__file__), 'benchmark_session3.json')


def load_benchmark(path=None):
    """Word error per reader on a scored session, in %, or None (the end screen then shows its placeholder).
    Without a path: $RECLAIM_WORKDIR/results/wer_session3.json if it exists, else the scored session 3 numbers that
    ship with the dashboard (benchmark_session3.json, from docs/SVR1_SPEC.md). `labels` renames a row, e.g. the third
    when it was scored without Claude."""
    if path is None:
        work = os.environ.get('RECLAIM_WORKDIR')
        path = os.path.join(work or '.', 'results', 'wer_session3.json')
        if not os.path.exists(path):
            path = SHIPPED_BENCHMARK
    try:
        with open(path) as f:
            b = json.load(f)
        out = {k: float(b[k]) for k in ('lips', 'lips_muscles', 'claude')} | {'session': str(b.get('session', ''))}
        labels = b.get('labels') or {}
        if labels:
            out['labels'] = {k: str(v) for k, v in labels.items() if k in ('lips', 'lips_muscles', 'claude')}
        return out
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


class Dashboard:
    def __init__(self, number='', endpoints=None, ctx=None, benchmark=None, channel='call', control=None):
        self.state = CallState(number, benchmark, channel)
        self.control = control.rstrip('/') if control else None     # the call server's mode switch (loopback)
        self.http = None
        self.endpoints = dict(endpoints or ENDPOINTS)
        self.ctx = ctx or zmq.asyncio.Context.instance()
        self.clients = set()
        self.frame = None                # newest packed preview frame
        self.viewers = {}                # websocket -> [unsent frame, asyncio.Event]
        self.faces = deque(maxlen=40)    # recent cam.face (t, pts)
        self.tasks = []

    def subscribe(self):
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2000)
        for addr in self.endpoints.values():
            sock.connect(addr)
        for t in TOPICS:
            sock.setsockopt(zmq.SUBSCRIBE, t.encode())
        return sock

    def handle(self, frames):
        """One bus message; a malformed one is skipped (the dashboard never stops the call)."""
        try:
            topic, msg = decode(frames, validate=False)
        except ValueError:
            return
        if topic == 'cam.preview':
            jpeg = msg.get('jpeg')
            if isinstance(jpeg, (bytes, bytearray)) and jpeg:
                try:
                    self.frame = pack_frame(msg, self.face_for(msg))
                except (TypeError, ValueError):
                    self.frame = pack_frame(msg, None)
                for slot in self.viewers.values():
                    slot[0] = self.frame
                    slot[1].set()
            return
        if topic == 'cam.face' and msg.get('face_ok') and msg.get('pts') is not None:
            try:
                self.faces.append((float(msg['t']), points(msg['pts'])))
            except (KeyError, TypeError, ValueError):
                pass
        try:
            self.state.feed(topic, msg, bus.now())
        except Exception:  # noqa: BLE001  one odd message must not stop the screen
            log.exception('bad %s message', topic)

    def face_for(self, msg):
        """The preview's own points, else the nearest cam.face within FACE_MATCH_S, else None."""
        if msg.get('pts') is not None:
            return points(msg['pts'])
        try:
            t = float(msg['t'])
        except (KeyError, TypeError, ValueError):
            return None
        best = min(self.faces, key=lambda f: abs(f[0] - t), default=None)
        return best[1] if best is not None and abs(best[0] - t) <= FACE_MATCH_S else None

    async def bus_loop(self):
        sock = self.subscribe()
        try:
            while True:
                self.handle(await sock.recv_multipart())
        finally:
            sock.close(linger=0)

    def check_logging(self):
        """The bus logger (buslog.record) wrote within the last few seconds: its newest file is fresh. The files sit in
        the newest session folder, or one folder down (the launcher's <run>-demo/bus/, bus-r1/ after a restart)."""
        try:
            sessions = sorted(log_root().iterdir())
            files = [*sessions[-1].glob('bus-*.msgpack'), *sessions[-1].glob('*/bus-*.msgpack')] if sessions else []
            newest = max((f.stat().st_mtime for f in files), default=0)
        except OSError:
            return
        if time.time() - newest < LOG_FRESH_S:
            self.state.logging(bus.now())

    async def send_frames(self, ws, slot):
        """Sends each page the newest frame only, so a slow page skips frames."""
        while True:
            await slot[1].wait()
            slot[1].clear()
            frame, slot[0] = slot[0], None
            if frame is not None:
                await ws.send_bytes(frame)

    async def push_loop(self):
        period, k = 1.0 / SNAPSHOT_HZ, 0
        while True:
            await asyncio.sleep(period)
            k += 1
            if k % SNAPSHOT_HZ == 0:
                self.check_logging()
            if not self.clients:
                continue
            try:
                text = json.dumps(self.state.snapshot(bus.now()), separators=(',', ':'))
            except Exception:  # noqa: BLE001  a bad state skips one frame; the screen keeps updating
                log.exception('snapshot failed')
                continue
            for ws in list(self.clients):
                try:
                    await ws.send_str(text)
                except (ConnectionError, RuntimeError):
                    self.clients.discard(ws)

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=10, compress=False)
        await ws.prepare(request)
        self.clients.add(ws)
        slot = self.viewers[ws] = [self.frame, asyncio.Event()]
        if self.frame is not None:
            slot[1].set()
        sender = asyncio.create_task(self.send_frames(ws, slot))
        try:
            async for m in ws:
                if m.type == WSMsgType.ERROR:
                    break
        finally:
            sender.cancel()
            self.clients.discard(ws)
            self.viewers.pop(ws, None)
        return ws

    # --- the Call / Text switch: the page asks us, we ask the call server (which owns the mode) ---------------
    async def _control(self, method, body=None):
        async with self.http.request(method, self.control + '/channel', json=body,
                                     headers={'X-Silent-Voice': '1'} if body else None) as r:
            if r.status != 200:
                raise ClientError(f'call server said {r.status}: {await r.text()}')
            return (await r.json())['channel']

    async def control_loop(self):
        """Keeps the switch honest: the call server's mode, and whether it can be switched (call server up)."""
        while True:
            try:
                self.state.set_channel(await self._control('GET'), switchable=True)
            except (ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError, TypeError):
                self.state.set_channel(None, switchable=False)
            await asyncio.sleep(CONTROL_EVERY_S)

    async def channel(self, request):
        """POST {"channel": "call" | "sms"} from the page. Same origin only: the header forces a browser preflight."""
        if request.headers.get('X-Silent-Voice') != '1':
            raise web.HTTPForbidden()
        if not self.control:
            raise web.HTTPServiceUnavailable(text='no call server to switch (start the demo with the launcher)')
        try:
            want = (await request.json()).get('channel')
            if want not in ('call', 'sms'):
                raise ValueError(want)
        except (ValueError, AttributeError, TypeError):
            raise web.HTTPBadRequest(text='channel must be "call" or "sms"') from None
        try:
            got = await self._control('POST', {'channel': want})
        except (ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError, TypeError) as e:
            log.warning('mode switch failed: %s', e)
            raise web.HTTPBadGateway(text='the call server did not switch: is it running?') from None
        self.state.set_channel(got, switchable=True)
        return web.json_response({'channel': got})

    async def index(self, request):
        return web.FileResponse(os.path.join(STATIC, 'index.html'))

    def app(self):
        app = web.Application()
        app.router.add_get('/', self.index)
        app.router.add_get('/ws', self.websocket)
        app.router.add_post('/api/channel', self.channel)
        app.router.add_static('/static/', STATIC)

        async def start(app):
            self.tasks = [asyncio.create_task(self.bus_loop()), asyncio.create_task(self.push_loop())]
            if self.control:
                self.http = ClientSession(timeout=ClientTimeout(total=4))
                self.tasks.append(asyncio.create_task(self.control_loop()))

        async def stop(app):
            for t in self.tasks:
                t.cancel()
            for ws in list(self.clients):
                await ws.close()
            if self.http is not None:
                await self.http.close()
        app.on_startup.append(start)
        app.on_cleanup.append(stop)
        return app


def points(pts):
    """Face points as (N, 2) or (N, 3) floats; ValueError otherwise."""
    a = np.asarray(pts, np.float64)
    if a.ndim != 2 or a.shape[1] not in (2, 3):
        a = a.reshape(-1, 2)
    return a


def pack_frame(msg, pts):
    """FRAME_MAGIC, 4-byte header length, JSON header {t, w, h, pts}, JPEG."""
    head = json.dumps({'t': msg.get('t'), 'w': msg.get('w'), 'h': msg.get('h'),
                       'pts': None if pts is None else np.round(points(pts), 4).tolist()},
                      separators=(',', ':')).encode()
    return FRAME_MAGIC + struct.pack('>I', len(head)) + head + bytes(msg['jpeg'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8790)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--number', default='', help="the judge's number as shown in the header, e.g. '(628) 888-6598'")
    ap.add_argument('--channel', choices=('call', 'sms'), default='call',
                    help="how the judge talks: on the call (Ink) or by text (SMS mode); call.state confirms it")
    ap.add_argument('--control', help="the call server's mode switch, e.g. http://127.0.0.1:8788 (the launcher sets it); "
                                      'without it the page has no Call / Text switch')
    ap.add_argument('--benchmark', help='scored word error JSON (default: $RECLAIM_WORKDIR/results/wer_session3.json)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    print(f'dashboard: http://{args.host}:{args.port}')
    benchmark = load_benchmark(args.benchmark)
    print('end screen word error:', f"from {benchmark['session'] or 'the benchmark file'}" if benchmark else 'placeholder')
    web.run_app(Dashboard(args.number, benchmark=benchmark, channel=args.channel, control=args.control).app(), host=args.host, port=args.port, print=None,
                access_log=None)


if __name__ == '__main__':
    main()
