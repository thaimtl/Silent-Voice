"""Start the whole Silent Voice demo with one command, in order, each part checked before the next starts.

    Double-click "Launch Silent Voice.bat" (Windows), or:
    venv-call/bin/python -m demo.launch                 # the live demo: camera + EMG + Claude + call + dashboard
    venv-call/bin/python -m demo.launch --check         # pre-flight checks only, nothing starts
    venv-call/bin/python -m demo.launch --no-emg        # pads off: camera + Claude
    venv-call/bin/python -m demo.launch --mode sms      # a loud room: the judge calls from our phone, listens on
                                                        # headphones, and texts the same number instead of speaking
    venv-call/bin/python -m demo.launch --rehearse      # no camera, no pads: session 3's camera log plays when a call
                                                        # connects (a dry run of everything after the camera)

Order (each waits until the part before it is really working, not just started):
    pre-flight -> logger -> EMG publisher -> camera reader -> Claude -> tunnel -> call server -> Twilio webhook
    -> decoder -> dashboard
The camera reader warms up (~40 s) while Claude, the tunnel, the call server and Twilio start; the decoder waits for
it. The decoder reuses the last EMG calibration; after re-padding, --calibrate 60 has Austin mouth for 60 s.
Ctrl-C stops everything in reverse order (the logger last, so it records it all). A part that crashes is restarted
(up to MAX_RESTARTS times; a call that was on is lost with its server). Each run gets a folder under $RECLAIM_WORKDIR/logs/
with every part's output and the bus log.

Keys stay in call/keys.env: they are only checked for presence and passed to the parts by their own loaders; no key
is ever printed or written to a log by this script. Call-audio recording (RECORD_INBOUND) is always off.
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WINDOWS = os.name == 'nt'
KEYS = REPO / 'call' / 'keys.env'
NEEDED_KEYS = ('ANTHROPIC_API_KEY', 'CARTESIA_API_KEY', 'CARTESIA_VOICE_ID', 'TWILIO_ACCOUNT_SID',
               'TWILIO_AUTH_TOKEN', 'TWILIO_NUMBER')
BUS_PORTS = {'emg': 5555, 'cam': 5556, 'dec': 5557, 'claude': 5558, 'call': 5559, 'svr': 5560}
CALL_PORT, DASH_PORT = 8787, 8790
MAX_RESTARTS = 3
REHEARSAL_LOG = 'cam_s3_test'
TUNNEL_URL = re.compile(r'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com')   # not cloudflared's own API


# ------------------------------------------------------------------------------------------------ small helpers
def say(msg, ok=None):
    mark = {True: '[ ok ]', False: '[FAIL]', None: '[ .. ]'}[ok]
    print(f'{time.strftime("%H:%M:%S")} {mark} {msg}', flush=True)


def workdir():
    """$RECLAIM_WORKDIR, else the nearest folder above the repo that holds venv-call."""
    env = os.environ.get('RECLAIM_WORKDIR')
    if env:
        return Path(env)
    for d in (REPO.parent, REPO.parent.parent, REPO):
        if venv_python(d, 'venv-call').exists():
            return d
    return REPO.parent


def venv_python(work, venv):
    return Path(work) / venv / ('Scripts/python.exe' if WINDOWS else 'bin/python')


def read_keys(path=KEYS):
    """The keys.env values, read as python-dotenv (the parts' own loader) reads them: `export ` allowed, quoted
    values kept whole, and on an unquoted value a ` #` starts a comment. Never printed."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not key or key.startswith('#'):
            continue
        if value[:1] in ('"', "'") and value[0] in value[1:]:
            value = value[1:value.index(value[0], 1)]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        out[key] = value
    return out


def endpoints(env=None):
    """Bus endpoints as the parts will see them (RECLAIM_<NAME>_PUB overrides, as bus/team.py)."""
    env = os.environ if env is None else env
    return {k: env.get(f'RECLAIM_{k.upper()}_PUB', f'tcp://127.0.0.1:{p}') for k, p in BUS_PORTS.items()}


def port_of(addr):
    return int(str(addr).rsplit(':', 1)[1])


def port_free(port):
    """Whether a part can bind it. POSIX: SO_REUSEADDR, as ZeroMQ and aiohttp bind, so a port a stopped part left in
    TIME_WAIT counts as free (it is). Windows: a plain bind (SO_REUSEADDR there would let two binds share a port)."""
    for host in ('127.0.0.1', '0.0.0.0'):   # a call server listens on all addresses: Windows would let 127.0.0.1 in
        s = socket.socket()
        if not WINDOWS:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def pretty_number(e164):
    d = re.sub(r'\D', '', e164 or '')
    return f'({d[-10:-7]}) {d[-7:-4]}-{d[-4:]}' if len(d) >= 10 else (e164 or '')


# ------------------------------------------------------------------------------------------------ readiness
def wait_bus(addr, topic, timeout_s, alive=lambda: True):
    """True once a message on `topic` arrives from `addr` (raw frames, nothing decoded)."""
    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
    sock.connect(addr)
    try:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and alive():
            if sock.poll(200):
                sock.recv_multipart()
                return True
        return False
    finally:
        sock.close()


UA = {'User-Agent': 'SilentVoice-launcher'}   # Cloudflare's bot check on our domain answers Python-urllib with 403


def http_status(url, timeout=3.0):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def public_status(url, path='/twiml', timeout=8.0):
    """HTTP status of a public https URL. Asks Cloudflare's DNS directly: a just-made tunnel name can take minutes to
    reach the laptop's own resolver, while Twilio (and the judge) see it at once."""
    host = urllib.parse.urlparse(url).hostname
    status = http_status(url + path, timeout)
    if status is not None:
        return status
    try:
        req = urllib.request.Request(f'https://1.1.1.1/dns-query?name={host}&type=A',
                                     headers={'accept': 'application/dns-json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ip = next(a['data'] for a in json.load(r).get('Answer', []) if a.get('type') == 1)
        raw = socket.create_connection((ip, 443), timeout=timeout)
        with ssl.create_default_context().wrap_socket(raw, server_hostname=host) as s:
            s.sendall(f'GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {UA["User-Agent"]}\r\n'
                      f'Connection: close\r\n\r\n'.encode())
            return int(s.recv(64).split(b' ', 2)[1])
    except (OSError, StopIteration, ValueError, IndexError, KeyError):
        return None


# ------------------------------------------------------------------------------------------------ Twilio
class Twilio:
    """Points the number's voice webhook (and in SMS mode its SMS webhook) at this run's tunnel, then reads them back.
    opener: injectable for tests."""

    def __init__(self, sid, token, number, opener=urllib.request.urlopen):
        self.sid, self.number, self.opener = sid, number, opener
        self.auth = 'Basic ' + b64encode(f'{sid}:{token}'.encode()).decode()
        self.base = f'https://api.twilio.com/2010-04-01/Accounts/{sid}/IncomingPhoneNumbers'

    def _call(self, url, data=None):
        req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None,
                                     headers={'Authorization': self.auth})
        try:
            with self.opener(req, timeout=15) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:   # Twilio's own message says what's wrong (it never holds the token)
            try:
                msg = json.loads(e.read()).get('message')
            except (ValueError, OSError, AttributeError):
                msg = None
            raise RuntimeError(f'Twilio said {e.code}: {msg or e.reason}') from None

    def point_at(self, voice_url, sms_url=None, sms_required=False):
        """Voice first (the call needs it: any failure raises). Then SMS, so the dashboard can switch to SMS mode at
        any time; a failure there raises only when SMS mode is required now, else it comes back as a warning."""
        found = self._call(f'{self.base}.json?' + urllib.parse.urlencode({'PhoneNumber': self.number}))
        nums = found.get('incoming_phone_numbers') or []
        if not nums:
            raise RuntimeError(f'{self.number} is not a number on this Twilio account (TWILIO_NUMBER)')
        url = f'{self.base}/{nums[0]["sid"]}.json'
        self._set(url, 'voice_url', {'VoiceUrl': voice_url, 'VoiceMethod': 'POST'})
        if not sms_url:
            return None
        try:
            self._set(url, 'sms_url', {'SmsUrl': sms_url, 'SmsMethod': 'POST'})
        except RuntimeError as e:
            if sms_required:
                raise
            return f'SMS webhook not set ({e}): texts won\'t arrive if you switch to SMS mode'
        return None

    def _set(self, url, key, data):
        self._call(url, data)
        back = self._call(url).get(key)
        if back != list(data.values())[0]:
            raise RuntimeError(f'Twilio still has {back!r} as {key}')


# ------------------------------------------------------------------------------------------------ parts
@dataclass
class Part:
    """One program. ready: ('line', regex) in its output, ('bus', endpoint, topic), or ('http', url, codes)."""
    name: str
    cmd: list
    ready: tuple
    timeout_s: float = 60.0
    restart: bool = True
    env: dict = field(default_factory=dict)
    proc: subprocess.Popen = None
    log: Path = None
    restarts: int = 0
    stopping: bool = False
    reported: bool = False
    offset: int = 0                         # where this start's output begins in the log (a restart appends)

    def start(self, folder, base_env):
        self.log = Path(folder) / f'{self.name.lower().replace(" ", "_")}.log'
        self.offset = self.log.stat().st_size if self.log.exists() else 0
        out = open(self.log, 'ab')
        kw = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if WINDOWS else dict(start_new_session=True)
        self.proc = subprocess.Popen(self.cmd, cwd=REPO, env={**base_env, **self.env}, stdout=out,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        out.close()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def output(self):
        """This start's output (a restarted part's earlier lines don't count)."""
        try:
            with open(self.log, 'rb') as f:
                f.seek(self.offset)
                return f.read().decode('utf-8', errors='replace')
        except OSError:
            return ''

    def tail(self, n=6):
        return '\n'.join('        ' + line for line in self.output().splitlines()[-n:])

    def wait_ready(self, on_line=None):
        """True when ready; False if it died or timed out. on_line(line): every new output line meanwhile."""
        kind, end, seen = self.ready[0], time.monotonic() + self.timeout_s, 0
        while time.monotonic() < end:
            if not self.alive():
                return False
            text = self.output()
            if on_line:
                for line in text.splitlines()[seen:]:
                    on_line(line)
                seen = len(text.splitlines())
            if kind == 'line' and re.search(self.ready[1], text):
                return True
            if kind == 'bus' and wait_bus(self.ready[1], self.ready[2], 0.5, self.alive):
                return True
            if kind == 'http' and http_status(self.ready[1]) in self.ready[2]:
                return True
            if kind != 'bus':
                time.sleep(0.25)
        return False

    def stop(self, grace_s=5.0):
        """Ctrl-C for the part's whole process group (demo/run_part.py turns CTRL_BREAK into it on Windows), then a
        hard kill of the whole tree: on Windows a venv's python.exe starts the real interpreter as its child."""
        self.stopping = True
        if not self.alive():
            return
        try:
            if WINDOWS:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.proc.pid, signal.SIGINT)
            self.proc.wait(grace_s)
            return
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        try:
            if WINDOWS:
                subprocess.run(['taskkill', '/T', '/F', '/PID', str(self.proc.pid)], capture_output=True, timeout=15)
            else:
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(5)
        except (subprocess.TimeoutExpired, OSError):
            pass


# ------------------------------------------------------------------------------------------------ the demo
class Launcher:
    def __init__(self, args, work, keys):
        self.args, self.work, self.keys = args, Path(work), keys
        self.folder = self.work / 'logs' / (time.strftime('%Y%m%d-%H%M%S') + '-demo')
        self.parts, self.public_url, self.in_call = [], None, False
        self.lock, self.shutting_down = threading.Lock(), False
        self.env = {**os.environ, 'RECLAIM_WORKDIR': str(self.work), 'PYTHONUNBUFFERED': '1',
                    'PYTHONIOENCODING': 'utf-8', 'RECORD_INBOUND': ''}
        if args.number:                     # the call server reads TWILIO_NUMBER too (its own number)
            self.env['TWILIO_NUMBER'] = args.number
        self.ep = endpoints(self.env)
        self.call_port, self.dash_port = args.call_port, args.dash_port
        self.control_port = args.call_port + 1          # the call server's mode switch, on the loopback only
        self.uno = args.port

    def py(self, venv):
        return str(venv_python(self.work, venv))

    def module(self, venv, name, *args):
        """A part's command: the module run through demo/run_part.py, so Ctrl-Break stops it cleanly on Windows."""
        return [self.py(venv), '-m', 'demo.run_part', name, *args]

    # ---------------------------------------------------------------- pre-flight
    def preflight(self):
        """Every check, each with its fix; returns True when nothing blocks the start."""
        a, problems = self.args, []

        def check(ok, what, fix):
            say(what if ok else f'{what}: {fix}', ok)
            if not ok:
                problems.append(what)

        missing = [k for k in NEEDED_KEYS if not self.keys.get(k)]
        check(KEYS.exists() and not missing, 'keys in call/keys.env',
              f'missing {", ".join(missing) or "the file"} (ask Frank; never commit it)')
        venvs = ['venv-call'] + ([] if a.rehearse else ['venv-live']) + ['venv-svr']
        for v in venvs:
            check(venv_python(self.work, v).exists(), f'{v} in {self.work}',
                  'create it (DEMO.md, one-time setup), or set RECLAIM_WORKDIR')
        ports = [port_of(x) for x in self.ep.values()] + [self.call_port, self.dash_port]
        busy = [p for p in ports if not port_free(p)]
        check(not busy, 'ports free', f'{busy} in use: close the other Silent Voice window, or restart the laptop')
        check(bool(self.cloudflared()), 'cloudflared', 'install it (DEMO.md) or put cloudflared(.exe) in '
                                                       f'{self.work}')
        check(gpu_ok(), 'NVIDIA GPU', 'nvidia-smi failed: check the driver, and that the laptop uses the RTX 5060')
        if a.rehearse:
            check((self.work / 'logs' / REHEARSAL_LOG).exists(), f'rehearsal log logs/{REHEARSAL_LOG}',
                  'copy it from Frank, or run without --rehearse')
        else:
            check(self.camera_opens(), f'webcam {a.cam}', 'plug in the C920, close other apps using it, or pass --cam 1')
            if not self.face_adapter().exists():    # not a blocker: the base model still reads, at twice the errors
                say(f'no face adapter at {self.face_adapter()}: the camera reads with the base model (2x the word '
                    'errors); copy it from the bucket', None)
        if not (a.no_emg or a.rehearse):
            self.uno = a.port or self.find_uno()
            check(bool(self.uno), f'Uno on {self.uno or "?"}', 'plug in its USB cable, pass --port COMx, or --no-emg')
            on_ac = charging()
            if on_ac and getattr(a, 'allow_charger', False):
                say('the laptop is charging: going ahead (--allow-charger)')
            elif on_ac:
                say('the laptop is charging: unplug the charger before the pads go on skin (no USB isolator)', False)
                problems.append('charger')
        return not problems

    def face_adapter(self):
        return self.work / 'data' / 'face_adapter' / 'austin.pth'

    def cloudflared(self):
        for p in (shutil.which('cloudflared'), self.work / ('cloudflared.exe' if WINDOWS else 'cloudflared'),
                  REPO / ('cloudflared.exe' if WINDOWS else 'cloudflared')):
            if p and Path(p).exists():
                return str(p)
        return None

    def camera_opens(self):
        code = ('import cv2,sys; c=cv2.VideoCapture(int(sys.argv[1]), cv2.CAP_DSHOW if sys.platform=="win32" else 0);'
                'ok=c.isOpened() and c.read()[0]; c.release(); sys.exit(0 if ok else 1)')
        try:
            return subprocess.run([self.py('venv-live'), '-c', code, str(self.args.cam)], cwd=REPO, env=self.env,
                                  capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def find_uno(self):
        # rig/signal_check.py imports its neighbours as a script does (`from filters import ...`): rig/ goes on the path
        code = ('import sys; sys.path.insert(0, "rig"); from serial.tools import list_ports;'
                'from signal_check import ARDUINO_VIDS;'
                'print(next((p.device for p in list_ports.comports() if p.vid in ARDUINO_VIDS), ""))')
        try:
            r = subprocess.run([self.py('venv-svr'), '-c', code], cwd=REPO, env=self.env, capture_output=True,
                               text=True, timeout=30)
            return r.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            return None

    # ---------------------------------------------------------------- the parts, in order
    def plan(self):
        a, ep, parts = self.args, self.ep, []
        parts.append(Part('Logger', self.module('venv-call', 'buslog.record', '--out', str(self.folder / 'bus')),
                          ('line', r'logging to'), 30))
        if not (a.no_emg or a.rehearse):
            parts.append(Part('EMG', self.module('venv-svr', 'rig.publish', self.uno),
                              ('bus', ep['emg'], 'emg.raw'), 30))
        if not a.rehearse:
            cam = self.module('venv-live', 'lipread.run', '--cam', str(a.cam), '--quiet')
            if self.face_adapter().exists():    # Austin's face: halves the camera's word errors (lipread/MEASURED.md)
                cam += ['--face-adapter', str(self.face_adapter())]
            if getattr(a, 'exposure', None) is not None:   # a stop under auto (-6 here) reads lips better (Oct 25)
                cam += ['--exposure', str(a.exposure)]
            parts.append(Part('Camera', cam + (['--camera-settings'] if a.camera_settings else []),
                              ('bus', ep['cam'], 'cam.preview'), 300 if a.camera_settings else 120))
        if not a.no_claude:
            parts.append(Part('Claude', self.module('venv-call', 'claude_service.service'),
                              ('line', r'claude: API key from'), 30))
        fusion = self.module('venv-svr', 'fusion.run', '--repair-s', str(a.repair_s))
        if a.no_emg or a.rehearse:
            fusion.append('--no-emg')
        elif a.calibrate > 0:
            fusion += ['--calibrate', str(a.calibrate)]
        if a.no_claude:
            fusion.append('--no-claude')
        parts.append(Part('Decoder', fusion, ('line', r'fusion: running'), 180 + a.calibrate))
        return parts

    def run(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        say(f'this run\'s logs: {self.folder}')
        try:
            *before, decoder = self.plan()
            camera = next((p for p in before if p.name == 'Camera'), None)
            for part in before:
                if part is camera:
                    self.launch(part)           # it warms up while the rest starts
                else:
                    self.start(part)
            self.start_call()                   # the network can fail: before the calibration, not after it
            if camera:
                self.ready(camera)
            self.start(decoder)
            self.start(Part('Dashboard', self.module('venv-call', 'dashboard.server', '--port', str(self.dash_port),
                                                     '--number', pretty_number(self.keys.get('TWILIO_NUMBER')),
                                                     '--channel', self.args.mode,
                                                     '--control', f'http://127.0.0.1:{self.control_port}'),
                            ('http', f'http://127.0.0.1:{self.dash_port}/', (200,)), 30))
            self.ready_banner()
            self.watch()
        except KeyboardInterrupt:
            print()
        except StartFailed as e:
            say(str(e), False)
        finally:
            self.shutdown()

    def start(self, part):
        self.launch(part)
        self.ready(part)

    def launch(self, part):
        say(f'starting {part.name}')
        part.start(self.folder, self.env)
        self.parts.append(part)

    def ready(self, part):
        if not part.wait_ready(self.prompt_for(part)):
            raise StartFailed(f'{part.name} did not come up ({"crashed" if not part.alive() else "timed out"}); '
                              f'its last lines:\n{part.tail()}\n        full log: {part.log}')
        say(f'{part.name} ready', True)

    def prompt_for(self, part):
        if part.name != 'Decoder':
            return None

        def on_line(line):
            if 'calibrating' in line:
                print('\n' + '=' * 72 + f'\n  AUSTIN: mouth a few normal sentences now, for {self.args.calibrate:.0f} s'
                      '\n  (the EMG model is learning your signal level)\n' + '=' * 72 + '\n', flush=True)
            elif line.startswith('fusion: calibration'):
                say('calibration done', True)
        return on_line

    def start_call(self):
        """Tunnel first (the call server must know its public address), then the server, then Twilio.
        --tunnel-host (RECLAIM_TUNNEL_HOST): our named Cloudflare tunnel, the same address every launch and no
        quick-tunnel rate limit (429 after many restarts); without it, a new quick tunnel each launch."""
        url = f'http://127.0.0.1:{self.call_port}'
        named = getattr(self.args, 'tunnel_host', None)
        if named:
            tunnel = Part('Tunnel', [self.cloudflared(), 'tunnel', '--no-autoupdate', 'run', '--url', url,
                                     getattr(self.args, 'tunnel_name', 'silentvoice')],
                          ('line', r'Registered tunnel connection'), 60, restart=False)
            self.start(tunnel)
            self.public_url = f'https://{named}'
        else:
            tunnel = Part('Tunnel', [self.cloudflared(), 'tunnel', '--no-autoupdate', '--url', url],
                          ('line', TUNNEL_URL.pattern), 60, restart=False)
            self.start(tunnel)
            self.public_url = TUNNEL_URL.search(tunnel.output()).group(0)
        host = urllib.parse.urlparse(self.public_url).hostname
        self.start(Part('Call', self.module('venv-call', 'call.server'), ('line', r'Sonic websocket open'), 60,
                        env={'PUBLIC_HOST': host, 'PORT': str(self.call_port), 'CALL_CHANNEL': self.args.mode,
                             'CALL_CONTROL_PORT': str(self.control_port)}))
        say('waiting for the tunnel to reach the call server from outside')
        end = time.monotonic() + 90
        while public_status(self.public_url) not in (200, 405):
            if time.monotonic() > end:
                raise StartFailed(f'{self.public_url} does not reach the call server: check the internet connection')
            time.sleep(2)
        say('tunnel reachable', True)
        if self.args.no_webhook:
            say('--no-webhook: Twilio left as it is', None)
            return
        try:
            warning = Twilio(self.keys['TWILIO_ACCOUNT_SID'], self.keys['TWILIO_AUTH_TOKEN'],
                             self.keys['TWILIO_NUMBER']).point_at(self.public_url + '/twiml', self.public_url + '/sms',
                                                                  sms_required=self.args.mode == 'sms')
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError) as e:
            raise StartFailed(f'Twilio webhook not updated: {e}') from None
        if warning:
            say(warning, None)
        say(f'Twilio number {pretty_number(self.keys["TWILIO_NUMBER"])} points at this laptop', True)

    def ready_banner(self):
        lines = ['SILENT VOICE IS READY',
                 f'Dashboard:  http://127.0.0.1:{self.dash_port}' + ('' if self.args.no_browser else
                                                                     '   (opened in the browser)'),
                 f'Call:       {pretty_number(self.keys.get("TWILIO_NUMBER"))}',
                 f'Running:    {", ".join(p.name for p in self.parts)}',
                 f'Logs:       {self.folder}',
                 'Stop:       Ctrl-C in this window']
        if self.args.mode == 'sms':
            lines.insert(3, 'SMS mode:   call the number from our phone (headphones on the judge), then the judge '
                            'texts it from that phone')
        lines.insert(3, 'Mode:       switch Call / Text on the dashboard (or press M there), even mid-call')
        if self.args.rehearse:
            lines.insert(3, 'Rehearsal:  call the number; session 3 plays once the call connects')
        print('\n' + '=' * 72 + '\n' + '\n'.join('  ' + x for x in lines) + '\n' + '=' * 72 + '\n', flush=True)
        if not self.args.no_browser:
            try:
                import webbrowser
                webbrowser.open(f'http://127.0.0.1:{self.dash_port}')
            except Exception:  # noqa: BLE001  a missing browser never stops the demo
                pass

    # ---------------------------------------------------------------- while it runs
    def watch(self):
        threading.Thread(target=self.follow_call, daemon=True).start()
        while True:
            self.check_parts()
            time.sleep(1)

    def check_parts(self, backoff_s=2.0):
        """One look at every part: a stopped one is reported, and restarted when it may be."""
        for part in list(self.parts):
            if part.alive() or part.stopping:
                continue
            if part.name == 'Rehearsal':
                say('rehearsal replay finished', True)
                self.parts.remove(part)
                continue
            if part.name == 'Call':
                self.in_call = False            # a dead call server has no call left: restart it now
            if not part.reported:
                say(f'{part.name} stopped (exit {part.proc.returncode}); last lines:\n{part.tail()}', False)
                part.reported = True
            if not part.restart or part.restarts >= MAX_RESTARTS:
                say(f'{part.name} is not restarted{"" if part.restart else " (Ctrl-C and launch again)"}: '
                    f'see {part.log}', False)
                part.stopping = True
                continue
            part.restarts += 1
            if part.name == 'Decoder':          # a restart skips the calibration: nobody is told to mouth for it
                part.cmd = [x for i, x in enumerate(part.cmd) if x != '--calibrate' and
                            not (i and part.cmd[i - 1] == '--calibrate')]
            if part.name == 'Logger':           # buslog never appends to a session folder: each restart gets its own
                part.cmd[part.cmd.index('--out') + 1] = str(self.folder / f'bus-r{part.restarts}')
            time.sleep(backoff_s * part.restarts)
            say(f'restarting {part.name} ({part.restarts}/{MAX_RESTARTS})')
            part.reported = False
            part.start(self.folder, self.env)
            back = part.wait_ready()
            say(f'{part.name} back' if back else f'{part.name} did not come back', back)

    def follow_call(self):
        """call.state from the call server: shown on screen, and starts the rehearsal replay."""
        import zmq
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'call.state')
        sock.connect(self.ep['call'])
        import msgpack
        while True:
            if self.shutting_down:
                return
            if not sock.poll(500):
                continue
            try:
                topic, payload = sock.recv_multipart()
                state = msgpack.unpackb(payload, raw=False).get('state')
            except Exception:  # noqa: BLE001  a malformed message is skipped
                time.sleep(0.1)
                continue
            if state == 'connected':
                self.in_call = True
                say('call connected', True)
                if self.args.rehearse and not any(p.name == 'Rehearsal' for p in self.parts):
                    time.sleep(3)
                    part = Part('Rehearsal', self.module('venv-call', 'buslog.replay',
                                                         str(self.work / 'logs' / REHEARSAL_LOG), '--topics', 'cam.'),
                                ('line', r'.'), 5, restart=False)
                    with self.lock:             # never after shutdown took its list: it would be left running
                        if self.shutting_down:
                            return
                        part.start(self.folder, self.env)
                        self.parts.append(part)
                    say('rehearsal: session 3 is playing into the call')
            elif state == 'ended':
                self.in_call = False
                say('call ended', None)

    def shutdown(self):
        with self.lock:
            self.shutting_down = True
            parts, self.parts = list(self.parts), []
        if not parts:
            return
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)     # a second Ctrl-C would orphan the parts left
        try:
            say('stopping everything (wait for it: this takes a few seconds)')
            logger = [p for p in parts if p.name == 'Logger']
            for part in [p for p in reversed(parts) if p.name != 'Logger'] + logger:
                part.stop()
                say(f'{part.name} stopped', True)
            say(f'logs kept in {self.folder}', True)
        finally:
            signal.signal(signal.SIGINT, old)


class StartFailed(Exception):
    pass


def charging():
    """True when the laptop runs on its charger (pads must not be on skin then)."""
    if WINDOWS:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [('ACLineStatus', ctypes.c_byte), ('BatteryFlag', ctypes.c_byte),
                        ('BatteryLifePercent', ctypes.c_byte), ('SystemStatusFlag', ctypes.c_byte),
                        ('BatteryLifeTime', ctypes.c_ulong), ('BatteryFullLifeTime', ctypes.c_ulong)]
        s = Status()
        return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))) and s.ACLineStatus == 1
    for p in Path('/sys/class/power_supply').glob('*/type'):
        try:
            if p.read_text().strip() == 'Mains' and (p.parent / 'online').read_text().strip() == '1':
                return True
        except OSError:
            pass
    return False


def gpu_ok():
    gpu = shutil.which('nvidia-smi')
    try:
        return bool(gpu) and subprocess.run([gpu, '-L'], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description='Start the Silent Voice demo, in order.')
    ap.add_argument('--check', action='store_true', help='pre-flight checks only')
    ap.add_argument('--mode', choices=('call', 'sms'), default='call',
                    help="call: the judge speaks on the call (default). sms: a loud room: the judge texts the number "
                         "from the phone on the call and hears Austin on it")
    ap.add_argument('--no-emg', action='store_true', help='pads off: camera + Claude only')
    ap.add_argument('--no-claude', action='store_true', help='without Claude (fallback)')
    ap.add_argument('--rehearse', action='store_true', help='no camera or pads: session 3 plays when a call connects')
    ap.add_argument('--cam', type=int, default=0, help='webcam index (default 0)')
    ap.add_argument('--camera-settings', action='store_true', help='Windows: lock exposure and focus in a dialog')
    ap.add_argument('--exposure', type=lambda s: None if s == 'auto' else float(s), default=-7.0,
                    help="webcam exposure, a stop darker than auto by default (-7; 'auto' for the camera's own)")
    ap.add_argument('--allow-charger', action='store_true',
                    help='start the pads with the charger plugged in (the new setup; default: refuse)')
    ap.add_argument('--port', default=None, help="the Uno's serial port, e.g. COM5 (default: found by itself)")
    ap.add_argument('--calibrate', type=float, default=0.0,
                    help='seconds of EMG calibration at the start, after re-padding (default 0: reuse the last one)')
    ap.add_argument('--repair-s', type=float, default=0.0,
                    help='wait before a locked word is spoken (default 0: repairs fix the captions only, never the voice)')
    ap.add_argument('--call-port', type=int, default=CALL_PORT)
    ap.add_argument('--dash-port', type=int, default=DASH_PORT)
    ap.add_argument('--number', default=None, help='use this Twilio number instead of TWILIO_NUMBER, e.g. +16282293851')
    ap.add_argument('--no-webhook', action='store_true', help="don't touch the Twilio number's webhook")
    ap.add_argument('--tunnel-host', default=os.environ.get('RECLAIM_TUNNEL_HOST'),
                    help='public host of our named Cloudflare tunnel (default $RECLAIM_TUNNEL_HOST; none: a quick tunnel)')
    ap.add_argument('--tunnel-name', default=os.environ.get('RECLAIM_TUNNEL_NAME', 'silentvoice'),
                    help='name of the named tunnel (cloudflared tunnel create NAME)')
    ap.add_argument('--no-browser', action='store_true', help="don't open the dashboard in the browser")
    ap.add_argument('--yes', action='store_true', help="don't ask about the signal check")
    args = ap.parse_args(argv)
    if WINDOWS:
        os.system('')                   # ANSI and UTF-8 friendly console
    try:                                # a part's odd character in its log never stops the launcher
        sys.stdout.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass
    work = workdir()
    keys = read_keys()
    if args.number:
        keys['TWILIO_NUMBER'] = args.number
    launcher = Launcher(args, work, keys)
    say(f'Silent Voice launcher: repo {REPO}, workdir {work}')
    ok = launcher.preflight()
    if args.check or not ok:
        say('pre-flight passed' if ok else 'pre-flight failed: fix the lines marked FAIL, then run again', ok)
        return 0 if ok else 1
    if not (args.no_emg or args.rehearse or args.yes):
        answer = input('Pads on and signal check done (.claude/skills/signal-check)? [Y/n] ').strip().lower()
        if answer not in ('', 'y', 'yes'):
            say('run the signal check first, then start again', False)
            return 1
    launcher.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
