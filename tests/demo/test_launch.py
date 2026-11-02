"""demo/launch.py: start order, readiness, crashes and restarts, shutdown order, Twilio, and that no key is printed.

    venv-call/bin/python -m pytest tests/demo -q
"""
import io
import os
import json
import socket
import sys
import time
import types
import urllib.parse
from contextlib import redirect_stdout

import pytest

from demo import launch
from demo.launch import Launcher, Part, StartFailed, Twilio


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def args(**kw):
    base = dict(check=False, no_emg=False, no_claude=False, rehearse=False, cam=0, camera_settings=False, port='COM9',
                calibrate=60.0, repair_s=0.3, call_port=free_port(), dash_port=free_port(), no_webhook=True,
                no_browser=True, yes=True, number=None, mode='call')
    return types.SimpleNamespace(**{**base, **kw})


def py(code):
    return [sys.executable, '-u', '-c', code]


def test_read_keys_and_number(tmp_path):
    f = tmp_path / 'keys.env'
    f.write_text('# comment\nTWILIO_NUMBER=+16288886598              # E.164, the number we call from\n'
                 'ANTHROPIC_API_KEY=sk-test\nexport CARTESIA_VOICE_ID="abc # not a comment"\nEMPTY=\n'
                 "PUBLIC_HOST='x.trycloudflare.com'  # old\n")
    keys = launch.read_keys(f)
    assert keys['TWILIO_NUMBER'] == '+16288886598' and keys['ANTHROPIC_API_KEY'] == 'sk-test'
    assert keys['CARTESIA_VOICE_ID'] == 'abc # not a comment' and keys['PUBLIC_HOST'] == 'x.trycloudflare.com'
    assert keys['EMPTY'] == ''
    assert launch.pretty_number(keys['TWILIO_NUMBER']) == '(628) 888-6598'


def test_ready_on_a_line(tmp_path):
    p = Part('Fake', py('import time; print("warming up"); time.sleep(0.3); print("fusion: running"); time.sleep(5)'),
             ('line', r'fusion: running'), 5)
    p.start(tmp_path, {})
    try:
        assert p.wait_ready()
    finally:
        p.stop(1)
    assert not p.alive()


def test_a_crash_or_a_hang_is_not_ready(tmp_path):
    crash = Part('Crash', py('import sys; print("Traceback: boom"); sys.exit(3)'), ('line', 'never'), 5)
    crash.start(tmp_path, {})
    assert not crash.wait_ready() and crash.proc.returncode == 3 and 'boom' in crash.tail()
    hang = Part('Hang', py('import time; time.sleep(30)'), ('line', 'never'), 0.5)
    hang.start(tmp_path, {})
    t0 = time.monotonic()
    assert not hang.wait_ready() and time.monotonic() - t0 < 3
    hang.stop(1)


def test_ready_on_the_bus(tmp_path):
    port = free_port()
    code = (f'import time, zmq; s = zmq.Context().socket(zmq.PUB); s.bind("tcp://127.0.0.1:{port}")\n'
            'while True:\n    s.send_multipart([b"emg.raw", b"x"]); time.sleep(0.05)')
    p = Part('EMG', py(code), ('bus', f'tcp://127.0.0.1:{port}', 'emg.raw'), 10)
    p.start(tmp_path, dict(os.environ))   # on Windows a process without SYSTEMROOT can't open a socket
    try:
        assert p.wait_ready()
    finally:
        p.stop(1)


def launcher(tmp_path, **kw):
    lc = Launcher(args(**kw), tmp_path, {'TWILIO_NUMBER': '+16288886598'})
    lc.folder = tmp_path / 'run'
    lc.folder.mkdir(exist_ok=True)
    return lc


def test_a_part_that_never_comes_up_stops_the_start_with_its_log(tmp_path):
    lc = launcher(tmp_path)
    with pytest.raises(StartFailed) as e:
        lc.start(Part('Camera', py('import sys; print("no camera at index 0"); sys.exit(1)'), ('line', 'x'), 5))
    assert 'Camera did not come up (crashed)' in str(e.value) and 'no camera at index 0' in str(e.value)


def test_shutdown_reverse_order_logger_last(tmp_path):
    lc = launcher(tmp_path)
    order = []
    for name in ('Logger', 'EMG', 'Camera', 'Decoder'):
        p = Part(name, py('import time; time.sleep(30)'), ('line', '.'), 5)
        p.start(lc.folder, {})
        p.stop = (lambda n, orig: lambda grace_s=5.0: (order.append(n), orig(1)))(name, p.stop)
        lc.parts.append(p)
    with redirect_stdout(io.StringIO()):
        lc.shutdown()
    assert order == ['Decoder', 'Camera', 'EMG', 'Logger']


def test_crashed_part_restarts_up_to_the_limit(tmp_path):
    lc = launcher(tmp_path)
    p = Part('Claude', py('import sys; sys.exit(1)'), ('line', 'never'), 0.3)
    p.start(lc.folder, {})
    p.proc.wait()
    lc.parts.append(p)
    out = io.StringIO()
    with redirect_stdout(out):
        for _ in range(launch.MAX_RESTARTS + 2):
            p.proc.wait()
            lc.check_parts(backoff_s=0)
    assert p.restarts == launch.MAX_RESTARTS and p.stopping
    assert 'is not restarted' in out.getvalue()


def test_a_dead_call_server_is_restarted_even_if_a_call_was_on(tmp_path):
    # it never sends call.state 'ended' when it crashes: waiting for one would leave the line dead
    lc = launcher(tmp_path)
    p = Part('Call', py('import sys; sys.exit(1)'), ('line', 'never'), 0.3)
    p.start(lc.folder, {})
    p.proc.wait()
    lc.parts.append(p)
    lc.in_call = True
    with redirect_stdout(io.StringIO()):
        lc.check_parts(backoff_s=0)
    assert p.restarts == 1 and not lc.in_call


def test_a_crashed_logger_comes_back_in_a_new_folder(tmp_path):
    # buslog never appends to an existing session folder: a restart into the same one dies at once
    lc = launcher(tmp_path)
    p = lc.plan()[0]
    p.cmd[0] = sys.executable
    p.start(lc.folder, lc.env)
    try:
        assert p.wait_ready()
        p.proc.kill()
        p.proc.wait()
        lc.parts.append(p)
        with redirect_stdout(io.StringIO()):
            lc.check_parts(backoff_s=0)
        time.sleep(1.5)
        assert p.alive() and p.restarts == 1 and (lc.folder / 'bus-r1').is_dir(), p.tail()
    finally:
        p.stop(1)


def test_the_call_line_is_up_before_the_calibration(tmp_path, monkeypatch):
    # a dead tunnel or Twilio must fail before Austin mouths for 60 s, not after
    lc = launcher(tmp_path)
    order = []
    monkeypatch.setattr(lc, 'start', lambda part: order.append(part.name))
    monkeypatch.setattr(lc, 'launch', lambda part: order.append(f'{part.name} starts'))
    monkeypatch.setattr(lc, 'ready', lambda part: order.append(f'{part.name} ready'))
    monkeypatch.setattr(lc, 'start_call', lambda: order.append('Call'))
    monkeypatch.setattr(lc, 'ready_banner', lambda: None)
    monkeypatch.setattr(lc, 'watch', lambda: None)
    with redirect_stdout(io.StringIO()):
        lc.run()
    # the camera warms up while Claude and the call side start; the decoder waits for it
    assert order == ['Logger', 'EMG', 'Camera starts', 'Claude', 'Call', 'Camera ready', 'Decoder', 'Dashboard']


@pytest.mark.parametrize('mode', ['call', 'sms'])
def test_the_mode_reaches_the_call_server_the_dashboard_and_twilio(tmp_path, monkeypatch, mode):
    lc = launcher(tmp_path, mode=mode, no_webhook=False)
    lc.keys = {'TWILIO_ACCOUNT_SID': 'AC1', 'TWILIO_AUTH_TOKEN': 't', 'TWILIO_NUMBER': '+16288886598'}
    parts, hooks = {}, []

    def start(part):
        parts[part.name] = part
        if part.name == 'Tunnel':
            part.output = lambda: '|  https://a-b.trycloudflare.com  |'
    monkeypatch.setattr(lc, 'start', start)
    monkeypatch.setattr(lc, 'launch', start)
    monkeypatch.setattr(lc, 'ready', lambda part: None)
    monkeypatch.setattr(lc, 'cloudflared', lambda: 'cloudflared')
    monkeypatch.setattr(launch, 'public_status', lambda url: 405)
    monkeypatch.setattr(launch.Twilio, 'point_at',
                        lambda self, voice, sms=None, sms_required=False: hooks.append((voice, sms, sms_required)))
    with redirect_stdout(io.StringIO()):
        lc.start_call()
    assert parts['Call'].env['CALL_CHANNEL'] == mode
    assert hooks == [('https://a-b.trycloudflare.com/twiml', 'https://a-b.trycloudflare.com/sms', mode == 'sms')]
    assert parts['Call'].env['CALL_CONTROL_PORT'] == str(lc.call_port + 1)
    for name in ('start_call', 'ready_banner', 'watch'):
        monkeypatch.setattr(lc, name, lambda: None)
    with redirect_stdout(io.StringIO()):
        lc.run()
    dash = parts['Dashboard'].cmd
    assert dash[dash.index('--channel') + 1] == mode
    assert dash[dash.index('--control') + 1] == f'http://127.0.0.1:{lc.call_port + 1}'


def test_a_restart_is_ready_only_from_its_own_output(tmp_path):
    p = Part('Decoder', py('print("fusion: running")'), ('line', r'fusion: running'), 0.5)
    p.start(tmp_path, {})
    assert p.wait_ready() or not p.alive()
    p.proc.wait()
    p.cmd = py('import time; print("loading"); time.sleep(5)')
    p.start(tmp_path, {})               # the log still holds the first run's ready line
    try:
        assert not p.wait_ready() and 'fusion: running' not in p.output() and 'loading' in p.output()
    finally:
        p.stop(1)


def test_tunnel_url_is_never_cloudflareds_own_api():
    text = ('ERR failed to request https://api.trycloudflare.com/tunnel\n'
            '|  https://berlin-closely-attachment-alt.trycloudflare.com  |')
    assert launch.TUNNEL_URL.search(text).group(0) == 'https://berlin-closely-attachment-alt.trycloudflare.com'


def test_run_part_runs_the_module_with_its_arguments(tmp_path):
    import subprocess
    mod = tmp_path / 'fake_part.py'
    mod.write_text('import sys\nprint("args", sys.argv[1:])\n')
    r = subprocess.run([sys.executable, '-m', 'demo.run_part', 'fake_part', '--port', '9'], capture_output=True,
                       text=True, cwd=launch.REPO, env={'PYTHONPATH': f'{tmp_path}{os.pathsep}{launch.REPO}'}, timeout=30)
    assert r.returncode == 0 and "args ['--port', '9']" in r.stdout, r.stderr


def test_shutdown_stops_a_part_started_by_the_call_thread_too(tmp_path):
    lc = launcher(tmp_path)
    with redirect_stdout(io.StringIO()):
        lc.shutdown()
    assert lc.shutting_down     # follow_call checks this under the lock before it adds the rehearsal


def test_decoder_restart_skips_calibration(tmp_path):
    lc = launcher(tmp_path)
    p = Part('Decoder', py('import sys; sys.exit(1)') + ['--calibrate', '60.0', '--repair-s', '0.3'], ('line', 'x'),
             0.3)
    p.start(lc.folder, {})
    p.proc.wait()
    lc.parts.append(p)
    with redirect_stdout(io.StringIO()):
        lc.check_parts(backoff_s=0)
    assert '--calibrate' not in p.cmd and '60.0' not in p.cmd and p.cmd[-2:] == ['--repair-s', '0.3']


def test_plan_follows_the_flags(tmp_path):
    names = lambda lc: [p.name for p in lc.plan()]                                     # noqa: E731
    full = launcher(tmp_path)
    assert names(full) == ['Logger', 'EMG', 'Camera', 'Claude', 'Decoder']
    dec = full.plan()[-1].cmd
    assert '--calibrate' in dec and '--no-emg' not in dec and full.plan()[1].cmd[-1] == 'COM9'
    no_emg = launcher(tmp_path, no_emg=True)
    assert names(no_emg) == ['Logger', 'Camera', 'Claude', 'Decoder'] and '--no-emg' in no_emg.plan()[-1].cmd
    rehearse = launcher(tmp_path, rehearse=True, no_claude=True)
    assert names(rehearse) == ['Logger', 'Decoder'] and {'--no-emg', '--no-claude'} <= set(rehearse.plan()[-1].cmd)


def test_record_inbound_is_always_off(tmp_path, monkeypatch):
    monkeypatch.setenv('RECORD_INBOUND', 'inbound.ulaw')
    assert launcher(tmp_path).env['RECORD_INBOUND'] == ''


class FakeTwilio:
    def __init__(self, back=None, sms_back=None):
        self.calls, self.url, self.back, self.sms_back, self.sent = [], None, back, sms_back, {}

    def __call__(self, req, timeout=None):
        self.calls.append((req.get_method(), req.full_url, req.data, req.headers.get('Authorization')))
        if req.data:
            self.sent.update(urllib.parse.parse_qsl(req.data.decode()))     # voice and SMS go in separate POSTs
            self.url = self.sent.get('VoiceUrl')
        body = ({'incoming_phone_numbers': [{'sid': 'PN1'}]} if 'PhoneNumber=' in req.full_url
                else {'voice_url': self.back or self.url, 'sms_url': self.sms_back or self.sent.get('SmsUrl')})
        return io.BytesIO(json.dumps(body).encode())


def test_twilio_points_the_number_and_reads_it_back():
    fake = FakeTwilio()
    base = 'https://a-b.trycloudflare.com'
    Twilio('AC1', 'secret-token', '+16288886598', opener=fake).point_at(base + '/twiml', base + '/sms')
    methods = [m for m, *_ in fake.calls]
    assert methods == ['GET', 'POST', 'GET', 'POST', 'GET'] and all(a.startswith('Basic ') for *_, a in fake.calls)
    assert fake.sent == {'VoiceUrl': base + '/twiml', 'VoiceMethod': 'POST', 'SmsUrl': base + '/sms',
                         'SmsMethod': 'POST'}
    # call mode: the SMS webhook is set too (the dashboard can switch), but a failure there only warns
    warn = Twilio('AC1', 'secret-token', '+16288886598', opener=FakeTwilio(sms_back='https://old/sms')).point_at(
        base + '/twiml', base + '/sms')
    assert 'SMS webhook not set' in warn
    with pytest.raises(RuntimeError, match='still has'):
        Twilio('AC1', 'secret-token', '+16288886598', opener=FakeTwilio(back='https://old/twiml')).point_at(
            'https://new.trycloudflare.com/twiml', 'https://new.trycloudflare.com/sms')
    with pytest.raises(RuntimeError, match='sms_url'):             # SMS mode needs it: the start fails
        Twilio('AC1', 'secret-token', '+16288886598', opener=FakeTwilio(sms_back='https://old/sms')).point_at(
            'https://new.trycloudflare.com/twiml', 'https://new.trycloudflare.com/sms', sms_required=True)


def test_preflight_never_prints_a_key(tmp_path, monkeypatch):
    secrets = {k: f'SECRET-{k}-9f8e7d' for k in launch.NEEDED_KEYS}
    lc = Launcher(args(rehearse=True), tmp_path, secrets)
    monkeypatch.setattr(launch, 'gpu_ok', lambda: True)
    out = io.StringIO()
    with redirect_stdout(out):
        lc.preflight()
    assert 'SECRET-' not in out.getvalue() and 'keys in call/keys.env' in out.getvalue()


def test_preflight_names_a_busy_port(tmp_path):
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    s.listen()
    busy = s.getsockname()[1]
    try:
        lc = Launcher(args(rehearse=True, dash_port=busy), tmp_path, {})
        out = io.StringIO()
        with redirect_stdout(out):
            assert not lc.preflight()
        assert f'[{busy}]' in out.getvalue() or str(busy) in out.getvalue()
    finally:
        s.close()


def test_no_repair_window_by_default():
    # Claude's repairs never change the voice, so a wait before speaking only adds delay to every word
    import argparse
    seen = {}

    def grab(self, argv=None, namespace=None):
        seen['ns'] = real(self, argv, namespace)
        raise SystemExit(0)
    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            launch.main([])
    finally:
        argparse.ArgumentParser.parse_args = real
    assert seen['ns'].repair_s == 0.0
