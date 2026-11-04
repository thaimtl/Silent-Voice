"""Runs tests/smoke_claude_service.py (its offline checks) under pytest, so the Claude service is tested with the rest."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_claude_service_smoke():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'smoke_claude_service.py')], cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, [line for line in r.stdout.splitlines() if line.startswith('FAIL')] or r.stderr[-2000:]


def test_keys_saved_with_a_bom_read_as_the_launcher_reads_them(tmp_path, monkeypatch):
    """Windows Notepad can save keys.env as UTF-8 with a BOM; the launcher's pre-flight accepts it, so the service must."""
    from claude_service.service import load_keys
    from demo.launch import read_keys
    (tmp_path / 'keys.env').write_bytes('\ufeffANTHROPIC_API_KEY=sk-test\n# Frank\u014d\n'.encode('utf-8'))
    monkeypatch.setenv('ANTHROPIC_API_KEY', '')
    assert load_keys(str(tmp_path)) == 'keys.env'
    assert os.environ['ANTHROPIC_API_KEY'] == read_keys(tmp_path / 'keys.env')['ANTHROPIC_API_KEY'] == 'sk-test'
