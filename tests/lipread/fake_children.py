"""Stand-ins for RemoteBeam's child process, importable by a spawned process without torch or weights."""
import os
import signal
import sys


def dies_on_start(jobs, results, *args):
    sys.exit(3)


def echo(jobs, results, *args):
    """Answers every job; a job of length 0 kills the process, as an out-of-memory kill would."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    results.put(('ready', 0.0))
    while (enc := jobs.get()) is not None:
        if len(enc) == 0:
            os._exit(9)
        results.put(((('HI', -1.0),), 1.0))


def imports(results, names):
    """Imports what a spawned child would, then reports; any failure reaches the parent as its text."""
    import importlib
    try:
        for name in names:
            importlib.import_module(name)
        results.put('ok')
    except BaseException as e:  # noqa: BLE001
        results.put(f'{type(e).__name__}: {e}')
