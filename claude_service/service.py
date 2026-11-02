"""The Claude service on the bus: reads partner.text, dec.lock, dec.hyp and dec.pause, publishes claude.predict,
claude.repair, claude.end and timing.claude (with every request and reply, for the logger).

One conversation-lane call at a time, so the conversation stays append-only (whatever arrives meanwhile goes into
the next one), and a repair-lane call for each dec.lock at once, up to brain.MAX_REPAIRS in flight (brain.py).
A live prediction is always on its way (brain.py, live lane): on every new word of the judge's live line (Cartesia
Ink), and otherwise as soon as the last one is back, up to brain.MAX_LIVE in flight; the final line still goes
through the conversation lane.
Each dec.pause (Austin's lips stopped; the decoder holds his sentence) starts an end-lane call, up to
brain.MAX_END in flight, whose claude.end says how likely he has finished.
While Austin mouths and nothing changed, the last claude.predict goes out again with a fresh ttl_s, without a call.
The bus loop never waits on Claude, and nothing that goes wrong (a malformed
message, a failed call, a reply that breaks apply) stops it: the error goes out on timing.claude and the
captions go on without Claude.

    python -m claude_service.service      # connects to 'dec' and 'call', binds 'claude'

The key: ANTHROPIC_API_KEY from the environment, else from call/keys.env (or call/.env), the git-ignored file that
already holds the call's keys, so every key lives in one place. Never in the repo or the logs.
"""
import argparse
import signal
import sys
import threading

import bus
from bus.team import Pub, Sub
from claude_service.brain import MAX_CAPTION, MAX_END, MAX_LIVE, MAX_REPAIRS, Brain

TOPICS = ("dec.sentence", "partner.text", "dec.lock", "dec.hyp", "dec.pause")
MOUTHING_S = 0.5  # dec.hyp showed unlocked words this recently


KEY_FILES = ("keys.env", ".env")  # in call/, as call/config.py reads them


def load_keys(call_dir=None):
    """ANTHROPIC_API_KEY from call/keys.env (or call/.env) unless the environment already has it. Returns where it came
    from ('environment', the file's name, or None), never the key."""
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "environment"
    call_dir = call_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "call")
    for name in KEY_FILES:
        path = os.path.join(call_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == "ANTHROPIC_API_KEY" and value.strip().strip("'\""):
                    os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")
                    return name
    return None


def report(pub, error):
    pub.send("timing.claude", {"stage": "claude", "error": error})
    print(f"claude: {error}", file=sys.stderr)


class Call:
    """brain.ask(messages) on a daemon thread, so quitting never waits for an unanswered call."""

    def __init__(self, brain, target, messages=None):
        self.target, self.result, self.error = target, None, None
        self.thread = threading.Thread(target=self._work, args=(brain, messages, target.get("lane")), daemon=True)
        self.thread.start()

    def _work(self, brain, messages, lane):
        try:
            self.result = brain.ask(messages) if messages is None else brain.ask(messages, lane=lane)
        except Exception as e:  # handed to the bus loop, which reports it
            self.error = e

    def done(self):
        return not self.thread.is_alive()

    def get(self):
        if self.error is not None:
            raise self.error
        return self.result


def finish(brain, call, pub, now):
    try:
        try:
            reply, log = call.get()
        except Exception as e:  # ask() itself broke: a failed call like any other
            reply, log = None, {"stage": "claude", "error": f"{type(e).__name__}: {e}"}
        if reply is None:
            log["kept"] = brain.failed(call.target, now, refused=log.get("refused", False))
        else:
            brain.succeeded()
            out, rejected = brain.apply(reply, call.target, now)
            log["rejected_edits"] = rejected
            for topic, msg in out:
                pub.send(topic, msg)
        pub.send("timing.claude", log)
        if "error" in log:
            print(f"claude: {log['error']}", file=sys.stderr)
    except Exception as e:
        report(pub, f"reply failed: {type(e).__name__}: {e}")


def run(brain, sub, pub, stop=lambda: False):
    """sub: a bus.Sub on TOPICS; pub: a bus.Pub. Runs until stop() is true."""
    convo, live, repairs, ends, captions, last_mouth = None, [], [], [], [], -1e9  # one conversation call, MAX_LIVE live,
    # MAX_REPAIRS repairs, MAX_END end checks
    while not stop():
        timeout = 20
        while True:  # wait up to 20 ms for the first message, then drain the rest
            try:
                topic, msg = sub.recv(timeout_ms=timeout)
                if topic is None:
                    break
                if topic == "partner.text":
                    brain.partner(msg)
                elif topic == "dec.lock":
                    if bad := brain.locked(msg):
                        report(pub, f"bad message: {bad} malformed word(s) in dec.lock {msg['utt_id']}")
                elif topic == "dec.hyp":
                    brain.hyp(msg)
                    if msg.get("unlocked"):
                        last_mouth = bus.now()
                elif topic == "dec.pause":
                    brain.pause(msg)
                elif topic == "dec.sentence":
                    brain.sentence(msg)
            except Exception as e:  # one bad message must not stop the service
                report(pub, f"bad message: {type(e).__name__}: {e}")
            timeout = 0
        now = bus.now()
        mouthing = now - last_mouth < MOUTHING_S
        for call in [c for c in repairs if c.done()]:
            finish(brain, call, pub, now)
            repairs.remove(call)
        if convo is not None and convo.done():
            finish(brain, convo, pub, now)
            convo = None
        for call in [c for c in live if c.done()]:
            finish(brain, call, pub, now)
            live.remove(call)
        for call in [c for c in ends if c.done()]:
            finish(brain, call, pub, now)
            ends.remove(call)
        for call in [c for c in captions if c.done()]:
            finish(brain, call, pub, now)
            captions.remove(call)
        if len(captions) < MAX_CAPTION:   # a sentence closed: Opus rewrites its caption (never the voice)
            try:
                if (req := brain.next_caption(now)) is not None:
                    captions.append(Call(brain, req[1], req[0]))
            except Exception as e:
                report(pub, f"caption request failed: {type(e).__name__}: {e}")
                brain.captions = []
        if len(ends) < MAX_END:     # Austin's lips stopped: has he finished? (the decoder holds his sentence)
            try:
                if (req := brain.next_end(now)) is not None:
                    ends.append(Call(brain, req[1], req[0]))
            except Exception as e:
                report(pub, f"end check failed: {type(e).__name__}: {e}")
                brain.end = None
        if len(live) < MAX_LIVE:   # a fresh prediction always on its way
            try:
                if (req := brain.next_live(now, len(live))) is not None:
                    live.append(Call(brain, req[1], req[0]))
            except Exception as e:
                report(pub, f"live request failed: {type(e).__name__}: {e}")
        while brain.review and len(repairs) < MAX_REPAIRS and now >= brain.retry_at:
            try:
                messages, target = brain.next_review(now)
                repairs.append(Call(brain, target, messages))
            except Exception as e:
                report(pub, f"repair request failed: {type(e).__name__}: {e}")
                brain.discard("repair")  # drop what broke it, or it fails again every loop
        if convo is None and brain.due(now, mouthing):
            try:
                convo = Call(brain, brain.next_request(now))
            except Exception as e:
                report(pub, f"request failed: {type(e).__name__}: {e}")
                brain.discard("conversation")  # drop what broke it
        if (again := brain.refresh(now, mouthing)) is not None:
            pub.send("claude.predict", again)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", action="append", default=[],
                    help="endpoint name or address to connect to, repeatable (default: dec and call)")
    ap.add_argument("--pub", default="claude", help="endpoint name or address to bind for our topics")
    args = ap.parse_args()
    source = load_keys()
    if source is None:
        sys.exit("no ANTHROPIC_API_KEY: put ANTHROPIC_API_KEY=... in call/keys.env (git-ignored) or the environment")
    print(f"claude: API key from {source}", file=sys.stderr)
    import anthropic  # here, so the tests run without the SDK
    client = anthropic.Anthropic(timeout=10.0, max_retries=1)  # key from ANTHROPIC_API_KEY, never from the repo
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # stop the same way as on Ctrl-C
    sub, pub = Sub(args.sub or ["dec", "call"], TOPICS), Pub(args.pub)
    try:
        run(Brain(client), sub, pub)
    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()  # the default linger lets the last timing.claude out


if __name__ == "__main__":
    main()
