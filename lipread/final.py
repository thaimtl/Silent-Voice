"""Final pass (E6): when a segment closes ('end' or the 8 s 'cap'), Auto-AVSR's beam search with the subword LM
(beam 5, LM weight 0.3 as Lipflow) re-reads the whole segment in the background -> cam.final with the 5 best.

    final = FinalPass(model, pub.send)          # model: the LiveModel the tick loop uses; ~25 s start (process)
    final.submit(seg.seg_id, rois, enc=r.enc)   # returns at once; enc: the last tick's encoder output (no re-encode)
    ...
    final.close(timeout=3.0)                    # finish what is queued within the timeout, drop the rest

One worker thread hands the segments over one after another to a spawned process (RemoteBeam): the beam runs there on
the CPU (4 torch threads, lower priority) with a LipReader whose encoder is deleted (decoder + CTC head + LM); only the
encoder output crosses. No VRAM, no second CUDA context, and no GIL shared with the tick loop. A beam in a thread of
this process was measured and dropped (CHANGELOG E6, 5090, noisy shared box): it took the tick (LiveModel.read) from
p50 15 / p95 27 ms to 52 / 69 ms (GIL: the tick is launch-bound Python); in the CPU process, 13 / 24 -> 18 / 32.
Beam (the search itself) still runs in-process for the process child and as the GPU reference in the tests.
Beam + LM on the CPU process: about 0.5 / 0.8 / 1.5 s for 2 / 4 / 8 s segments (GPU thread alone 0.35 / 0.65 / 1.35).

The child ignores Ctrl+C (the parent sends it the last segment, then stops it), and a child that dies (OOM, crash)
is started again on the next segment.

Queue policy: one beam running, at most max_pending (2) waiting. A segment closing while that is full drops the
OLDEST waiting one (logged, seg_id in .dropped): the newest segment is what is on screen, an older one already has its
live text from cam.tick. 2 waiting covers a capped 8 s segment, its continuation and one more before anything drops.

Scores: the beam's total weighted log-prob (0.9 decoder + 0.1 CTC + 0.3 LM), higher = better, best first, one entry
per distinct text. A segment with no hypothesis sends (('', score),).
"""
import collections
import logging
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import torch

import lipread  # noqa: F401  (puts Lipflow on sys.path)
from bus import CamFinal, bus_text, now

log = logging.getLogger('lipread.final')
BEAM, NBEST = 5, 5
LM_WEIGHT, CTC_WEIGHT = 0.3, 0.1
MAX_PENDING = 2


class Beam:
    """Beam search + subword LM over a whole segment: __call__(rois or enc) -> ((text, score), ...), best first.

    model_or_reader: a LiveModel (its decoder, CTC head and encoder are shared; the LM is loaded here) or a LipReader
    (its LM reused if it has one; the beam is rebuilt with beam_size). Set .encode = None on a reader whose encoder
    was deleted (the process worker): then only enc is accepted."""

    def __init__(self, model_or_reader, beam_size=BEAM, nbest=NBEST, lm_weight=LM_WEIGHT, ctc_weight=CTC_WEIGHT):
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.scorers.length_bonus import LengthBonus
        m = model_or_reader
        self.reader = getattr(m, 'reader', m)
        self.encode = m.encode
        self.nbest = nbest
        r = self.reader
        odim = len(r.token_list)
        lm = r.beam.full_scorers.get('lm') if lm_weight > 0 else None
        if lm is None and lm_weight > 0:
            lm = load_lm(odim).to(r.device)
        scorers = r.model.scorers()
        scorers.update(lm=lm, length_bonus=LengthBonus(odim))
        self.beam = BatchBeamSearch(
            beam_size=beam_size, vocab_size=odim, scorers=scorers, sos=odim - 1, eos=odim - 1,
            weights=dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=lm_weight if lm is not None else 0.0,
                         length_bonus=0.0),
            token_list=r.token_list, pre_beam_score_key='decoder').to(r.device).eval()
        self.has_lm = lm is not None

    @torch.inference_mode()
    def __call__(self, rois=None, enc=None):
        if enc is None:
            if rois is None or len(rois) == 0:
                return (('', 0.0),)
            enc = self.encode(rois)
        elif len(enc) == 0:
            return (('', 0.0),)
        enc = torch.as_tensor(enc).to(self.reader.device, torch.float32)
        for module in self.beam.full_scorers['decoder'].modules():
            if hasattr(module, '_mem_kv'):
                del module._mem_kv
        out, seen = [], set()
        for h in self.beam(enc):
            text = bus_text(''.join(self.reader.token_list[int(i)] for i in h.yseq[1:]))
            if text not in seen:
                seen.add(text)
                out.append((text, float(h.score)))
            if len(out) == self.nbest:
                break
        return tuple(out) or (('', 0.0),)


def load_lm(odim):
    """Auto-AVSR's subword RNN LM, as LipReader loads it (CPU, eval)."""
    from espnet.asr.asr_utils import get_model_conf, torch_load
    from espnet.nets.lm_interface import dynamic_import_lm
    path = os.path.join(lipread.LIPFLOW_MODELS, 'lm', 'model.pth')
    args = get_model_conf(path, os.path.join(lipread.LIPFLOW_MODELS, 'lm', 'model.json'))
    lm = dynamic_import_lm(getattr(args, 'model_module', 'default'), args.backend)(odim, args)
    torch_load(path, lm)
    return lm.eval()


def lower_priority(nice):
    """This process below the tick loop: nice on POSIX, BELOW_NORMAL_PRIORITY_CLASS on Windows."""
    if not nice:
        return
    if hasattr(os, 'nice'):
        os.nice(nice)
    elif sys.platform == 'win32':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)


def load_ctc_head(model, path):
    """The CTC head's tensors (ctc.*) of a face adapter trained with --train-ctc, over model's; any other tensor in
    the file is ignored (the encoder is not in this process). Returns the tensor count."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    ctc = {k: v for k, v in state.items() if k.startswith('ctc.')}
    if ctc:
        model.load_state_dict(ctc, strict=False)
    return len(ctc)


def _child(jobs, results, kw, device, threads, nice):
    """RemoteBeam's process: a LipReader without its encoder (decoder + CTC head + LM, on device) beams the encoder
    outputs it is sent. jobs: enc (T, D) float32, None stops -> results (nbest or error text, beam_ms).
    kw['face_adapter']: the adapter the encoder runs, whose CTC head (if trained) the beam must use."""
    kw = dict(kw)
    adapter = kw.pop('face_adapter', None)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(threads)
    lower_priority(nice)
    from lipflow.vsr import LipReader
    reader = LipReader(device=device, use_lm=kw['lm_weight'] > 0, lm_weight=kw['lm_weight'],
                       ctc_weight=kw['ctc_weight'], personal=False)
    if adapter:
        load_ctc_head(reader.model, adapter)
    del reader.model.encoder
    if reader.device.type == 'cuda':
        torch.cuda.empty_cache()
    beam = Beam(reader, **kw)
    beam.encode = None
    beam(enc=np.zeros((25, reader.model.adim), np.float32))
    cuda = reader.device.type == 'cuda'
    results.put(('ready', torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else 0.0))
    while (enc := jobs.get()) is not None:
        t0 = time.perf_counter()
        try:
            out = beam(enc=enc)
        except Exception as e:  # noqa: BLE001
            out = f'{type(e).__name__}: {e}'
        results.put((out, (time.perf_counter() - t0) * 1000))


class RemoteBeam:
    """Beam's interface, the search run in a spawned process (no GIL shared with the tick loop). Only the encoder output
    crosses (T x 768 float32, 0.6 MB at 8 s); rois without enc are encoded here first with model (the shared LiveModel).
    device 'cpu' (default): no VRAM and no CUDA context; measured about as fast as the GPU at threads=4.
    target: the child's function (tests swap in a fake one)."""

    def __init__(self, model=None, beam_size=BEAM, nbest=NBEST, lm_weight=LM_WEIGHT, ctc_weight=CTC_WEIGHT,
                 device='cpu', threads=4, nice=10, start_timeout=120, target=_child, face_adapter=None):
        self.model = model
        self.ctx = mp.get_context('spawn')
        kw = dict(beam_size=beam_size, nbest=nbest, lm_weight=lm_weight, ctc_weight=ctc_weight,
                  face_adapter=face_adapter)
        self.args = (kw, device, threads, nice)
        self.device, self.threads, self.start_timeout, self.target = device, threads, start_timeout, target
        self.proc = None
        self.restarts = 0
        self._start()

    def _start(self):
        """Spawn the child and wait for 'ready'; fails fast if it dies while loading."""
        self.jobs, self.results = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=self.target, args=(self.jobs, self.results, *self.args), daemon=True,
                                     name='final-beam')
        self.proc.start()
        t_end = now() + self.start_timeout
        while True:
            try:
                ready, self.child_mib = self.results.get(timeout=1.0)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f'final-pass process died while starting (exit code {self.proc.exitcode})') \
                        from None
                if now() > t_end:
                    self._stop(0.0)
                    raise RuntimeError(f'final-pass process not ready after {self.start_timeout} s') from None
        if ready != 'ready':
            raise RuntimeError('final-pass process did not start')
        log.info('final-pass process ready (%s, %d threads, GPU memory %.0f MiB)', self.device, self.threads,
                 self.child_mib)

    def __call__(self, rois=None, enc=None):
        if enc is None:
            if rois is None or len(rois) == 0:
                return (('', 0.0),)
            if self.model is None:
                raise ValueError('RemoteBeam without a model needs enc')
            enc = self.model.encode(rois)
        if isinstance(enc, torch.Tensor):
            enc = enc.float().cpu().numpy()
        if not self.proc.is_alive():
            log.warning('final-pass process had died (exit code %s): starting it again', self.proc.exitcode)
            self._stop(0.0)
            self.restarts += 1
            self._start()
        self.jobs.put(enc)
        while True:
            try:
                out, _ = self.results.get(timeout=1.0)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f'final-pass process died (exit code {self.proc.exitcode})') from None
        if isinstance(out, str):
            raise RuntimeError(out)
        return out

    def _stop(self, timeout):
        """Stop the child without ever blocking on a queue it no longer reads."""
        self.jobs.cancel_join_thread()
        if self.proc.is_alive():
            try:
                self.jobs.put_nowait(None)
            except (OSError, ValueError, queue.Full):
                pass
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1.0)

    def close(self, timeout=2.0):
        self._stop(timeout)


class FinalPass:
    """Background beam + LM over each closed segment -> publish(CamFinal). submit() never blocks.

    model: the LiveModel that encodes a segment whose last read can't be reused (None: until set on .beam.model).
    beam: a callable (rois, enc) -> ((text, score), ...) instead of the RemoteBeam (tests: a fake slow beam).
    stats: per published segment dict(seg_id, T, beam_ms, queued_ms); .dropped: seg_ids dropped by the queue policy;
    .errors: seg_ids whose beam raised (logged, nothing sent)."""

    def __init__(self, model, publish, beam_size=BEAM, nbest=NBEST, lm_weight=LM_WEIGHT, ctc_weight=CTC_WEIGHT,
                 max_pending=MAX_PENDING, device='cpu', threads=4, beam=None, face_adapter=None):
        if beam is None:
            beam = RemoteBeam(model, beam_size=beam_size, nbest=nbest, lm_weight=lm_weight, ctc_weight=ctc_weight,
                              device=device, threads=threads, face_adapter=face_adapter)
        self.beam, self.publish, self.max_pending = beam, publish, max_pending
        self.pending = collections.deque()
        self.cond = threading.Condition()
        self.busy = False
        self.closing = False
        self.stats, self.dropped, self.errors = [], [], []
        self.thread = threading.Thread(target=self._run, name='final-pass', daemon=True)
        self.thread.start()

    def submit(self, seg_id, rois, enc=None):
        """Queue a closed segment: rois (T, 96, 96) uint8; enc: its (T, D) encoder output if the tick has it (the
        final then skips the re-encode). Returns at once; False if closed."""
        with self.cond:
            if self.closing:
                return False
            if len(self.pending) >= self.max_pending:
                old = self.pending.popleft()[0]
                self.dropped.append(old)
                log.warning('final pass busy: dropped segment %d (%d dropped so far)', old, len(self.dropped))
            self.pending.append((seg_id, rois, enc, time.perf_counter()))
            self.cond.notify()
        return True

    def wait_idle(self, timeout=None):
        """Block until nothing is queued or running; True if idle (tests, benchmarks)."""
        with self.cond:
            return self.cond.wait_for(lambda: not self.pending and not self.busy, timeout)

    def _run(self):
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.pending or self.closing)
                if not self.pending:
                    return
                seg_id, rois, enc, t_sub = self.pending.popleft()
                self.busy = True
            t0 = time.perf_counter()
            try:
                nbest = self.beam(rois, enc)
                self.publish(CamFinal(seg_id, tuple(nbest)))
                self.stats.append(dict(seg_id=seg_id, T=len(rois) if rois is not None else len(enc),
                                       beam_ms=(time.perf_counter() - t0) * 1000, queued_ms=(t0 - t_sub) * 1000))
            except Exception:  # noqa: BLE001
                self.errors.append(seg_id)
                log.exception('final pass failed on segment %d', seg_id)
            with self.cond:
                self.busy = False
                self.cond.notify_all()

    def close(self, timeout=3.0):
        """Stop taking segments, finish what is queued within timeout, drop the rest (counted in .dropped). Returns
        True if the worker stopped in time (a beam already running cannot be interrupted)."""
        t_end = now() + timeout
        with self.cond:
            self.closing = True
            self.cond.notify_all()
        self.wait_idle(max(0.0, t_end - now()))
        with self.cond:
            while self.pending:
                self.dropped.append(self.pending.popleft()[0])
                log.warning('final pass closed: dropped segment %d', self.dropped[-1])
            self.cond.notify_all()
        self.thread.join(max(0.0, t_end - now()))
        done = not self.thread.is_alive()
        if isinstance(self.beam, RemoteBeam):
            self.beam.close(0.5 if done else 0.0)
        if not done:
            log.warning('final pass still running a beam after %.1f s', timeout)
        return done
