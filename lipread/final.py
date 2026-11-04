"""Final pass (E6): when a segment closes ('end' or the 8 s 'cap'), the whole segment is decoded once more with the
attention decoder and the subword LM -> cam.final with the 5 best. Two ways, chosen with lipread.run --final:

gpu (default): GpuFinal rescores on the tick thread, right after the closing read, on the encoder output that read
already has (Rescore): hypotheses from a CTC prefix beam over the live model's CTC top-k and from a batched
attention-decoder beam, each scored by exact CTC, the attention decoder and the LM in one batched pass, combined
0.9 attention + 0.1 CTC + 0.3 LM like the beam. It uses the live model's CTC head and decoder, so a face adapter that
trains the CTC head (recipe C) is honoured. cam.final goes out with the closing tick instead of 0.5 to 1.5 s later.
Measured (5090, data/avsr_experiments/rt/final_compare.py): WER of the best 0.175 vs the CPU beam's 0.191 on session 2
(adapter s1only), ~50 ms per segment. A worker thread instead took the tick read from p50 12 to 51 ms (GIL).

    final = GpuFinal(model, pub.send)           # model: the LiveModel the tick loop uses (loads the LM on its GPU)
    final.submit(seg.seg_id, rois, enc=r.enc)   # rescores and publishes before it returns
    final.close()

cpu: Auto-AVSR's beam search with the subword LM (beam 5, LM weight 0.3 as Lipflow) in the background:

    final = FinalPass(model, pub.send, face_adapter=path)   # ~25 s start (process)
    final.submit(seg.seg_id, rois, enc=r.enc)   # returns at once; enc: the last tick's encoder output (no re-encode)
    ...
    final.close(timeout=3.0)                    # finish what is queued within the timeout, drop the rest

One worker thread hands the segments over one after another to a spawned process (RemoteBeam): the beam runs there on
the CPU (4 torch threads, lower priority) with a LipReader whose encoder is deleted (decoder + CTC head + LM, the face
adapter's CTC head loaded over the base one); only the encoder output crosses. No VRAM, no second CUDA context, and
no GIL shared with the tick loop. A beam in a thread of
this process was measured and dropped (CHANGELOG E6, 5090, noisy shared box): it took the tick (LiveModel.read) from
p50 15 / p95 27 ms to 52 / 69 ms (GIL: the tick is launch-bound Python); in the CPU process, 13 / 24 -> 18 / 32.
Beam (the search itself) still runs in-process for the process child and as the GPU reference in the tests.
Beam + LM on the CPU process: about 0.5 / 0.8 / 1.5 s for 2 / 4 / 8 s segments (GPU thread alone 0.35 / 0.65 / 1.35).

The child ignores Ctrl+C (the parent sends it the last segment, then stops it), and a child that dies (OOM, crash)
is started again on the next segment.

Queue policy: one beam running, at most max_pending (2) waiting. A segment closing while that is full drops the
OLDEST waiting one (logged, seg_id in .dropped): the newest segment is what is on screen, an older one already has its
live text from cam.tick. 2 waiting covers a capped 8 s segment, its continuation and one more before anything drops.

Scores (both ways): the total weighted log-prob (0.9 decoder + 0.1 CTC + 0.3 LM), higher = better, best first, one
entry per distinct text. A segment with no hypothesis sends (('', score),).
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
from lipread.rescore import (LOG0, attention_beam, attention_scores_multi, combined, ctc_prefix_beam, ctc_scores,
                             lm_scores)

log = logging.getLogger('lipread.final')
BEAM, NBEST = 5, 5
LM_WEIGHT, CTC_WEIGHT = 0.3, 0.1
MAX_PENDING = 2
PREFIX_BEAM, PER_FRAME, ATT_BEAM = 8, 8, 5


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


def _child(jobs, results, kw, device, threads, nice, face_adapter=None):
    """RemoteBeam's process: a LipReader without its encoder (decoder + CTC head + LM, on device; the face adapter's
    CTC head if it has one) beams the encoder outputs it is sent. jobs: enc (T, D) float32, None stops -> results
    (nbest or error text, beam_ms)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, 'SIGBREAK'):     # Windows: the launcher's CTRL_BREAK reaches the whole group; the parent stops us
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    torch.set_num_threads(threads)
    lower_priority(nice)
    from lipflow.vsr import LipReader
    reader = LipReader(device=device, use_lm=kw['lm_weight'] > 0, lm_weight=kw['lm_weight'],
                       ctc_weight=kw['ctc_weight'], personal=False)
    if face_adapter:
        from lipread.model import load_face_adapter
        load_face_adapter(reader.model, face_adapter)
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
    face_adapter: the live model's adapter file, whose CTC head (recipe C, --train-ctc) the child loads.
    target: the child's function (tests swap in a fake one)."""

    def __init__(self, model=None, beam_size=BEAM, nbest=NBEST, lm_weight=LM_WEIGHT, ctc_weight=CTC_WEIGHT,
                 device='cpu', threads=4, nice=10, start_timeout=120, target=_child, face_adapter=None):
        self.model = model
        self.ctx = mp.get_context('spawn')
        kw = dict(beam_size=beam_size, nbest=nbest, lm_weight=lm_weight, ctc_weight=ctc_weight)
        self.args = (kw, device, threads, nice, face_adapter)
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


class Rescore:
    """The final pass on the live model's GPU: (rois, enc) -> ((text, score), ...), best first, as Beam returns it.

    Hypotheses: a CTC prefix beam (beam, per_frame) over the CTC top-k of enc from the live model's CTC head, and a
    batched attention-decoder beam (att_beam, with search_lm x LM; lipread/rescore.py attention_beam). The LM in the
    search (search_lm 0.3) cost ~50 ms more per segment on the 5090 for no better WER, so it is off by default. Every
    hypothesis is scored by exact CTC, the attention decoder and the LM in one batched pass each, then combined as the
    beam weighs them: ctc_weight x CTC + (1 - ctc_weight) x attention + lm_weight x LM + bonus x tokens."""

    def __init__(self, model, beam=PREFIX_BEAM, per_frame=PER_FRAME, att_beam=ATT_BEAM, search_lm=0.0,
                 nbest=NBEST, lm_weight=LM_WEIGHT, ctc_weight=CTC_WEIGHT, bonus=0.0):
        self.model = model
        self.beam, self.per_frame, self.att_beam, self.nbest = beam, per_frame, att_beam, nbest
        self.search_lm = search_lm
        self.lm_weight, self.ctc_weight, self.bonus = lm_weight, ctc_weight, bonus
        self.eos = model.reader.model.eos
        dev = next(model.reader.model.decoder.parameters()).device
        self.lm = load_lm(len(model.token_list)).to(dev) if lm_weight or search_lm else None
        self(enc=torch.zeros((25, model.reader.model.adim), device=dev))

    @torch.inference_mode()
    def prefix_hypotheses(self, enc):
        m = self.model
        logp = m.ctc.log_softmax(enc.float().unsqueeze(0))[0]
        top = logp.topk(self.per_frame, dim=-1)
        out = torch.cat([top.values, logp[:, :1], top.indices.float()], dim=1).cpu().numpy()
        k = self.per_frame
        return [h for h, s in ctc_prefix_beam(out[:, k + 1:].astype(np.int64), out[:, :k], out[:, k], self.beam,
                                              self.per_frame) if s > LOG0 / 2]

    @torch.inference_mode()
    def hypotheses(self, enc):
        """enc (T, D) -> (token tuples, CTC, attention, LM or None): both beams' hypotheses and their scores."""
        m = self.model
        enc = torch.as_tensor(enc).to(m.device)
        hyps = self.prefix_hypotheses(enc) if self.beam else []
        if self.att_beam:
            hyps += attention_beam(m, enc, self.att_beam, self.lm, self.search_lm, 1.0 - self.ctc_weight)
        hyps = list(dict.fromkeys(hyps))
        ctc = ctc_scores(m, enc, hyps)
        ok = np.isfinite(ctc)
        hyps, ctc = [h for h, k in zip(hyps, ok) if k], ctc[ok]
        if not hyps:
            return [()], np.zeros(1), np.zeros(1), None if self.lm is None else np.zeros(1)
        att = attention_scores_multi(m, enc.unsqueeze(0), [hyps])[0]
        lm = lm_scores(self.lm, [hyps], self.eos)[0] if self.lm is not None else None
        return hyps, ctc, att, lm

    def __call__(self, rois=None, enc=None):
        if enc is None:
            if rois is None or len(rois) == 0:
                return (('', 0.0),)
            enc = self.model.encode(rois)
        elif len(enc) == 0:
            return (('', 0.0),)
        hyps, ctc, att, lm = self.hypotheses(enc)
        score = combined(hyps, ctc, att, lm, self.ctc_weight, self.lm_weight, self.bonus)
        out, seen = [], set()
        for j in np.argsort(-score, kind='stable'):
            text = bus_text(''.join(self.model.token_list[i] for i in hyps[j]))
            if text not in seen:
                seen.add(text)
                out.append((text, float(score[j])))
            if len(out) == self.nbest:
                break
        return tuple(out)


class GpuFinal:
    """FinalPass's interface, rescored on the caller's thread (the tick loop, right after the closing read): submit()
    publishes the segment's CamFinal before it returns. A worker thread would share the GIL with the launch-bound
    tick; one rescoring costs a few ms per segment instead.

    rescore: a callable (rois, enc) -> ((text, score), ...); default Rescore(model).
    stats: per published segment dict(seg_id, T, beam_ms, queued_ms=0); .dropped stays empty; .errors: seg_ids whose
    rescoring raised (logged, nothing sent)."""

    def __init__(self, model, publish, rescore=None, **kw):
        self.rescore = rescore if rescore is not None else Rescore(model, **kw)
        self.publish = publish
        self.closing = False
        self.stats, self.dropped, self.errors = [], [], []

    def submit(self, seg_id, rois, enc=None):
        if self.closing:
            return False
        t0 = time.perf_counter()
        try:
            nbest = self.rescore(rois, enc)
            self.publish(CamFinal(seg_id, tuple(nbest)))
            self.stats.append(dict(seg_id=seg_id, T=len(rois) if rois is not None else len(enc),
                                   beam_ms=(time.perf_counter() - t0) * 1000, queued_ms=0.0))
        except Exception:  # noqa: BLE001
            self.errors.append(seg_id)
            log.exception('final pass failed on segment %d', seg_id)
        return True

    def wait_idle(self, timeout=None):
        return True

    def close(self, timeout=3.0):
        self.closing = True
        return True
