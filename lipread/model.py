"""Auto-AVSR (Lipflow's LipReader weights) as the camera reader needs it: re-read a whole segment, CTC top-k per frame.

    model = LiveModel(half=False)          # loads LRS3_V_WER19.1 on CUDA, warms up
    r = model.read(rois)                   # rois (T, 96, 96) uint8 crops at 25 fps -> Reading
    r.topk_ids (T, 32) int32, r.topk_logp (T, 32) float32, r.blank_logp (T,), r.greedy 'UPPERCASE TEXT', r.enc_ms ...
    r = model.read(rois, keep_enc=True)    # also r.enc, the (T, D) encoder output on the GPU (the final pass reuses it)
    model = LiveModel(face_adapter=path)   # with Austin's face adapter on top (lipread/adapt.py)
    model = LiveModel(graphs=False)        # eager reads only (lipread.run --no-graphs)

The encoder (3D-conv/ResNet front end + 12 Conformer blocks) runs on the whole segment every call (r.frontend_ms +
r.conformer_ms = r.enc_ms, split with CUDA events); the CTC head
(log_softmax, blank = 0) and top-k run on the GPU, one device-to-host copy per call. half=True runs the encoder in
fp16 via .half() (measured faster than autocast, CHANGELOG T0); the CTC head stays fp32.

On CUDA the read is captured as CUDA graphs in place of the warm-up: an eager read launches ~860 small kernels and
waits on Python between them, so it is launch-bound and slows down badly beside any busy Python thread; a replay
launches them all at once. Two graphs per read, with exactly the shapes the eager read uses, so the result is the same
to the bit (tests/lipread/test_graphs.py, at every length, with and without the FrontendCache): the front end on the n
frames the cache has not finalised (uint8 crops in, normalised on the GPU as LipReader.to_tensor; one graph per n),
then embed + Conformer + after_norm + CTC head + top-k on the whole segment's features (one graph per T). One graph of
the whole read would not do: the cached front end is only equal to a whole-segment one up to rounding, which the 12
Conformer blocks grow to ~0.9 nats (fp16). Lengths outside the captured set read eagerly.
"""
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

import lipread  # noqa: F401  (puts Lipflow on sys.path)
from bus import bus_text
from bus import BLANK, TOPK


@dataclass(frozen=True)
class Reading:
    topk_ids: np.ndarray
    topk_logp: np.ndarray
    blank_logp: np.ndarray
    greedy: str
    enc_ms: float
    ctc_ms: float
    enc: torch.Tensor | None = None
    frontend_ms: float = 0.0
    conformer_ms: float = 0.0


@dataclass(frozen=True)
class Captured:
    """One captured piece of a read: replaying graph reads inp and overwrites out.

    Front end (LiveModel.fe_graphs[n]): inp (n, 96, 96) uint8, out (1, n, F) features. Conformer (LiveModel.graphs[T]):
    inp (1, T, F) features, out = (packed (T, 2 * TOPK + 1) float32 in the eager read's layout: top-k log-probabilities,
    blank, top-k ids as floats; enc (T, D)). inp is a view of one static buffer per kind (outside the graph pool); out
    lives in the pool all graphs share, so any later replay may overwrite it: copy or clone it before the next replay."""
    graph: object
    inp: torch.Tensor
    out: object


FE_CONTEXT = 2


class FrontendCache:
    """Front-end features of one segment, computed once per frame (E3 step 2).

    A frame's features are final once the 2 frames after it exist; the newest 2 are recomputed each read with the
    same zero padding a whole-segment encode uses, so the result matches an encode of the whole segment
    (tests/lipread/test_run.py). One cache per segment: a new segment (also after a cap) starts a new one, because
    its first frames are zero-padded on the left."""

    def __init__(self, seg_id=None):
        self.seg_id = seg_id
        self.feats = None
        self.n_final = 0


FACE_ADAPTER_PREFIXES = ('encoder.frontend.', 'encoder.embed.', 'encoder.encoders.0.')
ADAPTER_ALLOWED = ('encoder.', 'ctc.ctc_lo.')     # what an adapter may change (lipread/adapt.py --extra-blocks/--train-ctc)


def load_face_adapter(model, path):
    """Lipflow's face adapter (lipread/adapt.py, as lipflow/train_vsr.py): new values for the front end, the embed and
    Conformer block 0 (a wider adapter: more encoder blocks and the CTC head's projection), loaded over the base
    weights. Same shapes, so a read costs the same. Returns the tensor count; ValueError for a file with any other
    tensor (the decoder), which would silently change the rest of the model."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    other = sorted(k for k in state if not k.startswith(ADAPTER_ALLOWED))
    if other:
        raise ValueError(f'{path}: not a face adapter, {len(other)} other tensors (first: {other[0]})')
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise ValueError(f'{path}: tensors the model does not have: {res.unexpected_keys[:3]}')
    return len(state)


class LiveModel:
    """graphs: capture the front end and the Conformer at each warmup length instead of warming them up eagerly (None:
    on CUDA). The face adapter loads before capture: a graph keeps the weights it was captured with."""

    def __init__(self, half=False, device='cuda', warmup=range(1, 201), cudnn_benchmark=False, face_adapter=None,
                 graphs=None):
        from lipflow.vsr import LipReader, _MEAN, _STD
        self.reader = LipReader(device=device, use_lm=False, personal=False)
        self.face_adapter = face_adapter
        if face_adapter:
            load_face_adapter(self.reader.model, face_adapter)
        self.device = self.reader.enc_device
        self.encoder = self.reader.model.encoder
        self.ctc = self.reader.model.ctc
        self.half = half
        torch.backends.cudnn.benchmark = cudnn_benchmark
        self.dtype = torch.float16 if half else torch.float32
        if half:
            self.encoder.half()
        self.token_list = self.reader.token_list
        self._events = ((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        if self.device.type == 'cuda' else None)
        self.fe_graphs, self.graphs = {}, {}
        t0 = time.perf_counter()
        if graphs is None:
            graphs = self.device.type == 'cuda'
        self.graphs_error = None
        if graphs:
            self._norm = torch.tensor([255.0, _MEAN, _STD], device=self.device)
            try:
                self._capture(tuple(warmup))
            except RuntimeError as e:   # e.g. out of memory on a small GPU: read eagerly instead of not starting
                self.graphs_error = str(e).splitlines()[0]
                print(f'lipread: CUDA graph capture failed ({self.graphs_error}); eager reads', file=sys.stderr,
                      flush=True)
                self.fe_graphs, self.graphs = {}, {}
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                graphs = False
        if not graphs:
            for t in warmup:
                self.read(np.zeros((t, 96, 96), np.uint8))
        self.warmup_s = time.perf_counter() - t0

    @torch.inference_mode()
    def _capture(self, lengths):
        """A front-end and a Conformer graph per length, all in one memory pool, each warmed up once on a side stream
        before capture. Every tensor a graph reads must live as long as the graph: the weights, the static inputs
        self._rois and self._feats, and self._norm = (255, mean, std), a GPU tensor because dividing by a CPU scalar
        would multiply by its reciprocal on the GPU, one ulp away from LipReader.to_tensor's division."""
        if not lengths:
            return
        n = max(lengths)
        self._rois = torch.zeros((n, 96, 96), dtype=torch.uint8, device=self.device)
        self._host = torch.zeros((n, 96, 96), dtype=torch.uint8).pin_memory()
        with torch.inference_mode():
            F = self._frontend(self._rois[:1]).shape[-1]
        self._feats = torch.zeros((1, n, F), dtype=self.dtype, device=self.device)
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream(self.device)

        def capture(fn, inp):
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                fn(inp)
            torch.cuda.current_stream(self.device).wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, capture_error_mode='thread_local'):
                out = fn(inp)
            return Captured(g, inp, out)

        for T in lengths:
            self.fe_graphs[T] = capture(self._frontend, self._rois[:T])
            self.graphs[T] = capture(self._encode_head, self._feats[:, :T])
        self._sync()

    def _frontend(self, rois):
        """(n, 96, 96) uint8 on the GPU -> (1, n, F): LipReader.to_tensor's centre crop and normalisation, front end."""
        o, norm = (rois.shape[-1] - 88) // 2, self._norm
        x = rois.float() / norm[0]
        x = ((x[:, o:o + 88, o:o + 88] - norm[1]) / norm[2]).to(self.dtype)
        return self.encoder.frontend(x[None, None])

    def _encode_head(self, feats):
        enc = self._conformer(feats)
        return self._head(enc), enc

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def encode(self, rois, events=None):
        """(T, 96, 96) uint8 -> (T, D) encoder output in the encoder's dtype: the whole segment, no cache."""
        return self.encode_cached(rois, FrontendCache(), events)

    @torch.inference_mode()
    def encode_cached(self, rois, cache, events=None):
        """encode() with the front end run only on frames it hasn't finalised yet (plus FE_CONTEXT frames of left
        context). rois: (T, 96, 96) array or a list of (96, 96) crops; only the slice it needs is stacked.

        After the front end: the same steps as the espnet Encoder.forward (Linear+pos-enc embed, Conformer blocks,
        after_norm; tests/lipread/test_run.py checks it against the encoder's own forward), split from the front end so
        that events = (start, end) CUDA events time the Conformer part."""
        e = self.encoder
        T, a = len(rois), cache.n_final
        lo = max(0, a - FE_CONTEXT)
        x = self.reader.to_tensor(np.stack(rois[lo:T])).unsqueeze(0).to(self.device, self.dtype)
        new = e.frontend(x)[:, a - lo:]
        feats = new if cache.feats is None else torch.cat([cache.feats, new], dim=1)
        n_final = max(a, T - FE_CONTEXT)
        cache.feats, cache.n_final = feats[:, :n_final], n_final
        if events is not None:
            events[0].record()
        x = self._conformer(feats)
        if events is not None:
            events[1].record()
        return x

    def _conformer(self, feats):
        """(1, T, F) front-end features -> (T, D): embed, Conformer blocks, after_norm."""
        e = self.encoder
        x = e.embed(feats)
        x, _ = e.encoders(x, None)
        if isinstance(x, tuple):
            x = x[0]
        if e.normalize_before:
            x = e.after_norm(x)
        return x[0]

    def _head(self, enc):
        """(T, D) encoder output -> (T, 2 * TOPK + 1) float32: CTC top-k log-probabilities, blank, top-k ids."""
        logp = self.ctc.log_softmax(enc.float().unsqueeze(0))[0]
        top = logp.topk(TOPK, dim=-1)
        return torch.cat([top.values, logp[:, BLANK:BLANK + 1], top.indices.float()], dim=1)

    @torch.inference_mode()
    def read(self, rois, keep_enc=False, cache=None):
        """rois: (T, 96, 96) array or a list of (96, 96) crops. cache: a FrontendCache for this segment (front end once
        per frame), or None (whole segment each read). The captured path does the same steps as encode_cached on the
        same shapes: the front-end graph for the frames not yet final, the cached features and the new ones into the
        static feature buffer, then the Conformer graph."""
        cache = FrontendCache() if cache is None else cache
        T, a = len(rois), cache.n_final
        lo = max(0, a - FE_CONTEXT)
        fe, cf = self.fe_graphs.get(T - lo), self.graphs.get(T)
        if fe is None or cf is None:
            return self._read_eager(rois, keep_enc, cache)
        events = self._events
        t0 = time.perf_counter()
        host = self._host[:T - lo]
        np.stack(rois[lo:T], out=host.numpy())
        fe.inp.copy_(host, non_blocking=True)
        fe.graph.replay()
        if a:
            self._feats[:, :a].copy_(cache.feats)
        self._feats[:, a:T].copy_(fe.out[:, a - lo:])
        n_final = max(a, T - FE_CONTEXT)
        cache.feats, cache.n_final = self._feats[:, :n_final].clone(), n_final
        events[0].record()
        cf.graph.replay()
        events[1].record()
        self._sync()
        t1 = time.perf_counter()
        packed, enc = cf.out
        out = packed.cpu().numpy()
        t2 = time.perf_counter()
        conformer = events[0].elapsed_time(events[1])
        return self._reading(out, enc.clone() if keep_enc else None, t1 - t0, t2 - t1,
                             max(0.0, (t1 - t0) * 1000 - conformer), conformer)

    def _read_eager(self, rois, keep_enc, cache):
        events = self._events
        t0 = time.perf_counter()
        enc = self.encode_cached(rois, cache, events)
        self._sync()
        t1 = time.perf_counter()
        conformer = events[0].elapsed_time(events[1]) if events is not None else 0.0
        out = self._head(enc).cpu().numpy()
        t2 = time.perf_counter()
        return self._reading(out, enc if keep_enc else None, t1 - t0, t2 - t1,
                             max(0.0, (t1 - t0) * 1000 - conformer), conformer)

    def _reading(self, out, enc, enc_s, ctc_s, frontend_ms, conformer_ms):
        ids = out[:, TOPK + 1:].astype(np.int32)
        return Reading(ids, np.ascontiguousarray(out[:, :TOPK], np.float32),
                       np.ascontiguousarray(out[:, TOPK], np.float32), self.text(ids[:, 0]),
                       enc_s * 1000, ctc_s * 1000, enc, frontend_ms, conformer_ms)

    def text(self, best):
        """CTC best path (argmax id per frame) -> bus text: repeats collapsed, then bus.bus_text."""
        out, prev = [], None
        for i in best.tolist():
            if i != prev:
                out.append(self.token_list[i])
            prev = i
        return bus_text(''.join(out))


def bench_encoder(T=250, n=10):
    """Median ms of LiveModel.read on T zero frames, fp32 and fp16 (.half()), plus peak GPU memory."""
    out = {}
    for half in (False, True):
        torch.cuda.reset_peak_memory_stats()
        m = LiveModel(half=half, warmup=(T,))
        x = np.zeros((T, 96, 96), np.uint8)
        rs = [m.read(x) for _ in range(n)]
        name = 'fp16' if half else 'fp32'
        out[name] = (f'read median {statistics.median(r.enc_ms + r.ctc_ms for r in rs):.1f} ms (encoder '
                     f'{statistics.median(r.enc_ms for r in rs):.1f}), peak mem '
                     f'{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB')
        del m
        torch.cuda.empty_cache()
    return out
