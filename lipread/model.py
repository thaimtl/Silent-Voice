"""Auto-AVSR (Lipflow's LipReader weights) as the camera reader needs it: re-read a whole segment, CTC top-k per frame.

    model = LiveModel(half=False)          # loads LRS3_V_WER19.1 on CUDA, warms up
    r = model.read(rois)                   # rois (T, 96, 96) uint8 crops at 25 fps -> Reading
    r.topk_ids (T, 32) int32, r.topk_logp (T, 32) float32, r.blank_logp (T,), r.greedy 'UPPERCASE TEXT', r.enc_ms ...
    r = model.read(rois, keep_enc=True)    # also r.enc, the (T, D) encoder output on the GPU (the final pass reuses it)
    model = LiveModel(face_adapter=path)   # with Austin's face adapter on top (lipread/adapt.py)

The encoder (3D-conv/ResNet front end + 12 Conformer blocks) runs on the whole segment every call (r.frontend_ms +
r.conformer_ms = r.enc_ms, split with CUDA events); the CTC head
(log_softmax, blank = 0) and top-k run on the GPU, one device-to-host copy per call. half=True runs the encoder in
fp16 via .half() (measured faster than autocast, CHANGELOG T0); the CTC head stays fp32.
"""
import statistics
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
    enc: object = None
    frontend_ms: float = 0.0
    conformer_ms: float = 0.0


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


def load_face_adapter(model, path):
    """Lipflow's face adapter (lipread/adapt.py, as lipflow/train_vsr.py): new values for the front end, the embed and
    Conformer block 0, loaded over the base weights. Same shapes, so a read costs the same. Returns the tensor count;
    ValueError for a file with any other tensor, which would silently change the rest of the model."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    other = sorted(k for k in state if not k.startswith(FACE_ADAPTER_PREFIXES))
    if other:
        raise ValueError(f'{path}: not a face adapter, {len(other)} other tensors (first: {other[0]})')
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise ValueError(f'{path}: tensors the model does not have: {res.unexpected_keys[:3]}')
    return len(state)


class LiveModel:
    def __init__(self, half=False, device='cuda', warmup=range(1, 201), cudnn_benchmark=False, face_adapter=None):
        from lipflow.vsr import LipReader
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
        t0 = time.perf_counter()
        for t in warmup:
            self.read(np.zeros((t, 96, 96), np.uint8))
        self.warmup_s = time.perf_counter() - t0

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
        x = e.embed(feats)
        x, _ = e.encoders(x, None)
        if isinstance(x, tuple):
            x = x[0]
        if e.normalize_before:
            x = e.after_norm(x)
        if events is not None:
            events[1].record()
        return x[0]

    @torch.inference_mode()
    def read(self, rois, keep_enc=False, cache=None):
        """cache: a FrontendCache for this segment (front end once per frame), or None (whole segment each read)."""
        events = self._events
        t0 = time.perf_counter()
        enc = self.encode_cached(rois, FrontendCache() if cache is None else cache, events)
        self._sync()
        t1 = time.perf_counter()
        conformer = events[0].elapsed_time(events[1]) if events is not None else 0.0
        logp = self.ctc.log_softmax(enc.float().unsqueeze(0))[0]
        top = logp.topk(TOPK, dim=-1)
        out = torch.cat([top.values, logp[:, BLANK:BLANK + 1], top.indices.float()], dim=1).cpu().numpy()
        t2 = time.perf_counter()
        ids = out[:, TOPK + 1:].astype(np.int32)
        return Reading(ids, np.ascontiguousarray(out[:, :TOPK], np.float32),
                       np.ascontiguousarray(out[:, TOPK], np.float32), self.text(ids[:, 0]),
                       (t1 - t0) * 1000, (t2 - t1) * 1000, enc if keep_enc else None,
                       max(0.0, (t1 - t0) * 1000 - conformer), conformer)

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
