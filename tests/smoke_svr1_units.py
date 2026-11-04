"""Unit checks for SVR-1 (CPU or GPU; FA2-vs-SDPA parity only where flash-attn and a CUDA bf16 GPU exist).
Needs no dataset.

    venv-svr/bin/python -m tests.smoke_svr1_units            # GPU if there is one
    CUDA_VISIBLE_DEVICES= venv-svr/bin/python -m tests.smoke_svr1_units   # CPU only
"""
import argparse
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from svr import augment
from svr import preprocess as P
from svr import train_svr1 as T
from svr.align import ctc_align, prefix_target, scale_emit, speak_target
from svr.phonemes import sentence_phonemes
from svr.svr1 import INFEASIBLE, build, candidate_tensors, ctc_fits, flash_attn_varlen_func, score_texts
from svr.text import BLANK, text_to_ids

TINY = dict(d_model=64, n_layers=2, n_heads=4, ffn_hidden=128, verifier_dim=64, verifier_layers=1)


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    m = build(TINY).to(device).eval()
    check('4-slot default = the rig\'s sites', T.parse_args(['--name', 'x']).keep_channels == '0,1,2,5'
          and P.KEEP == [0, 1, 2, 5] and P.RIG_TO_SLOT == [2, 1, 5, 0])
    cm = torch.zeros(8, device=device)
    cm[P.KEEP] = 1

    # shapes; frames = samples / 8 exactly
    x = torch.randn(2, 80 * 8, 8, device=device)
    lengths = torch.tensor([80, 50], device=device)
    with torch.no_grad():
        out = m(x, lengths, cm)
    check('letter/phone/artic shapes', out['letter_logp'].shape == (2, 80, 38) and out['phone_logp'].shape == (2, 80, 40)
          and out['artic'].shape == (2, 80, 13) and out['speak_logit'].shape == (2, 80))

    # padding invariance: row 1 alone == row 1 padded next to a longer row (no batch statistics, masked attention)
    with torch.no_grad():
        alone = m(x[1:2, :50 * 8], lengths[1:2], cm)['letter_logp'][0]
    err = (alone - out['letter_logp'][1, :50]).abs().max().item()
    check('padding does not change a row', err < 1e-4, f'max diff {err:.2e}')

    # absent channels have no effect (masked before and after the adapter)
    x2 = x.clone()
    x2[..., [3, 4, 6, 7]] = torch.randn_like(x2[..., [3, 4, 6, 7]]) * 10
    with torch.no_grad():
        err = (m(x2, lengths, cm)['letter_logp'] - out['letter_logp']).abs().max().item()
    check('masked channels are ignored', err < 1e-5, f'max diff {err:.2e}')

    # per-row channel masks
    with torch.no_grad():
        o2 = m(x, lengths, cm[None].expand(2, -1))['letter_logp']
    check('(B, C) channel mask == (C,) mask', (o2 - out['letter_logp']).abs().max().item() < 1e-5)

    # ctc_fits agrees with torch's CTC (inf loss <=> infeasible)
    for txt, frames in (('hello', 5), ('hello', 6), ('abc', 3), ('aa', 2), ('aa', 3)):
        ids = torch.tensor(text_to_ids(txt))
        lp = torch.randn(frames, 1, 38).log_softmax(-1)
        inf = torch.isinf(F.ctc_loss(lp, ids[None], [frames], [len(ids)], blank=37, reduction='none'))[0].item()
        fits = ctc_fits(ids, torch.tensor([len(ids)]), torch.tensor([frames]))[0].item()
        check(f'ctc_fits({txt!r}, {frames} frames) = {fits}', fits == (not inf))

    # candidate scoring: infeasible candidate -> INFEASIBLE, finite gradients everywhere
    m.train()
    out = m(x, torch.tensor([80, 12], device=device), cm)
    cands = ['the cat sat', 'the bat sat', 'the cat sat on a very long mat that cannot possibly fit here']
    s, detail = m.score_candidates(out, 1, *candidate_tensors(cands, device))
    check('too-long candidate scored INFEASIBLE', s[2].item() == INFEASIBLE and not detail['feasible'][2].item())
    loss = F.cross_entropy(s[None], torch.zeros(1, dtype=torch.long, device=device))
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    check('finite gradients with an infeasible candidate', not bad, ', '.join(bad[:3]))
    got = {n.split('.')[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    check('judge gradients reach verifier, mix, encoder', {'verifier', 'mix', 'blocks', 'front'} <= got, str(sorted(got)))
    m.zero_grad()

    # out-of-dictionary candidate does not get a free phoneme score
    oov = 'the cat zzqxv'
    check('test word is out of dictionary', sentence_phonemes(oov) is None)
    with torch.no_grad():
        out = m.eval()(x, lengths, cm)
        s_mixed, d_mixed = m.score_candidates(out, 0, *candidate_tensors(['the cat sat', oov], device))
    check('phone term dropped when any candidate is OOV', 'phone' not in d_mixed)
    check('mix weights on the simplex', abs(m.mix.softmax(0).sum().item() - 1) < 1e-6)
    check('score_texts finite', torch.isfinite(score_texts(m, out, 0, ['the cat sat', 'a bat sat'], device)).all())

    # augmentation: speed keeps whole frames and channel layout; channel_drop keeps >= 3
    rng = np.random.default_rng(0)
    xa = np.random.randn(800, 8).astype(np.float32)
    for f in (0.8, 1.0, 1.6):
        y = augment.speed(xa, f)
        check(f'speed {f}: whole frames', len(y) % 8 == 0 and y.shape[1] == 8)
    check('channel_drop min 3', all(len(augment.channel_drop(range(6), rng, 0.9)) >= 3 for _ in range(50)))

    # under autocast the attention inputs are bf16 (FA2 needs them; fp32 here silently fell back to SDPA)
    seen = []
    hook = m.blocks[0].attn.qkv.register_forward_hook(lambda mod, i, o: seen.append(o.dtype))
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        o = m(x, lengths, cm)
    hook.remove()
    check('qkv output is bf16 under autocast', seen == [torch.bfloat16], str(seen))
    check('heads return fp32 log-probs', o['letter_logp'].dtype == torch.float32)

    # FA2 vs SDPA (bf16, H100/H200)
    if device == 'cuda' and flash_attn_varlen_func is not None:
        torch.manual_seed(0)
        ms = build(TINY, attention='sdpa').cuda().eval()
        mf = build(TINY, attention='fa2').cuda().eval()
        mf.load_state_dict(ms.state_dict())
        xb = torch.randn(3, 300 * 8, 8, device='cuda')
        lb = torch.tensor([300, 211, 97], device='cuda')
        import svr.svr1 as S
        calls = []
        real = S.flash_attn_varlen_func
        S.flash_attn_varlen_func = lambda *a_, **k_: calls.append(1) or real(*a_, **k_)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = ms(xb, lb, cm)['letter_logp'], mf(xb, lb, cm)['letter_logp']
        S.flash_attn_varlen_func = real
        check('FA2 kernel actually runs', len(calls) == TINY['n_layers'], f'{len(calls)} calls')
        mask = torch.arange(300, device='cuda')[None] < lb[:, None]
        err = (a - b).abs()[mask].max().item()
        check('FA2 == SDPA on real frames (bf16)', err < 0.05, f'max diff {err:.3f}')
    else:
        print('SKIP FA2 parity (needs CUDA + flash-attn)')
    pipeline_checks()
    print('SVR1 UNITS PASS')


def pipeline_checks():
    """Preprocessing, alignment targets, augmentation and the live wrapper (CPU, no dataset)."""
    rng = np.random.default_rng(0)
    # causal filter: chunked (as live) == whole clip (as training), and never zero-phase
    raw = 512 + 40 * rng.standard_normal((3000, 4)) + 30 * np.sin(2 * np.pi * 60 * np.arange(3000) / 1000)[:, None]
    f = P.rig_filters.EmgFilter()
    f(raw[:1000][::-1])
    chunked = np.concatenate([f(raw[i:i + 10]) for i in range(0, len(raw), 10)])
    check('causal filter: 10-sample chunks == whole clip', np.abs(chunked - P.causal_filter(raw)).max() < 1e-9)
    y = P.causal_filter(raw)
    hum = np.abs(np.fft.rfft(y[1000:2000, 0]))[60] / np.abs(np.fft.rfft(raw[1000:2000, 0] - 512))[60]
    check('60 Hz hum removed', hum < 0.05, f'ratio {hum:.3f}')
    u = P.rig_unscaled(raw.astype(np.uint16))
    check('rig columns land in slots 2, 1, 5, 0', u.shape[1] == 8 and np.all(u[:, [3, 4, 6, 7]] == 0)
          and np.all(np.abs(u[:, P.KEEP]).sum(0) > 0) and len(u) % 8 == 0)
    g = P.rig_gain(P.robust_spread(u), np.full(8, 15.0))
    check('gain brings each used slot to the target spread',
          np.allclose(P.robust_spread(u * g)[P.KEEP], 15.0, rtol=1e-3) and np.all(g[[3, 4, 6, 7]] == 0))
    xs = P.apply_gain(u, g, P.KEEP)
    check('unscale inverts the soft clip', np.allclose(P.unscale(xs), u * g, rtol=1e-3, atol=1e-2))

    # forced alignment, speaking targets, partial-sentence targets
    ids = np.array(text_to_ids('hello world'))
    path = [BLANK] * 3 + [ids[0]] * 2 + [BLANK]
    for c in ids[1:]:
        path += [c, BLANK]
    path += [BLANK] * (40 - len(path))
    lp = np.full((40, 38), -8.0)
    lp[np.arange(40), path] = -0.01
    emit = ctc_align(lp, ids, BLANK)
    check('alignment recovers the path (repeated letter split by a blank)',
          emit is not None and emit.tolist() == [3, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    check('no alignment when the text cannot fit', ctc_align(lp[:5], ids, BLANK) is None)
    sp = speak_target(emit, 40)
    check('speaking target = first..last letter +- 50 ms', sp[:24 - 24].sum() == 0 and sp[0] == 1 and sp[28] == 1
          and sp[29] == 0, str(np.flatnonzero(sp)[[0, -1]]))
    check('prefix target = letters before the cut', prefix_target(emit, ids, 10).tolist() == ids[:3].tolist())
    check('alignment follows a speed change', scale_emit(emit, 40, 20).tolist()[:3] == [1, 3, 4])

    # make_batch: cuts shorten rows and targets, slow speech stays within the padded budget, effort keeps zeros zero
    data = SimpleNamespace(inputs=[], targets=[], artic=[], align=[], speak_neg=[], lengths=[], phones=[], clean=[])
    for n in (120, 200, 160):
        x = rng.standard_normal((n * 8, 8)).astype(np.float32) * 5
        x[:, [3, 4, 6, 7]] = 0
        data.inputs.append(x)
        data.targets.append(ids)
        data.artic.append(None)
        data.align.append(np.linspace(5, n - 10, len(ids)).astype(np.int64))
        data.speak_neg.append(False)
        data.lengths.append(n * 8)
    args = T.parse_args(['--name', 'x', '--no-hwsim', '--micro-batch-samples', str(3 * 200 * 8), '--slow-p', '1.0'])
    over = 0
    for seed in range(20):
        mb = T.make_batch(data, [0, 1, 2], np.random.default_rng(seed), args, P.KEEP, prefix_p=0.5)
        over = max(over, mb['x'].shape[0] * mb['x'].shape[1] / args.micro_batch_samples)
        for i, k in enumerate([0, 1, 2]):
            n = int(mb['lengths'][i])
            t = mb['targets'][i]
            ok = t.tolist() == ids[:len(t)].tolist() and (mb['cut'][i] or len(t) == len(ids))  # a cut keeps a prefix
            if not ok or not T.ctc_feasible(n, mb['targets'][i]) or mb['speak_mask'][i, :n].sum() != n:
                check('make_batch rows consistent', False, f'seed {seed} row {i}')
        if mb['x'][..., [3, 4, 6, 7]].abs().max() > 0:
            check('absent slots stay zero through augmentation', False, f'seed {seed}')
    check('make_batch: cuts, targets, speaking masks consistent; absent slots zero', True)
    check('slow speech (0.5x) never grows a batch past 1.25x the budget', over <= T.PADDED_GROWTH + 1e-6, f'{over:.2f}')

    # the live wrapper on a tiny model: feed 10 ms chunks, read every 100 ms, verify candidates
    torch.manual_seed(0)
    m = build(TINY)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'model.pt')
        torch.save({'config': m.config_dict(), 'model': m.state_dict(), 'keep_channels': P.KEEP, 'preprocess': 'causal',
                    'scale_target': [15.0] * 8}, path)
        from svr.live import LiveSVR1
        live = LiveSVR1(path, device='cpu')
        cal = live.calibrate(raw)
        check('live calibrate sets a gain for the 4 used slots', all(cal['gain'][c] > 0 for c in P.KEEP)
              and all(cal['gain'][c] == 0 for c in (3, 4, 6, 7)))
        outs = []
        for i in range(0, 3000, 10):
            live.feed(raw[i:i + 10], t0=100.0 + i / 1000)
            if i % 100 == 90:
                o = live.read(camera_active=(1000 <= i < 2000))
                if o is not None:
                    outs.append(o)
        o = outs[-1]
        check('live read: per-frame outputs line up', o['letter_logp'].shape[0] == len(o['frame_t']) == len(o['speak_prob'])
              and o['phone_logp'].shape[1] == 40)
        check('live frame times end near the last sample', abs(o['frame_t'][-1] - 102.999) < 0.02, f"{o['frame_t'][-1]:.3f}")
        check('live segments open on camera activity and close after 0.5 s',
              any(x['in_segment'] for x in outs) and any(x['segment_ended'] for x in outs))
        first = next(x for x in outs if x['segment_started'])
        check('the read where a segment starts reports it at once', first['in_segment'] and first['seg_start'] is not None)
        s = live.verify(['hello world', 'yellow word'])
        check('live verify scores candidates', s.shape == (2,) and np.isfinite(s).all())
        check("verify on a read's own output is verify after that read", np.array_equal(
            s, live.verify(['hello world', 'yellow word'], o['out'])))

        # fusion reads on another thread: a read of a window taken earlier ignores what feed() added since
        live2 = LiveSVR1(path, device='cpu')
        live2.gain, live2.effort_ref, live2.pace_ref = live.gain, live.effort_ref, live.pace_ref
        for i in range(0, 2000, 10):
            live2.feed(raw[i:i + 10], t0=100.0 + i / 1000)
        now = live2.read(True, 100.5)
        w = live2.window(100.5)
        live2.feed(raw[2000:2300], t0=102.0)
        late = live2.read(True, 100.5, window=w)
        check('a read of an earlier window equals the read made then',
              all(np.array_equal(now[k], late[k]) for k in ('frame_t', 'letter_logp', 'phone_logp', 'speak_prob'))
              and now['pace'] == late['pace'] and now['effort'] == late['effort'])
        w = live2.window(100.5)
        live2.feed(raw[2300:2400], t0=103.0)      # a gap in the stream: reset() between the window and its read
        gone = live2.read(True, 100.5, window=w)
        check('a window from before a gap still reads, but moves no segment',
              gone is not None and not gone['segment_started'] and not gone['in_segment'] and live2.seg_start is None)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    main()
