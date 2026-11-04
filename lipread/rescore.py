"""Rescoring a CTC read: a CTC prefix beam over the read's top-k, its n-best rescored by the attention decoder in one
batched teacher-forced pass (the two-pass decoding of U2, https://arxiv.org/abs/2012.05481) and by Auto-AVSR's
Transformer LM, as lipread/streaming_eval.py scores it offline; plus a batched attention-decoder beam and exact CTC
scores, which lipread/final.py's GPU final pass adds at segment end. No tick runs it: every tick it cost +26 ms on the
5090 (beam 14 ms on the CPU, decoder 5, LM 5).
"""
import numpy as np
import torch

LOG0 = -1e30


def ctc_prefix_beam(topk_ids, topk_logp, blank_logp, beam=8, per_frame=8, blank_skip=-0.001):
    """CTC prefix beam search over a tick's top-k -> [(token tuple, CTC log-prob)], best first. Frames whose blank
    log-prob is above blank_skip (blank prob > 0.999) only extend with blank."""
    beams = {(): (0.0, LOG0)}                                 # prefix -> (log p ending in blank, log p ending in label)
    lae = np.logaddexp
    for t in range(len(blank_logp)):
        b = float(blank_logp[t])
        nxt = {}

        def add(prefix, pb, pnb):
            ob, onb = nxt.get(prefix, (LOG0, LOG0))
            nxt[prefix] = (lae(ob, pb), lae(onb, pnb))

        if b > blank_skip:
            for prefix, (pb, pnb) in beams.items():
                add(prefix, lae(pb, pnb) + b, LOG0)
            beams = nxt
            continue
        ids, lps = topk_ids[t, :per_frame], topk_logp[t, :per_frame]
        for prefix, (pb, pnb) in beams.items():
            total = lae(pb, pnb)
            add(prefix, total + b, LOG0)
            last = prefix[-1] if prefix else None
            for c, lp in zip(ids.tolist(), lps.tolist()):
                if c == 0:
                    continue
                if c == last:
                    add(prefix, LOG0, pnb + lp)              # repeat collapses
                    add(prefix + (c,), LOG0, pb + lp)        # a repeat needs a blank between
                else:
                    add(prefix + (c,), LOG0, total + lp)
        beams = dict(sorted(nxt.items(), key=lambda kv: -lae(*kv[1]))[:beam])
    return [(p, float(lae(*v))) for p, v in sorted(beams.items(), key=lambda kv: -lae(*kv[1]))]


@torch.inference_mode()
def attention_scores_multi(model, encs, hyp_lists):
    """Attention-decoder log-prob of each hypothesis (token tuples, eos appended) for several reads in one batched
    teacher-forced pass: encs (B, T, D) of the same T, hyp_lists[b] the hypotheses of read b.
    -> per read an array of scores."""
    flat = [(b, h) for b, hs in enumerate(hyp_lists) for h in hs]
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    m = model.reader.model
    dev = next(m.decoder.parameters()).device
    eos, L = m.eos, max(len(h) for _, h in flat) + 1
    ys_in = torch.full((len(flat), L), eos, dtype=torch.long)
    ys_out = torch.full((len(flat), L), -1, dtype=torch.long)
    for j, (_, h) in enumerate(flat):
        if h:
            ys_in[j, 1:len(h) + 1] = torch.tensor(h)
            ys_out[j, :len(h)] = torch.tensor(h)
        ys_out[j, len(h)] = eos
    ys_in, ys_out = ys_in.to(dev), ys_out.to(dev)
    mem = encs.float().to(dev)[torch.tensor([b for b, _ in flat], device=dev)]
    pred, _ = m.decoder(ys_in, subsequent_mask(L, device=dev).unsqueeze(0), mem, None)
    tok = pred.float().log_softmax(-1).gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    scores = ((tok * (ys_out >= 0)).sum(-1)).cpu().numpy()
    out, k = [], 0
    for hs in hyp_lists:
        out.append(scores[k:k + len(hs)])
        k += len(hs)
    return out


@torch.inference_mode()
def lm_scores(lm, hyp_lists, eos):
    """Auto-AVSR's subword Transformer LM log-prob of each hypothesis (eos included), all in one causal pass, as the
    final pass's beam adds it token by token. lm: lipread.final.load_lm() on the decoder's device."""
    flat = [h for hs in hyp_lists for h in hs]
    dev = next(lm.parameters()).device
    L = max(len(h) for h in flat) + 1
    x = torch.zeros((len(flat), L), dtype=torch.long)         # 0 = padding, masked by the LM
    t = torch.full((len(flat), L), -1, dtype=torch.long)
    for j, h in enumerate(flat):
        x[j, 0] = eos
        if h:
            x[j, 1:len(h) + 1] = torch.tensor(h)
            t[j, :len(h)] = torch.tensor(h)
        t[j, len(h)] = eos
    x, t = x.to(dev), t.to(dev)
    h, _ = lm.encoder(lm.embed(x), lm._target_mask(x))
    logp = lm.decoder(h).float().log_softmax(-1)
    tok = logp.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    scores = (tok * (t >= 0)).sum(-1).cpu().numpy()
    out, k = [], 0
    for hs in hyp_lists:
        out.append(scores[k:k + len(hs)])
        k += len(hs)
    return out


def combined(hyps, ctc, att, lm, w_ctc, w_lm=0.0, bonus=0.0):
    """Score per hypothesis: w_ctc x CTC + (1 - w_ctc) x attention + w_lm x LM + bonus x tokens (lm may be None when
    w_lm is 0)."""
    score = w_ctc * ctc + (1 - w_ctc) * att + bonus * np.array([len(h) for h in hyps])
    if w_lm:
        score = score + w_lm * lm
    return score


@torch.inference_mode()
def ctc_scores(model, enc, hyps):
    """Exact CTC log-prob of each hypothesis (token tuples) given enc (T, D), from the model's CTC head; -inf when it
    can't fit in T frames."""
    logp = model.ctc.log_softmax(enc.float().unsqueeze(0))[0]                # (T, V)
    n = len(hyps)
    lens = torch.tensor([len(h) for h in hyps], dtype=torch.long)
    targets = torch.zeros((n, max(1, int(lens.max()))), dtype=torch.long)
    for j, h in enumerate(hyps):
        if h:
            targets[j, :len(h)] = torch.tensor(h)
    nll = torch.nn.functional.ctc_loss(
        logp.unsqueeze(1).expand(-1, n, -1), targets.to(logp.device), torch.full((n,), len(logp), dtype=torch.long),
        lens, blank=0, reduction='none', zero_infinity=False)
    return (-nll).cpu().numpy().astype(np.float64)


@torch.inference_mode()
def attention_beam(model, enc, beam=5, lm=None, lm_weight=0.0, att_weight=1.0, maxlen=None):
    """Batched beam search with the attention decoder (+ lm_weight x LM) and their key/value caches, no CTC: the
    hypotheses the decoder itself would write, for rescoring. -> [token tuples], best first (eos dropped).
    maxlen: most tokens per hypothesis (default T)."""
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    m = model.reader.model
    dec, eos = m.decoder, m.eos
    for module in dec.modules():
        if hasattr(module, '_mem_kv'):
            del module._mem_kv
    dev = next(dec.parameters()).device
    mem = enc.float().to(dev).unsqueeze(0)
    maxlen = maxlen or len(enc)
    ys = torch.full((1, 1), eos, dtype=torch.long, device=dev)
    score = torch.zeros(1, device=dev)
    dcache = lcache = None
    done = []
    for step in range(maxlen + 1):
        n = len(ys)
        att, dcache = dec.forward_one_step(ys, subsequent_mask(ys.size(1), device=dev).unsqueeze(0),
                                           mem.expand(n, -1, -1), cache=dcache)
        tot = score[:, None] + att_weight * att
        if lm is not None and lm_weight:
            emb = lm.embed(ys) if lm.embed_drop is None else lm.embed_drop(lm.embed(ys))
            h, _, lcache = lm.encoder.forward_one_step(emb, lm._target_mask(ys), cache=lcache)
            tot = tot + lm_weight * lm.decoder(h[:, -1]).log_softmax(-1)
        tot[:, 0] = -float('inf')                               # the CTC blank is no token to write
        if step == maxlen:                                      # only eos left
            tot[:, :eos] = -float('inf')
        V = tot.size(1)
        best = tot.view(-1).topk(min(beam, tot.numel()))
        src, tok = (best.indices // V).tolist(), (best.indices % V).tolist()
        vals = best.values.tolist()
        keep = []
        prefixes = ys.tolist()
        for s, t, v in zip(src, tok, vals):
            if t == eos:
                done.append((v, tuple(prefixes[s][1:])))
            else:
                keep.append((s, t, v))
        done.sort(key=lambda d: -d[0])
        if not keep or (len(done) >= beam and keep[0][2] < done[beam - 1][0]):
            break
        idx = torch.tensor([s for s, _, _ in keep], device=dev)
        ys = torch.cat([ys[idx], torch.tensor([[t] for _, t, _ in keep], device=dev)], dim=1)
        score = torch.tensor([v for _, _, v in keep], device=dev)
        dcache = [c[idx] for c in dcache]
        if lcache is not None:
            lcache = [c[idx] for c in lcache]
    return [h for _, h in done[:beam]]
