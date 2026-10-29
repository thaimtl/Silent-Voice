"""Rescoring a CTC read: a CTC prefix beam over the read's top-k, its n-best rescored by the attention decoder in one
batched teacher-forced pass (the two-pass decoding of U2, https://arxiv.org/abs/2012.05481) and by Auto-AVSR's
Transformer LM. lipread/streaming_eval.py scores it offline; lipread/run.py runs it on every tick.
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
    """attention_scores for several reads at once: encs (B, T, D) of the same T, hyp_lists[b] the hypotheses of read b.
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
