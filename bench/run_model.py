"""Run any svr checkpoint over the frozen splits, on CPU and GPU, and save raw outputs.

Writes benchmark_results/<name>_cpu/raw and <name>_gpu/raw (inputs, per-frame log-probs,
timings) for bench.score to turn into per-case results. GPU runs in strict fp32 (TF32 off).

    venv-svr/bin/python -m bench.run_model --name B1T_ours --checkpoint checkpoints/B1T/model.pt
    venv-svr/bin/python -m bench.run_model --name B2_ours --checkpoint checkpoints/B2/model.pt --channels 4ch

Inputs use the preprocessing the checkpoint was trained on (--preprocess auto): SVR-1 checkpoints say so in model.pt,
Gaddy-architecture runs in their config.json (svr.train --preprocess); anything else (Gaddy's released model) gets his
original zero-phase pipeline.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from bench.common import (CHANNEL_CONFIGS, CLIP_SECONDS, DATA_ROOT, EVAL_SPLITS, FRAME_RATE, RESULTS_DIR, SPLITS_DIR, Timer,
                          environment_info, latency_stats, load_split, md5, save_raw)
from svr.model import load_gaddy_checkpoint
from svr.text import clean_text
from svr.preprocess import load_utterance

WARMUP, REPEATS = 5, 30
DEVICE_SUFFIX = {'cpu': 'cpu', 'cuda': 'gpu'}


SPEEDS = (1.25, 1.5)  # fast-speech judge: EMG compressed in time


def compress_time(x, speed):
    """Linearly resample a (time, channels) model input to 1/speed of its length, whole frames only."""
    n = len(x)
    new_len = max(8, int(n / speed) // 8 * 8)
    src = np.linspace(0, n - 1, new_len)
    return np.stack([np.interp(src, np.arange(n), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def load_any(checkpoint, device):
    """Gaddy-architecture state dict, or an SVR-1 model.pt ({'config', 'model'})."""
    st = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(st, dict) and 'config' in st and 'model' in st:
        from svr.svr1 import build, config_of
        model = build(config_of(st))
        model.load_state_dict(st['model'])
        return model.to(device).eval(), 'svr1'
    return load_gaddy_checkpoint(checkpoint, device), 'gaddy'


def checkpoint_preprocess(checkpoint):
    """'causal' or 'gaddy': what this checkpoint was trained on."""
    st = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if isinstance(st, dict) and 'config' in st and 'model' in st:
        return st.get('preprocess', 'gaddy')
    cfg = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), 'config.json')
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f).get('preprocess', 'gaddy')
    return 'gaddy'


def model_trials(raw_split):
    """{trials name: {utterance id: candidates}} for every candidate benchmark scored on this raw split."""
    from bench.score import CANDIDATE_BENCHMARKS
    out = {}
    for rs, tname in CANDIDATE_BENCHMARKS:
        if rs == raw_split and os.path.exists(os.path.join(SPLITS_DIR, tname + '.json')):
            out[tname] = {t['id']: t['candidates'] for t in load_split(tname)['trials']}
    return out


def run(name, checkpoint, description, channels=tuple(CHANNEL_CONFIGS), devices=('cpu', 'cuda'), speeds=SPEEDS,
        preprocess='auto'):
    if preprocess == 'auto':
        preprocess = checkpoint_preprocess(checkpoint)
    print('preprocessing:', preprocess, flush=True)
    stats = None
    if preprocess == 'causal':
        from svr.data import scale_stats, session_gain
        stats = scale_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    # preprocessing is device independent: run it once, timed on CPU
    prepared = {}
    for ch in channels:
        for split in EVAL_SPLITS:
            for it in load_split(split)['items']:
                with Timer() as t:
                    x = load_utterance(os.path.join(DATA_ROOT, it['session_dir']), it['index'],
                                       keep_channels=CHANNEL_CONFIGS[ch]['keep_channels'], preprocess=preprocess,
                                       gain=session_gain(it['session_dir'], stats) if stats else None)
                prepared[ch, split, it['id']] = (x, t.ms)
                if split == 'open_vocab_test':
                    for sp in speeds:
                        prepared[ch, f'{split}@{sp:g}x', it['id']] = (compress_time(x, sp), t.ms)
        print('preprocessed', ch, flush=True)

    for device in devices:
        model_name = '%s_%s' % (name, DEVICE_SUFFIX[device])
        model, kind = load_any(checkpoint, device)
        sync = torch.cuda.synchronize if device == 'cuda' else None
        state = {}

        def forward(x):
            if kind == 'gaddy':
                return F.log_softmax(model(x), -1)
            lengths = torch.tensor([x.shape[1] // 8], device=x.device)
            out = model(x, lengths, state['cmask'])
            state['out'] = out
            return out['letter_logp']

        env = environment_info(torch)
        if device == 'cuda':
            env.update(gpu=torch.cuda.get_device_name(0), tf32=False)
        meta_common = {'model': model_name, 'description': description, 'checkpoint': os.path.abspath(checkpoint),
                       'preprocess': preprocess,
                       'checkpoint_md5': md5(checkpoint), 'device': device, 'environment': env,
                       'parameters': sum(p.numel() for p in model.parameters())}
        with torch.no_grad():
            for ch in channels:
                cfg = CHANNEL_CONFIGS[ch]
                cmask = torch.zeros(8, device=device)
                cmask[cfg['keep_channels']] = 1
                state['cmask'] = cmask
                first = load_split('open_vocab_test')['items'][0]
                clip_frames = int(round(CLIP_SECONDS * FRAME_RATE))
                clip_np = prepared[ch, 'open_vocab_test', first['id']][0][:clip_frames * 8]
                clip = torch.from_numpy(clip_np)[None].to(device)
                for _ in range(WARMUP):
                    forward(clip)
                clip_ms = []
                for _ in range(REPEATS):
                    with Timer(sync) as t:
                        forward(clip)
                    clip_ms.append(t.ms)

                for split in list(EVAL_SPLITS) + [f'open_vocab_test@{sp:g}x' for sp in speeds]:
                    items_out, logits, inputs = [], {}, {}
                    trials = model_trials(split) if kind == 'svr1' else {}
                    for it in load_split(split.split('@')[0])['items']:
                        x_np, pre_ms = prepared[ch, split, it['id']]
                        x = torch.from_numpy(x_np)[None].to(device)
                        with Timer(sync) as t_fwd:
                            lp = forward(x)
                        row = {'id': it['id'], 'n_frames': int(lp.shape[1]), 'duration_s': x_np.shape[0] / 689.06,
                               'preprocess_ms': pre_ms, 'forward_ms': t_fwd.ms}
                        if trials:  # SVR-1: score every candidate set with the full model (CTC + verifier)
                            from svr.svr1 import score_texts
                            row['candidate_scores'] = {}
                            with Timer(sync) as t_sc:
                                for tname, by_id in trials.items():
                                    cands = by_id.get(it['id'])
                                    if cands and len(cands) > 1:
                                        s = score_texts(model, state['out'], 0, [clean_text(c) for c in cands],
                                                        device)
                                        row['candidate_scores'][tname] = s.float().cpu().tolist()
                            row['scoring_ms'] = t_sc.ms
                        items_out.append(row)
                        logits[it['id']] = lp[0].float().cpu().numpy()
                        inputs[it['id']] = x_np
                    meta = dict(meta_common, split=split, channels=ch, channel_config=cfg,
                                latency_3s_clip_forward_ms=latency_stats(clip_ms),
                                notes=f'preprocess_ms is svr.preprocess.load_utterance ({preprocess}) on CPU (filters, resampling, '
                                      'scaling; reads only the audio file header for the frame count).')
                    save_raw(os.path.join(RESULTS_DIR, model_name, 'raw'), split, ch, meta, items_out, logits, inputs)
                    print(model_name, split, ch, 'done', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--description', default='')
    p.add_argument('--channels', default=','.join(CHANNEL_CONFIGS))
    p.add_argument('--devices', default='cpu,cuda')
    p.add_argument('--speeds', default=','.join(f'{s:g}' for s in SPEEDS), help='fast-speech compressions; "" for none')
    p.add_argument('--preprocess', default='auto', choices=['auto', 'causal', 'gaddy'])
    a = p.parse_args()
    run(a.name, a.checkpoint, a.description, a.channels.split(','), a.devices.split(','),
        tuple(float(x) for x in a.speeds.split(',') if x), a.preprocess)


if __name__ == '__main__':
    main()
