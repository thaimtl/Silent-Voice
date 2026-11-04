"""The live pipeline's costs in one table, with knobs that make a bigger box behave like Austin's laptop.

Two child processes, as live: the camera reader (venv-live: LiveModel, fp16, + the GPU final pass and its LM) and the
fusion process (venv-svr: SVR-1 + the decoder search), run side by side so the peak VRAM is the two together. Per row:

    camera tick, by segment band (T <= 50 / 100 / 200 frames), CUDA graphs and eager: wall p50 / p95 ms, host ms per
        tick (the tick thread's CPU time), launches per tick (CUDA runtime launch calls: kernels, graph launches,
        copies, memsets), GPU kernels and GPU busy ms per tick (torch.profiler, on a separate profiled segment)
    final pass at T 50 / 100 / 200 (2 / 4 / 8 s): wall, host ms, launches, GPU busy
    SVR-1 read over 2 / 4 / 8 s of a recorded session's EMG: wall, host ms, launches, GPU busy
    decoder pass (fusion.offline on the first --segments camera segments of a recorded session, camera + EMG + LM):
        wall p50 / p95, the same without the SVR-1 wait (live: the read runs in the gap between ticks)
    peak VRAM: each process's allocator peak (reserved) and the whole device's peak above its idle level

Launches and host ms per tick stand in for the laptop: neither depends on how fast the GPU is, and each launch costs
more under Windows' WDDM. A change is a laptop win when it cuts them (or VRAM) by more than the spread of --runs.

    python -m lipread.bench.laptop_profile                       # 3 runs, alone and beside a busy Python thread
    python -m lipread.bench.laptop_profile --vram-gib 8 --cpus 6 # emulate the laptop's 8 GB and fewer cores
    python -m lipread.bench.laptop_profile --runs 1 --no-hog     # quick look

--vram-gib caps each child's PyTorch allocator so that both, plus a CUDA context each (--ctx-gib), fit the budget
(an allocation past the cap fails as it would on the laptop). --cpus pins every process to the first N cores.
--hog adds a pure-Python busy thread in each child (MediaPipe, emotion, the preview hold the GIL like this live);
by default every run is done alone and then beside it. Results: logs/rt_win_profile_<time>/ (table, JSON, child
logs). Runs unchanged on Windows and Linux; venvs are found as the launcher finds them (RECLAIM_WORKDIR).
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WORK = os.environ.get('RECLAIM_WORKDIR', ROOT)
WINDOWS = sys.platform == 'win32'
BANDS = ((50, '2 s'), (100, '4 s'), (200, '8 s'))
FINAL_T = (50, 100, 200)
SVR_S = (2, 4, 8)
SPLIT = {'camera': 0.8, 'fusion': 0.2}       # share of the capped VRAM budget (after contexts) per process
LAUNCH_CALLS = ('cudaLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernel', 'cuLaunchKernelEx', 'cudaGraphLaunch',
                'cudaMemcpyAsync', 'cudaMemcpy', 'cudaMemsetAsync', 'cudaMemset')
VENV = {'camera': 'venv-live', 'fusion': 'venv-svr'}


# ------------------------------------------------------------------ in a child
def limit(args):
    """--cpus and --cap-gib in this process; --hog starts the busy thread. Returns the hog's stop event."""
    if args.cpus:
        cores = list(range(args.cpus))
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, cores)
        else:
            import psutil
            psutil.Process().cpu_affinity(cores)
    import torch
    if args.cap_gib:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.cap_gib * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    stop = threading.Event()
    if args.hog:
        threading.Thread(target=hog, args=(stop,), daemon=True).start()
    return stop


def hog(stop):
    while not stop.is_set():
        sum(k * k for k in range(20000))


def profiled(fn):
    """Runs fn() under torch.profiler; fn calls marks(label) as a context manager around each unit of work.
    -> {label: [(launch calls, GPU kernels, GPU busy ms)]}, kernels attributed by their start time."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
    labels = []

    def mark(label):
        labels.append(label)
        return record_function(f'lp#{len(labels) - 1}')

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(mark)
        torch.cuda.synchronize()
    spans, launches, kernels = {}, [], []
    for e in p.events():
        if e.name.startswith('lp#'):
            spans[int(e.name[3:])] = (e.time_range.start, e.time_range.end)
        elif e.device_type == torch.autograd.DeviceType.CUDA:
            kernels.append((e.time_range.start, getattr(e, 'device_time', None) or e.cuda_time))
        elif e.name in LAUNCH_CALLS:
            launches.append(e.time_range.start)
    out = {}
    for i, (a, b) in spans.items():
        n = sum(a <= t <= b for t in launches)
        ks = [d for t, d in kernels if a <= t <= b]
        out.setdefault(labels[i], []).append((n, len(ks), sum(ks) / 1000))
    return out


def summary(walls, hosts, prof=None):
    row = dict(n=len(walls), wall_p50=float(np.percentile(walls, 50)), wall_p95=float(np.percentile(walls, 95)),
               host_ms=float(np.mean(hosts)))
    if prof:
        row.update(launches=float(np.mean([x[0] for x in prof])), kernels=float(np.mean([x[1] for x in prof])),
                   gpu_ms=float(np.mean([x[2] for x in prof])))
    return row


def band(T):
    return next(name for top, name in BANDS if T <= top)


def camera(args):
    """The camera reader's rows: ticks (eager, then graphs as live), the final pass, its VRAM."""
    import contextlib
    import torch
    from bus import encode
    from lipread.bench.tick_bench import sample_rois
    from lipread.endpoint import Segmenter
    from lipread.final import Rescore
    from lipread.model import LiveModel
    from lipread.run import Runner
    from lipread.types import FPS, Crop
    rois, what = sample_rois(BANDS[-1][0])
    stop = limit(args)
    rows = {'_': dict(rois=what, gpu=torch.cuda.get_device_name(0))}

    def segment(model, gap, mark=None, budget=None):
        """One segment growing to 200 frames on the live tick schedule -> [(T, wall ms, host ms)]."""
        mark = mark or (lambda _: contextlib.nullcontext())
        r = Runner(model, encode, Segmenter(never_end=True, cap_s=len(rois) / FPS + 1), fe_cache=True)
        t, out, t_end = 1000.0, [], time.perf_counter() + (budget or 1e9)
        for i, roi in enumerate(rois):
            t += 1 / FPS
            h0 = time.thread_time()
            with mark(i):
                got = r.on_crops([(Crop(t=t, roi96=roi, mouth_open=0.0, face_ok=True), 0.0)])
            h = (time.thread_time() - h0) * 1000
            out += [(s.T, s.total_ms, h, i) for s in got]
            if got and gap:
                time.sleep(gap)
            if time.perf_counter() > t_end:
                break
        return out

    for mode in ('eager', 'graphs'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = LiveModel(half=True, face_adapter=args.face_adapter, graphs=mode == 'graphs')
        load_s = time.perf_counter() - t0
        ticks = []
        for _ in range(args.reps):
            ticks += segment(model, 0.1, budget=args.hog_budget if args.hog else None)
        prof = {}
        if not args.hog:
            prof = profiled(lambda mark: segment(model, 0.0, mark))
        tick_prof = {}
        for T, _, _, i in segment_ticks(prof, ticks):
            tick_prof.setdefault(band(T), []).extend(prof.get(i, []))
        for _, name in BANDS:
            sel = [x for x in ticks if band(x[0]) == name]
            if sel:
                rows[f'camera tick {mode} {name}'] = summary([x[1] for x in sel], [x[2] for x in sel],
                                                             tick_prof.get(name))
        rows[f'camera load {mode}'] = dict(load_s=load_s, warmup_s=model.warmup_s)
        if mode == 'graphs':
            rescore = Rescore(model)
            for T in FINAL_T:
                enc = model.read(rois[:T], keep_enc=True).enc
                walls, hosts = [], []
                for _ in range(args.final_reps):
                    w0, h0 = time.perf_counter(), time.thread_time()
                    rescore(enc=enc)
                    walls.append((time.perf_counter() - w0) * 1000)
                    hosts.append((time.thread_time() - h0) * 1000)
                p = None
                if not args.hog:
                    def run(mark):
                        for k in range(3):
                            with mark(k):
                                rescore(enc=enc)
                    p = [x for v in profiled(run).values() for x in v]
                rows[f'final pass T{T}'] = summary(walls, hosts, p)
            rows['vram camera'] = dict(reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                                       allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
        del model
    stop.set()
    return rows


def segment_ticks(prof, ticks):
    """The profiled segment's ticks are on the same crops as the timed ones: (T, -, -, crop index) per tick index."""
    seen = {}
    for T, w, h, i in ticks:
        seen.setdefault(i, (T, w, h, i))
    return [seen[i] for i in prof if i in seen]


def fusion(args):
    """The fusion process's rows: SVR-1 reads over recorded EMG, the decoder pass over a camera log, its VRAM."""
    import torch
    sys.path.insert(0, ROOT)
    from fusion import offline
    from svr.live import LiveSVR1
    from svr.finetune import usable_record
    stop = limit(args)
    rows = {}
    torch.cuda.reset_peak_memory_stats()
    svr = args.svr or os.path.join(WORK, 'checkpoints', 'W_s12_svr1_4ch', 'model.pt')
    if not os.path.exists(svr) or not os.path.isdir(os.path.join(offline.RECORDINGS, args.session)):
        rows['_fusion'] = dict(skipped=f'no {svr} or recordings/{args.session}')
        stop.set()
        return rows
    rows_ = [r for r in offline.session_rows(args.session) if usable_record(r, False, 0.05)]
    live = LiveSVR1(svr)
    offline.session_gain(live, rows_)
    fed = 0.0
    for t, _, msg in offline.emg_events(rows_):
        live.feed(msg['samples'])       # no t0: the clips back to back, else the gaps between them reset the buffer
        fed += len(msg['samples']) / 1000
        if fed >= max(SVR_S) + 1:
            break
    t_now = live.time_of(live.n)
    for s in SVR_S:
        live.read(True, t_now - s)
        walls, hosts = [], []
        for _ in range(args.svr_reps):
            w0, h0 = time.perf_counter(), time.thread_time()
            live.read(True, t_now - s)
            walls.append((time.perf_counter() - w0) * 1000)
            hosts.append((time.thread_time() - h0) * 1000)
        p = None
        if not args.hog:
            def run(mark):
                for k in range(3):
                    with mark(k):
                        live.read(True, t_now - s)
            p = [x for v in profiled(run).values() for x in v]
        rows[f'SVR-1 read {s} s'] = summary(walls, hosts, p)
    del live
    cam_log = args.cam_log or os.path.join(WORK, 'logs', 'rt_speed_all_s1')
    if os.path.exists(cam_log):
        gate, lock, _ = offline.load_params()
        c0 = time.process_time()
        recs, _, _ = offline.run_session(args.session, cam_log, svr, gate, lock, limit=args.segments, progress=False)
        cpu = (time.process_time() - c0) * 1000 / max(len(recs), 1)
        ms = [r['ms'] for r in recs if not r['final']]
        nowait = [r['ms'] - r.get('emg_wait_ms', 0.0) for r in recs if not r['final']]
        rows['decoder pass'] = dict(n=len(ms), wall_p50=float(np.percentile(ms, 50)),
                                    wall_p95=float(np.percentile(ms, 95)), host_ms=cpu)
        rows['decoder pass, no SVR-1 wait'] = dict(n=len(nowait), wall_p50=float(np.percentile(nowait, 50)),
                                                   wall_p95=float(np.percentile(nowait, 95)))
    else:
        rows['decoder pass'] = dict(skipped=f'no camera log {cam_log}')
    rows['vram fusion'] = dict(reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                               allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    stop.set()
    return rows


# ------------------------------------------------------------------ the parent
def venv_python(part):
    return os.path.join(WORK, VENV[part], 'Scripts' if WINDOWS else 'bin', 'python.exe' if WINDOWS else 'python')


def device_used_mib():
    """The whole GPU's used memory (nvidia-smi; per-process numbers are not available under WDDM)."""
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return None
    try:
        r = subprocess.run([smi, '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', '0'],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def run_once(args, hog, out_dir, tag):
    """Both children side by side; samples the device's memory meanwhile. -> merged rows."""
    idle = device_used_mib()
    procs = {}
    for part in ('camera', 'fusion'):
        py = getattr(args, f'py_{part}') or venv_python(part)
        if not os.path.exists(py):
            py = sys.executable
        out = os.path.join(out_dir, f'{tag}_{part}.json')
        cmd = [py, '-m', 'lipread.bench.laptop_profile', '--part', part, '--out', out,
               '--reps', str(args.reps), '--final-reps', str(args.final_reps), '--svr-reps', str(args.svr_reps),
               '--segments', str(args.segments), '--session', args.session, '--hog-budget', str(args.hog_budget)]
        for k in ('cpus', 'face_adapter', 'svr', 'cam_log'):
            if getattr(args, k):
                cmd += [f'--{k.replace("_", "-")}', str(getattr(args, k))]
        if args.vram_gib:
            share = (args.vram_gib - 2 * args.ctx_gib) * SPLIT[part]
            cmd += ['--cap-gib', f'{share:.3f}']
        if hog:
            cmd.append('--hog')
        log = open(os.path.join(out_dir, f'{tag}_{part}.log'), 'w', encoding='utf-8')
        procs[part] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'RECLAIM_WORKDIR': WORK, 'PYTHONUTF8': '1'}), out, log)
    peak = idle
    while any(p.poll() is None for p, _, _ in procs.values()):
        used = device_used_mib()
        if used is not None and peak is not None:
            peak = max(peak, used)
        time.sleep(0.25)
    rows = {}
    for part, (p, out, log) in procs.items():
        log.close()
        if p.returncode != 0 or not os.path.exists(out):
            rows[f'_{part}'] = dict(failed=f'exit {p.returncode}, see {os.path.join(out_dir, f"{tag}_{part}.log")}')
            continue
        with open(out, encoding='utf-8') as f:
            rows.update(json.load(f))
    if idle is not None and peak is not None:
        rows['vram device peak'] = dict(above_idle_gib=(peak - idle) / 1024, idle_gib=idle / 1024)
    return rows


COLS = ('wall_p50', 'wall_p95', 'host_ms', 'launches', 'kernels', 'gpu_ms', 'reserved_gib', 'above_idle_gib',
        'load_s', 'warmup_s', 'n')


def cell(vals):
    """Median of the runs, with the min-max spread when there are several."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return '-'
    fmt = (lambda v: f'{v:.2f}') if max(abs(v) for v in vals) < 20 else (lambda v: f'{v:.0f}')
    med = fmt(statistics.median(vals))
    return med if len(vals) == 1 or min(vals) == max(vals) else f'{med} ({fmt(min(vals))}-{fmt(max(vals))})'


def table(runs):
    """runs: {condition: [rows per run]} -> the printed table."""
    names = []
    for rs in runs.values():
        for rows in rs:
            names += [k for k in rows if k not in names]
    lines = []
    for cond, rs in runs.items():
        lines.append(f'\n== {cond} ({len(rs)} run{"s" * (len(rs) > 1)}; median (min-max) over runs)')
        cols = [c for c in COLS if any(c in rows.get(k, {}) for rows in rs for k in names)]
        lines.append(f'{"":34s}' + ''.join(f'{c:>20s}' for c in cols))
        for k in names:
            got = [rows.get(k, {}) for rows in rs]
            notes = sorted({str(g[x]) for g in got for x in ('skipped', 'failed', 'rois', 'gpu') if x in g})
            if notes:
                lines.append(f'{k:34s}' + '; '.join(notes))
                continue
            lines.append(f'{k:34s}' + ''.join(f'{cell([g.get(c) for g in got]):>20s}' for c in cols))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', type=int, default=3, help='repeat everything; the table shows the spread')
    ap.add_argument('--hog', action='store_true', help='only the runs beside a busy Python thread')
    ap.add_argument('--no-hog', action='store_true', help='skip the runs beside a busy Python thread')
    ap.add_argument('--vram-gib', type=float, default=None, help='cap both processes to fit this many GiB in all')
    ap.add_argument('--ctx-gib', type=float, default=0.6, help='a CUDA context per process, outside the allocator')
    ap.add_argument('--cpus', type=int, default=None, help='pin every process to the first N cores')
    ap.add_argument('--face-adapter', default=None)
    ap.add_argument('--svr', default=None, help='SVR-1 checkpoint (default checkpoints/W_s12_svr1_4ch/model.pt)')
    ap.add_argument('--session', default='session_01', help='recorded session for the EMG and the decoder pass')
    ap.add_argument('--cam-log', default=None, help='camera bus log of --session (default logs/rt_speed_all_s1)')
    ap.add_argument('--segments', type=int, default=20, help='camera segments the decoder pass replays')
    ap.add_argument('--reps', type=int, default=3, help='timed segments per tick mode')
    ap.add_argument('--final-reps', type=int, default=10)
    ap.add_argument('--svr-reps', type=int, default=20)
    ap.add_argument('--hog-budget', type=float, default=60.0, help='seconds per segment beside the busy thread')
    ap.add_argument('--py-camera', default=None)
    ap.add_argument('--py-fusion', default=None)
    ap.add_argument('--part', choices=('camera', 'fusion'), default=None, help=argparse.SUPPRESS)
    ap.add_argument('--out', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--cap-gib', type=float, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.part:
        rows = (camera if args.part == 'camera' else fusion)(args)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=1)
        return
    if args.cpus:
        limit(argparse.Namespace(cpus=args.cpus, cap_gib=None, hog=False))
    out_dir = os.path.join(WORK, 'logs', 'rt_win_profile_' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out_dir)
    conds = [c for c in ('alone', 'busy thread') if not (c == 'alone' and args.hog)
             and not (c == 'busy thread' and args.no_hog)]
    runs = {c: [] for c in conds}
    for i in range(args.runs):
        for c in conds:
            runs[c].append(run_once(args, c == 'busy thread', out_dir, f'run{i}_{c.replace(" ", "_")}'))
            print(f'[laptop_profile] run {i + 1}/{args.runs} {c} done', flush=True)
    head = (f'laptop_profile: --vram-gib {args.vram_gib} --cpus {args.cpus} runs {args.runs} '
            f'session {args.session} -> {out_dir}')
    text = head + table(runs)
    print(text)
    with open(os.path.join(out_dir, 'table.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'profile.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'runs': runs}, f, indent=1)


if __name__ == '__main__':
    main()
