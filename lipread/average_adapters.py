"""Average face adapters into one (a uniform weight soup): adapters trained on the same sentences with different seeds
(lipread/adapt.py --seed) read better averaged than any one of them: three recipe C seeds averaged cut the fused WER
from 19.6 to 17.8 (session 1) and from 26.6 to 24.0 (session 2), each read by a soup trained on the other session, at
the same tick cost. Floating-point tensors are averaged; integer buffers come from the first file.

    venv-live/bin/python -m lipread.average_adapters OUT.pth A_seed0.pth A_seed1.pth A_seed2.pth
"""
import argparse

import torch


def average(states):
    keys = list(states[0])
    if any(list(s) != keys for s in states[1:]):
        raise ValueError('the adapters do not have the same tensors')
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(states[0][k].dtype)
            if states[0][k].is_floating_point() else states[0][k].clone() for k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('adapters', nargs='+')
    a = ap.parse_args(argv)
    soup = average([torch.load(p, map_location='cpu', weights_only=True) for p in a.adapters])
    torch.save(soup, a.out)
    print(f'saved {len(soup)} tensors, the mean of {len(a.adapters)} adapters, to {a.out}')


if __name__ == '__main__':
    main()
