import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def parser(description, paths):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--paths', type=int, default=paths,
                   help=f'number of Monte Carlo paths (paper: {paths})')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--quick', action='store_true',
                   help='one tenth of the paths, for a fast check')
    return p


def n_paths(args):
    return max(1000, args.paths // 10) if args.quick else args.paths


class Report:
    """Prints lines and writes them to results/<name>.txt."""

    def __init__(self, name):
        self.path = os.path.join(RESULTS, name + '.txt')
        self.lines = []
        self.t0 = time.time()

    def __call__(self, line=''):
        print(line, flush=True)
        self.lines.append(line)

    def save(self):
        self(f'\n(run time {time.time() - self.t0:.0f} s)')
        with open(self.path, 'w') as fh:
            fh.write('\n'.join(self.lines) + '\n')
        print(f'written {os.path.relpath(self.path, ROOT)}')
