"""Loading S&P 500 daily closes from Investing.com or Yahoo Finance CSV exports.

The data are not redistributed with this repository.  Download them from
https://www.investing.com/indices/us-spx-500-historical-data (one or more files; overlapping
files are merged and checked for agreement) or from Yahoo Finance (ticker ^GSPC).
"""
import csv
import datetime as dt
import numpy as np


def _num(s):
    return float(s.replace(',', '').strip())


def _read(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'{path}: empty file')
    keys = rows[0].keys()
    out = {}
    if 'Price' in keys:                       # Investing.com: Date (MM/DD/YYYY), Price, ...
        for r in rows:
            d = dt.datetime.strptime(r['Date'].strip(), '%m/%d/%Y').date()
            out[d] = _num(r['Price'])
    elif 'Close' in keys or 'Adj Close' in keys:   # Yahoo Finance: Date (YYYY-MM-DD), ..., Close
        col = 'Close' if 'Close' in keys else 'Adj Close'
        for r in rows:
            if r[col] in ('', 'null'):
                continue
            out[dt.date.fromisoformat(r['Date'].strip()[:10])] = _num(r[col])
    else:
        raise ValueError(f'{path}: unrecognised format (columns {list(keys)})')
    return out


def load_closes(*paths, start=None, end=None):
    """Merge one or more CSV files; returns (dates, closes) sorted by date.

    Overlapping dates must agree to within 0.005 index points.
    """
    merged, common = {}, 0
    for p in paths:
        for d, c in _read(p).items():
            if d in merged:
                common += 1
                if abs(merged[d] - c) > 5e-3:
                    raise ValueError(f'files disagree on {d}: {merged[d]} vs {c}')
            merged[d] = c
    dates = sorted(d for d in merged if (start is None or d >= start) and (end is None or d <= end))
    return dates, np.array([merged[d] for d in dates]), common


def log_returns(closes):
    return np.diff(np.log(closes))
