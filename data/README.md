# Data

The empirical section uses daily closing levels of the S&P 500 index (SPX) from 3 January 2000 to
31 December 2025. The data are not redistributed here. Download them in either of two ways:

1. **Investing.com**: https://www.investing.com/indices/us-spx-500-historical-data
   - Set the period to daily and the range to 01/01/2000–12/31/2025.
   - Export as CSV; the site may split the range into several files.
   - The columns are `Date, Price, Open, High, Low, Vol., Change %`.
   - The paper used two overlapping files that agree on all 221 common dates.
2. **Yahoo Finance**: ticker `^GSPC`, with columns `Date, Open, High, Low, Close, Adj Close, Volume`.

Put the files in this folder and run, for example:

```bash
python scripts/empirical.py data/SP500_part1.csv data/SP500_part2.csv
```

The reader checks that overlapping files agree. Expected sample: 6,538 log-returns with an
annualised volatility of 19.4%.
