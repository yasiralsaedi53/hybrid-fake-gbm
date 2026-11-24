#!/bin/sh
# Reproduce all tables and figures. Arguments: S&P 500 CSV file(s) for Section 10.
set -e
cd "$(dirname "$0")"
python scripts/table1_closed_form.py
python scripts/table2_3_pde.py
python scripts/calibration_check.py
python scripts/mcconv.py
python scripts/table4_model_risk.py
python scripts/table5_hedging.py
if [ "$#" -gt 0 ]; then python scripts/empirical.py "$@"; else echo "Skipping Section 10: no data files given (see data/README.md)"; fi
