#!/usr/bin/env bash
# Weekly recent-data refresh: pull new Adzuna postings, fold them into the
# month-over-month trend, and re-render the chart. Invoked by cron (see
# CLAUDE.md for the crontab line) -- run manually any time with
# ./scripts/run_weekly.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== $(date -Iseconds) ==="
.venv/bin/python src/fetch_adzuna.py
.venv/bin/python src/analyze_recent_trend.py
.venv/bin/python src/plot_trends.py
