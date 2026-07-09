"""Renders results/skill_trend_recent_by_month.csv (from src/analyze_recent_trend.py)
as a line chart -- the CSV's pivoted numbers are real, but a table of ~12
skills x 14 months is hard to eyeball for direction/inflection points, which
is the whole point of having a trend in the first place.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available when run from cron
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TREND_PATH = RESULTS_DIR / "skill_trend_recent_by_month.csv"
OUT_PATH = RESULTS_DIR / "skill_trend_recent_by_month.png"


def main() -> None:
    trend = pd.read_csv(TREND_PATH)
    pivot = trend.pivot(index="month", columns="skill", values="pct").sort_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    for skill in pivot.columns:
        ax.plot(pivot.index, pivot[skill], marker="o", markersize=3, label=skill)

    ax.set_title("Skill demand trend, recent-data corpus (% of postings mentioning skill)")
    ax.set_xlabel("Month")
    ax.set_ylabel("% of postings")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
    fig.tight_layout()

    fig.savefig(OUT_PATH, dpi=150)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
