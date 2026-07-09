"""Skill co-occurrence and posting-date trend analysis, built on top of the
trusted rule-based extraction (results/job_skills_extracted.csv).

Trend-over-time caveat: this dataset's `first_seen` column spans only ~6 days
(2024-01-12 to 2024-01-17) -- it's a scrape snapshot, not a longitudinal
crawl. Day-over-day percentages here are noisy sample-size artifacts, NOT a
real "is this skill trending" signal. Kept as a placeholder that becomes
meaningful if/when a wider-dated dataset is used; don't treat its output as
an actionable finding today.
"""

import itertools
from collections import Counter
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"

TOP_N_FOR_COOCCURRENCE = 40
TOP_N_FOR_TREND = 10


def load_skill_sets() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "job_skills_extracted.csv")
    df["skill_set"] = df["extracted_skills"].fillna("").apply(
        lambda s: {skill.strip() for skill in s.split(",") if skill.strip()}
    )
    return df[["job_link", "skill_set"]]


def cooccurrence(skill_sets: pd.Series) -> pd.DataFrame:
    top_skills = set(pd.read_csv(RESULTS_DIR / "skill_counts.csv").head(TOP_N_FOR_COOCCURRENCE)["skill"])
    pair_counts = Counter()
    for skills in skill_sets:
        present = sorted(skills & top_skills)
        for a, b in itertools.combinations(present, 2):
            pair_counts[(a, b)] += 1

    rows = [{"skill_a": a, "skill_b": b, "postings_together": c} for (a, b), c in pair_counts.items()]
    return pd.DataFrame(rows).sort_values("postings_together", ascending=False)


def trend_by_day(skill_sets: pd.DataFrame) -> pd.DataFrame:
    postings = pd.read_csv(DATA_PATH, usecols=["job_link", "first_seen"])
    merged = postings.merge(skill_sets, on="job_link", how="inner")

    top_skills = pd.read_csv(RESULTS_DIR / "skill_counts.csv").head(TOP_N_FOR_TREND)["skill"].tolist()
    rows = []
    for day, group in merged.groupby("first_seen"):
        total = len(group)
        for skill in top_skills:
            mentions = int(group["skill_set"].apply(lambda s: skill in s).sum())
            rows.append({
                "first_seen": day,
                "skill": skill,
                "postings_that_day": total,
                "mentions": mentions,
                "pct": round(100 * mentions / total, 1),
            })
    return pd.DataFrame(rows)


def main() -> None:
    skill_sets = load_skill_sets()

    co = cooccurrence(skill_sets["skill_set"])
    co.to_csv(RESULTS_DIR / "skill_cooccurrence.csv", index=False)
    print(f"Co-occurrence among top {TOP_N_FOR_COOCCURRENCE} skills -> results/skill_cooccurrence.csv")
    print(co.head(15).to_string(index=False))

    trend = trend_by_day(skill_sets)
    trend.to_csv(RESULTS_DIR / "skill_trend_by_day.csv", index=False)
    n_days = trend["first_seen"].nunique()
    print(f"\nTrend by day -> results/skill_trend_by_day.csv")
    print(f"CAVEAT: only {n_days} distinct days in this snapshot dataset -- not a real trend signal, see module docstring.")
    print(trend.pivot(index="first_seen", columns="skill", values="pct").to_string())


if __name__ == "__main__":
    main()
