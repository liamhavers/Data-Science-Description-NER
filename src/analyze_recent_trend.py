"""Real month-over-month skill demand trend, built from the recent (2025-2026)
AI/data job postings gathered by src/ingest_recent.py (Kaggle snapshot) plus
whatever src/fetch_adzuna.py has accumulated since. Unlike
src/analyze_trends.py's day-by-day view over the primary ~2-year-old dataset
(which only spans 6 days and is pure noise), this corpus has real calendar
spread -- but it's thin before late 2025, so months below MIN_POSTINGS are
dropped rather than plotted as if they were a real signal.

Reuses the same rule-based EntityRuler as src/ner_spacy.py so results are on
the same skill taxonomy/canonical names as the rest of the project.
"""

from pathlib import Path

import pandas as pd

from ner_spacy import build_pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BASE_PATH = DATA_DIR / "ai_jobs_recent.csv"
ADZUNA_PATH = DATA_DIR / "adzuna_pulls.jsonl"

MIN_POSTINGS_PER_MONTH = 50
TOP_N_SKILLS = 12


def load_corpus() -> pd.DataFrame:
    # Each source's posted_date is parsed separately, before concatenating --
    # pandas infers a single date format from a string column's first values
    # and coerces every non-matching row to NaT, so parsing the combined
    # column in one pass would silently drop one source's dates depending on
    # concat order (verified: this dropped all live-fetched rows when the
    # base CSV's plain "YYYY-MM-DD" dates came first).
    base = pd.read_csv(BASE_PATH, usecols=["job_title", "posted_date", "job_description"])
    base["posted_date"] = pd.to_datetime(base["posted_date"], utc=True, errors="coerce")
    if ADZUNA_PATH.exists():
        live = pd.read_json(ADZUNA_PATH, lines=True)[["job_title", "posted_date", "job_description"]]
        live["posted_date"] = pd.to_datetime(live["posted_date"], utc=True, errors="coerce")
        base = pd.concat([base, live], ignore_index=True)
        print(f"included {len(live)} live-fetched postings from {ADZUNA_PATH.name}")
    base["posted_date"] = base["posted_date"].dt.tz_localize(None)
    return base.dropna(subset=["posted_date", "job_description"])


def main() -> None:
    df = load_corpus()
    nlp = build_pipeline()

    skill_sets = [
        {ent.ent_id_ for ent in doc.ents}
        for doc in nlp.pipe(df["job_description"].astype(str), batch_size=200)
    ]
    df = df.assign(skill_set=skill_sets, month=df["posted_date"].dt.to_period("M"))

    month_counts = df.groupby("month").size()
    usable_months = month_counts[month_counts >= MIN_POSTINGS_PER_MONTH].index
    dropped = month_counts[month_counts < MIN_POSTINGS_PER_MONTH]
    if len(dropped):
        print(f"dropping {len(dropped)} months below {MIN_POSTINGS_PER_MONTH} postings (too thin to trust): "
              f"{dict(dropped)}")

    usable = df[df["month"].isin(usable_months)]
    all_skills = pd.Series([s for skills in usable["skill_set"] for s in skills]).value_counts()
    top_skills = all_skills.head(TOP_N_SKILLS).index.tolist()

    rows = []
    for month, group in usable.groupby("month"):
        total = len(group)
        for skill in top_skills:
            mentions = int(group["skill_set"].apply(lambda s: skill in s).sum())
            rows.append({"month": str(month), "skill": skill, "postings": total,
                         "mentions": mentions, "pct": round(100 * mentions / total, 1)})
    trend = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trend.to_csv(RESULTS_DIR / "skill_trend_recent_by_month.csv", index=False)

    print(f"\n{len(usable)} postings across {len(usable_months)} usable months -> "
          f"results/skill_trend_recent_by_month.csv")
    print(trend.pivot(index="month", columns="skill", values="pct").to_string())


if __name__ == "__main__":
    main()
