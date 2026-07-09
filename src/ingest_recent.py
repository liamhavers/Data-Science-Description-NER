"""Downloads the freshest real dataset found for recent (2025-2026) AI/data job
postings: atharvasoundankar/ai-job-market-global-2026 on Kaggle, sourced from
the live Adzuna and USAJobs APIs. Unlike the primary dataset (a single scrape
from ~2 years ago), this one has real posted_date values reaching into 2026 --
but it's small (~5.8k rows) and heavily weighted toward its most recent few
months, so treat it as a recent-months snapshot to extend forward via
src/fetch_adzuna.py, not as deep historical trend data on its own.

Deduplicates on job_description: the source dataset repeats the same posting
once per city it's listed in, which would otherwise double-count skill
mentions.
"""

from pathlib import Path

import kagglehub
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "ai_jobs_recent.csv"

COLUMNS = ["job_title", "country", "posted_date", "job_description", "required_skills", "source"]


def main() -> None:
    path = kagglehub.dataset_download("atharvasoundankar/ai-job-market-global-2026")
    df = pd.read_csv(Path(path) / "ai_jobs_global.csv")

    before = len(df)
    df = df.drop_duplicates(subset="job_description")
    print(f"{before} rows -> {len(df)} after deduplicating repeated postings")

    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce").dt.date
    before = len(df)
    df = df.dropna(subset=["posted_date"])
    print(f"dropped {before - len(df)} rows with an unparseable posted_date")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(OUT_PATH, index=False)

    print(f"date range: {df['posted_date'].min()} to {df['posted_date'].max()}")
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
