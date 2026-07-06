"""Download the data-science job postings dataset and join it into one table.

Source: https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills
Splits the data across job_postings.csv (metadata), job_summary.csv (raw
description text), and job_skills.csv (the author's own pre-extracted skills,
kept here only as a reference column, not as NER ground truth).
"""

from pathlib import Path

import kagglehub
import pandas as pd

DATASET = "asaniczka/data-science-job-postings-and-skills"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"

COLUMNS = [
    "job_link",
    "job_title",
    "company",
    "job_location",
    "job_level",
    "job_type",
    "first_seen",
    "job_summary",
    "job_skills",
]


def download() -> Path:
    return Path(kagglehub.dataset_download(DATASET))


def load_and_join(dataset_dir: Path) -> pd.DataFrame:
    postings = pd.read_csv(dataset_dir / "job_postings.csv").drop_duplicates("job_link")
    summary = pd.read_csv(dataset_dir / "job_summary.csv").drop_duplicates("job_link")
    skills = pd.read_csv(dataset_dir / "job_skills.csv").drop_duplicates("job_link")

    df = postings.merge(summary, on="job_link", how="left").merge(skills, on="job_link", how="left")
    df = df.dropna(subset=["job_summary"])
    return df[COLUMNS]


def main() -> None:
    dataset_dir = download()
    df = load_and_join(dataset_dir)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Joined {len(df)} postings with descriptions -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
