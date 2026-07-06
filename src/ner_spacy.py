"""Rule-based skill NER over job_summary text using a spaCy EntityRuler.

This is the traditional-ML NER path from the README: an EntityRuler seeded
with a hand-curated skills taxonomy (skills_taxonomy.py) runs over each
posting's raw job_summary text independently. job_skills.csv (folded into
ingest.py's output as a reference column) is never consulted here.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language

from skills_taxonomy import TAXONOMY

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RANKING_PATH = RESULTS_DIR / "skill_counts.csv"
PER_POSTING_PATH = RESULTS_DIR / "job_skills_extracted.csv"


def build_pipeline() -> Language:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns = [
        {"label": category, "id": canonical, "pattern": surface}
        for canonical, category, surfaces in TAXONOMY
        for surface in surfaces
    ]
    ruler.add_patterns(patterns)
    return nlp


def main() -> None:
    nlp = build_pipeline()
    df = pd.read_csv(DATA_PATH)

    counts: Counter[str] = Counter()
    extracted_col = []
    for doc in nlp.pipe(df["job_summary"].astype(str), batch_size=200):
        skills = sorted({ent.ent_id_ for ent in doc.ents})
        counts.update(skills)
        extracted_col.append(", ".join(skills))

    df["extracted_skills"] = extracted_col

    ranking = pd.DataFrame(counts.most_common(), columns=["skill", "postings_mentioning"])
    ranking["pct_of_postings"] = (ranking["postings_mentioning"] / len(df) * 100).round(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(RANKING_PATH, index=False)
    df[["job_link", "job_title", "job_level", "job_type", "extracted_skills"]].to_csv(
        PER_POSTING_PATH, index=False
    )

    print(ranking.to_string(index=False))
    print(f"\n{len(df)} postings scanned")
    print(f"ranking -> {RANKING_PATH}")
    print(f"per-posting extractions -> {PER_POSTING_PATH}")


if __name__ == "__main__":
    main()
