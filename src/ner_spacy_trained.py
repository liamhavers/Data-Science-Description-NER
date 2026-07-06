"""Run the trained statistical NER model over the corpus.

Unlike ner_spacy.py (pure rule-based EntityRuler), this model can in
principle catch skill mentions that don't exactly match a gazetteer surface
form. Entities are split into "known" (matches a surface form already in
skills_taxonomy.py, mapped to its canonical name) vs. "novel" (a span the
model flagged that isn't in the taxonomy at all) so the novel bucket can be
reviewed as candidate additions to the taxonomy.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

from skills_taxonomy import TAXONOMY

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "spacy_ner"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

SURFACE_TO_CANONICAL = {
    surface: canonical for canonical, _category, surfaces in TAXONOMY for surface in surfaces
}


def normalize(span_text: str) -> tuple[str, str]:
    key = span_text.strip().lower()
    if key in SURFACE_TO_CANONICAL:
        return SURFACE_TO_CANONICAL[key], "known"
    return span_text.strip(), "novel"


def main() -> None:
    nlp = spacy.load(MODEL_DIR)
    df = pd.read_csv(DATA_PATH)

    known_counts: Counter[str] = Counter()
    novel_counts: Counter[str] = Counter()

    for doc in nlp.pipe(df["job_summary"].astype(str), batch_size=200):
        seen_known, seen_novel = set(), set()
        for ent in doc.ents:
            name, source = normalize(ent.text)
            if source == "known":
                seen_known.add(name)
            else:
                seen_novel.add(name.lower())
        known_counts.update(seen_known)
        novel_counts.update(seen_novel)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    known_df = pd.DataFrame(known_counts.most_common(), columns=["skill", "postings_mentioning"])
    known_df["pct_of_postings"] = (known_df["postings_mentioning"] / len(df) * 100).round(1)
    known_df.to_csv(RESULTS_DIR / "skill_counts_trained.csv", index=False)

    novel_df = pd.DataFrame(novel_counts.most_common(200), columns=["candidate_skill", "postings_mentioning"])
    novel_df.to_csv(RESULTS_DIR / "novel_skill_candidates.csv", index=False)

    print(f"{len(df)} postings scanned with the trained model")
    print(f"known skills found: {len(known_df)} (-> {RESULTS_DIR / 'skill_counts_trained.csv'})")
    print(f"novel candidate spans: {len(novel_counts)} unique (-> {RESULTS_DIR / 'novel_skill_candidates.csv'})")
    print("\nTop 30 novel candidates (model-flagged spans not in the taxonomy):")
    print(novel_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
