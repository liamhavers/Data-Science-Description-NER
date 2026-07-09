"""Aggregate the LLM extraction results (src/llm_extract.py output) into a
ranking comparable to the spaCy pipelines' results/skill_counts*.csv.

Skill names matching a known taxonomy surface form (case-insensitive) are
normalized to their canonical name; everything else is kept as the LLM's own
raw string in a separate "novel" ranking, since the LLM finds real skills the
hand-curated taxonomy doesn't cover (see the quality check in CLAUDE.md).
Category labels from the LLM are inconsistent across calls (e.g. Spark tagged
LIBRARY in one posting, METHOD in another), so they aren't used for grouping.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from skills_taxonomy import TAXONOMY

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EXTRACTED_PATH = RESULTS_DIR / "job_skills_extracted_llm.jsonl"

SURFACE_TO_CANONICAL = {
    surface: canonical for canonical, _category, surfaces in TAXONOMY for surface in surfaces
}


def normalize(skill_text: str) -> tuple[str, str]:
    key = skill_text.strip().lower()
    if key in SURFACE_TO_CANONICAL:
        return SURFACE_TO_CANONICAL[key], "known"
    return skill_text.strip(), "novel"


def main() -> None:
    known_counts: Counter[str] = Counter()
    novel_counts: Counter[str] = Counter()
    n_postings = 0
    n_errors = 0

    with open(EXTRACTED_PATH) as f:
        for line in f:
            rec = json.loads(line)
            n_postings += 1
            if rec.get("error"):
                n_errors += 1
                continue
            seen_known, seen_novel = set(), set()
            for entry in rec["skills"]:
                name, source = normalize(entry["skill"])
                (seen_known if source == "known" else seen_novel).add(name.lower() if source == "novel" else name)
            known_counts.update(seen_known)
            novel_counts.update(seen_novel)

    known_df = pd.DataFrame(known_counts.most_common(), columns=["skill", "postings_mentioning"])
    known_df["pct_of_postings"] = (known_df["postings_mentioning"] / n_postings * 100).round(1)
    known_df.to_csv(RESULTS_DIR / "skill_counts_llm.csv", index=False)

    novel_df = pd.DataFrame(novel_counts.most_common(300), columns=["candidate_skill", "postings_mentioning"])
    novel_df.to_csv(RESULTS_DIR / "novel_skill_candidates_llm.csv", index=False)

    print(f"{n_postings} postings processed ({n_errors} errors)")
    print(f"known skills: {len(known_df)} -> results/skill_counts_llm.csv")
    print(f"novel candidates: {len(novel_counts)} -> results/novel_skill_candidates_llm.csv")
    print("\nTop 20 known skills:")
    print(known_df.head(20).to_string(index=False))
    print("\nTop 20 novel candidates:")
    print(novel_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
