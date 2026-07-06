# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project purpose

This project uses Named Entity Recognition (NER) — traditional ML and/or LLM-based —
to extract skills/tools/technologies from **data science job descriptions**. The end
goal is a ranked view of in-demand skills that the user uses to decide what to build
next (e.g. deciding between an A/B testing project vs. an AWS project based on which
shows up more in real postings).

This is a personal learning/portfolio project, not production infrastructure. Favor
clarity and easy iteration (notebooks, small scripts) over heavy engineering.

## Current stage

Data ingestion, rule-based spaCy NER, and a first statistical spaCy NER model are all
built and verified working. Key decisions:

- **Data source (decided)**: Kaggle's [`asaniczka/data-science-job-postings-and-skills`]
  (https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills),
  downloaded via `kagglehub`. Credentials live at `~/.kaggle/access_token` (Kaggle's
  newer single-token format, not the legacy `kaggle.json`). Use `job_summary.csv` (raw
  description text) as the NER input. `job_skills.csv` holds the author's own
  pre-extracted skills — treat it as an optional comparison baseline, not ground truth.
  Fallback if row count after cleaning proves too small: `arshkon/linkedin-job-postings`
  (~124k rows, all roles, filter by title).
- **NER approach (spaCy stages decided, more to come)**:
  - Rule-based spaCy `EntityRuler` (`src/ner_spacy.py`), seeded by the taxonomy. No
    training data needed, immediately runnable, results verified sane (SQL/Python/AWS
    top the ranking, A/B Testing ~3%). **This is the trusted output** —
    `results/skill_counts.csv`.
  - Statistical spaCy `ner` component (`src/train_ner.py` trains it, weak-labeled from
    the rule-based pass run sentence-by-sentence; `src/ner_spacy_trained.py` runs it).
    Dev F1 ~0.998, but that's agreement-with-teacher, not a generalization measure
    (train/dev labels share the same source). Verified this distinction empirically by
    checking the model's "novel" spans (predictions outside the taxonomy) across the
    full corpus: found a few real misses now folded into the taxonomy (`a/b tests`,
    `ci / cd`, `machine vision`) plus a lot of noise (`jacks`, `probe`, `conceptualize`).
    **Don't present this model's dev F1 as a quality signal without this caveat**, and
    don't treat its "novel" bucket as ready to use unfiltered — it needs human review
    per term. It's a discovery aid for expanding the taxonomy, not yet a trustworthy
    standalone extractor.
  - An LLM-based extraction pass (per README) is still planned, to be compared against
    both of the above on the same posting set.
- **Skills taxonomy (~90 entries, still growing)**: hand-curated in
  `src/skills_taxonomy.py` as `(canonical_name, category, [surface forms])` tuples —
  category becomes the spaCy entity label, canonical_name is looked up via
  `ent.ent_id_` so spelling variants collapse to one skill. Expanded once already using
  frequency counts from `job_skills.csv` as an *empirical guide only* (which specific
  tools are actually mentioned a lot) — generic non-actionable terms from that column
  (e.g. "Communication", "Problem Solving", "Data Analysis") were deliberately excluded;
  it is still not used as ground truth for any posting's extraction. Expand this file
  directly when a new skill/tool needs tracking (including reviewed hits from
  `results/novel_skill_candidates.csv`); keep one skill's variants together in one
  tuple, and rerun `src/ner_spacy.py` after editing it.

## Conventions

- Python-first project (matches the parent `pythonProjects` workspace). Dependencies
  in `requirements.txt`, installed into a project-local `.venv` (gitignored) — run
  scripts via `.venv/bin/python`, not a system/global interpreter.
- `data/` (raw downloads + `data/processed/`) and `models/` (trained NER model) are
  gitignored — both fully regenerable via `src/ingest.py` / `src/train_ner.py` and
  shouldn't be committed.
- Prefer small, inspectable scripts/notebooks over a large framework — this is an
  analysis project, and results (which skills are in demand) matter more than
  pipeline sophistication.

## Working with the user

- The user's practical goal is *decision-making support* for future projects, not a
  polished product. When in doubt, optimize for "does this tell me something
  actionable about what to build next," not for engineering completeness.
- When multiple NER approaches are implemented, keep their outputs comparable
  (same entity schema/format) so results can be evaluated side by side.
