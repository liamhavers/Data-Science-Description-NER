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

Data ingestion (`src/ingest.py`) and a first spaCy NER pass (`src/ner_spacy.py` +
`src/skills_taxonomy.py`) are built and verified working. Key decisions:

- **Data source (decided)**: Kaggle's [`asaniczka/data-science-job-postings-and-skills`]
  (https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills),
  downloaded via `kagglehub`. Credentials live at `~/.kaggle/access_token` (Kaggle's
  newer single-token format, not the legacy `kaggle.json`). Use `job_summary.csv` (raw
  description text) as the NER input. `job_skills.csv` holds the author's own
  pre-extracted skills — treat it as an optional comparison baseline, not ground truth.
  Fallback if row count after cleaning proves too small: `arshkon/linkedin-job-postings`
  (~124k rows, all roles, filter by title).
- **NER approach (v1 decided, more to come)**: started with a rule-based spaCy
  `EntityRuler` (`src/ner_spacy.py`) seeded by a hand-curated gazetteer
  (`src/skills_taxonomy.py`) — no training data needed, immediately runnable, results
  verified sane (SQL/Python/AWS top the ranking, A/B Testing ~3%). This is a bootstrap,
  not the final word: a trained statistical NER component (spaCy or transformer) and
  an LLM-based extraction pass (per README) are still planned, to be compared against
  this baseline on the same posting set.
- **Skills taxonomy (v1 decided)**: hand-curated in `src/skills_taxonomy.py` as
  `(canonical_name, category, [surface forms])` tuples — category becomes the spaCy
  entity label, canonical_name is looked up via `ent.ent_id_` so spelling variants
  collapse to one skill. Deliberately *not* harvested from `job_skills.csv` — that
  file's terms are often too generic (e.g. "Programming", "Optimization") for the
  actionable, specific-tool signal the user wants. Expand this file directly when a
  new skill/tool needs tracking; keep one skill's variants together in one tuple.

## Conventions

- Python-first project (matches the parent `pythonProjects` workspace). Dependencies
  in `requirements.txt`, installed into a project-local `.venv` (gitignored) — run
  scripts via `.venv/bin/python`, not a system/global interpreter.
- `data/` (raw downloads + `data/processed/`) is gitignored — it's fully
  regenerable via `src/ingest.py` and shouldn't be committed.
- Prefer small, inspectable scripts/notebooks over a large framework — this is an
  analysis project, and results (which skills are in demand) matter more than
  pipeline sophistication.

## Working with the user

- The user's practical goal is *decision-making support* for future projects, not a
  polished product. When in doubt, optimize for "does this tell me something
  actionable about what to build next," not for engineering completeness.
- When multiple NER approaches are implemented, keep their outputs comparable
  (same entity schema/format) so results can be evaluated side by side.
