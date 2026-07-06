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

Scaffolding only — no code yet. Treat early requests as "help me stand this up," not
"extend the existing pipeline." Key open decisions (ask the user if a task depends on
one and it's not yet settled; update this file once decided):

- **Data source (decided)**: Kaggle's [`asaniczka/data-science-job-postings-and-skills`]
  (https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills),
  downloaded via `kagglehub`. Use `job_summary.csv` (raw description text) as the NER
  input. `job_skills.csv` holds the author's own pre-extracted skills — treat it as an
  optional comparison baseline, not ground truth, since building the extraction
  ourselves is the point of the project. Fallback if row count after cleaning proves
  too small: `arshkon/linkedin-job-postings` (~124k rows, all roles, filter by title).
- **NER approach**: spaCy custom/fine-tuned pipeline vs. fine-tuned transformer
  (token classification) vs. LLM prompt-based extraction (e.g. Claude). These may be
  built side by side for comparison rather than choosing one exclusively.
- **Skills taxonomy**: whether entities get normalized against a canonical list (so
  "AWS" / "Amazon Web Services" / "amazon-web-services" collapse to one entity) and
  where that taxonomy comes from (hand-curated vs. derived from the data).

## Conventions

- Python-first project (matches the parent `pythonProjects` workspace).
- No package manager / dependency file exists yet — when the first real code is
  added, set one up (e.g. `pyproject.toml` or `requirements.txt`) rather than
  installing ad hoc.
- Keep raw scraped/downloaded data out of git if it's large or has redistribution
  restrictions (add a `.gitignore` for `data/raw/` when data work starts).
- Prefer small, inspectable scripts/notebooks over a large framework — this is an
  analysis project, and results (which skills are in demand) matter more than
  pipeline sophistication.

## Working with the user

- The user's practical goal is *decision-making support* for future projects, not a
  polished product. When in doubt, optimize for "does this tell me something
  actionable about what to build next," not for engineering completeness.
- When multiple NER approaches are implemented, keep their outputs comparable
  (same entity schema/format) so results can be evaluated side by side.
