# jobDescNER

Named Entity Recognition (NER) pipeline for mining **data science job descriptions** to
find out which skills, tools, and technologies are currently in demand — and use that
signal to decide what to build/learn next (e.g. "is A/B testing more in-demand than
AWS right now?").

## Goal

Given a corpus of data science job postings, extract structured skill/tool/technology
entities (e.g. `AWS`, `A/B Testing`, `SQL`, `Airflow`, `Docker`, `Spark`) and aggregate
them into a ranked view of demand. The output is meant to directly inform **personal
project selection** — build toward the skills that show up most (or are trending up).

## Data source

[**Data Science Job Postings & Skills (2024)**](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills)
on Kaggle (author: `asaniczka`), a continuously-updated scrape of LinkedIn postings
already scoped to data science roles.

- ~12k+ deduplicated postings (more if duplicates are kept), each with a full raw
  text description — plenty of volume for NER.
- Three CSVs: `job_postings.csv` (title/company/location/level/type metadata),
  `job_summary.csv` (the **raw description text** — this is what we run our own NER
  over), and `job_skills.csv` (the author's own pre-extracted skills).
- `job_skills.csv` is *not* our source of truth — it's kept aside as an optional
  sanity check / comparison baseline for our own extraction, since the point of this
  project is to build that extraction ourselves.
- Fallback if this proves too small after cleaning: [LinkedIn Job Postings
  (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
  (~124k postings across all roles, ~29 columns including a full `description`
  field; filter by title for data science / data analyst / ML engineer roles).

Downloaded in Python via `kagglehub` (requires a Kaggle account + API token at
`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars):

```python
import kagglehub

path = kagglehub.dataset_download("asaniczka/data-science-job-postings-and-skills")
print("Path to dataset files:", path)
```

## Approach

Two NER strategies are supported, and can be compared against each other:

1. **Traditional ML NER (implemented, v1)** — a spaCy `EntityRuler` seeded with a
   hand-curated skills taxonomy (`src/skills_taxonomy.py`), run over each posting's
   raw `job_summary` text independently of the dataset's own `job_skills.csv`. Rule-based
   rather than trained for now; the extracted spans double as a bootstrap for training
   a statistical spaCy/transformer NER component later if the gazetteer proves too
   narrow.
2. **LLM-based extraction (planned)** — prompting an LLM (e.g. Claude) to pull
   structured skill/tool entities out of raw job description text, optionally
   normalizing them against the same canonical skills taxonomy for comparison against
   the spaCy results.

### Running the spaCy pipeline

```bash
.venv/bin/python src/ingest.py      # download + join the dataset -> data/processed/job_postings.csv
.venv/bin/python src/ner_spacy.py   # extract skills -> results/skill_counts.csv, results/job_skills_extracted.csv
```

`results/skill_counts.csv` is the ranked demand signal (skill, # postings mentioning it,
% of postings). `results/job_skills_extracted.csv` is the per-posting extraction,
kept for later analysis (trends by seniority, co-occurrence, etc.).

## Pipeline (planned)

```
raw job postings (scraped / dataset)
        |
        v
  text cleaning / preprocessing
        |
        v
   NER extraction  <-- traditional ML  or  LLM-based
        |
        v
 entity normalization (e.g. "AWS" == "Amazon Web Services")
        |
        v
  aggregation & ranking (frequency, trend over time, co-occurrence)
        |
        v
   analysis / reporting (which skills to prioritize)
```

## Status

Data ingestion and a first (rule-based) spaCy NER pass are working end-to-end. Still
open: LLM-based extraction, entity normalization beyond the taxonomy's own id-grouping,
and trend/co-occurrence analysis (see `CLAUDE.md` for current open questions).

## Project layout

```
data/processed/ joined job postings (gitignored; regenerate with src/ingest.py)
src/            pipeline code
  ingest.py           download dataset via kagglehub, join the 3 CSVs
  skills_taxonomy.py  hand-curated skill/tool gazetteer used by ner_spacy.py
  ner_spacy.py        spaCy EntityRuler NER pass -> results/
results/        skill demand rankings, per-posting extractions
```
