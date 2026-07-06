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

1. **Traditional ML NER** — e.g. spaCy (`en_core_web_*` + a custom/fine-tuned NER
   component) or a fine-tuned transformer (e.g. BERT/DistilBERT token classification)
   trained on a labeled set of job description spans.
2. **LLM-based extraction** — prompting an LLM (e.g. Claude) to pull structured
   skill/tool entities out of raw job description text, optionally normalizing them
   against a canonical skills taxonomy.

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

Early stage — project scaffolding only. Data source, taxonomy, and NER approach are
still being decided (see `CLAUDE.md` for current open questions).

## Project layout (planned)

```
data/           raw and processed job posting data
notebooks/      exploratory analysis
src/            pipeline code (ingestion, NER, normalization, aggregation)
models/         trained/fine-tuned NER models (if using traditional ML)
results/        skill demand rankings, charts, reports
```
