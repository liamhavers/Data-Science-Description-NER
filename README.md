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

Two families of NER strategy are implemented (three methods total), compared against
each other on the same posting set:

1. **Traditional ML NER (implemented)** — two stages, both in spaCy:
   - **Rule-based (`src/ner_spacy.py`)**: an `EntityRuler` seeded with a hand-curated
     skills taxonomy (`src/skills_taxonomy.py`, **157 skills** across LANGUAGE, CLOUD,
     DATABASE, LIBRARY, TOOL, METHOD — expanded twice: once using frequency counts from
     the dataset's own `job_skills.csv` column, once more from curating the LLM
     pipeline's novel-candidate bucket below, both times filtering out generic terms
     like "Communication" or near-duplicate rephrasings of existing entries). This is
     the primary, trusted ranking.
   - **Statistical (`src/train_ner.py` / `src/ner_spacy_trained.py`)**: a trained spaCy
     `ner` component, bootstrapped from the rule-based pass applied sentence-by-sentence
     (weak supervision — no hand-labeling). Dev-set F1 came back ~0.998, but that number
     is measuring agreement with its own teacher labels, not true generalization, since
     train and dev were both weak-labeled by the same rule-based ruler. A follow-up scan
     of the model's "novel" spans (things it flagged that aren't in the taxonomy at all)
     confirmed this: a handful of genuinely new phrasings turned up (`a/b tests`,
     `ci / cd`, `machine vision` — since folded into the taxonomy) alongside a lot of
     noise (`jacks`, `probe`, `conceptualize`, `solicit`). Retrained after the taxonomy
     expansion: novel spans dropped from 279 to 140 unique (most of what it used to
     flag as novel is now correctly recognized as known), and what's left is almost
     entirely noise (`amended`, `hcfa`, `stamford`, `junk`) — a good sign the taxonomy
     now covers the corpus well. **Current verdict: useful as a taxonomy-expansion
     discovery tool, not yet reliable as a standalone open-vocabulary extractor** — it'd
     need more diverse training data (full documents, not just sentences; harder
     negatives) to trust unsupervised.
2. **LLM-based extraction (implemented)** — `src/llm_extract.py` runs a local LLM
   (`qwen2.5:7b-instruct` via Ollama, on-GPU) over each posting's raw `job_summary`,
   prompted to return structured `(skill, category)` pairs directly (no gazetteer).
   Chosen over a paid API for this ~12k-posting volume: free and unlimited on local
   hardware, vs. rate-limited free-tier cloud APIs or a real cost on paid ones.
   Self-healing by necessity — a 10+ hour unattended run on a machine that sleeps and
   a resource-constrained WSL2 VM both killed the Ollama server mid-run at least
   twice; the script now auto-restarts it and retries on a dropped connection instead
   of recording a false per-posting result (see `src/llm_extract.py` docstring).
   `src/llm_aggregate.py` normalizes skill names matching a known taxonomy surface
   form to their canonical name (case-insensitive) and buckets everything else as
   "novel." Full-corpus run (12,217/12,217, 1 real error): **strong agreement with the
   rule-based ranking on named tools/languages/platforms** (A/B Testing 3.1% spaCy vs.
   3.3% LLM; PyTorch 4.9% vs. 4.7%; TensorFlow 6.0% vs. 5.8%) but **sharp divergence on
   broad methodology/concept terms**, where the LLM is far more conservative (Machine
   Learning 22.1% vs. 4.6%; Agile 20.8% vs. 2.5%; ETL 15.4% vs. 1.4%). Read: the
   rule-based ruler fires on every literal keyword occurrence including boilerplate
   mentions, while the LLM seems to only report a broad concept when it's a
   substantive focus, not a passing mention. **Trust both methods for specific
   tools/libraries; treat the LLM's methodology-term percentages as "how often this is
   a real focus," not "how often it's mentioned."** The novel bucket originally
   surfaced 19,805 unique spans — curated the clearest, most specific, least ambiguous
   ~65 of them into the taxonomy (Jenkins, Informatica, Azure Data Factory, T-SQL,
   SPSS, PL/SQL, Go, JavaScript, C#, and more), skipping generic soft-skill phrases,
   near-duplicate rephrasings, and off-topic noise from non-DS postings that slipped
   into the dataset. After that expansion the bucket sits at 19,722 (exact-string
   matching only, so it doesn't shrink 1:1 with taxonomy growth — e.g. "amazon web
   services (aws)" won't match the "amazon web services" surface it contains, unlike
   the rule-based ruler's substring matching). Bare `r` still tops it at 1,427
   mentions — the ambiguous single-letter language name deliberately excluded from the
   taxonomy's rule-based matching (and from `go`, `ai`, `lambda`, `glue` for the same
   reason), which the LLM handles fine via context.

### Running the pipeline

```bash
.venv/bin/python src/ingest.py            # download + join the dataset -> data/processed/job_postings.csv
.venv/bin/python src/ner_spacy.py          # rule-based extraction -> results/skill_counts.csv, results/job_skills_extracted.csv
.venv/bin/python src/train_ner.py          # train statistical model on weak labels -> models/spacy_ner/
.venv/bin/python src/ner_spacy_trained.py  # run trained model -> results/skill_counts_trained.csv, results/novel_skill_candidates.csv
.venv/bin/python src/llm_extract.py        # LLM extraction via local Ollama -> results/job_skills_extracted_llm.jsonl (resumable)
.venv/bin/python src/llm_aggregate.py      # aggregate -> results/skill_counts_llm.csv, results/novel_skill_candidates_llm.csv
.venv/bin/python src/combine_rankings.py   # merge all three rankings -> results/skill_counts_combined.csv
.venv/bin/python src/analyze_trends.py     # co-occurrence + posting-date trend -> results/skill_cooccurrence.csv, results/skill_trend_by_day.csv
```

`results/skill_counts.csv` (rule-based) is the trusted ranked demand signal (skill, #
postings mentioning it, % of postings) for specific tools — use this one for
project-selection decisions today. `results/skill_counts_llm.csv` corroborates it on
named tools and adds a second read on broad methodology terms (see caveat above).
`results/skill_counts_combined.csv` merges all three side by side and ranks skills by
how much the rule-based and LLM percentages diverge — useful for spotting which
"in-demand" terms are boilerplate-keyword inflation (Agile 20.8% rule vs. 2.5% LLM,
ETL 15.4% vs. 1.4%) versus a genuine, substantive focus (Python 39.6% vs. 41.9%,
A/B Testing 3.1% vs. 3.3%).
`results/novel_skill_candidates.csv` and `results/novel_skill_candidates_llm.csv` are
worth periodically skimming by hand to find real terms worth adding to the taxonomy.

`results/skill_cooccurrence.csv` (from `src/analyze_trends.py`, using the trusted
rule-based extraction) counts how often pairs of top-40 skills appear together in the
same posting — e.g. Python+SQL co-occur in 3,294 postings, Machine Learning+Python in
2,147 — useful for picking a *combination* of skills to build a project around, not
just a single one. The same script also buckets postings by `first_seen` date for a
day-over-day trend view, but this dataset's `first_seen` only spans 6 days
(2024-01-12 to 2024-01-17), a scrape snapshot rather than a longitudinal crawl — the
resulting swings (e.g. AWS 18.0%-35.5% day to day) are small-sample noise, not a real
trend signal. Kept in place as a placeholder for if/when a wider-dated dataset is
used.

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

All three extraction methods (rule-based spaCy, statistical spaCy, local-LLM via
Ollama) are working end-to-end and compared on the full corpus (see the caveats on
each above), with a combined cross-method ranking and a skill co-occurrence view on
top. Still open: entity normalization beyond the taxonomy's own id-grouping, a real
trend-over-time view (needs a wider-dated dataset — see the `first_seen` caveat
above), and actually picking/starting the next personal project based on the ranked
output (see `CLAUDE.md` for the project timeline and current open questions).

## Project layout

```
data/processed/ joined job postings (gitignored; regenerate with src/ingest.py)
models/          trained statistical NER model (gitignored; regenerate with src/train_ner.py)
src/             pipeline code
  ingest.py             download dataset via kagglehub, join the 3 CSVs
  skills_taxonomy.py    hand-curated skill/tool gazetteer
  ner_spacy.py          rule-based EntityRuler NER pass -> results/
  train_ner.py          bootstraps weak labels from ner_spacy.py, trains models/spacy_ner/
  ner_spacy_trained.py  runs the trained model -> results/ (known + novel skill spans)
  llm_extract.py        LLM extraction via local Ollama -> results/ (resumable, self-healing)
  llm_aggregate.py      aggregates llm_extract.py output -> results/ (known + novel skill spans)
  combine_rankings.py   merges all three methods' rankings -> results/skill_counts_combined.csv
  analyze_trends.py     skill co-occurrence + posting-date trend -> results/ (see trend caveat above)
results/         skill demand rankings, per-posting extractions, novel skill candidates
                 (from both the statistical spaCy model and the LLM), combined
                 cross-method ranking, co-occurrence and trend-by-day views
```
