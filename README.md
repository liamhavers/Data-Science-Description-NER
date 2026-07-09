# jobDescNER

Named Entity Recognition (NER) pipeline that tracks which skills, tools, and
technologies are in demand in **data science / AI job postings** — and, because that
demand shifts quickly, tracks **how it's changing over time** — to guide what to
build/learn next (e.g. "is A/B testing more in-demand than AWS right now, and is
either one rising or falling?").

## Goal

This started as a one-off question: extract skill/tool/technology entities from a job
posting corpus, rank them by frequency, and use that single ranking to pick the next
personal project. That ranking is still here and still trusted — see
`results/skill_counts.csv` — but a single point-in-time snapshot goes stale fast in a
field that moves this quickly, and there's no urgency to decide on a next project
right now (already two in progress). So the goal has grown into keeping a **running,
month-over-month view of skill demand**, so that whenever the next project decision
does come up, there's an up-to-date trend to look at instead of a two-year-old
snapshot.

That means two different questions, answered by two different pipelines:

- **"What's in demand right now?"** — the original snapshot corpus, extracted three
  independent ways (rule-based / statistical / LLM) and cross-compared. Trusted
  output: `results/skill_counts.csv`, cross-checked against
  `results/skill_counts_combined.csv`.
- **"Is it rising or falling?"** — an actively-refreshed recent-data corpus, extracted
  with the same rule-based method and aggregated by month. Output:
  `results/skill_trend_recent_by_month.csv`.

## Approach

### Data sourcing

Two separate corpora feed the two questions above — a single dataset can't answer
both, because ranking "what's in demand" wants volume and stability, while "is it
trending" wants real calendar spread across many months.

**Snapshot corpus** ([`asaniczka/data-science-job-postings-and-skills`](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills)
on Kaggle) — ~12k deduplicated LinkedIn postings scoped to data science roles, each
with a full raw text description. Three CSVs: `job_postings.csv` (metadata),
`job_summary.csv` (the raw description text NER runs over), and `job_skills.csv` (the
author's own pre-extracted skills — kept aside as an optional comparison baseline,
never treated as ground truth). Downloaded via `kagglehub` (requires a Kaggle account
+ API token). Its `first_seen` field turned out to span only 6 days
(2024-01-12 to 2024-01-17) once checked directly — a single scrape, not a
longitudinal crawl, so it's good for a demand ranking but useless for a trend. A
considered fallback, `arshkon/linkedin-job-postings` (~124k postings, all roles), had
the same problem when checked directly — half its rows dated within a 2-day
window — so it wasn't adopted either.

**Recent-data corpus** (for the trend question) — since no ready-made dataset with
real dates into 2026 and full posting volume existed (most "2020-2026 AI job market"
datasets found while searching turned out to be synthetic — one candidate had rows
dated *November 2026*, in the future, with only boolean skill flags and no real
text), this pipeline builds its own from two pieces:
- [`atharvasoundankar/ai-job-market-global-2026`](https://www.kaggle.com/datasets/atharvasoundankar/ai-job-market-global-2026)
  (`src/ingest_recent.py`) — real postings sourced from the live Adzuna and USAJobs
  APIs, genuine `posted_date` values reaching into February 2026 (verified directly).
  Small (5,773 rows, 3,750 unique after deduplicating same-posting-different-city
  repeats) and its own update pipeline had stalled at Feb 2026 as of when checked
  despite an advertised weekly refresh.
- A **live Adzuna API fetch** (`src/fetch_adzuna.py`) to extend the series forward
  from here — requires registering a free Adzuna developer account and re-running
  periodically (e.g. weekly); see the module docstring for setup.

This corpus's job-title mix (Data Scientist / AI Engineer / ML Engineer-heavy) is
narrower than the snapshot corpus's broader data-science scope, so **absolute
percentages between the two corpora aren't comparable** — e.g. AWS/Azure read far
lower in the recent corpus (0-3%) than in the snapshot ranking (16-21%), most likely
reflecting the title mix, not an actual demand drop. Use the snapshot corpus for
absolute current-demand ranking, the recent corpus only for within-corpus
month-over-month movement.

### Extraction: three NER methods, compared

Both corpora are extracted with the same rule-based method; the snapshot corpus is
additionally extracted two more independent ways so the ranking can be
cross-checked, not taken on faith:

1. **Rule-based (`src/ner_spacy.py`)** — a spaCy `EntityRuler` seeded with a
   hand-curated skills taxonomy (`src/skills_taxonomy.py`, **157 skills** across
   LANGUAGE, CLOUD, DATABASE, LIBRARY, TOOL, METHOD — expanded twice, first from
   `job_skills.csv` frequency counts, then from curating the LLM pipeline's
   novel-candidate bucket below, both times filtering out generic terms like
   "Communication"). No training data needed, immediately runnable. **This is the
   primary, trusted extraction method**, used for both corpora.
2. **Statistical (`src/train_ner.py` / `src/ner_spacy_trained.py`)** — a trained
   spaCy `ner` component, bootstrapped from the rule-based pass applied
   sentence-by-sentence (weak supervision, no hand-labeling). Dev-set F1 came back
   ~0.998, but that's agreement with its own teacher labels, not true generalization,
   since train and dev share the same weak-label source. A scan of the model's
   "novel" spans (predictions outside the taxonomy) confirmed this: a handful of
   genuinely new phrasings turned up (`a/b tests`, `ci / cd`, `machine vision` —
   since folded into the taxonomy) alongside a lot of noise (`jacks`, `probe`,
   `conceptualize`). Retrained after the taxonomy expansion: novel spans dropped from
   279 to 140 unique, and what's left is almost entirely noise (`amended`, `hcfa`,
   `stamford`) — a good sign the taxonomy covers the corpus well. **Verdict: useful
   as a taxonomy-expansion discovery tool, not yet reliable as a standalone
   open-vocabulary extractor.**
3. **LLM-based (`src/llm_extract.py` / `src/llm_aggregate.py`)** — a local LLM
   (`qwen2.5:7b-instruct` via Ollama, on-GPU) reads each posting's raw text and
   returns structured `(skill, category)` pairs directly, no gazetteer. Chosen over a
   paid API for this posting volume: free and unlimited on local hardware. Made
   self-healing after a 10+ hour unattended run got killed mid-way by both a
   sleeping host and a resource-constrained WSL2 VM — it now auto-restarts the
   Ollama server and retries on a dropped connection instead of recording a false
   per-posting result (see the module docstring). Full-corpus result: **strong
   agreement with the rule-based ranking on named tools/languages/platforms**
   (A/B Testing 3.1% rule vs. 3.3% LLM; PyTorch 4.9% vs. 4.7%) but **sharp
   divergence on broad methodology terms**, where the LLM is far more conservative
   (Machine Learning 22.1% vs. 4.6%; Agile 20.8% vs. 2.5%). Read: the rule-based
   ruler fires on every literal keyword occurrence including boilerplate mentions,
   while the LLM only reports a broad concept when it's a substantive focus. **Use
   the LLM numbers for "how often is this a real focus"; use the rule-based numbers
   for "how often is this word present."**

### Aggregation, comparison & trend

- `results/skill_counts_combined.csv` (`src/combine_rankings.py`) merges all three
  snapshot-corpus rankings side by side and ranks skills by how much the rule-based
  and LLM percentages diverge — useful for spotting boilerplate-keyword inflation
  (Agile 20.8% rule vs. 2.5% LLM) versus genuine substantive focus (Python 39.6% vs.
  41.9%).
- `results/skill_cooccurrence.csv` (`src/analyze_trends.py`) counts how often pairs
  of top-40 skills appear together in the same snapshot-corpus posting — e.g.
  Python+SQL together in 3,294 postings — useful for picking a *combination* of
  skills to build toward, not just a single one. The same script's day-by-day
  breakdown of the snapshot corpus (`results/skill_trend_by_day.csv`) is a dead end,
  kept only as a documented placeholder — 6 days of `first_seen` spread produces
  pure sampling noise, not a trend.
- `results/skill_trend_recent_by_month.csv` (`src/analyze_recent_trend.py`) is the
  real trend signal — the recent corpus's postings run through the same rule-based
  ruler, grouped by month, with any month under 50 postings dropped as too thin to
  trust. First run (recent-corpus base data only, before any live Adzuna pulls):
  3,513 postings across 9 usable months (May 2025 - Feb 2026), showing genuine
  movement — Machine Learning holding steady ~15-24%, Generative AI/LLMs
  fluctuating 3-11%, Computer Vision drifting down from ~8-9% toward ~1-3% by late
  2025.
- `results/novel_skill_candidates.csv` and `results/novel_skill_candidates_llm.csv`
  are worth periodically skimming by hand to find real terms worth adding to the
  taxonomy.

## Running the pipeline

```bash
# Snapshot corpus -> current-demand ranking
.venv/bin/python src/ingest.py            # download + join the dataset -> data/processed/job_postings.csv
.venv/bin/python src/ner_spacy.py          # rule-based extraction -> results/skill_counts.csv, results/job_skills_extracted.csv
.venv/bin/python src/train_ner.py          # train statistical model on weak labels -> models/spacy_ner/
.venv/bin/python src/ner_spacy_trained.py  # run trained model -> results/skill_counts_trained.csv, results/novel_skill_candidates.csv
.venv/bin/python src/llm_extract.py        # LLM extraction via local Ollama -> results/job_skills_extracted_llm.jsonl (resumable)
.venv/bin/python src/llm_aggregate.py      # aggregate -> results/skill_counts_llm.csv, results/novel_skill_candidates_llm.csv
.venv/bin/python src/combine_rankings.py   # merge all three rankings -> results/skill_counts_combined.csv
.venv/bin/python src/analyze_trends.py     # co-occurrence + posting-date trend -> results/skill_cooccurrence.csv, results/skill_trend_by_day.csv

# Recent corpus -> trend-over-time
.venv/bin/python src/ingest_recent.py         # download recent Kaggle snapshot -> data/processed/ai_jobs_recent.csv
.venv/bin/python src/fetch_adzuna.py          # live-fetch current postings (needs your own Adzuna API key) -> data/processed/adzuna_pulls.jsonl
.venv/bin/python src/analyze_recent_trend.py  # month-over-month skill trend -> results/skill_trend_recent_by_month.csv
```

## Pipeline

```
snapshot corpus (one scrape)         recent corpus (Kaggle base + live Adzuna pulls)
        |                                         |
        v                                         v
  rule-based / statistical / LLM NER        rule-based NER (same taxonomy)
        |                                         |
        v                                         v
 cross-method ranking + co-occurrence      month-over-month aggregation
        |                                         |
        v                                         v
  "what's in demand right now"              "is it rising or falling"
                        \                    /
                         v                  v
                    project-selection decision
```

## Status

All three extraction methods (rule-based spaCy, statistical spaCy, local-LLM via
Ollama) are working end-to-end and cross-compared on the snapshot corpus, with a
combined ranking and co-occurrence view on top. The recent-data trend pipeline is
running and already shows real month-over-month movement (May 2025-Feb 2026), though
still thin before late 2025 and not yet extended by any live Adzuna pulls. Open
items: registering an Adzuna API key to start extending the trend series forward via
`src/fetch_adzuna.py`, entity normalization beyond the taxonomy's own id-grouping,
and — once the trend pipeline has accumulated enough live data to be worth
consulting — actually picking the next personal project (see `CLAUDE.md` for the
full project timeline).

## Project layout

```
data/processed/ joined/downloaded job postings (gitignored; regenerate with src/ingest.py, src/ingest_recent.py)
models/          trained statistical NER model (gitignored; regenerate with src/train_ner.py)
src/             pipeline code
  ingest.py               download the snapshot dataset via kagglehub, join the 3 CSVs
  skills_taxonomy.py      hand-curated skill/tool gazetteer
  ner_spacy.py            rule-based EntityRuler NER pass -> results/
  train_ner.py            bootstraps weak labels from ner_spacy.py, trains models/spacy_ner/
  ner_spacy_trained.py    runs the trained model -> results/ (known + novel skill spans)
  llm_extract.py          LLM extraction via local Ollama -> results/ (resumable, self-healing)
  llm_aggregate.py        aggregates llm_extract.py output -> results/ (known + novel skill spans)
  combine_rankings.py     merges all three methods' rankings -> results/skill_counts_combined.csv
  analyze_trends.py       skill co-occurrence + posting-date trend -> results/ (day-trend is a noise placeholder)
  ingest_recent.py        downloads the recent (real, dated into 2026) Kaggle snapshot -> data/processed/
  fetch_adzuna.py         live-fetches current postings via the Adzuna API (needs your own key) -> data/processed/
  analyze_recent_trend.py month-over-month skill trend over the recent corpus -> results/
results/         skill demand rankings, per-posting extractions, novel skill candidates,
                 combined cross-method ranking, co-occurrence, and the real
                 month-over-month trend from the recent-data pipeline
```
