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

Data ingestion and all three NER approaches (rule-based spaCy, statistical spaCy,
local-LLM via Ollama) are built, verified working, and compared on the full corpus.
Key decisions:

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
    standalone extractor. Retrained after the taxonomy expansion below: novel spans
    dropped from 279 to 140 unique, and what's left is almost entirely noise
    (`amended`, `hcfa`, `stamford`, `junk`) rather than real misses — a good sign the
    taxonomy now covers the corpus well; diminishing returns from this discovery loop
    going forward.
  - LLM-based extraction (`src/llm_extract.py` / `src/llm_aggregate.py`): a local
    `qwen2.5:7b-instruct` model via Ollama, running on-GPU rather than a paid API —
    free and unlimited for this ~12k-posting volume, versus rate-limited free-tier
    cloud APIs or real cost on paid ones. Prompted to return `(skill, category)` pairs
    directly, no gazetteer involved. Had to be made self-healing: a 10+ hour
    unattended run outlasted both a sleeping host machine and a resource-constrained
    WSL2 VM (`.wslconfig` capped at 2GB/2 cores at the time), each of which silently
    killed the Ollama server mid-run and caused the naive first version to record
    "connection failed" as if it were a real per-posting result for ~12k postings in a
    row. Fixed by retrying with backoff (including relaunching the server itself) on a
    dropped connection instead of writing a result for it — see the module docstring.
    Full-corpus result: strong agreement with the rule-based ranking on named
    tools/languages/platforms (A/B Testing 3.1% vs. 3.3%, PyTorch 4.9% vs. 4.7%), but
    the LLM reports broad methodology/concept terms (Machine Learning, Agile, ETL) at
    roughly 5-10x lower rates than the rule-based ruler. This isn't the LLM being
    wrong — the ruler fires on every literal keyword occurrence including boilerplate
    ("familiarity with agile methodologies a plus"), while the LLM appears to only
    report a broad concept when it's a substantive focus. **Use the LLM numbers for
    "how often is this a real focus"; use the rule-based numbers for "how often is
    this word present."** They're answering different questions, not disagreeing.
    Novel bucket surfaced bare `r` at 1,427 mentions — confirms the single-letter
    ambiguity concern that made us exclude it from the taxonomy's rule-based
    surfaces, but the LLM disambiguates it fine via context.
- **Skills taxonomy (157 entries, still growing)**: hand-curated in
  `src/skills_taxonomy.py` as `(canonical_name, category, [surface forms])` tuples —
  category becomes the spaCy entity label, canonical_name is looked up via
  `ent.ent_id_` so spelling variants collapse to one skill. It is still not used as
  ground truth for any posting's extraction — only as the seed vocabulary for the
  rule-based ruler and the known/novel normalization key for the LLM pipeline.
  Expanded twice so far, both times from empirical evidence rather than guessing:
  1. From `job_skills.csv` frequency counts (which specific tools are actually
     mentioned a lot) — 40 → 90 entries.
  2. From curating the LLM pipeline's novel-candidate bucket (~65 more) — 90 → 157
     entries. Added the clearest, most specific, least ambiguous hits (Jenkins,
     Informatica, Azure Data Factory, T-SQL, SPSS, PL/SQL, Go, JavaScript, C#, MLOps,
     Generative AI, LLMs, Reinforcement Learning, and more); skipped generic
     soft-skill phrases ("project management", "communication skills"),
     near-duplicate rephrasings of existing entries ("data analytics", "etl
     processes"), enterprise-BI tools with low personal-project value (Cognos,
     MicroStrategy, Collibra, ServiceNow), and embedded-hardware terms that were
     clearly off-topic noise from a non-DS posting that slipped into the dataset.
  Both times, generic non-actionable terms were deliberately excluded (e.g.
  "Communication", "Problem Solving", "Data Analysis" from `job_skills.csv`).
  **Ambiguous single-token surfaces are deliberately excluded from rule-based
  matching** even when the underlying skill is added: bare `r` (too ambiguous a
  token), bare `go` (an ordinary English word, riskier than `r`), bare `ai`, bare
  `lambda` (also an anonymous-function keyword), bare `glue` (also "glue code"). Keep
  this exclusion pattern for any future single-token/common-word addition — it's
  cheap insurance against false positives in the mechanical phrase matcher, and the
  LLM pipeline disambiguates these fine via context anyway. Expand this file directly
  when a new skill/tool needs tracking (including reviewed hits from
  `results/novel_skill_candidates.csv` or `results/novel_skill_candidates_llm.csv`);
  keep one skill's variants together in one tuple, and rerun `src/ner_spacy.py` (and
  `src/train_ner.py` + `src/ner_spacy_trained.py` if retraining is warranted, and
  `src/llm_aggregate.py` to reclassify known-vs-novel LLM output) after editing it.

- **Trend-over-time / recent-data source (decided)**: the primary dataset's
  `first_seen` only spans 6 days, so it can't show whether a skill is rising or
  falling — and the user wants this since AI/data-science skills shift quickly.
  Researched alternatives before building anything:
  - `arshkon/linkedin-job-postings` (README's own documented fallback): checked
    directly, turned out to be another single-scrape snapshot (half its 124k rows
    dated within 2 days) — same problem, not usable for trend.
  - Adzuna's historical API: real 24-month history, but salary-by-category, not
    skill-keyword vacancy counts — not directly useful without much more work.
  - Several Kaggle "20XX-2026 AI/data job market" datasets found by search are
    **synthetic**: `shree0910/ai-and-data-science-job-market-dataset-20202026`
    checked directly and has rows dated **November 2026** (in the future) with
    only boolean skill flags (`skills_python`, `skills_ml`, etc.) and no real
    description text. Treat any of this genre of dataset as suspect until
    verified by direct download — don't trust the listing description alone.
  - `atharvasoundankar/ai-job-market-global-2026`: verified real — sourced from
    the live Adzuna + USAJobs APIs, genuine `posted_date` values reaching
    2026-02-22. But small (5,773 rows, 3,750 unique after dedup — the source
    repeats a posting once per city), description text truncated to ~500 chars
    (Adzuna's free-tier snippet limit, not an artifact we introduced), and 79% of
    it concentrated in its last 3 months — a recent-months snapshot, not deep
    history, and despite an advertised weekly refresh it hadn't moved past Feb
    2026 as of when checked (mid-2026) — the update pipeline may have stalled.
  - **Decision (user's explicit choice among three presented options)**: hybrid —
    use the Kaggle snapshot as a historical base (`src/ingest_recent.py`) and
    live-fetch via the Adzuna API (`src/fetch_adzuna.py`) to extend the series
    forward from here. Required the user to register their own free Adzuna API
    account (https://developer.adzuna.com/) — done; credentials live in
    `.adzuna/credentials` (repo root, gitignored — note this is a different
    convention from `~/.kaggle/access_token`'s home-directory location, because
    the user had already created the credentials file at the repo-root path
    before it came up, and it's safely gitignored either way).
  - **This recent-data corpus's job-title mix differs from the primary dataset's**
    (skews Data Scientist / AI Engineer / ML Engineer, vs. the primary dataset's
    broader data-science scope) — e.g. AWS/Azure read far lower here than in the
    primary ranking. **Don't treat these two corpora's absolute percentages as
    comparable** — use the primary dataset for current-demand ranking, this one
    only for within-corpus month-over-month movement.
  - **Rate limiting (`src/fetch_adzuna.py`)**: Adzuna's terms of service
    (developer.adzuna.com/docs/terms_of_service) cap free-tier usage at 25
    hits/minute, 250/day, 1000/week, 2500/month. The script paces calls at one
    every 3s (~20/min) and tracks a persisted call log
    (`data/processed/adzuna_usage_log.json`) against all three longer windows,
    each shaded to 90% of the real limit, stopping itself early rather than risk
    the account — this matters because the script is meant to be re-run
    routinely (see cron below), so usage accumulates across runs, not just
    within one.
  - **Bugs caught while first running this for real**: (1) a transient 502 from
    Adzuna's own servers crashed the whole run under `set -euo pipefail` in
    `scripts/run_weekly.sh` — fixed by retrying transient `requests` errors with
    backoff (5s/15s/30s) inside `fetch_page`, giving up on just that one
    page/query rather than the whole script. (2) `src/analyze_recent_trend.py`'s
    `load_corpus()` originally parsed the combined (base CSV + live JSONL)
    `posted_date` column in one `pd.to_datetime(..., errors="coerce")` call —
    pandas infers a date format from a string column's first values and coerces
    every non-matching row to NaT, so with the base CSV's plain `YYYY-MM-DD`
    dates first, **every one of the 4,678 live-fetched postings silently
    disappeared** (parsed to NaT, dropped) despite the script printing that it
    included them. Fixed by parsing each source's dates separately before
    concatenating. **Lesson: never parse dates across concatenated sources in
    one pass if their string formats might differ — parse each source first.**
  - **Automation**: `scripts/run_weekly.sh` chains
    `fetch_adzuna.py` → `analyze_recent_trend.py` → `src/plot_trends.py` (the
    last renders the trend as a line chart, `results/skill_trend_recent_by_month.png`
    — a table of skills x months is hard to eyeball for direction). Runs every
    Monday 17:00 via cron (same time as the user's pre-existing `quantProject`
    entry; `crontab -l` to inspect). Logs to `logs/` (gitignored). Cron + systemd
    are confirmed actually running in this WSL2 instance (checked
    directly, not assumed), so this should fire reliably as long as the WSL2 VM
    itself is up at that time — same host-sleep caveat as the original LLM
    extraction run, not re-solved here.
  - **Chart was initially too generic to be useful**: the first version picked
    the top-12 skills by raw mention count, which put broad umbrella terms
    (Machine Learning, Artificial Intelligence, NLP, ...) front and center —
    these show up in nearly every posting regardless of what the role actually
    focuses on, so they dominate the chart by volume without differentiating
    anything, and crowd out specific tools (PyTorch, Snowflake, Databricks) that
    are the actually-interesting trend signal. Fixed in
    `src/analyze_recent_trend.py` with an explicit `GENERIC_SKILLS` exclusion set
    (Machine Learning, Artificial Intelligence, Deep Learning, NLP, Statistics,
    Big Data, Agile, DevOps, Business Intelligence, Data Mining) applied before
    picking the top skills to chart (raised to top-15 to compensate) — these
    terms are still counted normally everywhere else (`skill_counts.csv`,
    `combine_rankings.py`), this exclusion is chart-selection-only. **If future
    additions to the chart look generic again, extend this set rather than
    increasing TOP_N_SKILLS** — more of the same umbrella terms isn't more
    signal.
  - **The trend chart is now embedded in README.md**, but publishing it to GitHub
    is deliberately **not** part of the automation. User's explicit choice between
    (1) cron also commits+pushes weekly, fully hands-off, vs. (2) cron only
    regenerates the local file, user pushes manually after a check-in: chose
    (2), because (1) means unattended, recurring write access to a public repo,
    a meaningfully bigger authorization than "regenerate a local file" — flagged
    this distinction before building anything, per the git-safety norm of not
    auto-pushing without explicit sign-off, and per-repo sign-off doesn't cover
    an indefinite recurring push. **Consequence: the chart in README.md is only
    as fresh as the last manual `git push`, not as fresh as `results/` locally**
    — don't assume they're in sync when reading the repo from GitHub's side.

## Project timeline (rough outline)

Roughly chronological, for orientation rather than a changelog (see `git log` for
exact history):

1. **Scoping** — README/CLAUDE.md drafted, data source decided (Kaggle
   `asaniczka/data-science-job-postings-and-skills` via `kagglehub`), venv set up.
2. **Data ingestion** — `src/ingest.py` joins the three source CSVs into
   `data/processed/job_postings.csv`.
3. **Rule-based NER** — `src/skills_taxonomy.py` (v1, ~40 entries) +
   `src/ner_spacy.py`. First trustworthy ranking (`results/skill_counts.csv`).
4. **Taxonomy expansion round 1 + statistical NER** — taxonomy grown 40 → 90 entries
   from `job_skills.csv` frequency counts; `src/train_ner.py` /
   `src/ner_spacy_trained.py` added as a weak-supervision-trained spaCy model, with
   the dev-F1-is-not-generalization caveat established immediately.
5. **LLM-based extraction** — `src/llm_extract.py` (local Ollama, `qwen2.5:7b-instruct`)
   + `src/llm_aggregate.py`. Required making the ~10+ hour unattended run self-healing
   (retry/backoff + auto-restart of the Ollama server) after a host-sleep/WSL2-VM
   resource crash silently corrupted an early run's resumable state; also required
   raising `.wslconfig` memory/CPU limits for a stable overnight run.
6. **Taxonomy expansion round 2** — curated the LLM pipeline's novel-candidate bucket
   (~19.8k unique spans) down to ~65 clear additions; taxonomy 90 → 157 entries.
   Statistical model retrained and rule-based/LLM pipelines rerun against it.
7. **Cross-method comparison + relationship analysis** —
   `src/combine_rankings.py` merges all three methods' rankings into one table and
   surfaces where they agree/diverge; `src/analyze_trends.py` adds skill co-occurrence
   and a first pass at posting-date trend (see its caveat: the dataset's `first_seen`
   only spans ~6 days, so this isn't a real trend signal).
8. **Recent-data trend pipeline** — `src/ingest_recent.py` +
   `src/fetch_adzuna.py` + `src/analyze_recent_trend.py` build a real
   month-over-month skill trend from actively-sourced 2025-2026 data, since the
   user is already committed to two other projects and wants ongoing signal on
   what to pick up next rather than deciding right now.
9. **Automation + visualization** (current) — Adzuna credentials registered and
   live-fetch confirmed working (caught and fixed a date-parsing bug that had
   been silently dropping every live-fetched posting, plus a transient-HTTP-error
   crash — see above); `scripts/run_weekly.sh` now runs the fetch → aggregate →
   `src/plot_trends.py` chain automatically every Monday via cron, so the trend
   series and its chart extend themselves without a manual step.
10. **Not yet started** — deciding and acting on an actual next personal project
    based on the ranked skill demand output (the project's actual end goal),
    once the trend pipeline has had time to accumulate enough live data to be
    useful.

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
