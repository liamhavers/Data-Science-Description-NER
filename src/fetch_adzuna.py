"""Live-fetches current AI/data-science job postings from the Adzuna API, to
extend results/skill_trend_recent.csv forward past wherever the Kaggle
snapshot (src/ingest_recent.py) stops. Meant to be re-run periodically (e.g.
weekly) rather than once -- each run only adds postings not already seen.

Requires a free Adzuna API account (https://developer.adzuna.com/) -- there's
no way around registering one, it's an account-creation step only you can
do. Put the credentials in .adzuna/credentials (repo root, gitignored) as:
    APP_ID=your_app_id
    APP_KEY=your_app_key

Adzuna's free-tier search endpoint only returns a truncated description
snippet, not the full posting text -- this matches the truncation already
present in the Kaggle base dataset (which is itself Adzuna-sourced), so it's
not a new limitation introduced here.

Rate limiting: Adzuna's terms of service (developer.adzuna.com/docs/terms_of_service)
cap free-tier usage at 25 hits/minute, 250/day, 1000/week, 2500/month.
REQUEST_DELAY paces individual calls to stay under the per-minute cap, and a
persisted call log (data/processed/adzuna_usage_log.json) tracks how many
calls have been made in the last day/week/month so a run stops itself early
rather than risk the account's quota -- this matters because the script is
meant to be re-run routinely, so usage adds up across runs, not just within
one.
"""

import json
import time
from pathlib import Path

import requests

CREDENTIALS_PATH = Path(__file__).resolve().parent.parent / ".adzuna" / "credentials"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = DATA_DIR / "adzuna_pulls.jsonl"
USAGE_LOG_PATH = DATA_DIR / "adzuna_usage_log.json"

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
COUNTRIES = ["us", "gb", "ca", "de", "au"]
SEARCH_TERMS = ["data scientist", "machine learning engineer", "ai engineer", "data analyst"]
RESULTS_PER_PAGE = 50
MAX_PAGES_PER_QUERY = 5  # per-run cap, independent of the budget tracker below

REQUEST_DELAY = 3.0  # seconds between calls -> 20 calls/min, safely under the 25/min cap

# Adzuna's published free-tier limits, each shaded down by SAFETY_MARGIN so a
# run stops before actually touching the real ceiling (clock skew, a manual
# test run earlier the same day, etc. shouldn't be able to tip it over).
SAFETY_MARGIN = 0.9
LIMITS = {
    "day": (250, 1 * 24 * 60 * 60),
    "week": (1000, 7 * 24 * 60 * 60),
    "month": (2500, 30 * 24 * 60 * 60),
}
LOG_RETENTION_SECONDS = LIMITS["month"][1]


class BudgetExhausted(Exception):
    pass


def load_credentials() -> tuple[str, str]:
    if not CREDENTIALS_PATH.exists():
        raise SystemExit(
            f"No credentials at {CREDENTIALS_PATH}. Register a free account at "
            "https://developer.adzuna.com/ and save APP_ID=... / APP_KEY=... there."
        )
    values = {}
    for line in CREDENTIALS_PATH.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values["APP_ID"], values["APP_KEY"]


def load_seen_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    seen = set()
    with open(OUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["id"])
    return seen


def load_call_log() -> list[float]:
    if not USAGE_LOG_PATH.exists():
        return []
    now = time.time()
    timestamps = json.loads(USAGE_LOG_PATH.read_text())
    return [t for t in timestamps if now - t < LOG_RETENTION_SECONDS]


def save_call_log(timestamps: list[float]) -> None:
    USAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_LOG_PATH.write_text(json.dumps(timestamps))


def check_budget(timestamps: list[float]) -> None:
    now = time.time()
    for window, (limit, seconds) in LIMITS.items():
        used = sum(1 for t in timestamps if now - t < seconds)
        allowed = int(limit * SAFETY_MARGIN)
        if used >= allowed:
            raise BudgetExhausted(
                f"stopping: {used}/{allowed} calls already used this {window} "
                f"({SAFETY_MARGIN:.0%} of Adzuna's {limit}/{window} limit) -- try again later"
            )


def fetch_page(app_id: str, app_key: str, country: str, what: str, page: int, call_log: list[float]) -> list[dict]:
    check_budget(call_log)
    resp = requests.get(
        BASE_URL.format(country=country, page=page),
        params={
            "app_id": app_id,
            "app_key": app_key,
            "what": what,
            "results_per_page": RESULTS_PER_PAGE,
            "content-type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    call_log.append(time.time())
    save_call_log(call_log)
    return resp.json().get("results", [])


def main() -> None:
    app_id, app_key = load_credentials()
    seen = load_seen_ids()
    call_log = load_call_log()
    new_count = 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUT_PATH, "a") as out:
            for country in COUNTRIES:
                for what in SEARCH_TERMS:
                    for page in range(1, MAX_PAGES_PER_QUERY + 1):
                        results = fetch_page(app_id, app_key, country, what, page, call_log)
                        if not results:
                            break
                        for job in results:
                            if job["id"] in seen:
                                continue
                            seen.add(job["id"])
                            new_count += 1
                            out.write(json.dumps({
                                "id": job["id"],
                                "job_title": job.get("title"),
                                "country": country,
                                "posted_date": job.get("created"),
                                "job_description": job.get("description"),
                                "search_term": what,
                            }) + "\n")
                        out.flush()
                        time.sleep(REQUEST_DELAY)
    except BudgetExhausted as e:
        print(e)

    print(f"{new_count} new postings -> {OUT_PATH} ({len(call_log)} calls used in the last 30 days)")


if __name__ == "__main__":
    main()
