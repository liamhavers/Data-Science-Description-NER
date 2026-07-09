"""Live-fetches current AI/data-science job postings from the Adzuna API, to
extend results/skill_trend_recent.csv forward past wherever the Kaggle
snapshot (src/ingest_recent.py) stops. Meant to be re-run periodically (e.g.
weekly) rather than once -- each run only adds postings not already seen.

Requires a free Adzuna API account (https://developer.adzuna.com/) -- there's
no way around registering one, it's an account-creation step only you can
do. Put the credentials in ~/.adzuna/credentials as:
    APP_ID=your_app_id
    APP_KEY=your_app_key

Adzuna's free-tier search endpoint only returns a truncated description
snippet, not the full posting text -- this matches the truncation already
present in the Kaggle base dataset (which is itself Adzuna-sourced), so it's
not a new limitation introduced here.
"""

import json
import time
from pathlib import Path

import requests

CREDENTIALS_PATH = Path.home() / ".adzuna" / "credentials"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "adzuna_pulls.jsonl"

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
COUNTRIES = ["us", "gb", "ca", "de", "au"]
SEARCH_TERMS = ["data scientist", "machine learning engineer", "ai engineer", "data analyst"]
RESULTS_PER_PAGE = 50
MAX_PAGES_PER_QUERY = 5  # keep each run modest against the free-tier call quota

REQUEST_DELAY = 1.0  # seconds between calls, polite to the API


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


def fetch_page(app_id: str, app_key: str, country: str, what: str, page: int) -> list[dict]:
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
    return resp.json().get("results", [])


def main() -> None:
    app_id, app_key = load_credentials()
    seen = load_seen_ids()
    new_count = 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a") as out:
        for country in COUNTRIES:
            for what in SEARCH_TERMS:
                for page in range(1, MAX_PAGES_PER_QUERY + 1):
                    results = fetch_page(app_id, app_key, country, what, page)
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

    print(f"{new_count} new postings -> {OUT_PATH}")


if __name__ == "__main__":
    main()
