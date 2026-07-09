"""LLM-based skill extraction using a local Ollama model (qwen2.5:7b-instruct).

Complements the spaCy pipelines: instead of a gazetteer or weak-labeled
statistical model, an instruction-following LLM reads each raw job_summary
and returns structured (skill, category) pairs directly, using the same
category labels as skills_taxonomy.py (LANGUAGE, CLOUD, DATABASE, LIBRARY,
TOOL, METHOD) so results are directly comparable across all three approaches.

Resumable: writes one JSON line per posting to RESULTS_PATH as it goes, and
skips job_links already present in that file on restart -- a full run over
~12k postings takes hours on local hardware, so interruptions are expected.

Self-healing: a run this long will likely outlast the machine staying awake
(sleep/suspend kills the WSL2 VM, taking `ollama serve` down with it). A
dropped connection is retried with backoff -- including relaunching the
Ollama server itself -- rather than being recorded as a per-posting result,
so an interrupted posting is retried on the next attempt instead of being
wrongly marked "done" with an empty extraction.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Literal

import ollama
import pandas as pd
from pydantic import BaseModel

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "job_skills_extracted_llm.jsonl"

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b-instruct"

OLLAMA_BIN = Path.home() / ".local" / "ollama" / "bin" / "ollama"
OLLAMA_MODELS_DIR = Path.home() / ".local" / "ollama" / "models"
OLLAMA_SERVE_LOG = Path.home() / ".local" / "ollama" / "serve.log"

MAX_RECONNECT_ATTEMPTS = 20
RECONNECT_BACKOFF_CAP = 300  # seconds

SYSTEM_PROMPT = (
    "Extract specific data-science-relevant skills, tools, languages, platforms, "
    "databases, and methodologies mentioned in the job description. "
    "For each, classify it into one category: LANGUAGE, CLOUD, DATABASE, LIBRARY, TOOL, or METHOD. "
    "Only include specific named technologies/skills (e.g. Python, AWS, A/B Testing) -- "
    "not generic soft skills like communication or teamwork."
)


class SkillEntry(BaseModel):
    skill: str
    category: Literal["LANGUAGE", "CLOUD", "DATABASE", "LIBRARY", "TOOL", "METHOD"]


class SkillExtraction(BaseModel):
    skills: list[SkillEntry]


def is_server_up(client: ollama.Client) -> bool:
    try:
        client.list()
        return True
    except ConnectionError:
        return False


def start_server() -> None:
    print("Ollama server unreachable -- starting it...")
    env = os.environ | {"OLLAMA_MODELS": str(OLLAMA_MODELS_DIR)}
    with open(OLLAMA_SERVE_LOG, "a") as log:
        subprocess.Popen(
            [str(OLLAMA_BIN), "serve"],
            env=env,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def ensure_server_running(client: ollama.Client) -> None:
    """Block until the Ollama server responds, (re)starting it if needed."""
    if is_server_up(client):
        return
    start_server()
    for attempt in range(30):
        time.sleep(2)
        if is_server_up(client):
            print("Ollama server is back up.")
            return
    raise RuntimeError("Ollama server did not come back up after 60s of waiting.")


def load_done_links() -> set[str]:
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["job_link"])
    return done


def extract_one(client: ollama.Client, text: str) -> list[dict]:
    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        format=SkillExtraction.model_json_schema(),
        options={"temperature": 0},
    )
    parsed = SkillExtraction.model_validate_json(resp["message"]["content"])
    return [entry.model_dump() for entry in parsed.skills]


def extract_with_retry(client: ollama.Client, text: str) -> list[dict]:
    """Extract skills, transparently reconnecting (and restarting the server
    if needed) on a dropped connection. Only a non-connection failure (e.g.
    the model returning malformed output for this particular posting) is
    allowed to surface as a per-posting error."""
    for attempt in range(MAX_RECONNECT_ATTEMPTS):
        try:
            ensure_server_running(client)
            return extract_one(client, text)
        except ConnectionError:
            backoff = min(2**attempt, RECONNECT_BACKOFF_CAP)
            print(f"  connection dropped, retrying in {backoff}s (attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS})")
            time.sleep(backoff)
    raise ConnectionError(f"Ollama unreachable after {MAX_RECONNECT_ATTEMPTS} reconnect attempts")


def main(limit: int | None = None) -> None:
    client = ollama.Client(host=OLLAMA_HOST)
    df = pd.read_csv(DATA_PATH)
    if limit:
        df = df.head(limit)

    done = load_done_links()
    todo = df[~df["job_link"].isin(done)]
    print(f"{len(done)} already done, {len(todo)} remaining")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as out:
        for pos, (_, row) in enumerate(todo.iterrows(), start=1):
            t0 = time.time()
            try:
                skills = extract_with_retry(client, str(row["job_summary"]))
                error = None
            except ConnectionError:
                # Ollama is down and wouldn't come back after repeated retries --
                # something is fundamentally broken (not a transient blip). Stop
                # the whole run rather than racing through every remaining
                # posting recording the same failure; this posting is left
                # unwritten so it's retried on the next invocation.
                raise
            except Exception as e:
                skills = []
                error = str(e)
            elapsed = time.time() - t0

            out.write(json.dumps({"job_link": row["job_link"], "skills": skills, "error": error}) + "\n")
            out.flush()

            print(f"[{len(done) + pos}/{len(df)}] {elapsed:.1f}s  {len(skills)} skills  {row['job_link'][:60]}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=n)
