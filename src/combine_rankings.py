"""Merge the three NER methods' skill rankings into one comparison table.

Rule-based (skill_counts.csv) and statistical (skill_counts_trained.csv) are
highly correlated by construction -- the statistical model was weak-labeled
from the rule-based ruler's own output, so it isn't an independent signal on
top of it. The interesting comparison is rule-based vs. LLM: they agree
closely on named tools/languages/platforms but diverge sharply on broad
methodology terms (see README.md/CLAUDE.md), so this script surfaces that
divergence directly rather than just averaging all three numbers together.
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    rule = pd.read_csv(RESULTS_DIR / "skill_counts.csv").set_index("skill")
    stat = pd.read_csv(RESULTS_DIR / "skill_counts_trained.csv").set_index("skill")
    llm = pd.read_csv(RESULTS_DIR / "skill_counts_llm.csv").set_index("skill")

    combined = pd.DataFrame({
        "pct_rule": rule["pct_of_postings"],
        "pct_statistical": stat["pct_of_postings"],
        "pct_llm": llm["pct_of_postings"],
    })
    combined["rank_rule"] = combined["pct_rule"].rank(ascending=False, method="min")
    combined["rank_llm"] = combined["pct_llm"].rank(ascending=False, method="min")
    combined["llm_minus_rule"] = (combined["pct_llm"] - combined["pct_rule"]).round(1)
    combined = combined.sort_values("pct_rule", ascending=False)
    combined.to_csv(RESULTS_DIR / "skill_counts_combined.csv")

    print(f"{len(combined)} skills compared across methods -> results/skill_counts_combined.csv")

    print("\nTop 15 by rule-based ranking (the trusted output):")
    print(combined.head(15)[["pct_rule", "pct_llm", "llm_minus_rule"]].to_string())

    print("\nLLM reports it as a bigger deal than the rule-based keyword count would suggest:")
    print(combined.sort_values("llm_minus_rule", ascending=False).head(10)[["pct_rule", "pct_llm", "llm_minus_rule"]].to_string())

    print("\nRule-based count is inflated relative to the LLM (likely boilerplate-keyword matches):")
    print(combined.sort_values("llm_minus_rule").head(10)[["pct_rule", "pct_llm", "llm_minus_rule"]].to_string())


if __name__ == "__main__":
    main()
