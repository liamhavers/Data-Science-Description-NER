"""Train a statistical spaCy NER model on weakly-labeled skill spans.

Bootstraps training data from the rule-based EntityRuler pipeline in
ner_spacy.py (applied sentence-by-sentence) rather than hand-labeling data.
The trained model learns to recognize skill/tool/language/etc. mentions by
*category* (LANGUAGE, CLOUD, DATABASE, LIBRARY, TOOL, METHOD -- not the exact
canonical skill name), so it can in principle generalize to phrasings the
hand-curated gazetteer in skills_taxonomy.py doesn't cover.
"""

import random
from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import minibatch

from ner_spacy import build_pipeline
from skills_taxonomy import TAXONOMY

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "job_postings.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "spacy_ner"

RANDOM_SEED = 42
MAX_TRAIN_SENTENCES = 15_000
MAX_DEV_SENTENCES = 2_000
NEGATIVE_RATIO = 1.0  # non-entity sentences kept per entity-bearing sentence
N_EPOCHS = 6
DROPOUT = 0.2
BATCH_SIZE = 64

CATEGORIES = sorted({category for _, category, _ in TAXONOMY})


def weak_label_sentences(nlp, texts):
    """Run the rule-based pipeline and split into (sentence_text, entities) pairs."""
    positives, negatives = [], []
    for doc in nlp.pipe(texts, batch_size=100):
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) < 4:
                continue
            ents = [
                (ent.start_char - sent.start_char, ent.end_char - sent.start_char, ent.label_)
                for ent in doc.ents
                if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char
            ]
            (positives if ents else negatives).append((sent.text, ents))
    return positives, negatives


def make_dataset(nlp, df, rng, max_positive):
    positives, negatives = weak_label_sentences(nlp, df["job_summary"].astype(str))
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positives = positives[:max_positive]
    negatives = negatives[: int(len(positives) * NEGATIVE_RATIO)]

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def build_examples(nlp_blank, pairs):
    examples = []
    for text, ents in pairs:
        doc = nlp_blank.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": ents}))
    return examples


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n_dev_docs = max(1, int(len(df) * 0.1))
    dev_docs, train_docs = df.iloc[:n_dev_docs], df.iloc[n_dev_docs:]

    weak_nlp = build_pipeline()
    weak_nlp.add_pipe("sentencizer", before="entity_ruler")

    print("Weak-labeling training sentences...")
    train_pairs = make_dataset(weak_nlp, train_docs, rng, MAX_TRAIN_SENTENCES)
    print("Weak-labeling dev sentences...")
    dev_pairs = make_dataset(weak_nlp, dev_docs, rng, MAX_DEV_SENTENCES)
    print(f"train sentences: {len(train_pairs)}, dev sentences: {len(dev_pairs)}")

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for category in CATEGORIES:
        ner.add_label(category)

    train_examples = build_examples(nlp, train_pairs)
    dev_examples = build_examples(nlp, dev_pairs)

    optimizer = nlp.initialize(lambda: train_examples)
    for epoch in range(1, N_EPOCHS + 1):
        rng.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=BATCH_SIZE):
            nlp.update(batch, drop=DROPOUT, sgd=optimizer, losses=losses)
        scores = nlp.evaluate(dev_examples)
        print(
            f"epoch {epoch:2d}  loss={losses.get('ner', 0):8.1f}  "
            f"P={scores['ents_p']:.3f} R={scores['ents_r']:.3f} F={scores['ents_f']:.3f}"
        )

    scores = nlp.evaluate(dev_examples)
    print("\nFinal dev set scores:")
    print(f"  overall    P={scores['ents_p']:.3f} R={scores['ents_r']:.3f} F={scores['ents_f']:.3f}")
    for label, s in sorted(scores["ents_per_type"].items()):
        print(f"  {label:10s} P={s['p']:.3f} R={s['r']:.3f} F={s['f']:.3f}")

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"\nModel saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
