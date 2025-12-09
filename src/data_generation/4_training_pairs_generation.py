'''Generating training data for fine-tuning (late-interaction model, cross-encoder..).
The goal is to generate positive pairs of span-phenotype,
and leverage hyperbolic distance to create hard negative candidates
(distance between the negative candidate and the input span)'''

from hyperrag.config import *
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
HPO_SPAN_FILE = data_path / "hpo/hpo_spans_filtered.json"  # file containing HPO labels + spans
OUTPUT_PATH = data_path / "hpo/training_pairs.jsonl"
NUM_SPANS_PER_HPO = 5
NUM_NEGATIVE_PER_SPAN = 5
NEGATIVE_SAME_BRANCH_RATIO = 0.5  # % of negatives from same branch
MAX_HOPS = 3  # max number of hops to consider as same branch
GLOBAL_MAX_DIST = 43.3674
OUT_BRANCH_MAX_SCORE = 0.4  # max score for negatives outside branch
BRANCH_SCORE_THRESHOLD = 0.4  # min score for negatives in branch
MIN_DELTA = 0.1  # min delta between branch score and out-branch to accept both

assert OUT_BRANCH_MAX_SCORE <= BRANCH_SCORE_THRESHOLD, \
    "OUT_BRANCH_MAX_SCORE must be <= BRANCH_SCORE_THRESHOLD to ensure proper scoring order."


# ============================
# Utils
# ============================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# ============================
# Core Dataset Generator
# ============================
def generate_training_pairs():
    # Load all required data
    hpo_spans = load_json(HPO_SPAN_FILE)["entries"]
    hpo_info = load_json(HPO_RELATIONSHIPS_FILE)
    term_to_index = load_json(TERM_TO_INDEX_FILE)
    distance_matrix = np.load(DISTANCE_MATRIX_FILE)
    original_terms = load_json(ORIGINAL_TERMS_FILE)

    all_hpo_ids = list(original_terms.keys())
    dataset = []

    for entry in hpo_spans:
        hpo_iri = entry.get("hpo_id")
        if not hpo_iri:
            continue

        hpo_id = hpo_iri.split("/")[-1].replace("_", ":")
        if hpo_id not in hpo_info:
            continue

        hpo_label = hpo_info[hpo_id]["label"]
        if hpo_label not in term_to_index:
            continue

        anchor_idx = term_to_index[hpo_label]
        spans = entry["spans"][:NUM_SPANS_PER_HPO]

        for span in spans:
            print(f"Generating training pairs for {span}...")
            dataset.append({
                "span": span,
                "hpo_label": hpo_label,
                "hpo_id": hpo_id,
                "score": 1.0
            })

            negatives = []

            same_branch_ids = set([
                aid for aid, hop in hpo_info[hpo_id].get("hops", {}).items()
                if hop <= MAX_HOPS and aid in hpo_info and aid in original_terms and hpo_info[aid][
                    "label"] in term_to_index
            ])

            other_branch_ids = set([
                oid for oid in all_hpo_ids
                if oid != hpo_id and oid not in same_branch_ids and oid in hpo_info and hpo_info[oid][
                    "label"] in term_to_index
            ])

            num_same_branch = int(NUM_NEGATIVE_PER_SPAN * NEGATIVE_SAME_BRANCH_RATIO)
            num_other_branch = NUM_NEGATIVE_PER_SPAN - num_same_branch

            branch_scores = []

            # Negatives in same branch (hard)
            for neg_id in random.sample(list(same_branch_ids), min(num_same_branch, len(same_branch_ids))):
                neg_label = hpo_info[neg_id]["label"]
                neg_idx = term_to_index[neg_label]
                dist = float(distance_matrix[anchor_idx, neg_idx])
                score = 1.0 - (dist / GLOBAL_MAX_DIST)

                if score >= BRANCH_SCORE_THRESHOLD:
                    negatives.append({
                        "span": span,
                        "hpo_label": neg_label,
                        "hpo_id": neg_id,
                        "score": round(score, 4)
                    })
                    branch_scores.append(score)

            max_branch_score = max(branch_scores) if branch_scores else 0.0

            # Negatives in other branches (easy)
            for neg_id in random.sample(list(other_branch_ids), min(num_other_branch, len(other_branch_ids))):
                neg_label = hpo_info[neg_id]["label"]
                neg_idx = term_to_index[neg_label]
                dist = float(distance_matrix[anchor_idx, neg_idx])
                score = 1.0 - (dist / GLOBAL_MAX_DIST)

                if score <= OUT_BRANCH_MAX_SCORE and score + MIN_DELTA < max_branch_score:
                    negatives.append({
                        "span": span,
                        "hpo_label": neg_label,
                        "hpo_id": neg_id,
                        "score": round(score, 4)
                    })

            dataset.extend(negatives)

    print(f"✅ Generated {len(dataset)} training pairs")
    save_jsonl(dataset, OUTPUT_PATH)
    print(f"📁 Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_training_pairs()