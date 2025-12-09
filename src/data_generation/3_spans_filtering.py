from hyperrag.config import data_path
import json
from pathlib import Path


# ==============================
# CONFIGURATION
# ==============================
HPO_SPAN_FILE = data_path / "hpo/hpo_sentences_with_spans_sample.json"
FILTERED_SPANS_OUTPUT = data_path / "hpo/hpo_spans_filtered.json"
FILTERED_SPANS_LOG = data_path / "hpo/filtered_spans.txt"

SPAN_SCORE_THRESHOLD = 0.2

GENERIC_WORDS = set([
    "normal", "expected", "usual", "unusual", "typical", "irregularities", "range", "value", "parameter",
    "shape", "structure", "feature", "texture", "thick", "thin", "small", "large", "short", "long",
    "movement", "growth", "size", "height", "width", "volume", "anomaly", "abnormality", "below"
])

WHITELIST_WORDS = set([
    "photopsia", "amyloid", "schizencephaly", "csf", "glycosylation", "retinal", "cerebellum", "cochlear",
    "iris", "chorioretinal", "choroidal", "enamel", "diaphragm", "thymic", "bladder", "genitalia", "echogenicity",
    "scrotal", "kidney", "renal", "tubule", "ovarian", "palate", "cranial", "maxillary", "thyroid", "parathyroid",
    "adrenal", "adrenocortical", "menses", "irregular periods", "clavicles", "skeletal", "vascular", "macular",
    "metacarpal", "axial", "acral", "skin", "pupil", "astigmatism", "atlantoaxial", "spinal", "EMG", "myopathic",
    "myelin", "glycoprotein", "mtDNA", "helix", "Canine", "EEG", "pituitary", "lung", "Leukocyte", "platelet",
    "Granulocyte", "coagulation", "CNS", "glabellar", "respiratory", "ureteral", "epiphysis", "Metaphyseal",
    "cognitive", "cardiac", "forebrain", "esophageal", "Oligodendrocyte", "Fingernail", "periungual", "EKG", "ECG",
    "heartbeat", "acetabular", "penile length of", "tear production", "level", "scored", "cell", "concentration", "bone",
    "ankle", "mediastinal", "measured", "joint", "fingers", "finger", "muscles", "phalanx", "immunoglobulin", "hemolytic",
    "urine", "cervical", "elbow", "shoulder", "motor"
])

BLACKLIST_PATTERNS = [
    "below the normal range",
    "outside the expected parameters",
    "greater than normal",
    "lower than typical",
    "values outside the expected",
    "an unusual shape",
    "restricted movement",
    "limited during the examination"
]

# ============================
# UTILS
# ============================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_span_score(span: str) -> float:
    span = span.lower().strip()
    words = span.split()
    if not words:
        return 0.0

    # Reject span if blacklisted
    for black_phrase in BLACKLIST_PATTERNS:
        if black_phrase in span:
            if any(white in span for white in WHITELIST_WORDS):
                return 1.0
            else:
                return 0.0

    if any(white in span for white in WHITELIST_WORDS):
        return 1.0  # Always keep if whitelisted

    score = 1.0 if len(words) >= 3 else 0.3

    generic_count = sum(1 for w in words if w.strip(".,!?\"'") in GENERIC_WORDS)
    generic_ratio = generic_count / len(words)
    if generic_ratio > 0.6:
        score -= 0.5

    vague_patterns = [
        "unusual", "abnormal", "irregular", "unspecified",
        "outside expected", "greater than normal", "lower than typical"
    ]
    if any(pat in span for pat in vague_patterns):
        score -= 0.3

    return max(score, 0.0)

# ============================
# MAIN FILTER FUNCTION
# ============================

def filter_hpo_spans():
    data = load_json(HPO_SPAN_FILE)
    filtered_entries = []
    filtered_spans = []

    for entry in data.get("entries", []):
        spans = entry.get("spans", [])
        kept_spans = []

        for span in spans:
            score = compute_span_score(span)
            if score >= SPAN_SCORE_THRESHOLD:
                kept_spans.append(span)
            else:
                filtered_spans.append(span)

        if kept_spans:
            entry_copy = entry.copy()
            entry_copy["spans"] = kept_spans
            filtered_entries.append(entry_copy)

    save_json({"entries": filtered_entries}, FILTERED_SPANS_OUTPUT)
    print(f"✅ Saved {len(filtered_entries)} entries with clean spans to {FILTERED_SPANS_OUTPUT}")

    with open(FILTERED_SPANS_LOG, "w", encoding="utf-8") as f:
        for span in filtered_spans:
            f.write(span + "\n")

    print(f"📝 Logged {len(filtered_spans)} filtered spans to {FILTERED_SPANS_LOG}")


if __name__ == "__main__":
    filter_hpo_spans()