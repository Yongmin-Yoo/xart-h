import re
from collections import Counter
from datasets import load_dataset
from huggingface_hub import HfApi

REPO = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

EXPECTED = {
    "train": {
        "rows": 25187, "X": 8462, "A": 8450, "U": 8275
    },
    "validation": {
        "rows": 5282, "X": 1813, "A": 1813, "U": 1656
    },
    "test": {
        "rows": 5316, "X": 1822, "A": 1816, "U": 1678
    },
}

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

dataset = load_dataset(REPO, revision=REVISION)
commit = HfApi().dataset_info(REPO, revision=REVISION).sha
split_sets = {}

for split, expected in EXPECTED.items():
    data = dataset[split]
    labels = Counter(data["label"])

    assert len(data) == expected["rows"]
    assert labels[1] == expected["X"]
    assert labels[0] == expected["A"]
    assert labels[2] == expected["U"]

    pairs = {
        (normalize(claim), normalize(passage))
        for claim, passage in zip(data["text"], data["text_b"])
    }
    assert len(pairs) == len(data)

    split_sets[split] = {
        "claims": {normalize(x) for x in data["text"]},
        "passages": {normalize(x) for x in data["text_b"]},
        "applications": set(data["patent_application_id"]),
        "documents": set(data["cited_document_id"]),
        "components": set(data["_priority_component_id"]),
    }

splits = list(EXPECTED)
for i, left in enumerate(splits):
    for right in splits[i + 1:]:
        for relation in split_sets[left]:
            overlap = (
                split_sets[left][relation]
                & split_sets[right][relation]
            )
            assert not overlap, (
                f"{left}/{right}: {relation} overlap "
                f"({len(overlap)})"
            )

print("XART-H verification passed.")
print("Revision:", REVISION)
print("Commit:", commit)
