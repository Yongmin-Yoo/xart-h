from datasets import load_dataset

REPO = "yongminyoo91/xart-h"
REVISION = "v1.0.1"

dataset = load_dataset(
    REPO,
    revision=REVISION
)

print(dataset)
