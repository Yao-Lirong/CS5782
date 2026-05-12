# Data directory

The notebooks and Python exports under `code/` **do not read from this `data/` folder**. They obtain data exactly as follows (same pattern in `lora_efficiency_roberta_glue_sst2_mrpc.*`, `lora_rank_sensitivity_roberta_glue.*`, and `lora_efficiency_gpt2_e2e.*`).

## What the experiments use

### GLUE (SST-2, MRPC)

```python
from datasets import load_dataset

load_dataset("glue", "sst2")
load_dataset("glue", "mrpc")
```

Splits are downloaded and cached by the Hugging Face `datasets` library from the Hub (not from `data/`).

### E2E NLG Challenge (CSV)

```python
from datasets import load_dataset

data_files = {
    "train": "https://github.com/tuetschek/e2e-dataset/raw/master/trainset.csv",
    "validation": "https://github.com/tuetschek/e2e-dataset/raw/master/devset.csv",
}
load_dataset("csv", data_files=data_files)
```

Train and validation are loaded straight from those GitHub raw URLs via `datasets` 
