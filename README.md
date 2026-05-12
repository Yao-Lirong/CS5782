# CS5782 — LoRA experiments

This repository contains notebooks and scripts for LoRA efficiency and rank-sensitivity experiments (RoBERTa on GLUE, GPT-2 on E2E NLG). The code under `code/` loads data programmatically; you do not need to hand-place GLUE files unless you want an offline mirror.

## Obtaining the datasets

### GLUE: SST-2 and MRPC

The experiments use the [GLUE benchmark](https://gluebenchmark.com/) tasks **SST-2** and **MRPC** via the Hugging Face [`datasets`](https://huggingface.co/docs/datasets) library:

```python
from datasets import load_dataset

sst2 = load_dataset("glue", "sst2")
mrpc = load_dataset("glue", "mrpc")
```

On first use, `datasets` downloads and caches the prepared GLUE splits from the Hugging Face Hub. Ensure you have internet access for that first run (and enough disk space for the cache). The default cache location follows Hugging Face conventions; you can override it with the environment variable `HF_HOME` (or `HF_DATASETS_CACHE` for dataset files only). If you are on a restricted network, configure your environment for Hub access (for example proxy or mirror settings documented by Hugging Face).

No separate registration is required for these public GLUE configurations beyond normal Hub usage.

### E2E NLG Challenge

The GPT-2 / NLG experiments use the **E2E NLG Challenge** data (CSV with `mr` and `ref` columns). The notebooks load the official CSVs over HTTPS, for example:

- Train: `https://github.com/tuetschek/e2e-dataset/raw/master/trainset.csv`
- Validation: `https://github.com/tuetschek/e2e-dataset/raw/master/devset.csv`

With `datasets`, that matches:

```python
from datasets import load_dataset

data_files = {
    "train": "https://github.com/tuetschek/e2e-dataset/raw/master/trainset.csv",
    "validation": "https://github.com/tuetschek/e2e-dataset/raw/master/devset.csv",
}
e2e = load_dataset("csv", data_files=data_files)
```

For a fully offline workflow, clone or download the repository [tuetschek/e2e-dataset](https://github.com/tuetschek/e2e-dataset) and point `data_files` to local paths (for example `data/e2e/trainset.csv` and `data/e2e/devset.csv`). Respect the dataset’s license and citation requirements stated in that project.
