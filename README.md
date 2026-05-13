# CS5782 — LoRA Re-implementation

## 1. Introduction

This repo is a **CS5782 course re-implementation** of *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021): we rerun key experiments to check the paper’s claims under limited compute. **LoRA** freezes pretrained weights and trains small **low-rank adapters** in selected layers so adaptation uses far fewer parameters than full fine-tuning while staying competitive on NLU/NLG.

## 2. Chosen result

We reproduce whether **LoRA nears full fine-tuning with a tiny trainable budget** (paper **Tables 2–3, 9–11**) and whether **rank / target-module choices** matter at fixed budget (spirit of **Table 5**, originally on GPT-3). **Figure:** rank–target sweep on RoBERTa (this repo).

![Rank sensitivity — RoBERTa GLUE](results/rank_sensitivity_roberta.png)

## 3. GitHub contents

**`code/`** — notebooks + `.py` exports: RoBERTa GLUE LoRA vs FT (`lora_efficiency_roberta_glue_sst2_mrpc`), RoBERTa rank/target sweep (`lora_rank_sensitivity_roberta_glue`), GPT-2 E2E (`lora_efficiency_gpt2_e2e`). **`results/`** — logs/plots. **`data/`** — only notes (data load via Hugging Face). **`poster/`** — poster PDF.

## 4. Re-implementation details

**Setup:** `roberta-base` on GLUE **SST-2** & **MRPC**; **`gpt2-medium`** on **E2E NLG**; metrics = **accuracy** / **BLEU**; stack = PyTorch, `transformers`, `datasets`, `evaluate`, `accelerate`, **sacrebleu** (E2E). **Changes:** paper used **multi V100**; we used **one Colab A100** and a ~24–28h budget—narrower scope, same hyperparameter tables where listed. Scripts are **Colab exports** (see below). Encoder **RoBERTa** tests whether Table-5-style **Q+V vs Q+K+V+O** guidance transfers from the paper’s decoder GPT-3 setting.

## 5. Reproduction steps

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets evaluate accelerate tqdm numpy sacrebleu
```

**GPU:** CUDA strongly recommended (we used **1× A100**). **Run:** open `code/*.ipynb` or the matching `.py`; **comment out** `google.colab` / `drive.mount` for local use. **No CLI flags**—edit in-file hyperparameters (match paper **Table 9** / **Table 11**). Data: `load_dataset("glue", ...)` and E2E CSV URLs as in `data/README.md`.

## 6. Results / insights

**RoBERTa:** LoRA (~**0.3M** params) matched paper-scale SST-2; MRPC slightly below paper mean (see report **Table 1**). **GPT-2 E2E:** our LoRA BLEU **~65.9** vs paper **70.4±0.1**; NLG runs were noisier. **Rank sweep:** **Q+V** best, **Q+K+V+O** second on SST-2—aligned with the paper’s engineering takeaway.

## 7. Conclusion

LoRA’s **parameter efficiency vs accuracy** story reproduced most cleanly on **GLUE**; **NLG** needed more runs/seeds to match published BLEU. **Lesson:** fixed paper hyperparameters carry classification well; generation may need extra tuning or averaging.

## 8. References

Hu et al. (2021). *LoRA: Low-rank adaptation of large language models.* https://arxiv.org/abs/2106.09685 · Wang et al. (2018). *GLUE.* · E2E NLG dataset (Tuetschek et al., via URLs in code) · Hugging Face docs for `transformers` / `datasets` / `evaluate` / `accelerate`.

## 9. Acknowledgements

Project for **CS5782** (graded coursework). **Group:** Lirong Yao, Laurence Yang, Haoyu Yan, Yaxi Zeng.
