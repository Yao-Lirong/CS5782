# CS5782 — LoRA Re-implementation

## 1. Introduction

This repo is a **CS5782 course re-implementation** of *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021): we rerun key experiments to check the paper’s claims under limited compute. **LoRA** freezes pretrained weights and trains small **low-rank adapters** in selected layers so adaptation uses far fewer parameters than full fine-tuning while staying competitive on NLU/NLG.

## 2. Chosen result

We reproduce whether **LoRA nears full fine-tuning with a tiny trainable budget** (paper **Tables 2–3, 9–11**) and whether **rank / target-module choices** matter at fixed budget (spirit of **Table 5**, originally on GPT-3). The rank-sensitivity figure for our RoBERTa runs is shown in **§6**.

## 3. GitHub contents

Top-level **write-ups:** `README.md` (this file), `CS5782 Final Report.md` (full write-up; may be large if figures are embedded).

**`code/`** — all runnable experiments (Jupyter + Colab-exported Python). Each pair shares the same logic; prefer notebooks for interactive runs.

| File | Description |
|------|-------------|
| `lora_efficiency_roberta_glue_sst2_mrpc.ipynb` / `.py` | RoBERTa-base on GLUE **SST-2** and **MRPC**: LoRA vs full fine-tuning (and related baselines in-notebook), accuracy-focused. |
| `lora_rank_sensitivity_roberta_glue.ipynb` / `.py` | RoBERTa on GLUE **SST-2**: sweep **LoRA rank** and **target modules** (Q / K / V / O and combinations) at comparable trainable-parameter budgets; produces sensitivity plots/metrics. |
| `lora_efficiency_gpt2_e2e.ipynb` / `.py` | **GPT-2 Medium** on the **E2E NLG** data-to-text task: LoRA vs full fine-tuning; **BLEU** vs references. |

**`results/`** — committed outputs from runs (e.g. `rank_sensitivity_roberta.png`, text logs such as `lora_efficiency_exp_results`); safe to regenerate by re-running `code/`.

**`data/`** — no checked-in corpora; `data/README.md` documents how experiments pull **GLUE** and **E2E** through Hugging Face `datasets` (Hub / URLs).

**`poster/`** — course poster PDF 

## 4. Re-implementation details

**Setup:** `roberta-base` on GLUE **SST-2** & **MRPC**; **`gpt2-medium`** on **E2E NLG**; metrics = **accuracy** / **BLEU**; stack = PyTorch, `transformers`, `datasets`, `evaluate`, `accelerate`, **sacrebleu** (E2E). **Changes:** paper used **multi V100**; we used **one Colab A100** and a ~24–28h budget—narrower scope, same hyperparameter tables where listed. Scripts are **Colab exports** (see below). Encoder **RoBERTa** tests whether Table-5-style **Q+V vs Q+K+V+O** guidance transfers from the paper’s decoder GPT-3 setting.

## 5. Reproduction steps

1. set up environement, including installing the following dependencies: 
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets evaluate accelerate tqdm numpy sacrebleu
```

**GPU:** CUDA strongly recommended (we used **1× A100**). **Run:** open `code/*.ipynb` or the matching `.py`; **comment out** `google.colab` / `drive.mount` for local use. **No CLI flags**—edit in-file hyperparameters (match paper **Table 9** / **Table 11**). Data: `load_dataset("glue", ...)` and E2E CSV URLs as in `data/README.md`.

## 6. Results / insights

**Versus the paper (same trainable-parameter scale).** On **RoBERTa-base + LoRA (~0.3M params)**, Hu et al. report SST-2 **95.1±0.2** and MRPC **89.7±0.7** (Table 2); our run reached **95.1** / **88.7** (report **Table 1**)—SST-2 matches the paper mean; MRPC is a bit below their central value. On **GPT-2 Medium + LoRA (~0.35M params)** on E2E, the paper reports **70.4±0.1** BLEU (Table 3); ours was **65.88** (report **Table 2**)—same *ranking story* vs full fine-tuning in our logs, but not the same absolute BLEU. **Rank / target modules:** our RoBERTa SST-2 sweep puts **Q+V** first and **Q+K+V+O** second (report **Figure 1**), consistent with the paper’s **Table 5** takeaway that query–value (and full-attention) placements are strong defaults.

![Rank sensitivity — RoBERTa GLUE](results/rank_sensitivity_roberta.png)

**What you should expect from this repo.** After §5 you get **JSON/metric logs**, **RoBERTa GLUE** numbers close to the paper’s LoRA row on SST-2 (MRPC varies more by seed), **GPT-2 E2E** where LoRA trains but **BLEU often trails ~70.4** without extra runs/tuning, and a **rank/target sweep** (table + figure above). **Not** a bit-for-bit PDF match—especially E2E—**but** a runnable harness for the paper’s **qualitative** takeaways (efficient LoRA, Q+V / full-attention placements).

## 7. Conclusion

LoRA’s **parameter efficiency vs accuracy** story reproduced most cleanly on **GLUE**; **NLG** needed more runs/seeds to match published BLEU. **Lesson:** fixed paper hyperparameters carry classification well; generation may need extra tuning or averaging.

## 8. References

Hu et al. (2021). *LoRA: Low-rank adaptation of large language models.* https://arxiv.org/abs/2106.09685 · Wang et al. (2018). *GLUE.* · E2E NLG dataset (Tuetschek et al., via URLs in code) · Hugging Face docs for `transformers` / `datasets` / `evaluate` / `accelerate`.

## 9. Acknowledgements

Project for **CS5782** (graded coursework). **Group:** Lirong Yao, Laurence Yang, Haoyu Yan, Yaxi Zeng.
