# Token-Budget Sensitivity Results

## Setup Summary

This experiment tests whether LitePack's advantage is larger when the input context budget is tighter. The evaluation entrypoint was `src/evaluation/evaluation.py`, using the same evaluation set as the prior baseline run: `results/evaluation_data/eval_data.json`.

Prior comparable baseline run located: `results/runs/20260425_191613_eval_002` (`gpt-4o-mini`, `text-embedding-3-small`, budget 1800, methods `similarity`, `mmr`, `litepack`, keyword and recency features enabled).

New budget-sensitivity runs:

| Budget | Run directory |
|---:|---|
| 1200 | `results/runs/budget_sensitivity_1200_gpt4o_mini` |
| 1800 | `results/runs/budget_sensitivity_1800_gpt4o_mini` |
| 2400 | `results/runs/budget_sensitivity_2400_gpt4o_mini` |

Fixed settings across the new runs:

| Setting | Value |
|---|---|
| Evaluation set | `results/evaluation_data/eval_data.json` |
| Methods | `similarity`, `mmr`, `litepack` |
| Embedding model | `text-embedding-3-small` |
| LLM model | `gpt-4o-mini` |
| Temperature | `0.1` |
| Seed | `100` |
| Max output tokens | `150` |
| Top-N retrieved candidates | `20` |
| Feature flags | `use_keywords=true`, `use_recency=true`, `use_publication_type=false`, `use_mesh=false` |
| Reference year | `2026` |

## Aggregate Results

All values below are from the generated `summary.json` files in the three run directories.

| Budget | Method | Runs | Recall@5 | NDCG@5 | Hit Rate | Redundancy | Avg Tokens | Utilization | ROUGE-L | Keyword Support |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1200 | similarity | 20 | 1.000 | 1.000 | 1.000 | 0.617 | 1158.2 | 0.965 | 0.4501 | 0.9039 |
| 1200 | mmr | 20 | 1.000 | 1.000 | 1.000 | 0.561 | 1158.0 | 0.965 | 0.4456 | 0.8941 |
| 1200 | litepack | 20 | 1.000 | 1.000 | 1.000 | 0.582 | 1133.7 | 0.945 | 0.4832 | 0.9088 |
| 1800 | similarity | 20 | 1.000 | 1.000 | 1.000 | 0.574 | 1765.0 | 0.981 | 0.4480 | 0.9258 |
| 1800 | mmr | 20 | 1.000 | 1.000 | 1.000 | 0.544 | 1757.4 | 0.976 | 0.4654 | 0.9198 |
| 1800 | litepack | 20 | 1.000 | 1.000 | 1.000 | 0.552 | 1733.8 | 0.963 | 0.4730 | 0.9281 |
| 2400 | similarity | 20 | 1.000 | 1.000 | 1.000 | 0.553 | 2332.5 | 0.972 | 0.4488 | 0.9268 |
| 2400 | mmr | 20 | 1.000 | 1.000 | 1.000 | 0.541 | 2335.5 | 0.973 | 0.4607 | 0.9330 |
| 2400 | litepack | 20 | 1.000 | 1.000 | 1.000 | 0.545 | 2334.3 | 0.973 | 0.4725 | 0.9343 |

## Per-Budget Comparison

At the 1200-token budget, retrieval metrics are saturated for all methods, but LitePack has the strongest generation-facing metrics. It has the highest ROUGE-L (0.4832), highest keyword support (0.9088), and uses fewer context tokens than both baselines. Compared with the best baseline ROUGE-L score at this budget, LitePack gains +0.0332.

At the 1800-token budget, LitePack again has the highest ROUGE-L (0.4730) and keyword support (0.9281), while using fewer average context tokens than similarity and MMR. The ROUGE-L margin over the strongest baseline drops to +0.0076.

At the 2400-token budget, LitePack still has the highest ROUGE-L (0.4725) and keyword support (0.9343), but the token-efficiency gap nearly disappears. LitePack uses slightly fewer tokens than MMR but slightly more than similarity. The ROUGE-L margin over the strongest baseline is +0.0119.

| Budget | Best baseline ROUGE-L | LitePack ROUGE-L | LitePack ROUGE-L Gain | Best baseline keyword support | LitePack keyword support | LitePack keyword gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1200 | 0.4501 | 0.4832 | +0.0332 | 0.9039 | 0.9088 | +0.0048 |
| 1800 | 0.4654 | 0.4730 | +0.0076 | 0.9258 | 0.9281 | +0.0023 |
| 2400 | 0.4607 | 0.4725 | +0.0119 | 0.9330 | 0.9343 | +0.0012 |

## Does LitePack Gain More Under Tighter Budgets?

Yes, for the main generation-quality signal in this evaluation. LitePack's ROUGE-L margin over the best baseline at each budget is largest at 1200 tokens (+0.0332), with smaller margins at 1800 (+0.0076) and 2400 (+0.0119). When compared against the strongest competing method at each budget, keyword support follows the same pattern: the gain peaks at 1200 (+0.0048) and narrows as the budget expands (+0.0023 at 1800, +0.0012 at 2400). Note that keyword support is a lexical overlap metric, not a measure of semantic faithfulness.

The retrieval metrics do not distinguish the methods in this experiment: Recall@5, NDCG@5, and evidence hit rate are all 1.000 for every method and budget. This means the budget-sensitivity claim should be framed around how well the selected context supports the generated answer under a constrained input budget, not around whether the gold PMID appears in the top evidence.

## Redundancy And Token Efficiency

LitePack consistently reduces redundancy relative to the pure similarity baseline, and the redundancy reduction is strongest under the tightest budget:

| Budget | Similarity redundancy | LitePack redundancy | LitePack reduction vs similarity |
|---:|---:|---:|---:|
| 1200 | 0.6165 | 0.5817 | -0.0348 |
| 1800 | 0.5739 | 0.5522 | -0.0218 |
| 2400 | 0.5527 | 0.5448 | -0.0079 |

MMR remains the lowest-redundancy method at all budgets, which is expected because MMR directly optimizes diversity. LitePack is not the least redundant method overall, but it provides the best answer-quality metrics while also reducing redundancy versus similarity.

LitePack's token-efficiency advantage is also strongest when the context budget is tight. At 1200 tokens, LitePack uses 1133.7 average context tokens versus about 1158 for both baselines, reducing budget utilization from roughly 0.965 to 0.945. At 1800 tokens, LitePack still uses fewer tokens than both baselines. At 2400 tokens, utilization converges across all methods, suggesting the larger budget leaves less pressure for LitePack's length-aware selection to matter.

## Presentation Takeaway

The sensitivity experiment supports the LitePack claim most clearly at the tightest input budget, where context scarcity makes selection quality most consequential. When constrained to 1200 context tokens, LitePack achieves the best ROUGE-L and keyword support while using fewer tokens and reducing redundancy relative to similarity. As the budget expands to 1800 and 2400 tokens, all methods can include more evidence and margins narrow accordingly. These results support LitePack as a budget-aware context selector rather than a uniformly superior retriever.
