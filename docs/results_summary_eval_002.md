## Methods summary for the paper

LitePack-RAG was evaluated as a token-budget-aware context selection component for a retrieval-augmented generation pipeline over a small curated medical literature corpus. For each question, the system embedded the query with `text-embedding-3-small` using 512 dimensions and retrieved the top 20 candidate chunks from a Pinecone index, including metadata and stored vector values. Candidate chunks were enriched with token lengths, stored embeddings, keyword overlap, and recency scores. The prompt budget was fixed at 1,800 context tokens. All compared methods used the same retriever, the same retrieved candidate set size, the same prompt format, and the same generator, `gpt-4o-mini`, with temperature 0.1, seed 100, and a maximum generation length of 150 tokens. Thus, the experiment isolates differences in context ranking and selection rather than differences in retrieval or generation.

The three methods were similarity ordering, MMR, and LitePack. The similarity baseline filled the budget using retrieval-score ordering. MMR used a marginal relevance and redundancy criterion under the same budget. LitePack used a greedy marginal selector that combined query relevance, redundancy penalty, keyword coverage, length penalty, and a metadata bonus. In the reported run, keyword overlap and recency were enabled; publication type and MeSH features existed only as inactive stubs and should not be described as active experimental features. After selection, chunks were packed with source labels, and the generator was instructed to answer only from the provided sources and cite source labels inline. Evaluation was performed on 20 questions, with one output per method per question, for 60 JSONL records. The reported aggregates below were recomputed directly from `results.jsonl`.

## Main results summary

Retrieval-side metrics were saturated in this run. All 60 method-question records retrieved 20 chunks, and all methods achieved mean recall@5, nDCG@5, and evidence hit rate of 1.0000. Because these retrieval metrics are identical across similarity, MMR, and LitePack, the observed differences should not be interpreted as retrieval improvements. They reflect changes in how the same retrieved candidate pool was selected and packed into the generator prompt.

On answer-quality metrics, LitePack had the highest mean ROUGE-L, 0.4761, compared with 0.4673 for MMR and 0.4523 for the similarity baseline. LitePack also had the highest mean lexical keyword-support rate, 0.9308, compared with 0.9202 for MMR and 0.9181 for similarity. These differences are modest and should be reported as improvements in lexical overlap metrics, not as proof of clinical correctness or factual faithfulness. Per-question ROUGE-L results were mixed: counting ties as wins, MMR won 10 questions, LitePack won 8, and similarity won 7. Excluding ties, MMR and similarity each had 6 sole wins, while LitePack had 4 sole wins. Four questions had ties, including one three-way tie and three ties between MMR and LitePack.

LitePack was the most token-efficient method on average. It used 1,733.8 prompt tokens per question, compared with 1,757.4 for MMR and 1,765.0 for similarity, and its mean budget utilization was 0.9632 compared with 0.9763 for MMR and 0.9806 for similarity. LitePack also selected slightly more chunks on average, 12.95, than MMR, 12.60, or similarity, 12.55, while using fewer tokens. This pattern is consistent with the selector favoring shorter, budget-efficient chunks in some cases. Redundancy results were more favorable to MMR: MMR had the lowest mean redundancy score, 0.5442, followed by LitePack at 0.5522 and similarity at 0.5739. Therefore, LitePack's contribution in this run is best characterized as improving average ROUGE-L and keyword coverage while reducing prompt token use, rather than as the strongest redundancy-reduction method.

Because retrieval was saturated for every method, the gains are primarily packing and selection gains. LitePack did not improve the measured retrieval stage; it changed which already-retrieved chunks entered the prompt. The results support the project premise that context selection under a fixed budget can affect answer overlap and token efficiency even when the retriever and generator are held constant.

## Results table draft

| Method | Recall@5 | nDCG@5 | Evidence hit | Redundancy (lower) | Prompt tokens (lower) | Budget used (lower) | ROUGE-L (higher) | Keyword support (higher) | Mean chunks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Similarity | 1.0000 | 1.0000 | 1.0000 | 0.5739 | 1765.0 | 0.9806 | 0.4523 | 0.9181 | 12.55 |
| MMR | 1.0000 | 1.0000 | 1.0000 | **0.5442** | 1757.4 | 0.9763 | 0.4673 | 0.9202 | 12.60 |
| LitePack | 1.0000 | 1.0000 | 1.0000 | 0.5522 | **1733.8** | **0.9632** | **0.4761** | **0.9308** | 12.95 |

## Generator variant: gpt-4o comparison (eval_002_gpt4o)

A secondary run, tagged `eval_002_gpt4o` and timestamped `20260426_092514`, repeated the exact same evaluation protocol as `eval_002` with one change: the generator model was switched from `gpt-4o-mini` to `gpt-4o`. All other settings were held constant — same embedding model (`text-embedding-3-small`, 512 dims), same Pinecone index, same 1,800-token prompt budget, same temperature (0.1), same seed (100), and same maximum generation length (150 tokens). Because the retriever and selector are deterministic given the same inputs, all retrieval-side and packing-side values (`recall_at_5`, `ndcg_at_5`, `evidence_hit_rate`, `redundancy_score`, `prompt_token_count`, `budget_utilization`) are byte-identical between the two runs, confirming that the only source of difference is the generator.

On answer-quality metrics, gpt-4o improved ROUGE-L and keyword-support rate across all three methods. ROUGE-L gains ranged from +1.9 points for LitePack to +3.6 points for the similarity baseline. Keyword-support rate improved by +0.8 to +2.2 points. The largest relative improvement appeared on the similarity method, which benefited the most from a stronger generator despite having the highest prompt redundancy. The improvement was smallest for LitePack, consistent with the hypothesis that more focused, budget-efficient context packs reduce the room for a better generator to differentiate itself. Despite a shared 150-token output cap, gpt-4o packed higher-quality content within that limit; lifting the cap would likely widen the quality gap further.

Generation latency increased substantially with gpt-4o. Across all 60 calls, `generation_ms` averaged roughly 2,200–2,300 ms per call compared to approximately 500–1,500 ms for gpt-4o-mini in the prior run, representing a 2–3× slowdown. Several individual calls were considerably longer: the MMR run on the fNIRS neurofeedback CBT question reached 10,544 ms, and several other calls exceeded 4,000 ms. This variance is typical of frontier model API latency under concurrent load and is not a property of the pipeline itself.

### Generator comparison table (gpt-4o-mini vs gpt-4o, mean over 20 questions per method)

| Method | Model | Redundancy ↓ | Prompt tokens ↓ | Budget util ↓ | ROUGE-L ↑ | Keyword support ↑ |
|--------|-------|---:|---:|---:|---:|---:|
| Similarity | gpt-4o-mini | 0.5739 | 1765.0 | 0.9806 | 0.4523 | 0.9181 |
| Similarity | **gpt-4o** | 0.5739 | 1765.0 | 0.9806 | **0.4879** | **0.9402** |
| MMR | gpt-4o-mini | 0.5442 | 1757.4 | 0.9763 | 0.4673 | 0.9202 |
| MMR | **gpt-4o** | 0.5442 | 1757.4 | 0.9763 | **0.4916** | **0.9408** |
| LitePack | gpt-4o-mini | 0.5522 | 1733.8 | 0.9632 | 0.4761 | 0.9308 |
| LitePack | **gpt-4o** | 0.5522 | 1733.8 | 0.9632 | **0.4948** | **0.9384** |

Redundancy, prompt token count, and budget utilization are identical across model variants by design. ROUGE-L and keyword-support rate are the only metrics that differ. All retrieval metrics (`recall_at_5`, `ndcg_at_5`, `evidence_hit_rate`) remained 1.0000 for every (method, model) combination and are omitted from this table for brevity.

This comparison should not be interpreted as evidence that gpt-4o is categorically superior for this task. The evaluation set is small (20 questions), ROUGE-L is a lexical metric not a faithfulness metric, and the 150-token output cap constrains both models equally. The result does support the narrower claim that, under identical retrieval and selection conditions, a stronger generator produces answers with higher lexical overlap with the reference and better keyword coverage, regardless of which selection method is used.

## Ablation paragraph

No before/after feature-ablation result should be reported from the provided files. The pipeline notes describe earlier implementation behavior, including a legacy retrieval path and inactive publication-type and MeSH stubs, but the provided `results.jsonl` contains only the reported `eval_002` comparison among similarity, MMR, and LitePack. Without a directly comparable earlier results file with the same question set, methods, budget, metrics, and generator settings, an eval_001 versus eval_002 ablation would be unsupported.

## Error analysis paragraph

The hardest questions by mean ROUGE-L across methods were the cognitive therapy versus short-term dynamic psychotherapy comparison, mean 0.2159; the Internet Cognitive Therapy for Prolonged Grief symptom-reduction question, mean 0.2471; and the tacrolimus and cyclosporine adverse-event reporting question, mean 0.2963. These questions likely required more specific phrasing or multi-part factual alignment than the lexical metrics rewarded. Several outliers show that selection can help or hurt depending on the question: similarity strongly outperformed both alternatives on the burnout question, 0.9231 versus 0.6000 for MMR and 0.7500 for LitePack, while LitePack clearly outperformed the other methods on the cluster B personality disorder utilization question, 0.7731 versus 0.6167 for MMR and 0.5760 for similarity. LitePack also underperformed on the Turkish PARDI-AR-Q question, 0.4575 versus 0.6099 for similarity and 0.5833 for MMR.

## Threats to validity / limitations paragraph

The evaluation is small, with 20 questions and one generated answer per method per question, so the reported differences should be treated as exploratory. ROUGE-L measures lexical overlap with a reference answer and does not establish clinical correctness, factual consistency, or citation faithfulness. The keyword-support metric is also lexical and should not be described as a faithfulness metric. The provided files do not document whether the evaluation set was synthetically generated, so synthetic-eval bias should not be asserted from these files alone; if the questions or references were produced synthetically, that would further limit generalization. Finally, the saturated retrieval metrics indicate that this run does not test retrieval robustness under harder candidate pools or incomplete evidence retrieval.

## Presentation takeaway

In this experiment, all three methods used the same retriever and generator, so the comparison is about what gets packed into the prompt. Retrieval was already saturated, but LitePack produced the best average ROUGE-L and keyword-overlap scores while using fewer prompt tokens. The improvement is modest and lexical, not a claim of clinical correctness, but it supports the idea that budget-aware context selection can improve RAG behavior even without changing the retriever or the language model.

## Consistency audit: claims that must be worded carefully

- Do not claim that LitePack improved retrieval. Recall@5, nDCG@5, and evidence hit rate were 1.0000 for every method.
- Do not call keyword-support rate a faithfulness metric. It is lexical keyword overlap.
- Do not claim publication type or MeSH features were active. The pipeline notes identify them as inactive stubs for this run.
- Do not claim that LitePack was always best. MMR had the best mean redundancy score, and per-question ROUGE-L winners were mixed.
- Do not overstate the ROUGE-L improvement. LitePack's mean ROUGE-L advantage was modest: 0.4761 versus 0.4673 for MMR and 0.4523 for similarity.
- Do not report an eval_001 versus eval_002 ablation unless a comparable earlier results file is supplied and analyzed.
- Do not describe saturated retrieval metrics as evidence of real-world retrieval robustness. They only show that the provided evaluation run had perfect measured retrieval-side scores.
- Do not assert synthetic-eval bias as a documented fact from these files. The provided files do not establish whether the evaluation set was synthetic.
