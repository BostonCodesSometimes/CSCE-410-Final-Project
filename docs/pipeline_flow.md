Good, I have everything I need. Here is the complete picture:

---

## Full Pipeline Flow — Fresh Run

The default method when you run `pipeline.py` is **LitePack + greedy selector**. Using a concrete example query throughout:

> **Query: `"What are effective treatments for treatment-resistant depression?"`**

---

### Step 0 — Startup

`pipeline.py main()` runs. It:
1. Reads `LLM_MODEL`, `EMBEDDING_MODEL` from env → builds `PipelineConfig` (`llm_model="gpt-4o-mini"`, `embedding_model="text-embedding-3-small"`, `budget=1800`, `top_n=20`)
2. Instantiates `Retriever(config)` → connects to Pinecone, detects index metric (`cosine`), creates `OpenAIEmbeddings`
3. Instantiates `Generator(config)` → reads `TEMPERATURE=0.1`, `SEED=100`, `MAX_OUTPUT_TOKENS=150` from env; opens `openai.OpenAI()` client

---

### Step 1 — Retrieval (`retriever.py`)

The query string is embedded using OpenAI:

```
OpenAI API call:
  model: text-embedding-3-small
  dimensions: 512
  input: "What are effective treatments for treatment-resistant depression?"
  → returns a 512-dimensional vector
```

That vector is passed directly to the Pinecone index via `self.index.query(vector=..., top_k=20, include_metadata=True, include_values=True)`. Pinecone returns the 20 most similar chunks **including their stored vector values**. Each result becomes a `Candidate` object with fields like `chunk_id`, `pmid`, `text`, `retrieval_score`, `title`, `year`, `journal`, `keywords`, and — critically — `embedding` pre-populated from the returned vector values.

> **Architecture note:** The previous implementation used `PineconeVectorStore.similarity_search_with_score()`, which does not return stored vectors. A legacy version of that method is preserved as `Retriever._retrieve_langchain_legacy()` for debugging/fallback, but it is not called by default.

---

### Step 2 — Enrichment (`features.py`)

All 20 candidates are enriched in-place:

- **`add_token_lengths`** — tiktoken counts tokens in each chunk's text
- **`add_embeddings`** — checks whether `Candidate.embedding` is already populated.
  Because Step 1 now uses `include_values=True`, all 20 candidates arrive with
  embeddings pre-filled from the Pinecone-stored vectors.  `add_embeddings()`
  detects this and **skips the OpenAI call entirely**:
  ```
  # No OpenAI API call made — embeddings already present from Pinecone retrieval.
  ```
  If the legacy LangChain retrieval path were used instead, `Candidate.embedding`
  would be `None` and `add_embeddings()` would fall back to its original behaviour
  (one batched `embed_documents` call for all missing candidates).
- **`add_keyword_overlap`** (if `--use-keywords`) — Jaccard similarity between query tokens and each chunk's keyword/title tokens → `candidate.keyword_overlap`
- **`add_recency_feature`** (if `--use-recency`) — scores each chunk by how recent its publication year is → `candidate.recency_score`

> **Reported experiment note:** In the evaluation run described in this document, only keyword overlap and recency were enabled. Publication type and MeSH term features exist as stubs in `features.py` and are wired into the `metadata_bonus` term, but they were not activated for this run and should not be treated as active experimental features.

---

### Step 3 — Ranking (`rankers.py` — `LitePackRanker`)

`LitePackRanker.rank()` does a **static pre-sort** by `retrieval_score` descending. No composite scoring yet — that happens per-step inside the greedy loop.

> **Method mapping (as implemented in `pipeline.py`):** `similarity` = retrieval-score ordering + baseline budget fill; `mmr` = MMR marginal scoring + greedy budgeted selection; `litepack` = composite marginal scoring (relevance, redundancy, coverage, length penalty, metadata bonus) + greedy budgeted selection.

---

### Step 4 — Greedy Selection (`selector.py` — `select_greedy`)

The greedy loop runs up to 20 iterations. At each step it:
1. Evaluates every remaining candidate using `LitePackRanker.marginal_score()`:

```
score = α × relevance       (cosine sim of chunk embedding to query)
      - β × redundancy      (max cosine sim to already-selected chunks)
      + γ × coverage        (keyword_overlap Jaccard score)
      - δ × length_penalty  (token_length / 300, capped at 1.0)
      + ε × metadata_bonus  (weighted avg of recency/pubtype/mesh if enabled)

With defaults: α=1.0, β=0.6, γ=0.3, δ=0.2, ε=0.2
```

2. Picks the highest-scoring candidate, checks if its token cost fits within the remaining budget (1800 tokens total)
3. If it fits: assigns it a label (`S1`, `S2`, …), adds it to selected set, subtracts its token cost
4. If it doesn't fit: drops it and continues
5. Loop ends when budget is exhausted or all candidates are evaluated

Selection continues until no remaining candidate fits within the token budget.

---

### Step 5 — Packing (`packer.py`)

The selected candidates are assembled into a single context string. Each chunk becomes a block:

```
[S1] PMID: 38291047 | Year: 2023 | Title: Ketamine and esketamine for treatment-resistant depression
Ketamine has emerged as a rapid-acting antidepressant for patients with treatment-resistant
depression (TRD). A systematic review of 23 RCTs found that intravenous ketamine produced
significant antidepressant effects within hours, with response rates of 40–70%...

[S2] PMID: 37104852 | Year: 2022 | Title: Transcranial magnetic stimulation in TRD
Repetitive transcranial magnetic stimulation (rTMS) targeting the left dorsolateral
prefrontal cortex showed response rates of 30–55% in patients who failed ≥2 antidepressants...

[S3] PMID: 36841209 | Year: 2023 | Title: Electroconvulsive therapy outcomes
ECT remains the most effective treatment for severe TRD with remission rates of 60–80%
in appropriately selected patients...
```

Token count is verified to be ≤ 1800.

---

### Step 6 — Generation (`generator.py`)

The OpenAI chat completion call is configured as:

```
OpenAI API call:
  model: gpt-4o-mini
  temperature: 0.1
  seed: 100
  max_tokens: 150
  messages:
    [
      {
        "role": "user",
        "content":
          "You are a medical literature assistant.
           Answer the question using ONLY the sources provided below.
           Cite each source you use inline as [S1], [S2], etc., where the
           label matches the source header.
           Be concise and factual. Do not include information that is not
           present in the sources.

           --- SOURCES ---
           [S1] PMID: 38291047 | Year: 2023 | Title: Ketamine and esketamine...
           Ketamine has emerged as a rapid-acting antidepressant...

           [S2] PMID: 37104852 | Year: 2022 | Title: Transcranial magnetic stimulation...
           Repetitive transcranial magnetic stimulation...

           [S3] PMID: 36841209 | Year: 2023 | Title: Electroconvulsive therapy...
           ECT remains the most effective treatment...
           --- END SOURCES ---

           Question: What are effective treatments for treatment-resistant depression?

           Answer:"
      }
    ]
```

The model replies with something like:

```
For treatment-resistant depression, ketamine and esketamine have shown rapid
antidepressant effects within hours, with response rates of 40–70% in RCTs [S1].
Repetitive TMS targeting the left DLPFC achieves response rates of 30–55%
after prior antidepressant failures [S2]. ECT remains the most effective option
for severe cases, with remission rates of 60–80% [S3].
```

The pipeline extracts `["[S1]", "[S2]", "[S3]"]` as cited labels, looks up the corresponding PMIDs from the `citation_map`, and packages everything into the final `RunResult`.

---

### What You Actually Run

```bash
cd src
python pipeline.py --query "What are effective treatments for treatment-resistant depression?"
```

Or with all methods compared side by side:

```bash
python pipeline.py \
  --query "What are effective treatments for treatment-resistant depression?" \
  --compare similarity mmr litepack \
  --show-scores
```

Or to run the full evaluation suite:

```bash
python evaluation/evaluation.py \
  --eval-set ../results/eval_data.json \
  --methods similarity mmr litepack
```

---

### Total API Calls Per Query (default LitePack run)

| Call | Where | Purpose |
|---|---|---|
| 1× embeddings | `retriever.py` | Embed the query string |
| 1× Pinecone query | `retriever.py` | Retrieve top-20 chunks **with stored vectors** (`include_values=True`) |
| 1× chat completion | `generator.py` | Generate the answer |

So **2 OpenAI API calls** per query (down from 3). The second embedding call
that `features.py` previously made to embed all 20 candidate chunk texts is
**eliminated** because `Retriever.retrieve()` now fetches stored vector values
directly from Pinecone in the same retrieval round-trip.

> **Previous behaviour (now legacy):** When the LangChain `similarity_search_with_score`
> path was used, Pinecone did not return stored vectors, so `features.add_embeddings()`
> made a second `embed_documents(...)` call for all 20 chunks. That path is still
> available as `Retriever._retrieve_langchain_legacy()` for debugging.