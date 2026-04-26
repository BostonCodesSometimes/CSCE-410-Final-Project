## CSCE-410-Final-Project

# LitePack-RAG for Medical Information Retrieval Systems
### Team C1
Aquib Raza Raja, Audrey Vazzana, Irfan Gazi, Kishan Agrahari, Boston Bailey, Liam Makela, Alex Varvil

---

### Project Goals
- Retrieval-Augmented Generation (RAG) system optimizing document selection under fixed token budgets.
- Implements token-aware context packing to improve information density and reduce inference cost.
- Evaluates performance using Recall@K, nDCG, faithfulness scoring, and quality-per-token metrics.
- Compares baseline retrieval pipelines against budget-aware strategies through controlled experiments.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Environment Configuration](#environment-configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [Running Evaluation](#running-evaluation)
6. [Running Tests](#running-tests)
7. [Project Structure](#project-structure)
8. [Pipeline Configuration Options](#pipeline-configuration-options)

---

## Prerequisites

- Python 3.10 or higher
- Access to an OpenAI API key (`gpt-4o-mini` + `text-embedding-3-small`)
- Access to the shared Pinecone index (`lite-rag`) — ask a teammate for the credentials
- (Optional) NCBI/Entrez credentials if re-fetching PubMed data

---

## Setup

```bash
# 1. Clone the repo and enter the project directory
git clone <repo-url>
cd CSCE-410-Final-Project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Configuration

All secrets and model settings are loaded from a `.env` file in the project root.

```bash
# Copy the template and fill in your credentials
cp .env.template .env
```

Then open `.env` and replace every `<placeholder>` with a real value. The keys you **must** set are:

| Key | Description |
|-----|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `PINECONE_API_KEY` | Pinecone API key for the shared index |
| `PINECONE_INDEX_NAME` | Pinecone index name (e.g. `lite-rag`) |
| `PINECONE_HOST` | Full Pinecone host URL |
| `PINECONE_REGION` | Pinecone region (e.g. `us-east-1`) |
| `NCBI_EMAIL` | Your email address (required by Biopython Entrez) |

> `.env` is git-ignored. Never commit it. The tracked file is `.env.template`.

---

## Running the Pipeline

All pipeline commands are run from the **`src/`** directory (imports are relative to it).

```bash
cd src
```

### Single query — default method (LitePack)

```bash
python pipeline.py --query "What are the risk factors for type 2 diabetes?"
```

### Single query — choose a specific method

```bash
# Baseline cosine-similarity ranker
python pipeline.py --query "What are the risk factors for type 2 diabetes?" \
    --ranker similarity --selector baseline

# MMR ranker (diversity-aware)
python pipeline.py --query "What are the risk factors for type 2 diabetes?" \
    --ranker mmr --selector greedy

# LitePack ranker (default, token-budget-aware)
python pipeline.py --query "What are the risk factors for type 2 diabetes?" \
    --ranker litepack --selector greedy
```

### Compare all three methods side by side

```bash
python pipeline.py \
    --query "What are the risk factors for type 2 diabetes?" \
    --compare similarity mmr litepack
```

### Useful optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--budget INT` | `1800` | Token budget for context packing |
| `--top-n INT` | `20` | Number of candidates retrieved from Pinecone |
| `--use-keywords` | off | Enable keyword-overlap scoring feature |
| `--use-recency` | off | Enable publication-recency scoring feature |
| `--show-scores` | off | Print per-candidate scores after selection |
| `--show-trace` | off | Print the greedy selection trace |
| `--save-run PATH` | — | Save full run result to a JSON file |
| `--log-level LEVEL` | `WARNING` | Verbosity: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Example: verbose run with recency + keyword features, saved output

```bash
python pipeline.py \
    --query "What are the risk factors for type 2 diabetes?" \
    --ranker litepack --selector greedy \
    --budget 2000 \
    --use-keywords --use-recency \
    --show-scores --show-trace \
    --save-run ../results/my_run.json \
    --log-level INFO
```

---

## Running Evaluation

The evaluation runner scores all three methods over the full eval set and writes a timestamped results directory.

Commands are also run from **`src/`**.

```bash
cd src
```

### Basic evaluation run (all three methods, default settings)

```bash
python evaluation/evaluation.py \
    --eval-set ../results/evaluation_data/eval_data.json
```

### Evaluation with a custom tag (used in the output directory name)

```bash
python evaluation/evaluation.py \
    --eval-set ../results/evaluation_data/eval_data.json \
    --run-tag eval_003
```

Results are written to `results/runs/<timestamp>_<tag>/`:

| File | Contents |
|------|----------|
| `results.jsonl` | One JSON record per (question, method) pair |
| `summary.json` | Aggregate metrics by method |
| `metrics.csv` | Per-question metric rows (importable to Excel / pandas) |
| `config.json` | Snapshot of PipelineConfig used for the run |
| `eval_set_snapshot.json` | Copy of the eval set used |
| `logs.txt` | Full debug log |

### Evaluation with specific methods or custom budget

```bash
python evaluation/evaluation.py \
    --eval-set ../results/evaluation_data/eval_data.json \
    --methods litepack mmr \
    --budget 2000 \
    --top-n 25 \
    --use-keywords \
    --run-tag my_ablation \
    --log-level INFO
```

### Write results to a custom directory

```bash
python evaluation/evaluation.py \
    --eval-set ../results/evaluation_data/eval_data.json \
    --output-dir ../results/runs/my_custom_run
```

---

## Running Tests

Tests are run from the **project root** using `pytest`.

```bash
# From project root
python -m pytest tests/ -v
```

### Run a specific test file

```bash
python -m pytest tests/test_add_embeddings.py -v
```

### Run without pytest (plain Python)

```bash
python tests/test_add_embeddings.py
```

> The test suite mocks all external API calls (OpenAI, Pinecone) so no credentials are needed.

---

## Project Structure

```
CSCE-410-Final-Project/
├── .env.template            # Copy to .env and fill in credentials
├── requirements.txt         # Python dependencies
├── src/
│   ├── pipeline.py          # Entrypoint: single-query and compare-mode CLI
│   ├── retriever.py         # Pinecone vector search + candidate construction
│   ├── rankers.py           # Similarity, MMR, and LitePack rankers
│   ├── selector.py          # Baseline and greedy token-budget selectors
│   ├── packer.py            # Context packing and ordering policies
│   ├── generator.py         # OpenAI answer generation
│   ├── features.py          # Candidate enrichment (embeddings, recency, keywords)
│   ├── metrics.py           # evaluate_run() — Recall@K, nDCG, ROUGE-L, etc.
│   ├── run_dir.py           # Timestamped results directory helpers
│   ├── pinecone_settings.py # Credential loading from .env / environment
│   ├── config/
│   │   └── config.py        # PipelineConfig dataclass (single source of truth)
│   ├── data/
│   │   ├── data_load.py     # PubMed fetching via NCBI Entrez
│   │   └── queries.py       # Predefined query sets
│   └── evaluation/
│       ├── evaluation.py    # Evaluation runner CLI
│       └── build_eval_set.py
├── tests/
│   └── test_add_embeddings.py
├── results/
│   ├── evaluation_data/
│   │   └── eval_data.json   # Eval question set with gold PMIDs
│   └── runs/                # Auto-created timestamped run directories
└── docs/
    ├── pipeline_flow.md     # Detailed pipeline architecture notes
    └── results_summary_eval_002.md
```

---

## Pipeline Configuration Options

All settings live in `src/config/config.py` as `PipelineConfig`. Key defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ranker` | `litepack` | `similarity` / `mmr` / `litepack` |
| `selector` | `greedy` | `baseline` / `greedy` |
| `budget` | `1800` | Token budget for context window |
| `top_n` | `20` | Candidates fetched from Pinecone |
| `mmr_lambda` | `0.6` | MMR diversity vs. relevance trade-off |
| `litepack_alpha` | `1.0` | Relevance coefficient |
| `litepack_beta` | `0.6` | Redundancy penalty coefficient |
| `litepack_gamma` | `0.3` | Keyword-coverage bonus |
| `litepack_delta` | `0.2` | Length-penalty coefficient |
| `litepack_epsilon` | `0.2` | Metadata-bonus coefficient |
| `use_recency` | `False` | Publication-recency scoring |
| `use_keywords` | `False` | Keyword-overlap scoring |
| `ordering_policy` | `score_desc` | `score_desc` / `support_first` / `year_desc` / `original_rank` |

Models and token limits are read from the `.env` file (`LLM_MODEL`, `EMBEDDING_MODEL`, `MAX_OUTPUT_TOKENS`, `TEMPERATURE`, `SEED`).
