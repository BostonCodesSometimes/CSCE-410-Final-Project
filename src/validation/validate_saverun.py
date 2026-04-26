"""Save-run + no-embedding validation — run with the project venv active."""
from __future__ import annotations

import json
import os
import tempfile

from config import PipelineConfig
from retriever import Retriever
from generator import Generator
from pipeline import run_once, _run_result_to_dict, _NumpyEncoder

print("=== Save-run / serialization validation ===\n")

cfg = PipelineConfig(
    top_n=5,
    budget=1800,
    use_keywords=True,
    use_recency=True,
    ranker="litepack",
    selector="greedy",
)
retriever = Retriever(cfg)
generator = Generator(cfg)

query = "What are treatments for PTSD?"
print(f"Running pipeline for: {query!r}\n")

result = run_once(query, cfg, retriever, retriever.embeddings_model, generator)

print(f"method_name     : {result.method_name}")
print(f"chunks selected : {result.metrics.get('chunks_selected')}")
print(f"tokens used     : {result.metrics.get('tokens_used')}")
print(f"selected_pmids  : {result.packed_context.selected_pmids}")
print()

# Serialize
payload = _run_result_to_dict(result)
out_path = os.path.join(tempfile.gettempdir(), "litepack_run_test.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
print(f"Saved to: {out_path}")

# Reload and verify
with open(out_path, "r", encoding="utf-8") as fh:
    reloaded = json.load(fh)

print(f"method_name in JSON : {reloaded['method_name']}")
print(f"selected_pmids      : {reloaded['packed_context']['selected_pmids']}")
print()

# --- Assertions ---
for c in reloaded["retrieved_candidates"]:
    assert "embedding" not in c, f"FAIL: embedding found in candidate {c['chunk_id']}"
print("PASS: no embedding field in any retrieved_candidate dict")

for c in reloaded["ranked_candidates"]:
    assert "embedding" not in c
print("PASS: no embedding field in any ranked_candidate dict")

for c in reloaded["selection_result"]["selected_candidates"]:
    assert "embedding" not in c
print("PASS: no embedding field in any selected_candidate dict")

for label, c in reloaded["packed_context"]["citation_map"].items():
    assert "embedding" not in c, f"FAIL: embedding in citation_map[{label!r}]"
print("PASS: no embedding field in citation_map values")

# Confirm the JSON is valid (already reloaded above)
print("PASS: JSON is well-formed and fully reloadable")

# Spot-check key fields exist
assert reloaded["generation_result"]["answer_text"], "FAIL: answer_text is empty"
assert "timings" in reloaded
assert set(reloaded["timings"].keys()) == {
    "retrieval_ms", "enrichment_ms", "ranking_ms",
    "selection_ms", "packing_ms", "generation_ms", "total_ms",
}, f"FAIL: unexpected timings keys: {reloaded['timings'].keys()}"
print("PASS: all 7 timing keys present")

print()
print(f"=== ALL save-run assertions PASSED  (file: {out_path}) ===")
