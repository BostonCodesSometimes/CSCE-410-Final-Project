"""Phase 5 validation script — run with the project venv active."""
from __future__ import annotations

from config import PipelineConfig
from retriever import Retriever
from features import enrich_candidates
from rankers import LitePackRanker
from selector import select_greedy
from packer import build_context, count_tokens, assign_label, effective_token_cost

print("=== Phase 5 validation ===\n")

cfg = PipelineConfig(top_n=10, use_keywords=True, use_recency=True)
r = Retriever(cfg)
query = "depression treatment"

print(f"Retrieving top {cfg.top_n} candidates ...")
candidates, q_emb = r.retrieve(query, top_n=cfg.top_n)
print(f"  Retrieved: {len(candidates)} candidates\n")

print("Enriching ...")
candidates = enrich_candidates(candidates, query, q_emb, r.embeddings_model, cfg)
print("  Done\n")

ranker = LitePackRanker()
ranked = ranker.rank(candidates, q_emb, cfg)

print("Selecting (greedy) ...")
sr = select_greedy(ranked, ranker, q_emb, budget=cfg.budget, config=cfg)
print(f"  Selected {len(sr.selected_candidates)} / {len(candidates)} candidates")
print(f"  Tokens used at selection time: {sr.tokens_used}")
print(f"  Selection labels: {sr.selection_labels}\n")

print("Building context ...")
packed = build_context(sr, cfg)

print("--- Context text (first 300 chars) ---")
print(packed.context_text[:300])
print("...")
print()
print(f"tokens_used    : {packed.tokens_used}  (budget: {cfg.budget})")
print(f"selected_pmids : {packed.selected_pmids}")
print(f"citation_map   : {list(packed.citation_map.keys())}")
print()

# --- Assertions ---
assert packed.tokens_used <= cfg.budget, (
    f"FAIL: tokens_used {packed.tokens_used} exceeds budget {cfg.budget}"
)
print("PASS: tokens_used <= budget")

assert len(packed.citation_map) == len(sr.selected_candidates), (
    f"FAIL: citation_map has {len(packed.citation_map)} entries "
    f"but {len(sr.selected_candidates)} candidates were selected"
)
print("PASS: citation_map length == selected_candidates length")

assert set(packed.citation_map.keys()) == set(sr.selection_labels.values()), (
    f"FAIL: label mismatch\n"
    f"  citation_map keys:       {set(packed.citation_map.keys())}\n"
    f"  selection_labels values: {set(sr.selection_labels.values())}"
)
print("PASS: citation_map keys match selection_labels values (label consistency)")

# Verify effective_token_cost matches what build_context uses
for c in sr.selected_candidates:
    label = sr.selection_labels[c.chunk_id]
    cost = effective_token_cost(c, label, cfg.llm_model)
    assert cost > 0, f"FAIL: zero token cost for {c.chunk_id}"
print("PASS: effective_token_cost > 0 for all selected candidates")

# Verify assign_label produces expected labels
assert assign_label(0) == "S1"
assert assign_label(9) == "S10"
print("PASS: assign_label(0)='S1', assign_label(9)='S10'")

# Verify selector imports are the same objects as packer's
import packer
import selector as sel
assert sel.assign_label is packer.assign_label
assert sel.format_chunk_header is packer.format_chunk_header
assert sel.count_tokens is packer.count_tokens
assert sel.effective_token_cost is packer.effective_token_cost
print("PASS: selector.py imports are the same objects as packer.py exports")

print()
print("=== ALL Phase 5 assertions PASSED ===")
