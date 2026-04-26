import { useState } from "react";
import type { RunResult } from "./types";
import { comparePipeline } from "./api";
import { motion } from "framer-motion";

interface CompareViewProps {
  onCompare?: (results: RunResult[]) => void;
}

export default function CompareView({ onCompare }: CompareViewProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RunResult[] | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCompare = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const data = await comparePipeline(query);
      setResults(data);

      if (onCompare) {
        onCompare(data);
      }
    } catch (e) {
      console.error(e);
      alert("Comparison failed");
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Compare Methods</h2>
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter query..."
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-zinc-500"
        />
        <button
          onClick={handleCompare}
          className="px-5 py-3 rounded-xl bg-white text-black font-medium hover:bg-zinc-200 transition"
        >
          Compare
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="text-sm text-zinc-400">Running comparison...</div>
      )}

      {/* Results */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {results.map((r) => (
            <motion.div
              key={r.method_name}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow space-y-4"
            >
              {/* Title */}
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-lg">{r.method_name}</h3>
                <span className="text-xs text-zinc-400">
                  {r.timings.total_ms.toFixed(0)} ms
                </span>
              </div>

              {/* Metrics */}
              <div className="flex gap-4 text-sm">
                <div className="bg-zinc-800 px-3 py-2 rounded-lg">
                  <div className="text-zinc-400 text-xs">Tokens</div>
                  <div className="font-medium">
                    {r.packed_context.tokens_used}
                  </div>
                </div>

                <div className="bg-zinc-800 px-3 py-2 rounded-lg">
                  <div className="text-zinc-400 text-xs">Chunks</div>
                  <div className="font-medium">{r.metrics.chunks_selected}</div>
                </div>
              </div>

              {/* Answer */}
              <div>
                <div className="text-xs text-zinc-400 mb-1">Answer</div>
                <div className="text-sm leading-relaxed line-clamp-6">
                  {r.generation_result.answer_text}
                </div>
              </div>

              {/* Sources */}
              <div>
                <div className="text-xs text-zinc-400 mb-1">Sources</div>
                <div className="space-y-1 text-sm">
                  {Object.entries(r.packed_context.citation_map).map(
                    ([label, s]) => (
                      <div
                        key={label}
                        className="flex justify-between bg-zinc-800 px-2 py-1 rounded"
                      >
                        <span className="text-zinc-400">{label}</span>
                        <span className="truncate ml-2">{s.title}</span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
