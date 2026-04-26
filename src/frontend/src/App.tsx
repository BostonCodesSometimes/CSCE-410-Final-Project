import { useState } from "react";
import { runPipeline } from "./api";
import type { RunResult } from "./types";
import AnswerPanel from "./Results/AnswerPanel";
import SourcesPanel from "./Results/SourcesPanel";
import TracePanel from "./Results/SelectionTrace";
import MetricsPanel from "./Results/MetricsPanel";
import PipelineVisualizer from "./PipelineVisualizer";
import CompareView from "./CompareView";
import { motion } from "framer-motion";
import CompareMetricsView from "./CompareMetrixsView";

export default function App() {
  // --- Pipeline State ---
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("litepack");
  const [topN, setTopN] = useState(20);
  const [budget, setBudget] = useState(1800);
  const [useKeywords, setUseKeywords] = useState(true);
  const [useRecency, setUseRecency] = useState(true);

  // --- UI State ---
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [compareResults, setCompareResults] = useState<RunResult[] | null>(
    null,
  );

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      // Pass the full configuration object to the API
      const data = await runPipeline({
        query,
        method,
        top_n: topN,
        budget,
        use_keywords: useKeywords,
        use_recency: useRecency,
      });
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error running pipeline");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-900 to-zinc-800 text-white p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-100">
            LitePack-RAG
          </h1>
          <span className="text-sm text-zinc-400">
            Semantic Search Pipeline
          </span>
        </div>

        {/* Search & Settings Card */}
        <div className="bg-zinc-900/60 backdrop-blur border border-zinc-700 rounded-2xl p-6 shadow-xl space-y-5">
          {/* Main Search Row */}
          <div className="flex gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a medical question..."
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-zinc-500 transition-all text-zinc-100"
            />
            <button
              onClick={handleRun}
              disabled={loading}
              className="px-8 py-3 rounded-xl bg-white text-black font-bold hover:bg-zinc-200 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Running..." : "Run"}
            </button>
          </div>

          {/* Parameters Row */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-6 pt-4 border-t border-zinc-800">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-zinc-500 font-bold uppercase tracking-wider">
                Method
              </label>
              <select
                value={method}
                onChange={(e) => setMethod(e.target.value)}
                className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-200 outline-none focus:border-zinc-500"
              >
                <option value="litepack">LitePack</option>
                <option value="mmr">MMR</option>
                <option value="similarity">Similarity</option>
              </select>
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-zinc-500 font-bold uppercase tracking-wider">
                Budget
              </label>
              <input
                type="number"
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
                className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-200 outline-none focus:border-zinc-500"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs text-zinc-500 font-bold uppercase tracking-wider">
                Top N
              </label>
              <input
                type="number"
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-200 outline-none focus:border-zinc-500"
              />
            </div>

            <div className="flex items-center gap-3 pt-5">
              <input
                type="checkbox"
                id="keywords"
                checked={useKeywords}
                onChange={(e) => setUseKeywords(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-700 bg-zinc-800 text-zinc-500 focus:ring-zinc-500"
              />
              <label
                htmlFor="keywords"
                className="text-sm text-zinc-300 cursor-pointer select-none"
              >
                Keywords
              </label>
            </div>

            <div className="flex items-center gap-3 pt-5">
              <input
                type="checkbox"
                id="recency"
                checked={useRecency}
                onChange={(e) => setUseRecency(e.target.checked)}
                className="w-4 h-4 rounded border-zinc-700 bg-zinc-800 text-zinc-500 focus:ring-zinc-500"
              />
              <label
                htmlFor="recency"
                className="text-sm text-zinc-300 cursor-pointer select-none"
              >
                Recency
              </label>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center justify-center py-12 space-y-4"
          >
            <div className="w-8 h-8 border-4 border-zinc-700 border-t-white rounded-full animate-spin" />
            <p className="text-zinc-400 animate-pulse">
              Processing through LitePack pipeline...
            </p>
          </motion.div>
        )}

        {/* Results Panels */}
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
              <AnswerPanel result={result} />
            </div>

            <div className="text-sm text-zinc-500 font-medium uppercase tracking-widest px-1">
              Pipeline Analysis
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow hover:border-zinc-600 transition">
                <SourcesPanel result={result} />
              </div>

              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow hover:border-zinc-600 transition">
                <TracePanel result={result} />
              </div>

              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow hover:border-zinc-600 transition">
                <MetricsPanel result={result} />
              </div>
            </div>

            <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
              <PipelineVisualizer result={result} />
            </div>
          </motion.div>
        )}

        {/* Comparison Section */}
        <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
          <CompareView onCompare={setCompareResults} />
        </div>

        {compareResults && (
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
            <CompareMetricsView results={compareResults} />
          </div>
        )}
      </div>
    </div>
  );
}
