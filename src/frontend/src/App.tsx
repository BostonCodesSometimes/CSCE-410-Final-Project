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
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [compareResults, setCompareResults] = useState<RunResult[] | null>(
    null,
  );

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const data = await runPipeline(query);
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
          <h1 className="text-3xl font-semibold tracking-tight">
            LitePack-RAG
          </h1>
          <span className="text-sm text-zinc-400">
            Semantic Search Pipeline
          </span>
        </div>

        {/* Search Bar */}
        <div className="bg-zinc-900/60 backdrop-blur border border-zinc-700 rounded-2xl p-4 shadow-lg">
          <div className="flex gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything..."
              className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-zinc-500"
            />
            <button
              onClick={handleRun}
              className="px-5 py-3 rounded-xl bg-white text-black font-medium hover:bg-zinc-200 transition"
            >
              Run
            </button>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center text-zinc-400"
          >
            Running pipeline...
          </motion.div>
        )}

        {/* Results */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Answer (FULL WIDTH PRIMARY) */}
            <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
              <AnswerPanel result={result} />
            </div>

            <div className="text-sm text-zinc-400">Pipeline Details</div>
            {/* Secondary Panels */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
                <SourcesPanel result={result} />
              </div>

              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
                <TracePanel result={result} />
              </div>

              <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
                <MetricsPanel result={result} />
              </div>
            </div>
          </motion.div>
        )}

        {/* Pipeline Visualizer */}
        {result && (
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-5 shadow">
            <PipelineVisualizer result={result} />
          </div>
        )}

        {/* Compare View */}
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
