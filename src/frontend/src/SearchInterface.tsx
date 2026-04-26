import React, { useState } from "react";
import { runPipeline } from "./api";

export default function SearchInterface() {
  // Existing state
  const [query, setQuery] = useState("");

  // New parameter states based on your JSON example
  const [method, setMethod] = useState("litepack");
  const [topN, setTopN] = useState(20);
  const [budget, setBudget] = useState(1800);
  const [useKeywords, setUseKeywords] = useState(true);
  const [useRecency, setUseRecency] = useState(true);

  const handleRun = async () => {
    try {
      const results = await runPipeline({
        query,
        method,
        top_n: topN,
        budget,
        use_keywords: useKeywords,
        use_recency: useRecency,
      });
      console.log(results);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="bg-zinc-900/60 backdrop-blur border border-zinc-700 rounded-2xl p-6 shadow-lg space-y-4">
      {/* Search Bar Row */}
      <div className="flex gap-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask anything..."
          className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 outline-none focus:ring-2 focus:ring-zinc-500 text-white"
        />
        <button
          onClick={handleRun}
          className="px-5 py-3 rounded-xl bg-white text-black font-bold hover:bg-zinc-200 transition"
        >
          Run
        </button>
      </div>

      {/* Settings Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pt-2 border-t border-zinc-800">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-500 font-medium">Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-300 outline-none"
          >
            <option value="litepack">LitePack</option>
            <option value="mmr">MMR</option>
            <option value="similarity">Similarity</option>
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-500 font-medium">
            Budget (Tokens)
          </label>
          <input
            type="number"
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
            className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-300 outline-none"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-500 font-medium">Top N</label>
          <input
            type="number"
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            className="bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-sm text-zinc-300 outline-none"
          />
        </div>

        <div className="flex items-center gap-2 pt-5">
          <input
            type="checkbox"
            checked={useKeywords}
            onChange={(e) => setUseKeywords(e.target.checked)}
            className="accent-zinc-500"
          />
          <label className="text-xs text-zinc-300">Keywords</label>
        </div>

        <div className="flex items-center gap-2 pt-5">
          <input
            type="checkbox"
            checked={useRecency}
            onChange={(e) => setUseRecency(e.target.checked)}
            className="accent-zinc-500"
          />
          <label className="text-xs text-zinc-300">Recency</label>
        </div>
      </div>
    </div>
  );
}
