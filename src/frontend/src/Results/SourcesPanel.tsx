import type { RunResult } from "../types";

export default function SourcesPanel({ result }: { result: RunResult }) {
  const sources = result.packed_context.citation_map;

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold">Sources</h3>

      {/* Scroll Container */}
      <div className="max-h-64 overflow-y-auto pr-1 space-y-2">
        {Object.entries(sources).map(([label, source]) => (
          <div
            key={label}
            className="bg-zinc-800 border border-zinc-700 rounded-xl p-3 flex justify-between"
          >
            <span className="text-zinc-400 text-sm">{label}</span>

            <div className="text-right text-sm">
              <div className="font-medium">{source.title}</div>
              <div className="text-zinc-400 text-xs">{source.year}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
