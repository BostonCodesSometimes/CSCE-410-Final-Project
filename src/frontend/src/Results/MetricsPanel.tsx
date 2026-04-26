import type { RunResult } from "../types";

export default function MetricsPanel({ result }: { result: RunResult }) {
  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold">Metrics</h3>

      <div className="grid grid-cols-3 gap-3">
        <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-3">
          <div className="text-xs text-zinc-400">Tokens</div>
          <div className="text-lg font-semibold">
            {result.metrics.tokens_used}
          </div>
        </div>

        <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-3">
          <div className="text-xs text-zinc-400">Chunks</div>
          <div className="text-lg font-semibold">
            {result.metrics.chunks_selected}
          </div>
        </div>

        <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-3">
          <div className="text-xs text-zinc-400">Latency</div>
          <div className="text-lg font-semibold">
            {result.timings.total_ms.toFixed(0)} ms
          </div>
        </div>
      </div>
    </div>
  );
}
