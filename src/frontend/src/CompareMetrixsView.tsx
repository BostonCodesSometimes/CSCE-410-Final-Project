import type { RunResult } from "./types";

type Props = {
  results: RunResult[];
};

export default function CompareMetricsView({ results }: Props) {
  const metrics = results.map((r) => {
    const tokens = r.packed_context.tokens_used;
    const chunks = r.metrics.chunks_selected;
    const latency = r.timings.total_ms;
    const answerLength = r.generation_result.answer_text.length;

    const avgRetrieval =
      r.ranked_candidates.reduce((acc, c) => acc + c.retrieval_score, 0) /
      (r.ranked_candidates.length || 1);

    return {
      method: r.method_name,
      tokens,
      chunks,
      latency,
      answerLength,
      coverage: avgRetrieval,
    };
  });

  const best = {
    tokens: Math.min(...metrics.map((m) => m.tokens)),
    latency: Math.min(...metrics.map((m) => m.latency)),
    coverage: Math.max(...metrics.map((m) => m.coverage)),
  };

  return (
    <div className="space-y-4 mt-6">
      <h2 className="text-lg font-semibold">Comparison Metrics</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm border border-zinc-800 rounded-xl overflow-hidden">
          <thead className="bg-zinc-900 text-zinc-400">
            <tr>
              <th className="text-left p-3">Method</th>
              <th className="text-left p-3">Tokens</th>
              <th className="text-left p-3">Chunks</th>
              <th className="text-left p-3">Latency</th>
              <th className="text-left p-3">Coverage</th>
              <th className="text-left p-3">Answer Length</th>
            </tr>
          </thead>

          <tbody>
            {metrics.map((m) => {
              const isBestTokens = m.tokens === best.tokens;
              const isBestLatency = m.latency === best.latency;
              const isBestCoverage = m.coverage === best.coverage;

              return (
                <tr
                  key={m.method}
                  className="border-t border-zinc-800 hover:bg-zinc-900/60 transition"
                >
                  <td className="p-3 font-medium text-white">{m.method}</td>

                  <td
                    className={`p-3 ${
                      isBestTokens
                        ? "text-green-400 font-semibold"
                        : "text-zinc-300"
                    }`}
                  >
                    {m.tokens}
                  </td>

                  <td className="p-3 text-zinc-300">{m.chunks}</td>

                  <td
                    className={`p-3 ${
                      isBestLatency
                        ? "text-green-400 font-semibold"
                        : "text-zinc-300"
                    }`}
                  >
                    {m.latency.toFixed(0)} ms
                  </td>

                  <td
                    className={`p-3 ${
                      isBestCoverage
                        ? "text-green-400 font-semibold"
                        : "text-zinc-300"
                    }`}
                  >
                    {m.coverage.toFixed(3)}
                  </td>

                  <td className="p-3 text-zinc-300">{m.answerLength}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
