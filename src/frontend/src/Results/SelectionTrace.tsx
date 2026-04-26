import type { RunResult } from "../types";

export default function TracePanel({ result }: { result: RunResult }) {
  const trace = result.selection_result.trace;

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold">Selection Trace</h3>

      {/* Scroll container */}
      <div className="max-h-64 overflow-y-auto pr-1 space-y-2">
        {trace.map((step) => (
          <div
            key={step.step_index}
            className="flex justify-between items-center bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2"
          >
            <div className="text-sm">
              Step {step.step_index}:{" "}
              <span className="font-medium">{step.label}</span>
            </div>

            <div className="text-sm text-zinc-300">
              {step.marginal_score.toFixed(3)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
