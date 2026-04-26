import { useState } from "react";
import type { RunResult } from "./types";
import { motion } from "framer-motion";

const steps = [
  "retrieval",
  "enrichment",
  "ranking",
  "selection",
  "packing",
  "generation",
];

export default function PipelineVisualizer({ result }: { result: RunResult }) {
  const [activeStep, setActiveStep] = useState<string>("retrieval");

  const renderContent = () => {
    switch (activeStep) {
      case "retrieval":
        return (
          <div className="space-y-2">
            {result.ranked_candidates?.slice(0, 5).map((c) => (
              <div
                key={c.chunk_id}
                className="p-3 rounded-xl bg-zinc-800 border border-zinc-700 flex justify-between"
              >
                <div>
                  <div className="font-medium">{c.title}</div>
                  <div className="text-xs text-zinc-400">{c.year}</div>
                </div>
                <div className="text-sm text-zinc-300">
                  {c.retrieval_score.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        );

      case "selection":
        return (
          <div className="space-y-2">
            {result.selection_result.trace.map((step) => (
              <div
                key={step.step_index}
                className="p-3 rounded-xl bg-zinc-800 border border-zinc-700 flex justify-between"
              >
                <div>
                  Step {step.step_index}:{" "}
                  <span className="font-medium">{step.label}</span>
                </div>
                <div className="text-sm text-zinc-300">
                  {step.marginal_score.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        );

      case "packing":
        return (
          <div className="p-4 rounded-xl bg-zinc-800 border border-zinc-700">
            <div className="text-sm text-zinc-400">Token Usage</div>
            <div className="text-2xl font-semibold">
              {result.packed_context.tokens_used}
            </div>
          </div>
        );

      case "generation":
        return (
          <div className="p-4 rounded-xl bg-zinc-800 border border-zinc-700">
            <div className="text-sm text-zinc-400 mb-2">Generated Answer</div>
            <div className="leading-relaxed whitespace-pre-wrap">
              {result.generation_result.answer_text}
            </div>
          </div>
        );

      default:
        return <div className="text-zinc-400">No data</div>;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Pipeline</h2>
        <div className="text-sm text-zinc-400">
          {result.timings.total_ms.toFixed(0)} ms
        </div>
      </div>

      {/* Step Bar */}
      <div className="flex gap-2 flex-wrap">
        {steps.map((step) => {
          const active = activeStep === step;
          return (
            <button
              key={step}
              onClick={() => setActiveStep(step)}
              className={`px-3 py-1.5 rounded-lg text-sm capitalize transition
                ${
                  active
                    ? "bg-white text-black"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }`}
            >
              {step}
            </button>
          );
        })}
      </div>

      {/* Animated Content */}
      <motion.div
        key={activeStep}
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="mt-2"
      >
        {renderContent()}
      </motion.div>
    </div>
  );
}
