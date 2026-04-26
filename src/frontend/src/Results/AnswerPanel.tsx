import type { RunResult } from "../types";

export default function AnswerPanel({ result }: { result: RunResult }) {
  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">Answer</h2>

      <div className="bg-zinc-800 border border-zinc-700 rounded-xl p-4 leading-relaxed whitespace-pre-wrap text-sm">
        {result.generation_result.answer_text}
      </div>
    </div>
  );
}
