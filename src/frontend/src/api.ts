// src/api.ts
import type { RunResult, PipelineRequest } from "./types";

export async function runPipeline(params: PipelineRequest): Promise<RunResult> {
  const res = await fetch("http://localhost:8000/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    // We stringify the entire params object to match your required JSON body
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "API request failed");
  }

  return res.json();
}

export async function comparePipeline(query: string) {
  const res = await fetch("http://localhost:8000/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  return res.json();
}