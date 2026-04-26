// src/api.ts
import type { RunResult } from "./types";

export async function runPipeline(query: string): Promise<RunResult> {
  const res = await fetch("http://localhost:8000/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      method: "litepack",
    }),
  });

  if (!res.ok) {
    throw new Error("API request failed");
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