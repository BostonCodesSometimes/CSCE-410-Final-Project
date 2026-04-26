export interface GenerationResult {
  answer_text: string;
  cited_source_labels: string[];
  model_name: string;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface Candidate {
  chunk_id: string;
  pmid: string;
  title: string;
  year: number;
  retrieval_score: number;
  final_score?: number;
  token_length: number;
}

export interface SelectionTraceStep {
  step_index: number;
  label: string;
  candidate_id: string;
  marginal_score: number;
  tokens_used_so_far: number;
}

export type CompareResult = RunResult[];

export interface RunResult {
  method_name: string;

  generation_result: GenerationResult;

  retrieved_candidates: Candidate[];
  ranked_candidates: Candidate[];

  packed_context: {
    citation_map: Record<string, Candidate>;
    tokens_used: number;
  };

  selection_result: {
    selected_candidates: Candidate[];
    trace: SelectionTraceStep[];
  };

  metrics: {
    tokens_used: number;
    chunks_selected: number;
  };

  timings: {
    total_ms: number;
  };
}

export interface PipelineRequest {
  query: string;
  method: string;
  top_n: number;
  budget: number;
  use_keywords: boolean;
  use_recency: boolean;
}