"""Answer generation module for LitePack-RAG.

Responsibilities:
  - fixed generation layer: same model and prompt style for all methods
  - accept a PackedContext and query, call the LLM, return a GenerationResult
  - extract citation labels from the generated answer

Nothing here performs retrieval, ranking, selection, or evaluation.
The Generator is intentionally blind to which selection method produced the
PackedContext so that generation is never a confound in method comparisons.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import openai
from dotenv import load_dotenv

from config import PipelineConfig

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from packer import PackedContext

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[S\d+\]")

# Generation parameters read from .env at module load time (after load_dotenv).
# CLI callers can override by setting env vars before import.
_TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
_MAX_OUTPUT_TOKENS: int | None = (
    int(os.getenv("MAX_OUTPUT_TOKENS")) if os.getenv("MAX_OUTPUT_TOKENS") else None
)
_SEED: int | None = (
    int(os.getenv("SEED")) if os.getenv("SEED") else None
)


# --------------------------------------------------------------------------- #
# GenerationResult dataclass                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class GenerationResult:
    """Output of Generator.generate().

    Attributes:
        answer_text:          model's answer string.
        cited_source_labels:  deduplicated citation labels found in answer_text
                              (e.g. ["[S1]", "[S3]"]), in order of first appearance.
        raw_response:         the raw answer string from the API (same as answer_text
                              for chat completions; kept for traceability).
        model_name:           the LLM model used.
        prompt_tokens:        tokens consumed by the prompt (from API usage object).
        completion_tokens:    tokens in the completion (from API usage object).
    """

    answer_text: str
    cited_source_labels: list[str]
    raw_response: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int


# --------------------------------------------------------------------------- #
# Generator class                                                              #
# --------------------------------------------------------------------------- #

class Generator:
    """Fixed LLM generation layer.

    Uses openai.OpenAI() with a stable prompt template.  The same template
    and model are applied for all selection methods (similarity, MMR, LitePack)
    so that generation quality differences across methods come only from the
    packed context, not from prompt variation.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the OpenAI client and store the configured model name.

        Args:
            config: PipelineConfig with llm_model.
        """
        self.client = openai.OpenAI()
        self.model_name = config.llm_model

    def build_prompt(self, query: str, packed_context: PackedContext) -> str:
        """Build the full prompt string to send to the LLM.

        Instructs the model to:
          - answer using only the provided sources
          - cite each source as [S1], [S2], etc.
          - be concise and factual

        Args:
            query:          the user's question.
            packed_context: PackedContext from packer.build_context().

        Returns:
            Complete prompt string ready to send as a user message.
        """
        return (
            "You are a medical literature assistant. "
            "Answer the question using ONLY the sources provided below. "
            "Cite each source you use inline as [S1], [S2], etc., where the "
            "label matches the source header. "
            "Be concise and factual. Do not include information that is not "
            "present in the sources.\n\n"
            "--- SOURCES ---\n"
            f"{packed_context.context_text}\n"
            "--- END SOURCES ---\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def generate(
        self,
        query: str,
        packed_context: PackedContext,
        config: PipelineConfig,
    ) -> GenerationResult:
        """Call the LLM and return a GenerationResult.

        Citation labels are extracted with the regex r"\\[S\\d+\\]" and
        deduplicated while preserving order of first appearance.

        Prompt and completion token counts come from the API usage object;
        both default to 0 if the field is absent.

        Args:
            query:          the user's question.
            packed_context: PackedContext from packer.build_context().
            config:         PipelineConfig; llm_model is read at call time so
                            callers can override it via dataclasses.replace().

        Returns:
            GenerationResult with answer, citations, token counts, and raw
            response populated.
        """
        model = config.llm_model
        prompt = self.build_prompt(query, packed_context)

        logger.info(
            "Generator.generate: calling model=%s context_tokens=%d",
            model,
            packed_context.tokens_used,
        )

        create_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _TEMPERATURE,
        }
        if _MAX_OUTPUT_TOKENS is not None:
            create_kwargs["max_tokens"] = _MAX_OUTPUT_TOKENS
        if _SEED is not None:
            create_kwargs["seed"] = _SEED

        response = self.client.chat.completions.create(**create_kwargs)

        raw_answer: str = response.choices[0].message.content or ""

        # Extract citation labels in order of first appearance, deduplicated.
        seen: set[str] = set()
        cited_labels: list[str] = []
        for match in _CITATION_RE.finditer(raw_answer):
            label = match.group()
            if label not in seen:
                seen.add(label)
                cited_labels.append(label)

        # Token counts from API usage; fall back to 0 if absent.
        usage = getattr(response, "usage", None)
        prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0

        logger.info(
            "Generator.generate: done  prompt_tokens=%d  completion_tokens=%d  "
            "citations=%d",
            prompt_tokens,
            completion_tokens,
            len(cited_labels),
        )

        return GenerationResult(
            answer_text=raw_answer,
            cited_source_labels=cited_labels,
            raw_response=raw_answer,
            model_name=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
