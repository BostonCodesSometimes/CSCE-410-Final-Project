"""Load Pinecone settings from .env and/or environment variables.

Priority (highest to lowest):
  1. Existing environment variables (already exported in the shell)
  2. .env file in the project root (loaded via python-dotenv)
  3. pinecone_creds.txt in the src/ directory (legacy fallback, optional)
  4. Hard-coded defaults
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Locate the project root (.env lives two levels above src/pinecone_settings.py)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _normalize_pinecone_host(host: str) -> str:
    h = host.strip()
    for prefix in ("https://", "http://"):
        if h.startswith(prefix):
            h = h[len(prefix) :]
    return h.rstrip("/")


def load_pinecone_creds_file(path: str | Path) -> dict[str, str]:
    """Parse key: value lines; values may be wrapped in [...]."""
    path = Path(path)
    if not path.is_file():
        return {}
    config: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, rest = line.partition(":")
        norm = key.strip().lower().replace(" ", "_")
        val = rest.strip()
        bracket = re.match(r"^\[(.*)\]\s*$", val, re.DOTALL)
        if bracket:
            val = bracket.group(1).strip()
        config[norm] = val
    return config


def get_pinecone_config(
    creds_path: str | Path | None = None,
) -> tuple[str, str, str | None, str | None]:
    """Return (api_key, index_name, host, embedding_model).

    Loads .env from the project root before reading environment variables so
    callers do not need to call load_dotenv() themselves.  An explicit
    creds_path (or the legacy pinecone_creds.txt) is consulted last.
    """
    # Load .env; override=False so a pre-exported shell variable wins.
    load_dotenv(_ENV_FILE, override=False)

    root = Path(__file__).resolve().parent
    path = Path(creds_path) if creds_path else root / "pinecone_creds.txt"
    file_vals = load_pinecone_creds_file(path)

    api_key = os.environ.get("PINECONE_API_KEY") or file_vals.get("api_key", "")
    index_name = (
        os.environ.get("PINECONE_INDEX_NAME")
        or file_vals.get("index_name")
        or "lite-rag"
    )
    raw_host = os.environ.get("PINECONE_HOST") or file_vals.get("host")
    host = _normalize_pinecone_host(raw_host) if raw_host else None
    embedding_model = (
        os.environ.get("EMBEDDING_MODEL")
        or os.environ.get("PINECONE_EMBEDDING_MODEL")
        or file_vals.get("embedding_model")
        or "text-embedding-3-small"
    )
    return api_key, index_name, host, embedding_model
