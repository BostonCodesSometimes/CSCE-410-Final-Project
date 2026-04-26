"""Verify Pinecone credentials and index reachability (no writes)."""

from pinecone import Pinecone

from pinecone_settings import get_pinecone_config


def main() -> int:
    api_key, index_name, host, embedding_model = get_pinecone_config()
    if not api_key:
        print("Missing PINECONE_API_KEY or api_key in pinecone_creds.txt")
        return 1

    pc = Pinecone(api_key=api_key)
    desc = pc.describe_index(index_name)
    print("describe_index OK:", getattr(desc, "name", index_name))
    stats = host or getattr(desc, "host", None)
    if stats:
        print("index host:", stats)
    print("embedding_model (from config):", embedding_model)

    idx = pc.Index(index_name, host=host) if host else pc.Index(index_name)
    st = idx.describe_index_stats()
    print("describe_index_stats OK:", st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
