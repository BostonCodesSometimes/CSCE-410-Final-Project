from fastapi import APIRouter
from pydantic import BaseModel
import dataclasses

from pipeline import run_once, _run_result_to_dict, compare_methods
from config.config import PipelineConfig
from retriever import Retriever
from generator import Generator

router = APIRouter()

# Initialize ONCE (important for performance)
config = PipelineConfig()
retriever = Retriever(config)
generator = Generator(config)


class QueryRequest(BaseModel):
    query: str
    method: str = "litepack"
    top_n: int = 20
    budget: int = 1800
    use_keywords: bool = True
    use_recency: bool = True


@router.post("/run")
def run_pipeline_api(req: QueryRequest):
    # Map method → ranker/selector
    method_map = {
        "similarity": ("similarity", "baseline"),
        "mmr": ("mmr", "greedy"),
        "litepack": ("litepack", "greedy"),
    }

    ranker, selector = method_map.get(req.method, ("litepack", "greedy"))

    # Build config dynamically
    cfg = dataclasses.replace(
        config,
        top_n=req.top_n,
        budget=req.budget,
        use_keywords=req.use_keywords,
        use_recency=req.use_recency,
        ranker=ranker,
        selector=selector,
    )

    result = run_once(
        req.query,
        cfg,
        retriever,
        retriever.embeddings_model,
        generator,
    )

    return _run_result_to_dict(result)

@router.post("/compare")
def compare(req: QueryRequest):
    methods = ["similarity", "mmr", "litepack"]

    results = compare_methods(
        req.query,
        methods,
        config,
        retriever,
        retriever.embeddings_model,
        generator,
    )

    return [_run_result_to_dict(r) for r in results]