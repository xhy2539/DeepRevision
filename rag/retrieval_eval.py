"""Offline retrieval evaluation with human-labelled relevant chunk IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Iterable


METRIC_NAMES = ("precision_at_k", "recall_at_k", "hit_at_k", "mrr", "ndcg_at_k")


def evaluate_query(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    """Calculate binary-relevance retrieval metrics for one query."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    ranked = [str(item) for item in retrieved_ids[:k]]
    relevant = {str(item) for item in relevant_ids}
    hit_count = sum(1 for item in ranked if item in relevant)

    reciprocal_rank = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(ranked, start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant) if relevant else 0.0,
        "hit_at_k": 1.0 if hit_count else 0.0,
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def aggregate_metrics(results: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average query-level metrics without inventing values for an empty dataset."""
    rows = list(results)
    if not rows:
        return {"case_count": 0, **{name: 0.0 for name in METRIC_NAMES}}
    return {
        "case_count": len(rows),
        **{
            name: sum(float(row.get(name, 0.0)) for row in rows) / len(rows)
            for name in METRIC_NAMES
        },
    }


def load_cases(dataset: str | Path) -> list[dict[str, Any]]:
    """Load and validate the labelled JSON evaluation dataset."""
    path = Path(dataset)
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty 'cases' list")
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not str(case.get("query", "")).strip():
            raise ValueError(f"case {index} must contain a non-empty query")
        relevant = case.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"case {index} must contain human-labelled relevant_chunk_ids")
    return cases


async def _retrieve_ids(query: str, session_id: str, k: int) -> list[str]:
    from rag.rag_service import RagSummarizeService
    from utils.session_context import current_session_id

    token = current_session_id.set(session_id)
    try:
        service = RagSummarizeService()
        trace: list[dict[str, Any]] = []
        await service.retrieve_context(query, mode="eval", trace=trace)
        final = next((item for item in reversed(trace) if item.get("stage") == "finalize"), {})
        chunks = final.get("selected_chunks", []) if isinstance(final, dict) else []
        return [str(item.get("chunk_id")) for item in chunks[:k] if item.get("chunk_id")]
    finally:
        current_session_id.reset(token)


async def evaluate_dataset(
    dataset: str | Path,
    *,
    k: int,
    session_id: str = "",
    variant: str = "",
) -> dict[str, Any]:
    """Evaluate stored rankings or run the live retriever against labelled cases."""
    cases = load_cases(dataset)
    details = []
    for case in cases:
        variants = case.get("variants") if isinstance(case.get("variants"), dict) else {}
        supplied = variants.get(variant) if variant else case.get("retrieved_ids")
        if isinstance(supplied, list):
            retrieved_ids = [str(item) for item in supplied]
        elif session_id:
            retrieved_ids = await _retrieve_ids(str(case["query"]), session_id, k)
        elif variant:
            raise ValueError(f"case '{case['query']}' does not provide variant '{variant}'")
        else:
            raise ValueError("session_id is required when a case does not provide retrieved_ids")
        metrics = evaluate_query(retrieved_ids, set(case["relevant_chunk_ids"]), k)
        details.append({"query": case["query"], "retrieved_ids": retrieved_ids[:k], **metrics})
    return {"k": k, "variant": variant or "live", "aggregate": aggregate_metrics(details), "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval against human-labelled chunk IDs")
    parser.add_argument("--dataset", required=True, help="JSON file containing evaluation cases")
    parser.add_argument("--session-id", default="", help="Knowledge-base session used for live retrieval")
    parser.add_argument("--variant", default="", help="Stored ranking variant to evaluate")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = asyncio.run(evaluate_dataset(args.dataset, k=args.k, session_id=args.session_id, variant=args.variant))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
