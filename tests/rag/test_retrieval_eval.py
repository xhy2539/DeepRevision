import asyncio
import json
import math

import pytest

from rag.retrieval_eval import aggregate_metrics, evaluate_dataset, evaluate_query


def test_evaluate_query_calculates_standard_metrics():
    result = evaluate_query(["x", "b", "a", "z"], {"a", "b", "c"}, k=4)

    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(2 / 3)
    assert result["hit_at_k"] == 1.0
    assert result["mrr"] == pytest.approx(0.5)
    expected_dcg = 1 / math.log2(3) + 1 / math.log2(4)
    expected_idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert result["ndcg_at_k"] == pytest.approx(expected_dcg / expected_idcg)


def test_evaluate_query_handles_no_relevant_results():
    result = evaluate_query(["x", "y"], {"a"}, k=2)
    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "hit_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
    }


def test_aggregate_metrics_reports_empty_dataset_honestly():
    assert aggregate_metrics([])["case_count"] == 0


def test_evaluate_query_rejects_invalid_k():
    with pytest.raises(ValueError, match="greater than 0"):
        evaluate_query([], set(), k=0)


def test_evaluate_dataset_supports_stored_variants(tmp_path):
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps({"cases": [{
        "query": "q",
        "relevant_chunk_ids": ["a"],
        "variants": {"baseline": ["x", "a"]},
    }]}), encoding="utf-8")

    report = asyncio.run(evaluate_dataset(dataset, k=2, variant="baseline"))

    assert report["variant"] == "baseline"
    assert report["aggregate"]["recall_at_k"] == 1.0
    assert report["aggregate"]["mrr"] == 0.5
