from typing import Dict, Any, List
import threading


_LOCK = threading.Lock()

RAG_RUNTIME_METRICS: Dict[str, Any] = {
    "rag_retrieve_calls": 0,
    "rag_cache_eligible_calls": 0,
    "rag_retrieve_cache_hit": 0,
    "rag_retrieve_cache_miss": 0,
    "rag_empty_context_count": 0,
    "rag_grounded_pass_count": 0,
    "rag_grounded_partial_count": 0,
    "rag_grounded_fail_count": 0,
    "rag_retrieve_latency_samples_ms": [],
    "comprehensive_calls": 0,
    "comprehensive_parallel_ms": 0,
    "comprehensive_parallel_ms_samples": [],
    "fast_to_full_escalations": 0,
    "hyde_trigger_count": 0,
}


def rag_inc(metric_key: str, delta: int = 1) -> None:
    with _LOCK:
        RAG_RUNTIME_METRICS[metric_key] = int(RAG_RUNTIME_METRICS.get(metric_key, 0)) + int(delta)


def rag_append_sample(metric_key: str, value: int, max_len: int = 500) -> None:
    with _LOCK:
        samples: List[int] = RAG_RUNTIME_METRICS.setdefault(metric_key, [])
        samples.append(int(value))
        if len(samples) > max_len:
            del samples[: len(samples) - max_len]


def rag_set(metric_key: str, value: Any) -> None:
    with _LOCK:
        RAG_RUNTIME_METRICS[metric_key] = value


def rag_get_metrics_snapshot() -> Dict[str, Any]:
    with _LOCK:
        data: Dict[str, Any] = {}
        for k, v in RAG_RUNTIME_METRICS.items():
            data[k] = list(v) if isinstance(v, list) else v
        return data
