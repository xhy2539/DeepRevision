import json

import pytest

from agent.multi_agent.supervisor import _extract_grounded_evidence, rag_subagent_node
from utils.rag_metrics import rag_get_metrics_snapshot, rag_set


def test_failure_hint_cannot_be_grounded_evidence():
    hint = "知识库检索遇到问题，请确保已上传课件"
    parsed = {"evidence": [{"source": "检索到的课件资料", "quote": hint}]}
    assert _extract_grounded_evidence(parsed, hint, retrieval_status="failed") == []


def test_empty_context_cannot_be_grounded_evidence():
    parsed = {"evidence": [{"source": "课件", "quote": "进程是资源分配的基本单位"}]}
    assert _extract_grounded_evidence(parsed, "", retrieval_status="empty") == []


def test_real_retrieved_quote_is_grounded():
    quote = "进程是资源分配的基本单位"
    parsed = {"evidence": [{"source": "操作系统.pdf", "quote": quote}]}
    assert _extract_grounded_evidence(parsed, f"[参考资料1]: {quote}") == [
        {"source": "操作系统.pdf", "quote": quote}
    ]


@pytest.mark.asyncio
async def test_rag_node_failure_does_not_increment_grounded_pass(monkeypatch):
    class FailingRag:
        async def retrieve_context(self, _query, mode="rag_chat", trace=None):
            raise RuntimeError("embedding unavailable")

    async def fake_get_rag_service():
        return FailingRag()

    async def fake_call_llm(_prompt, **_kwargs):
        return json.dumps(
            {
                "answer": "我目前没有检索到相关课件内容。",
                "evidence": [
                    {"source": "检索到的课件资料", "quote": "知识库检索遇到问题，请确保已上传课件"}
                ],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("agent.multi_agent.supervisor.get_rag_service", fake_get_rag_service)
    monkeypatch.setattr("agent.multi_agent.supervisor._call_llm", fake_call_llm)
    rag_set("rag_grounded_pass_count", 0)

    result = await rag_subagent_node(
        {
            "input": "根据课件解释进程和线程",
            "chat_history": [],
            "memory_context": "",
            "session_id": "操作系统",
            "route_params": {},
        }
    )

    structured = result["subagent_result"]
    assert structured["payload"]["evidence_cards"] == []
    assert structured["meta"]["retrieval_status"] == "failed"
    assert rag_get_metrics_snapshot()["rag_grounded_pass_count"] == 0


@pytest.mark.asyncio
async def test_failed_orchestrator_observation_is_not_reused_as_evidence(monkeypatch):
    async def fail_if_retrieved():
        raise AssertionError("failed observation should be classified without another retrieval")

    async def fake_call_llm(_prompt, **_kwargs):
        return json.dumps(
            {
                "answer": "当前课件检索不可用。",
                "evidence": [{"source": "课件", "quote": "课件检索失败：embedding unavailable"}],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr("agent.multi_agent.supervisor.get_rag_service", fail_if_retrieved)
    monkeypatch.setattr("agent.multi_agent.supervisor._call_llm", fake_call_llm)
    rag_set("rag_grounded_pass_count", 0)
    result = await rag_subagent_node(
        {
            "input": "根据课件解释进程和线程",
            "chat_history": [],
            "memory_context": "",
            "session_id": "操作系统",
            "route_params": {
                "orchestrator": {
                    "observations": [
                        {
                            "tool_name": "search_courseware",
                            "content": "课件检索失败：embedding unavailable",
                        }
                    ]
                }
            },
        }
    )

    structured = result["subagent_result"]
    assert structured["payload"]["evidence_cards"] == []
    assert structured["meta"]["retrieval_status"] == "failed"
    assert rag_get_metrics_snapshot()["rag_grounded_pass_count"] == 0
