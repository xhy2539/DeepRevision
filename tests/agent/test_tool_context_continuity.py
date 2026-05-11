import pytest


def test_tool_context_reference_detector_targets_followups():
    from utils.tool_context import should_use_tool_context

    assert should_use_tool_context("基于刚才搜索结果生成综合测试卷")
    assert should_use_tool_context("按上次那套卷再出三道题")
    assert should_use_tool_context("参考前面查到的题型分布")
    assert not should_use_tool_context("你好")


def test_tool_context_formatter_keeps_structured_labels():
    from utils.tool_context import format_tool_contexts_for_prompt

    text = format_tool_contexts_for_prompt(
        [
            {
                "tool_name": "search_exam_format",
                "args": {"course_name": "操作系统"},
                "summary": "操作系统题型参考",
                "content": "选择题、判断题、填空题、简答题",
                "created_at": 1778234420920,
            }
        ],
        query="基于刚才搜索结果出卷",
    )

    assert "【近期工具上下文】" in text
    assert "search_exam_format" in text
    assert "操作系统题型参考" in text
    assert "选择题、判断题、填空题、简答题" in text


def test_memory_manager_persists_recent_tool_contexts(tmp_path, monkeypatch):
    import utils.memory_service as memory_service

    monkeypatch.setattr(memory_service, "_db_path", lambda: str(tmp_path / "sessions.db"))
    manager = memory_service.SessionMemoryManager(max_recent_turns=2)

    saved = manager.append_tool_contexts(
        "操作系统",
        [{"tool_name": "web_search", "args": {"query": "CLOCK"}, "content": "CLOCK 页面置换算法参考"}],
        turn_query="联网查一下 CLOCK 页面置换算法",
    )
    recent = manager.get_recent_tool_contexts("操作系统", limit=1)

    assert saved == 1
    assert recent[0]["tool_name"] == "web_search"
    assert recent[0]["args"]["query"] == "CLOCK"
    assert "CLOCK 页面置换算法参考" in recent[0]["content"]

    assert manager.clear_session("操作系统") is True
    assert manager.get_agent_state("操作系统", "tool_context") == {}


def test_exam_message_becomes_reusable_tool_context():
    from api.routers.chat import _build_assistant_artifact_tool_context

    item = _build_assistant_artifact_tool_context(
        query="生成综合测试卷",
        message={"kind": "exam_paper", "meta": {"route": "exam"}},
        answer="操作系统综合测试卷\n一、选择题...",
    )

    assert item["tool_name"] == "generated_exam_paper"
    assert item["summary"] == "生成的综合试卷"
    assert "操作系统综合测试卷" in item["content"]


@pytest.mark.asyncio
async def test_tool_orchestrator_persists_observations(monkeypatch):
    from agent.multi_agent import supervisor

    class FakeMemory:
        def __init__(self):
            self.saved = []

        def append_tool_contexts(self, session_id, observations, *, turn_query="", source="react_orchestrator"):
            self.saved.append((session_id, observations, turn_query, source))
            return len(observations)

    async def fake_react(**kwargs):
        return {
            "next_agent": "ops",
            "tool_plan": {"needs_tools": True},
            "task_plan": {"goal": "查题型", "steps": []},
            "plan_execution": [],
            "guard_blocked": False,
            "observations": [{"tool_name": "search_exam_format", "args": {}, "content": "选择题、判断题"}],
            "trace": [{"step": "observation", "tool_name": "search_exam_format", "status": "success"}],
            "react_status": "pass",
            "delegate": {},
            "final_answer": "",
        }

    fake_memory = FakeMemory()
    monkeypatch.setattr(supervisor, "memory_manager", fake_memory)
    monkeypatch.setattr(supervisor, "run_react_orchestrator", fake_react)

    result = await supervisor.tool_orchestrator_node(
        {
            "input": "网络搜索一下操作系统期末考试题型",
            "chat_history": [],
            "memory_context": "",
            "tool_context": "",
            "session_id": "操作系统",
            "client_action": "",
            "exam_stage_plan": True,
            "exam_rerun_stage": None,
            "exam_partial_questions": [],
            "exam_fast_mode": True,
            "quiz_force_llm_critic": False,
            "route": "ops",
            "route_reason": "test",
            "route_params": {},
            "supervisor_precomputed": False,
            "subagent_result": "",
            "final_answer": "",
        }
    )

    assert result["route_params"]["orchestrator"]["observations"][0]["tool_name"] == "search_exam_format"
    assert fake_memory.saved[0][0] == "操作系统"
    assert fake_memory.saved[0][2] == "网络搜索一下操作系统期末考试题型"


@pytest.mark.asyncio
async def test_tool_orchestrator_passes_tool_context_to_planner(monkeypatch):
    from agent.multi_agent import supervisor

    async def fake_react(**kwargs):
        assert "【近期工具上下文】" in kwargs["recent_history"]
        assert "search_exam_format" in kwargs["recent_history"]
        return {
            "next_agent": "exam",
            "tool_plan": {},
            "task_plan": {"goal": "出卷", "steps": []},
            "plan_execution": [],
            "guard_blocked": False,
            "observations": [],
            "trace": [],
            "react_status": "pass",
            "delegate": {},
            "final_answer": "",
        }

    monkeypatch.setattr(supervisor, "run_react_orchestrator", fake_react)

    result = await supervisor.tool_orchestrator_node(
        {
            "input": "基于刚才搜索结果生成综合测试卷",
            "chat_history": [],
            "memory_context": "",
            "tool_context": "【近期工具上下文】\n1. search_exam_format\n选择题、判断题",
            "session_id": "操作系统",
            "client_action": "",
            "exam_stage_plan": True,
            "exam_rerun_stage": None,
            "exam_partial_questions": [],
            "exam_fast_mode": True,
            "quiz_force_llm_critic": False,
            "route": "exam",
            "route_reason": "test",
            "route_params": {},
            "supervisor_precomputed": False,
            "subagent_result": "",
            "final_answer": "",
        }
    )

    assert result["route"] == "exam"
