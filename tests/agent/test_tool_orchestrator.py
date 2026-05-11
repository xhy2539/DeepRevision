import pytest

from agent.multi_agent.tool_orchestrator import build_tool_plan, execute_tool_plan, run_react_orchestrator


def test_explicit_web_search_uses_exam_format_tool():
    plan = build_tool_plan("网络搜索一下操作系统期末考试题型", session_id="操作系统", route_hint="ops")

    assert plan["needs_tools"] is True
    assert plan["next_agent"] == "ops"
    assert plan["tool_calls"][0]["tool_name"] == "search_exam_format"
    assert plan["tool_calls"][0]["args"]["course_name"] == "操作系统"


def test_web_search_without_topic_asks_for_clarification():
    plan = build_tool_plan("联网搜索一下啊", session_id="操作系统", route_hint="ops")

    assert plan["needs_tools"] is False
    assert plan["next_agent"] == "ops"
    assert "搜索什么" in plan["final_answer"]


def test_web_search_topic_strips_network_trigger_words():
    plan = build_tool_plan("联网查一下 CLOCK 页面置换算法", session_id="操作系统", route_hint="ops")

    assert plan["tool_calls"][0]["args"]["query"] == "CLOCK 页面置换算法"


@pytest.mark.asyncio
async def test_web_search_topic_can_use_recent_history(monkeypatch):
    class FakeTool:
        name = "web_search"

        async def ainvoke(self, args):
            return f"searched: {args['query']}"

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._tool_map", lambda: {"web_search": FakeTool()})

    result = await run_react_orchestrator(
        query="联网搜索一下啊",
        session_id="操作系统",
        route_hint="ops",
        recent_history="学生: 死锁的四个必要条件是什么？\n助手: 互斥、占有并等待、不可抢占、循环等待。",
    )

    assert result["observations"][0]["tool_name"] == "web_search"
    assert result["observations"][0]["args"]["query"] == "死锁的四个必要条件"


def test_courseware_question_uses_rag_search_tool():
    plan = build_tool_plan("C3 Processes 讲了什么", session_id="操作系统", route_hint="rag")

    assert plan["needs_tools"] is True
    assert plan["next_agent"] == "rag"
    assert plan["tool_calls"][0]["tool_name"] == "search_courseware"


def test_delete_courseware_requires_confirmation_guard():
    plan = build_tool_plan("删除 C3-Processes.pdf", session_id="操作系统", route_hint="ops")

    assert plan["needs_tools"] is True
    assert plan["next_agent"] == "ops"
    assert plan["requires_confirmation"] is True
    assert plan["tool_calls"][0]["tool_name"] == "delete_knowledge_file_tool"


@pytest.mark.asyncio
async def test_dangerous_tool_is_not_executed_without_confirmation():
    plan = build_tool_plan("删除 C3-Processes.pdf", session_id="操作系统", route_hint="ops")

    result = await execute_tool_plan(plan)

    assert result["guard_blocked"] is True
    assert result["observations"] == []
    assert result["trace"][-1]["status"] == "blocked"


@pytest.mark.asyncio
async def test_web_search_plan_executes_registered_tool(monkeypatch):
    class FakeTool:
        name = "search_exam_format"

        async def ainvoke(self, args):
            return "网络题型参考：选择题、判断题、填空题、简答题"

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._tool_map", lambda: {"search_exam_format": FakeTool()})
    plan = build_tool_plan("网络搜索一下操作系统期末考试题型", session_id="操作系统", route_hint="ops")

    result = await execute_tool_plan(plan)

    assert result["guard_blocked"] is False
    assert result["observations"][0]["tool_name"] == "search_exam_format"
    assert "选择题" in result["observations"][0]["content"]


@pytest.mark.asyncio
async def test_react_planner_delegates_quiz_without_running_generation_tool(monkeypatch):
    async def fake_planner(**kwargs):
        return {
            "tool_calls": [{"tool_name": "generate_quiz", "args": {"topic": "死锁", "quiz_type": "选择题", "num": 3}}],
            "delegate_route": "",
            "final_answer": "",
            "stop_reason": "need_quiz_agent",
        }

    def fail_tool_map():
        raise AssertionError("delegate tools must not be executed directly by the orchestrator")

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._call_react_planner", fake_planner)
    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._tool_map", fail_tool_map)

    result = await run_react_orchestrator(
        query="根据错题和课件安排复习并出题",
        session_id="操作系统",
        route_hint="planner",
        force_react=True,
    )

    assert result["next_agent"] == "quiz"
    assert result["react_status"] == "pass"
    assert result["delegate"]["route"] == "quiz"
    assert result["delegate"]["route_params"]["topic"] == "死锁"
    assert any(item.get("step") == "delegate" for item in result["trace"])


@pytest.mark.asyncio
async def test_react_planner_blocks_dangerous_write_tool(monkeypatch):
    async def fake_planner(**kwargs):
        return {
            "tool_calls": [{"tool_name": "delete_knowledge_file_tool", "args": {"filename": "C3-Processes.pdf", "session_id": "操作系统"}}],
            "delegate_route": "",
            "final_answer": "",
            "stop_reason": "need_guard",
        }

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._call_react_planner", fake_planner)

    result = await run_react_orchestrator(
        query="删除 C3-Processes.pdf",
        session_id="操作系统",
        route_hint="ops",
        force_react=True,
    )

    assert result["next_agent"] == "ops"
    assert result["guard_blocked"] is True
    assert result["react_status"] == "guarded"
    assert result["tool_plan"]["tool_calls"][0]["tool_name"] == "delete_knowledge_file_tool"


@pytest.mark.asyncio
async def test_react_planner_invalid_json_falls_back_to_route(monkeypatch):
    async def fake_planner(**kwargs):
        return "not json"

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._call_react_planner", fake_planner)

    result = await run_react_orchestrator(
        query="根据错题和课件安排复习并出题",
        session_id="操作系统",
        route_hint="planner",
        force_react=True,
    )

    assert result["next_agent"] == "planner"
    assert result["react_status"] == "partial"
    assert result["observations"] == []


@pytest.mark.asyncio
async def test_react_planner_stops_after_max_steps(monkeypatch):
    async def fake_planner(**kwargs):
        return {
            "tool_calls": [{"tool_name": "get_practice_stats_tool", "args": {"session_id": "操作系统"}}],
            "delegate_route": "",
            "final_answer": "",
            "stop_reason": "continue",
        }

    class FakeTool:
        name = "get_practice_stats_tool"

        async def ainvoke(self, args):
            return "练习统计：共 3 题"

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._call_react_planner", fake_planner)
    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._tool_map", lambda: {"get_practice_stats_tool": FakeTool()})

    result = await run_react_orchestrator(
        query="根据错题和课件安排复习并出题",
        session_id="操作系统",
        route_hint="planner",
        force_react=True,
        max_steps=2,
    )

    assert result["next_agent"] == "planner"
    assert result["react_status"] == "partial"
    assert len(result["observations"]) == 2
