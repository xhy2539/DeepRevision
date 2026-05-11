import pytest

from agent.multi_agent.planning_agent import build_default_task_plan
from agent.multi_agent.tool_orchestrator import run_react_orchestrator


def _step_tools(plan):
    return [step.get("tool_name") for step in plan.get("steps", [])]


def test_complex_exam_network_request_builds_task_plan():
    plan = build_default_task_plan(
        "根据课件和网络题型生成操作系统综合试卷",
        session_id="操作系统",
        route_hint="exam",
    )

    assert plan["goal"]
    assert plan["completion_criteria"]
    assert _step_tools(plan)[:3] == ["list_knowledge_files_tool", "search_exam_format", "exam"]
    assert plan["steps"][0]["action_type"] == "tool"
    assert plan["steps"][1]["action_type"] == "tool"
    assert plan["steps"][2]["action_type"] == "delegate"


@pytest.mark.asyncio
async def test_task_plan_executes_readonly_tools_then_delegates_exam(monkeypatch):
    class FakeTool:
        def __init__(self, name, content):
            self.name = name
            self.content = content

        async def ainvoke(self, args):
            return self.content

    monkeypatch.setattr(
        "agent.multi_agent.tool_orchestrator._tool_map",
        lambda: {
            "list_knowledge_files_tool": FakeTool("list_knowledge_files_tool", "C3-Processes.pdf | completed"),
            "search_exam_format": FakeTool("search_exam_format", "网络题型参考：选择题10道、判断题5道"),
        },
    )

    result = await run_react_orchestrator(
        query="根据课件和网络题型生成操作系统综合试卷",
        session_id="操作系统",
        route_hint="exam",
        force_react=True,
        max_steps=6,
    )

    assert result["next_agent"] == "exam"
    assert result["react_status"] == "pass"
    assert result["delegate"]["route"] == "exam"
    assert result["task_plan"]["steps"][0]["status"] == "completed"
    assert result["task_plan"]["steps"][1]["status"] == "completed"
    assert result["task_plan"]["steps"][2]["status"] == "delegated"
    assert [item["tool_name"] for item in result["observations"]] == ["list_knowledge_files_tool", "search_exam_format"]
    assert result["plan_execution"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_task_planner_invalid_json_falls_back_to_current_route(monkeypatch):
    async def fake_planner(**kwargs):
        return "not json"

    monkeypatch.setattr("agent.multi_agent.tool_orchestrator._call_react_planner", fake_planner)

    result = await run_react_orchestrator(
        query="请自主规划步骤并调用工具完成这个复杂学习任务",
        session_id="操作系统",
        route_hint="planner",
        force_react=True,
    )

    assert result["next_agent"] == "planner"
    assert result["react_status"] == "partial"
    assert result["task_plan"]["fallback_policy"]
    assert result["plan_execution"][-1]["status"] == "failed"
