import pytest

from agent.multi_agent.supervisor import learning_loop_subagent_node, ops_subagent_node, planner_subagent_node, rag_subagent_node, supervisor_node, tool_orchestrator_node


BASE_STATE = {
    "input": "",
    "chat_history": [],
    "memory_context": "",
    "session_id": "操作系统",
    "client_action": "",
    "exam_stage_plan": True,
    "exam_rerun_stage": None,
    "exam_partial_questions": [],
    "exam_fast_mode": True,
    "quiz_force_llm_critic": False,
    "route": "",
    "route_reason": "",
    "route_params": {},
    "supervisor_precomputed": False,
    "subagent_result": "",
    "final_answer": "",
}


@pytest.mark.asyncio
async def test_web_search_routes_to_orchestrator_then_ops(monkeypatch):
    async def fake_react(**kwargs):
        return {
            "next_agent": "ops",
            "tool_plan": {"needs_tools": True, "tool_calls": [{"tool_name": "search_exam_format", "args": {"course_name": "操作系统"}}]},
            "task_plan": {"goal": "查题型", "steps": [], "completion_criteria": ["返回结果"], "fallback_policy": "降级"},
            "plan_execution": [{"step_id": "1", "status": "completed", "summary": "查题型完成"}],
            "guard_blocked": False,
            "observations": [{"tool_name": "search_exam_format", "args": {}, "content": "选择题、判断题"}],
            "trace": [{"step": "intent", "status": "success"}, {"step": "action", "tool_name": "search_exam_format", "status": "success"}],
            "react_status": "pass",
            "delegate": {},
            "final_answer": "",
        }

    monkeypatch.setattr("agent.multi_agent.supervisor.run_react_orchestrator", fake_react)
    routed = await supervisor_node({**BASE_STATE, "input": "网络搜索一下操作系统期末考试题型"})

    assert routed["route"] == "ops"

    orchestrated = await tool_orchestrator_node({**BASE_STATE, **routed, "input": routed["input"]})

    assert orchestrated["route"] == "ops"
    orchestrator = orchestrated["route_params"]["orchestrator"]
    assert orchestrator["tool_plan"]["needs_tools"] is True
    assert orchestrator["task_plan"]["goal"] == "查题型"
    assert orchestrator["plan_execution"][0]["status"] == "completed"
    assert orchestrator["observations"][0]["tool_name"] == "search_exam_format"


@pytest.mark.asyncio
async def test_button_exam_bypasses_tool_orchestration_to_exam():
    routed = await supervisor_node({**BASE_STATE, "client_action": "generate_exam", "input": "生成综合测试卷"})

    assert routed["route"] == "exam"

    orchestrated = await tool_orchestrator_node({**BASE_STATE, **routed, "client_action": "generate_exam"})

    assert orchestrated["route"] == "exam"
    assert "orchestrator" not in orchestrated.get("route_params", {})


@pytest.mark.asyncio
async def test_ops_returns_existing_web_observation_without_replanning(monkeypatch):
    def fail_tool_map():
        raise AssertionError("ops should consume orchestrator observations before rebuilding tool map")

    monkeypatch.setattr("agent.multi_agent.supervisor._build_ops_tool_map", fail_tool_map)
    state = {
        **BASE_STATE,
        "input": "网络搜索一下操作系统期末考试题型",
        "route": "ops",
        "route_params": {
            "orchestrator": {
                "observations": [
                    {
                        "tool_name": "search_exam_format",
                        "args": {"course_name": "操作系统"},
                        "content": "选择题、判断题、填空题、简答题",
                    }
                ],
                "trace": [{"step": "action", "tool_name": "search_exam_format", "status": "success"}],
                "guard_blocked": False,
            }
        },
    }

    result = await ops_subagent_node(state)

    text = result["final_answer"]
    assert "选择题" in text
    assert "不具有实时联网" not in text
    assert result["subagent_result"]["meta"]["orchestrator_used"] is True


@pytest.mark.asyncio
async def test_ops_returns_orchestrator_clarification_without_replanning(monkeypatch):
    def fail_tool_map():
        raise AssertionError("clarification should not rebuild ops tool map")

    monkeypatch.setattr("agent.multi_agent.supervisor._build_ops_tool_map", fail_tool_map)
    state = {
        **BASE_STATE,
        "input": "联网搜索一下啊",
        "route": "ops",
        "route_params": {
            "orchestrator": {
                "observations": [],
                "trace": [{"step": "intent", "status": "clarify"}],
                "guard_blocked": False,
                "final_answer": "你想搜索什么内容？",
            }
        },
    }

    result = await ops_subagent_node(state)

    assert result["final_answer"] == "你想搜索什么内容？"
    assert result["subagent_result"]["meta"]["orchestrator_used"] is True


@pytest.mark.asyncio
async def test_simple_greeting_does_not_trigger_react_planner(monkeypatch):
    async def fail_react(**kwargs):
        raise AssertionError("simple greeting should not enter ReAct planner")

    monkeypatch.setattr("agent.multi_agent.supervisor.run_react_orchestrator", fail_react)
    routed = await supervisor_node({**BASE_STATE, "input": "你好"})

    assert routed["route"] == "chitchat"

    orchestrated = await tool_orchestrator_node({**BASE_STATE, **routed, "input": routed["input"]})

    assert orchestrated["route"] == "chitchat"


@pytest.mark.asyncio
async def test_react_delegate_route_params_are_merged(monkeypatch):
    async def fake_react(**kwargs):
        return {
            "next_agent": "quiz",
            "tool_plan": {"needs_tools": True, "tool_calls": [{"tool_name": "generate_quiz", "args": {"topic": "死锁"}}]},
            "observations": [{"tool_name": "get_weak_points_tool", "args": {}, "content": "死锁"}],
            "trace": [{"step": "delegate", "tool_name": "generate_quiz", "status": "success", "summary": "route=quiz"}],
            "guard_blocked": False,
            "react_status": "pass",
            "delegate": {"route": "quiz", "route_params": {"topic": "死锁", "quiz_type": "选择题", "num": 3}},
            "final_answer": "",
        }

    monkeypatch.setattr("agent.multi_agent.supervisor.run_react_orchestrator", fake_react)

    orchestrated = await tool_orchestrator_node(
        {
            **BASE_STATE,
            "input": "根据错题和课件安排复习并出题",
            "route": "planner",
            "route_reason": "test",
        }
    )

    assert orchestrated["route"] == "quiz"
    assert orchestrated["route_params"]["topic"] == "死锁"
    assert orchestrated["route_params"]["orchestrator"]["react_status"] == "pass"
    assert orchestrated["route_params"]["orchestrator"]["route_before_orchestrator"] == "planner"
    assert orchestrated["route_params"]["orchestrator"]["route_after_orchestrator"] == "quiz"
    assert orchestrated["route_params"]["orchestrator"]["route_changed_by_orchestrator"] is True


@pytest.mark.asyncio
async def test_ops_turns_orchestrator_guard_into_confirmation_card(monkeypatch):
    def fail_tool_map():
        raise AssertionError("guarded orchestrator action should not replan through ops tools")

    monkeypatch.setattr("agent.multi_agent.supervisor._build_ops_tool_map", fail_tool_map)
    result = await ops_subagent_node(
        {
            **BASE_STATE,
            "input": "把重复课件处理掉",
            "route": "ops",
            "route_params": {
                "orchestrator": {
                    "guard_blocked": True,
                    "react_status": "guarded",
                    "tool_plan": {
                        "tool_calls": [
                            {
                                "tool_name": "delete_knowledge_file_tool",
                                "args": {"filename": "C3-Processes.pdf", "session_id": "操作系统"},
                            }
                        ]
                    },
                    "trace": [{"step": "guard", "tool_name": "delete_knowledge_file_tool", "status": "blocked"}],
                }
            },
        }
    )

    structured = result["subagent_result"]
    assert structured["meta"]["danger_confirmation_required"] is True
    assert structured["meta"]["pending_tool"] == "delete_knowledge_file_tool"
    assert "确认卡" in result["final_answer"]


@pytest.mark.asyncio
async def test_rag_uses_orchestrator_courseware_observation(monkeypatch):
    async def fail_get_rag_service():
        raise AssertionError("RAG should not retrieve again when orchestrator supplied courseware observation")

    async def fake_call_llm(prompt_template, **kwargs):
        assert "进程是正在执行的程序" in kwargs["context"]
        return '{"answer":"进程是正在执行的程序。","evidence":[{"source":"C3-Processes.pdf","quote":"进程是正在执行的程序"}]}'

    monkeypatch.setattr("agent.multi_agent.supervisor.get_rag_service", fail_get_rag_service)
    monkeypatch.setattr("agent.multi_agent.supervisor._call_llm", fake_call_llm)

    result = await rag_subagent_node(
        {
            **BASE_STATE,
            "input": "C3 Processes 讲了什么",
            "route": "rag",
            "route_params": {
                "orchestrator": {
                    "observations": [
                        {
                            "tool_name": "search_courseware",
                            "args": {"query": "C3 Processes"},
                            "content": "【课件检索结果】关键词：C3 Processes\n1. C3-Processes.pdf：进程是正在执行的程序",
                        }
                    ],
                    "trace": [{"step": "observation", "tool_name": "search_courseware", "status": "success"}],
                }
            },
        }
    )

    assert "进程" in result["final_answer"]


@pytest.mark.asyncio
async def test_planner_injects_orchestrator_practice_observations(monkeypatch):
    async def fake_call_llm(prompt_template, **kwargs):
        snapshot = kwargs["practice_snapshot"]
        assert "死锁" in snapshot
        assert "正确率" in snapshot
        return "{}"

    monkeypatch.setattr("agent.multi_agent.supervisor._call_llm", fake_call_llm)
    result = await planner_subagent_node(
        {
            **BASE_STATE,
            "input": "根据错题制定3天计划",
            "route": "planner",
            "route_params": {
                "orchestrator": {
                    "observations": [
                        {"tool_name": "get_weak_points_tool", "args": {}, "content": "薄弱点：死锁 正确率 33%"},
                        {"tool_name": "get_practice_stats_tool", "args": {}, "content": "练习统计：总数 3，正确率 66.7%"},
                    ],
                    "trace": [{"step": "observation", "tool_name": "get_weak_points_tool", "status": "success"}],
                }
            },
        }
    )

    assert "复习计划" in result["final_answer"]


@pytest.mark.asyncio
async def test_learning_loop_uses_orchestrator_courseware_observation(monkeypatch):
    async def fail_get_rag_service():
        raise AssertionError("learning loop should reuse orchestrator evidence before retrieving")

    monkeypatch.setattr("agent.multi_agent.supervisor.get_rag_service", fail_get_rag_service)
    result = await learning_loop_subagent_node(
        {
            **BASE_STATE,
            "session_id": "react_test_session",
            "input": "开始自主复习",
            "route": "learning_loop",
            "route_params": {
                "orchestrator": {
                    "observations": [
                        {
                            "tool_name": "search_courseware",
                            "args": {"query": "进程"},
                            "content": "[source=C3-Processes.pdf page=2]\n进程是正在执行的程序。",
                        }
                    ],
                    "trace": [{"step": "observation", "tool_name": "search_courseware", "status": "success"}],
                }
            },
        }
    )

    payload = result["subagent_result"]["payload"]
    assert payload["evidence_cards"]
    assert payload["evidence_cards"][0]["source"] == "C3-Processes.pdf"
