from api.message_protocol import build_assistant_message


def test_build_assistant_message_merges_react_trace_from_route_params():
    trace = [{"step": "plan", "status": "success", "summary": "planner selected quiz"}]
    task_plan = {
        "goal": "生成练习",
        "constraints": [],
        "steps": [],
        "completion_criteria": ["返回题卡"],
        "fallback_policy": "降级到 quiz",
    }
    plan_execution = [{"step_id": "1", "status": "completed", "summary": "已完成"}]

    message = build_assistant_message(
        "quiz",
        "练习题生成完成",
        "操作系统",
        route_params={
            "orchestrator": {
                "trace": trace,
                "react_status": "pass",
                "task_plan": task_plan,
                "plan_execution": plan_execution,
            }
        },
        structured_result={"kind": "chat", "render_mode": "markdown", "text": "练习题生成完成", "payload": {}, "meta": {}},
    )

    assert message["meta"]["agent_mode"] == "react"
    assert message["meta"]["react_status"] == "pass"
    assert message["meta"]["agent_trace"] == trace
    assert message["payload"]["agent_trace"] == trace
    assert message["payload"]["task_plan"] == task_plan
    assert message["payload"]["plan_execution"] == plan_execution
