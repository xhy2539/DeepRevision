import pytest

from agent.multi_agent.supervisor import exam_subagent_node, quiz_subagent_node
from agent.multi_agent.quiz_parser import build_quiz_payload, build_quiz_text_from_questions


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
    "route": "quiz",
    "route_reason": "",
    "route_params": {},
    "supervisor_precomputed": False,
    "subagent_result": "",
    "final_answer": "",
}


@pytest.mark.asyncio
async def test_quiz_subagent_merges_user_requested_multiple_question_types(monkeypatch):
    async def fake_run_quiz_agent(topic, quiz_type, num, sample_ctx, **kwargs):
        questions = [
            {
                "id": idx,
                "type": quiz_type,
                "question": f"{quiz_type}题干{idx}",
                "options": ["A. 对", "B. 错", "C. 其他", "D. 无关"] if quiz_type == "选择题" else None,
                "answer": "A" if quiz_type == "选择题" else "参考答案",
                "explanation": "解析",
                "score": "5分",
                "difficulty": "中等",
            }
            for idx in range(1, int(num) + 1)
        ]
        payload = build_quiz_payload("进程", questions)
        return {
            "kind": "quiz_set",
            "render_mode": "interactive_cards",
            "text": build_quiz_text_from_questions(questions),
            "payload": payload,
            "meta": {"delivery_mode": "full"},
        }

    monkeypatch.setattr("agent.multi_agent.quiz_agent.run_quiz_agent", fake_run_quiz_agent)
    monkeypatch.setattr("api.routers.knowledge.get_sample_paper_context", lambda sid: "")

    result = await quiz_subagent_node(
        {
            **BASE_STATE,
            "input": "围绕进程出选择题2道、填空题2道",
            "route_params": {"topic": "进程"},
        }
    )

    structured = result["subagent_result"]
    questions = structured["payload"]["questions"]
    assert structured["kind"] == "quiz_set"
    assert [q["type"] for q in questions] == ["选择题", "选择题", "填空题", "填空题"]
    assert structured["meta"]["question_type_plan"]["source"] == "user_explicit"


@pytest.mark.asyncio
async def test_exam_subagent_uses_web_type_reference_over_default_distribution(monkeypatch):
    captured = {}

    async def fake_run_exam_agent(topics, quiz_types, total, sample_ctx, quantity_dist, **kwargs):
        captured["quiz_types"] = quiz_types
        captured["total"] = total
        captured["quantity_dist"] = quantity_dist
        return {
            "kind": "exam_paper",
            "render_mode": "exam_canvas",
            "text": "试卷内容",
            "payload": {"exam_data": {"questions": []}},
            "meta": {"delivery_mode": "full"},
        }

    monkeypatch.setattr("agent.multi_agent.quiz_agent.run_exam_agent", fake_run_exam_agent)
    monkeypatch.setattr("api.routers.knowledge.get_sample_paper_context", lambda sid: "")

    result = await exam_subagent_node(
        {
            **BASE_STATE,
            "input": "生成综合测试卷，题型你参考网络决定",
            "route": "exam",
            "route_params": {
                "orchestrator": {
                    "observations": [
                        {
                            "tool_name": "search_exam_format",
                            "content": "网络题型参考：选择题8道、判断题4道、填空题4道、简答题2道",
                        }
                    ]
                }
            },
        }
    )

    assert captured["quantity_dist"] == {"choice": 8, "fill": 4, "judge": 4, "essay": 2}
    assert captured["total"] == 18
    assert result["subagent_result"]["meta"]["question_type_plan"]["source"] == "web_search"
