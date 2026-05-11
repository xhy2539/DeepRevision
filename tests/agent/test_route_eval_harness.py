import pytest

from agent.multi_agent.route_eval import (
    ROUTE_EVAL_REQUIRED_ROUTES,
    RouteEvalCase,
    evaluate_route_case,
    list_route_eval_cases,
    summarize_route_eval_results,
)


def test_route_eval_cases_cover_each_top_level_agent():
    cases = list_route_eval_cases()

    covered = {case.expected_route for case in cases}

    assert ROUTE_EVAL_REQUIRED_ROUTES <= covered
    assert all(case.name and case.input and case.expected_route for case in cases)


@pytest.mark.asyncio
async def test_route_eval_case_reports_initial_and_final_route_mismatches():
    case = RouteEvalCase(
        name="courseware_summary",
        input="C3 Processes 讲了什么",
        expected_route="rag",
        expected_final_route="rag",
        run_orchestrator=True,
        tags=("rag",),
    )

    async def fake_supervisor(state):
        return {"route": "ops", "route_reason": "bad test route", "route_params": {}, "input": state["input"]}

    async def fake_orchestrator(state):
        return {**state, "route": "history"}

    result = await evaluate_route_case(case, fake_supervisor, fake_orchestrator)

    assert result["passed"] is False
    assert result["initial_route"] == "ops"
    assert result["final_route"] == "history"
    assert result["mismatches"] == [
        "expected initial route rag, got ops",
        "expected final route rag, got history",
    ]


def test_route_eval_summary_counts_pass_rate_and_failures():
    summary = summarize_route_eval_results(
        [
            {"name": "a", "passed": True, "mismatches": []},
            {"name": "b", "passed": False, "mismatches": ["expected initial route rag, got ops"]},
        ]
    )

    assert summary == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
        "failures": [{"name": "b", "mismatches": ["expected initial route rag, got ops"]}],
    }
