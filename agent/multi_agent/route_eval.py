from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable


ROUTE_EVAL_REQUIRED_ROUTES: set[str] = {
    "rag",
    "quiz",
    "exam",
    "ops",
    "planner",
    "history",
    "learning_loop",
    "chitchat",
}


@dataclass(frozen=True)
class RouteEvalCase:
    """描述一个稳定路由样例，用于回归评测和人工审阅。"""

    name: str
    input: str
    expected_route: str
    expected_final_route: str | None = None
    client_action: str = ""
    run_orchestrator: bool = False
    route_params: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    reason: str = ""


SupervisorFn = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]
OrchestratorFn = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


_ROUTE_EVAL_CASES: tuple[RouteEvalCase, ...] = (
    RouteEvalCase(
        name="courseware_summary",
        input="C3 Processes 讲了什么",
        expected_route="rag",
        expected_final_route="rag",
        run_orchestrator=True,
        tags=("rag", "courseware"),
        reason="课件内容理解应优先进入 RAG，并允许编排层先检索课件。",
    ),
    RouteEvalCase(
        name="quiz_generation",
        input="围绕死锁出3道选择题",
        expected_route="quiz",
        tags=("quiz",),
        reason="明确出少量题目应进入 Quiz Agent。",
    ),
    RouteEvalCase(
        name="button_exam_generation",
        input="生成综合测试卷",
        expected_route="exam",
        expected_final_route="exam",
        client_action="generate_exam",
        run_orchestrator=True,
        tags=("exam", "button"),
        reason="前端生成试卷按钮应绕过工具编排直达 Exam Agent。",
    ),
    RouteEvalCase(
        name="network_exam_format",
        input="网络搜索一下操作系统期末考试题型",
        expected_route="ops",
        expected_final_route="ops",
        run_orchestrator=True,
        tags=("ops", "web_search"),
        reason="显式联网检索属于工具操作，由 Ops 展示 observation。",
    ),
    RouteEvalCase(
        name="review_plan_from_wrong_questions",
        input="根据最近错题制定3天复习计划",
        expected_route="planner",
        tags=("planner", "mastery"),
        reason="复习安排请求应进入 Planner Agent。",
    ),
    RouteEvalCase(
        name="practice_history",
        input="查看最近10条错题记录",
        expected_route="history",
        tags=("history",),
        reason="查看练习历史应进入 History Agent。",
    ),
    RouteEvalCase(
        name="learning_loop_start",
        input="开始自主复习闭环",
        expected_route="learning_loop",
        tags=("learning_loop",),
        reason="自主复习闭环应由 LearningLoopAgent 管理状态。",
    ),
    RouteEvalCase(
        name="greeting",
        input="你好",
        expected_route="chitchat",
        expected_final_route="chitchat",
        run_orchestrator=True,
        tags=("chitchat",),
        reason="简单问候应走确定性 chitchat 快路径。",
    ),
)


def list_route_eval_cases(tags: Iterable[str] | None = None) -> list[RouteEvalCase]:
    """按标签过滤并返回路由评测样例，默认返回全量样例。"""

    wanted = {str(tag).strip() for tag in (tags or []) if str(tag).strip()}
    if not wanted:
        return list(_ROUTE_EVAL_CASES)
    return [case for case in _ROUTE_EVAL_CASES if wanted.intersection(case.tags)]


def build_route_eval_state(case: RouteEvalCase, session_id: str = "操作系统") -> dict[str, Any]:
    """构造与 supervisor_workflow 兼容的最小评测 state。"""

    return {
        "input": case.input,
        "chat_history": [],
        "memory_context": "",
        "tool_context": "",
        "session_id": session_id,
        "client_action": case.client_action,
        "exam_stage_plan": True,
        "exam_rerun_stage": None,
        "exam_partial_questions": [],
        "exam_fast_mode": True,
        "quiz_force_llm_critic": False,
        "route": "",
        "route_reason": "",
        "route_params": dict(case.route_params or {}),
        "supervisor_precomputed": False,
        "subagent_result": "",
        "final_answer": "",
    }


async def _maybe_await(value: Any) -> Any:
    """兼容同步/异步测试替身，避免评测调用方写重复分支。"""

    if inspect.isawaitable(value):
        return await value
    return value


async def evaluate_route_case(
    case: RouteEvalCase,
    supervisor_fn: SupervisorFn,
    orchestrator_fn: OrchestratorFn | None = None,
    *,
    session_id: str = "操作系统",
) -> dict[str, Any]:
    """执行单条路由样例，并返回可展示的 pass/fail 结果。"""

    state = build_route_eval_state(case, session_id=session_id)
    routed = await _maybe_await(supervisor_fn(dict(state)))
    initial_route = str((routed or {}).get("route") or "")
    final_route = initial_route
    orchestrated: dict[str, Any] | None = None

    if case.run_orchestrator:
        if orchestrator_fn is None:
            final_route = ""
        else:
            orchestrator_state = {**state, **(routed or {})}
            orchestrated = await _maybe_await(orchestrator_fn(orchestrator_state))
            final_route = str((orchestrated or {}).get("route") or initial_route)

    expected_final = case.expected_final_route or case.expected_route
    mismatches: list[str] = []
    if initial_route != case.expected_route:
        mismatches.append(f"expected initial route {case.expected_route}, got {initial_route}")
    if case.run_orchestrator and final_route != expected_final:
        mismatches.append(f"expected final route {expected_final}, got {final_route}")

    return {
        "name": case.name,
        "input": case.input,
        "expected_route": case.expected_route,
        "expected_final_route": expected_final,
        "initial_route": initial_route,
        "final_route": final_route,
        "passed": not mismatches,
        "mismatches": mismatches,
        "route_reason": str((routed or {}).get("route_reason") or ""),
        "routed": routed or {},
        "orchestrated": orchestrated or {},
    }


def summarize_route_eval_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """汇总路由评测结果，便于日志、CI 或后续看板消费。"""

    items = list(results)
    passed = [item for item in items if bool(item.get("passed"))]
    failures = [
        {
            "name": str(item.get("name") or ""),
            "mismatches": list(item.get("mismatches") or []),
        }
        for item in items
        if not bool(item.get("passed"))
    ]
    total = len(items)
    return {
        "total": total,
        "passed": len(passed),
        "failed": len(failures),
        "pass_rate": (len(passed) / total) if total else 0.0,
        "failures": failures,
    }
