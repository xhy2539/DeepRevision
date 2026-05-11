import json
import re
from typing import Any, Dict, List


VALID_ACTION_TYPES = {"tool", "delegate", "final", "clarify"}
VALID_STEP_STATUSES = {"pending", "completed", "failed", "skipped", "blocked", "delegated", "partial"}


def make_plan_step(
    step_id: str,
    title: str,
    action_type: str,
    tool_name: str = "",
    args: Dict[str, Any] | None = None,
    expected_observation: str = "",
    fallback: str = "",
) -> Dict[str, Any]:
    """创建稳定的计划步骤结构，供后端执行和前端展示共用。"""
    safe_action = action_type if action_type in VALID_ACTION_TYPES else "tool"
    return {
        "id": str(step_id),
        "title": str(title or "").strip() or f"步骤{step_id}",
        "action_type": safe_action,
        "tool_name": str(tool_name or "").strip(),
        "args": args if isinstance(args, dict) else {},
        "expected_observation": str(expected_observation or "").strip(),
        "status": "pending",
        "fallback": str(fallback or "").strip(),
    }


def make_task_plan(
    goal: str,
    route_hint: str,
    steps: List[Dict[str, Any]] | None = None,
    constraints: List[str] | None = None,
    completion_criteria: List[str] | None = None,
    fallback_policy: str = "",
) -> Dict[str, Any]:
    """创建 TaskPlan 骨架，避免每个调用点拼接不同字段。"""
    return {
        "goal": str(goal or "").strip() or "完成用户请求",
        "route_hint": str(route_hint or "").strip() or "rag",
        "constraints": [str(item).strip() for item in (constraints or []) if str(item).strip()],
        "steps": steps or [],
        "completion_criteria": [
            str(item).strip() for item in (completion_criteria or []) if str(item).strip()
        ],
        "fallback_policy": str(fallback_policy or "").strip() or "规划失败时降级到当前专家 Agent。",
    }


def task_plan_has_steps(plan: Dict[str, Any]) -> bool:
    """判断计划是否包含可执行步骤。"""
    return bool(isinstance(plan, dict) and isinstance(plan.get("steps"), list) and plan.get("steps"))


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _course_name_from_query(query: str, fallback: str) -> str:
    match = re.search(r"(操作系统|数据结构|计算机网络|数据库|软件工程|高等数学|线性代数|大学物理|英语)", str(query or ""))
    return match.group(1) if match else str(fallback or "大学课程")


def _filename_from_query(query: str) -> str:
    match = re.search(
        r"([\w\u4e00-\u9fa5 .+\-&()（）]+?\.(?:pdf|docx?|pptx?|txt|png|jpe?g|webp|gif|bmp))",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return str(match.group(1)).strip(" ，,。；;：:") if match else ""


def _is_explicit_web_search(query: str) -> bool:
    text = _compact_text(query)
    return bool(
        re.search(r"(联网|网络|网上|互联网|网页).{0,12}(搜索|搜|查|查询|找|检索)", text)
        or re.search(r"(搜索|搜|查|查询|找|检索).{0,12}(联网|网络|网上|互联网|网页)", text)
    )


def _strip_web_trigger(query: str) -> str:
    text = str(query or "").strip()
    text = re.sub(
        r"(联网|网络|网上|互联网|网页)\s*(搜索一下|搜索|搜一下|搜|查询一下|查询|查一下|查|找一下|找|检索一下|检索)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(搜索一下|搜索|搜一下|搜|查询一下|查询|查一下|查|找一下|找|检索一下|检索)\s*(联网|网络|网上|互联网|网页)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(请|帮我|麻烦|能不能|可以|给我|帮忙|一下)", "", text, flags=re.IGNORECASE)
    return re.sub(r"^[\s，,。；;：:！？!?、]+|[\s，,。；;：:！？!?、]+$", "", text).strip()


def _is_exam_generation_request(query: str, route_hint: str) -> bool:
    text = _compact_text(query)
    return route_hint == "exam" or bool(re.search(r"(综合测试卷|综合试卷|期末试卷|生成试卷|出一套试卷|出卷)", text))


def _wants_network_question_types(query: str) -> bool:
    text = _compact_text(query)
    return bool(re.search(r"(网络|联网|网上|互联网).{0,10}(题型|试卷|格式|参考|决定)", text) or re.search(r"(题型).{0,10}(网络|联网|网上|参考)", text))


def build_default_task_plan(
    query: str,
    session_id: str,
    route_hint: str = "",
    recent_history: str = "",
) -> Dict[str, Any]:
    """为高确定性复杂任务生成无需 LLM 的 TaskPlan。"""
    del recent_history
    text = str(query or "").strip()
    route = str(route_hint or "").strip() or "rag"
    compact = _compact_text(text)
    course_name = _course_name_from_query(text, session_id)

    if _is_exam_generation_request(text, route) and _wants_network_question_types(text):
        return make_task_plan(
            goal=f"基于课件和网络题型参考生成{course_name}综合试卷",
            route_hint=route,
            constraints=["用户要求参考网络题型时，题型参考可联网；知识依据仍优先来自课件。"],
            steps=[
                make_plan_step(
                    "1",
                    "读取课件状态",
                    "tool",
                    "list_knowledge_files_tool",
                    {"session_id": session_id},
                    "返回已上传课件列表与向量化状态",
                    "若课件为空，委派 exam 时要求保守提示依据不足。",
                ),
                make_plan_step(
                    "2",
                    "搜索题型参考",
                    "tool",
                    "search_exam_format",
                    {"course_name": course_name},
                    "返回网络试卷题型结构参考",
                    "搜索失败时使用默认综合卷分布。",
                ),
                make_plan_step(
                    "3",
                    "委派组卷专家",
                    "delegate",
                    "exam",
                    {"question_type_source": "web_search", "topics": [course_name]},
                    "返回结构化 exam_paper",
                    "组卷失败时返回可解释失败原因。",
                ),
            ],
            completion_criteria=["生成 exam_paper 结构化结果", "payload 中保留任务计划和题型来源"],
            fallback_policy="网络题型参考不可用时，降级为默认综合卷 10/5/5/3。",
        )

    if _is_explicit_web_search(text):
        wants_format = bool(re.search(r"(题型|试卷格式|考试格式|卷面结构|题型分布|期末试卷)", text))
        tool_name = "search_exam_format" if wants_format else "web_search"
        args = {"course_name": course_name} if wants_format else {"query": _strip_web_trigger(text)}
        return make_task_plan(
            goal="执行联网检索并返回可读结果",
            route_hint=route,
            constraints=["联网结果只能作为参考，不能替代课件证据。"],
            steps=[
                make_plan_step(
                    "1",
                    "联网检索",
                    "tool",
                    tool_name,
                    args,
                    "返回搜索结果摘要与来源链接",
                    "搜索失败时说明暂不可用。",
                )
            ],
            completion_criteria=["将 observation 交给 ops 直接展示"],
            fallback_policy="搜索不可用时返回明确失败原因。",
        )

    if re.fullmatch(r"确认删除(?:重复)?课件[:：].+", text):
        return make_task_plan(
            goal="执行已确认的课件删除操作",
            route_hint="ops",
            steps=[],
            completion_criteria=["交给 ops 原确认链执行"],
            fallback_policy="若确认语义无法识别则回到 ops。",
        )

    if re.search(r"(删除|移除)\s*(课件|文件)?", text):
        filename = _filename_from_query(text)
        if filename:
            return make_task_plan(
                goal=f"删除课件文件 {filename}",
                route_hint="ops",
                constraints=["危险写操作必须生成确认卡，不能自动执行。"],
                steps=[
                    make_plan_step(
                        "1",
                        "请求删除课件",
                        "tool",
                        "delete_knowledge_file_tool",
                        {"filename": filename, "session_id": session_id},
                        "返回删除结果",
                        "必须先让用户点击确认卡。",
                    )
                ],
                completion_criteria=["返回安全确认卡"],
                fallback_policy="无法确认目标文件时交给 ops 询问。",
            )

    if ("课件" in text or re.search(r"\bC\d+", text, flags=re.IGNORECASE)) and re.search(
        r"(讲了什么|主要讲|内容|总结|要点|解释)", text
    ):
        return make_task_plan(
            goal="检索课件并回答用户问题",
            route_hint=route,
            constraints=["回答必须优先使用课件片段。"],
            steps=[
                make_plan_step(
                    "1",
                    "检索课件片段",
                    "tool",
                    "search_courseware",
                    {"query": text},
                    "返回相关课件片段",
                    "检索为空时提示用户换关键词。",
                )
            ],
            completion_criteria=["将课件 observation 交给 RAG 专家回答"],
            fallback_policy="检索失败时由 RAG 专家给保守提示。",
        )

    if re.search(r"(错题|薄弱点|练习画像|掌握度).*(安排|规划|复习).*(出题|练习|题)", compact):
        return make_task_plan(
            goal="读取学习画像、检索课件并安排练习",
            route_hint=route,
            constraints=["不能在空画像时声称基于最近错题。"],
            steps=[
                make_plan_step("1", "读取练习统计", "tool", "get_practice_stats_tool", {"session_id": session_id}, "返回练习统计"),
                make_plan_step("2", "读取薄弱点", "tool", "get_weak_points_tool", {"session_id": session_id}, "返回薄弱知识点"),
                make_plan_step("3", "检索相关课件", "tool", "search_courseware", {"query": text}, "返回相关课件证据"),
                make_plan_step("4", "委派出题专家", "delegate", "quiz", {"topic": text, "quiz_type": "选择题", "num": 3}, "返回 quiz_set"),
            ],
            completion_criteria=["优先覆盖薄弱点", "返回可执行练习"],
            fallback_policy="画像为空时使用课件结构做基线诊断。",
        )

    return make_task_plan(
        goal=text or "完成用户请求",
        route_hint=route,
        steps=[],
        completion_criteria=["由当前专家 Agent 完成回复"],
        fallback_policy="不需要工具规划时直接走当前路由。",
    )


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty planner output")
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("planner output is not json")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("planner output must be object")
    return parsed


def _normalize_steps(value: Any) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return steps
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        action_type = str(item.get("action_type") or "tool").strip()
        if action_type not in VALID_ACTION_TYPES:
            action_type = "tool"
        status = str(item.get("status") or "pending").strip()
        if status not in VALID_STEP_STATUSES:
            status = "pending"
        args = item.get("args") if isinstance(item.get("args"), dict) else {}
        step = make_plan_step(
            str(item.get("id") or idx),
            str(item.get("title") or f"步骤{idx}"),
            action_type,
            str(item.get("tool_name") or item.get("tool") or ""),
            args,
            str(item.get("expected_observation") or ""),
            str(item.get("fallback") or ""),
        )
        step["status"] = status
        steps.append(step)
    return steps[:8]


def _legacy_decision_to_task_plan(parsed: Dict[str, Any], query: str, route_hint: str) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    for idx, call in enumerate(parsed.get("tool_calls") or [], start=1):
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool_name") or call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if name:
            steps.append(make_plan_step(str(idx), f"调用 {name}", "tool", name, args, "返回工具 observation"))
    delegate_route = str(parsed.get("delegate_route") or "").strip()
    if delegate_route:
        steps.append(
            make_plan_step(
                str(len(steps) + 1),
                f"委派 {delegate_route} 专家",
                "delegate",
                delegate_route,
                {},
                "返回专家结果",
            )
        )
    final_answer = str(parsed.get("final_answer") or "").strip()
    if final_answer and not steps:
        steps.append(
            make_plan_step(
                "1",
                "生成最终回答",
                "final",
                "",
                {"answer": final_answer},
                "返回最终回答",
            )
        )
    return make_task_plan(
        goal=str(parsed.get("goal") or query or "完成用户请求"),
        route_hint=route_hint,
        steps=steps,
        constraints=[],
        completion_criteria=[str(parsed.get("stop_reason") or "完成本轮任务")],
        fallback_policy="Planner 决策不可执行时降级到当前专家 Agent。",
    )


def normalize_task_plan(raw: Any, query: str, route_hint: str = "") -> Dict[str, Any]:
    """把 LLM/旧决策输出归一为 TaskPlan。"""
    parsed = _safe_json_loads(raw)
    if "steps" not in parsed:
        return _legacy_decision_to_task_plan(parsed, query, route_hint)
    plan = make_task_plan(
        goal=str(parsed.get("goal") or query or "完成用户请求"),
        route_hint=str(parsed.get("route_hint") or route_hint or "rag"),
        steps=_normalize_steps(parsed.get("steps")),
        constraints=parsed.get("constraints") if isinstance(parsed.get("constraints"), list) else [],
        completion_criteria=parsed.get("completion_criteria") if isinstance(parsed.get("completion_criteria"), list) else [],
        fallback_policy=str(parsed.get("fallback_policy") or "Planner 失败时降级到当前专家 Agent。"),
    )
    return plan


def failed_task_plan(query: str, route_hint: str, reason: str) -> Dict[str, Any]:
    """构造规划失败时仍可展示的 TaskPlan。"""
    plan = make_task_plan(
        goal=query or "完成用户请求",
        route_hint=route_hint,
        steps=[],
        completion_criteria=["降级到当前专家 Agent"],
        fallback_policy=f"规划失败：{str(reason or '')[:180]}",
    )
    return plan


def task_planner_prompt(
    *,
    query: str,
    route_hint: str,
    session_id: str,
    recent_history: str,
    observations: str,
    tool_catalog: str,
) -> str:
    """构造 Task Planner 提示词，要求只输出结构化 JSON。"""
    return f"""你是 DeepRevision 的 ReAct Task Planner。只输出 JSON，不输出 Thought、提示词或解释。

session_id: {session_id}
route_hint: {route_hint}
用户请求: {query}

近期历史:
{recent_history or "无"}

已有 Observation:
{observations or "无"}

工具目录:
{tool_catalog}

输出严格 JSON，字段必须完整：
{{
  "goal": "本轮任务目标",
  "constraints": ["约束1"],
  "steps": [
    {{
      "id": "1",
      "title": "步骤标题",
      "action_type": "tool|delegate|final|clarify",
      "tool_name": "工具名或专家路由",
      "args": {{}},
      "expected_observation": "预期观察",
      "status": "pending",
      "fallback": "失败兜底"
    }}
  ],
  "completion_criteria": ["完成标准"],
  "fallback_policy": "整体兜底策略"
}}

规则：
1. 复杂任务先取证再委派；出题/出卷使用 delegate 到 quiz/exam，不直接生成裸文本。
2. read_only 工具可以自动执行；删除/清空/销毁类工具只能生成确认卡。
3. 用户显式要求联网搜索题型时，先 search_exam_format，再委派 exam。
4. 不要编造工具名；如果不需要工具，给 final 或 delegate 步骤。"""
