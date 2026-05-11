import asyncio
import re
from typing import Any, Dict, List

from agent.multi_agent.planning_agent import (
    build_default_task_plan,
    failed_task_plan,
    normalize_task_plan,
    task_plan_has_steps,
    task_planner_prompt,
)


DANGEROUS_TOOLS = {
    "delete_practice_record_tool",
    "clear_practice_history_tool",
    "delete_chat_message_tool",
    "clear_chat_history_tool",
    "delete_session_tool",
    "delete_knowledge_file_tool",
    "delete_duplicate_knowledge_files_tool",
}

DELEGATE_TOOLS = {
    "generate_quiz": "quiz",
    "generate_exam_paper": "exam",
}

EXPORT_TOOLS = {
    "export_exam_docx_tool",
    "export_answer_sheet_tool",
}

MANAGED_WRITE_TOOLS = {
    "create_session_tool",
    "rename_session_tool",
    "backfill_practice_kp_tool",
}

VALID_DELEGATE_ROUTES = {"rag", "quiz", "exam", "ops", "planner", "history", "learning_loop", "chitchat"}


WEB_SEARCH_FILLER_WORDS = {
    "啊",
    "呀",
    "吧",
    "呢",
    "嘛",
    "哈",
    "啦",
    "哦",
    "喔",
    "额",
    "呃",
}


def _course_name_from_query(query: str, fallback: str) -> str:
    """从联网题型检索请求里提取课程名。"""
    match = re.search(r"(操作系统|数据结构|计算机网络|数据库|软件工程|高等数学|线性代数|大学物理|英语)", query)
    return match.group(1) if match else fallback or "大学课程"


def _filename_from_query(query: str) -> str:
    """从文件管理请求里提取一个课件文件名。"""
    match = re.search(
        r"([\w\u4e00-\u9fa5 .+\-&()（）]+?\.(?:pdf|docx?|pptx?|txt|png|jpe?g|webp|gif|bmp))",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    return str(match.group(1)).strip(" ，,。；;：:") if match else ""


def is_explicit_web_search_intent(query: str) -> bool:
    """识别用户明确要求联网/网络搜索的请求。"""
    text = re.sub(r"\s+", "", str(query or "").strip())
    if not text:
        return False
    return bool(
        re.search(r"(联网|网络|网上|互联网|网页).{0,12}(搜索|搜|查|查询|找|检索)", text)
        or re.search(r"(搜索|搜|查|查询|找|检索).{0,12}(联网|网络|网上|互联网|网页)", text)
    )


def _strip_web_search_trigger(query: str) -> str:
    """去掉搜索触发词，保留真正要查的主题。"""
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
    text = re.sub(
        r"(请|帮我|麻烦|能不能|可以|给我|帮忙|联网搜索|网络搜索|网上搜索|互联网搜索|网页搜索|搜索一下|查询一下|查一下|搜一下|一下)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^[\s，,。；;：:！？!?、]+|[\s，,。；;：:！？!?、]+$", "", text)
    text = re.sub(r"[啊呀吧呢嘛哈啦哦喔额呃]+$", "", text).strip()
    return text


def _is_low_information_topic(topic: str) -> bool:
    """判断清洗后的主题是否只剩语气词或无意义短词。"""
    text = re.sub(r"[\s，,。；;：:！？!?、]+", "", str(topic or ""))
    if not text:
        return True
    if text in WEB_SEARCH_FILLER_WORDS:
        return True
    return len(text) <= 1 and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text)


def _normalize_history_topic(content: str) -> str:
    """把最近用户问题压成适合搜索的短主题。"""
    topic = str(content or "").strip()
    topic = re.sub(r"^[\s，,。；;：:]+", "", topic)
    topic = re.sub(r"[。！？!?]+$", "", topic).strip()
    topic = re.sub(r"^(请|帮我|麻烦|能不能|可以)?(解释|讲讲|讲一下|说明|介绍|分析|总结)\s*", "", topic)
    topic = re.sub(r"(是什么|有哪些|有哪[些几]个|分别是什么|是啥)$", "", topic).strip()
    topic = re.sub(r"[吗呢啊吧]+$", "", topic).strip()
    return topic


def _topic_from_recent_history(recent_history: str) -> str:
    """从近期对话里取最近一个有信息量的学生话题作为搜索兜底。"""
    lines = [line.strip() for line in str(recent_history or "").splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.match(r"^(学生|用户|Human|human)\s*[:：]\s*(.+)$", line)
        if not match:
            continue
        topic = _strip_web_search_trigger(match.group(2))
        topic = _normalize_history_topic(topic)
        if topic and not is_explicit_web_search_intent(topic) and not _is_low_information_topic(topic):
            return topic
    return ""


def _web_search_topic(query: str, recent_history: str = "") -> str:
    """优先用当前输入提取搜索主题，缺省时回看近期上下文。"""
    topic = _strip_web_search_trigger(query)
    if not _is_low_information_topic(topic):
        return topic
    return _topic_from_recent_history(recent_history)


def _base_plan(route: str) -> Dict[str, Any]:
    """创建统一计划骨架，便于前端展示 ReAct 轨迹。"""
    return {
        "needs_tools": False,
        "next_agent": route,
        "requires_confirmation": False,
        "tool_calls": [],
        "final_answer": "",
        "trace": [{"step": "intent", "status": "success", "summary": f"route_hint={route}"}],
    }


def classify_tool(tool_name: str) -> str:
    """按执行风险给工具分类，供 ReAct 循环决定执行或委派。"""
    name = str(tool_name or "").strip()
    if name in DANGEROUS_TOOLS:
        return "dangerous_write"
    if name in DELEGATE_TOOLS:
        return "delegate"
    if name in EXPORT_TOOLS:
        return "export_high_cost"
    if name in MANAGED_WRITE_TOOLS:
        return "managed_write"
    return "read_only"


def should_use_react_planner(query: str, route_hint: str = "", force_react: bool = False) -> bool:
    """混合触发：复杂、多步、工具型请求才进入 LLM ReAct Planner。"""
    if force_react:
        return True
    text = str(query or "").strip()
    route = str(route_hint or "").strip()
    if not text:
        return False
    if route == "chitchat":
        return False
    if route in {"learning_loop", "ops"}:
        return True
    return bool(
        re.search(r"(先|再|然后|并且|同时|结合|根据).*(课件|错题|练习|联网|搜索|出题|计划)", text)
        or re.search(r"(课件|错题|练习|画像|薄弱点).*(安排|规划|复习).*(出题|练习|题)", text)
        or re.search(r"(自主规划|多步|ReAct|react|工具调用|调用工具)", text, flags=re.IGNORECASE)
    )


def build_tool_plan(query: str, session_id: str, route_hint: str = "", recent_history: str = "") -> Dict[str, Any]:
    """为高确定性的工具需求生成第一版 ReAct 工具计划。"""
    text = str(query or "").strip()
    compact = re.sub(r"\s+", "", text)
    route = str(route_hint or "").strip() or "rag"
    plan = _base_plan(route)

    if is_explicit_web_search_intent(text):
        if re.search(r"(题型|试卷格式|考试格式|卷面结构|题型分布|期末试卷)", text):
            plan.update(
                {
                    "needs_tools": True,
                    "next_agent": "ops",
                    "tool_calls": [
                        {
                            "tool_name": "search_exam_format",
                            "args": {"course_name": _course_name_from_query(text, session_id)},
                        }
                    ],
                }
            )
        else:
            topic = _web_search_topic(text, recent_history)
            if not topic:
                plan["final_answer"] = "你想搜索什么内容？"
                plan["trace"].append({"step": "clarify", "status": "blocked", "summary": "web_search_topic_missing"})
                return plan
            plan.update(
                {
                    "needs_tools": True,
                    "next_agent": "ops",
                    "tool_calls": [{"tool_name": "web_search", "args": {"query": topic}}],
                }
            )
        return plan

    # 已确认的危险操作继续交给 ops 原有确认执行链，避免被编排层二次拦截。
    if re.fullmatch(r"确认删除(?:重复)?课件[:：].+", text):
        plan["next_agent"] = "ops"
        return plan

    if re.search(r"(删除|移除)\s*(课件|文件)?", text):
        filename = _filename_from_query(text)
        if filename:
            plan.update(
                {
                    "needs_tools": True,
                    "next_agent": "ops",
                    "requires_confirmation": True,
                    "tool_calls": [
                        {
                            "tool_name": "delete_knowledge_file_tool",
                            "args": {"filename": filename, "session_id": session_id},
                        }
                    ],
                }
            )
            return plan

    if ("课件" in text or re.search(r"\bC\d+", text, flags=re.IGNORECASE)) and re.search(
        r"(讲了什么|主要讲|内容|总结|要点|解释)", text
    ):
        plan.update(
            {
                "needs_tools": True,
                "next_agent": "rag",
                "tool_calls": [{"tool_name": "search_courseware", "args": {"query": text}}],
            }
        )
        return plan

    return plan


def _tool_map() -> Dict[str, Any]:
    """按工具名懒加载 LangChain 工具，降低普通路由导入成本。"""
    from agent.tools.agent_tools import tools as registered_tools

    return {getattr(tool, "name", ""): tool for tool in registered_tools if getattr(tool, "name", "")}


def _extract_tool_args(tool_obj: Any) -> str:
    """提取工具参数名，避免把完整 schema 塞进 Planner 提示词。"""
    try:
        schema = getattr(tool_obj, "args_schema", None)
        fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None) or {}
        if isinstance(fields, dict):
            return ", ".join(str(key) for key in fields.keys())
    except Exception:
        return ""
    return ""


def _tool_catalog_text() -> str:
    """生成 ReAct Planner 可读的工具目录，包含工具名、参数和策略。"""
    lines: List[str] = []
    for name, tool_obj in sorted(_tool_map().items()):
        desc = str(getattr(tool_obj, "description", "") or "").strip()
        args = _extract_tool_args(tool_obj)
        policy = classify_tool(name)
        lines.append(f"- {name}({args}) [{policy}]: {desc[:180]}")
    return "\n".join(lines) if lines else "- no tools"


def _clip_observations(observations: List[Dict[str, Any]]) -> str:
    """压缩 observation 文本，控制 Planner 上下文长度。"""
    lines: List[str] = []
    for idx, item in enumerate(observations[-6:], start=1):
        name = str(item.get("tool_name") or "tool")
        content = str(item.get("content") or "").replace("\n", " ")[:700]
        lines.append(f"{idx}. {name}: {content}")
    return "\n".join(lines) if lines else "无"


def _react_planner_prompt(
    *,
    query: str,
    route_hint: str,
    session_id: str,
    recent_history: str,
    observations: List[Dict[str, Any]],
) -> str:
    """构造 TaskPlan Planner 提示词；保留旧函数名便于测试 monkeypatch。"""
    return task_planner_prompt(
        query=query,
        route_hint=route_hint,
        session_id=session_id,
        recent_history=recent_history,
        observations=_clip_observations(observations),
        tool_catalog=_tool_catalog_text(),
    )


async def _call_react_planner(**kwargs: Any) -> Any:
    """调用轻量模型生成 ReAct 规划；测试中可直接 monkeypatch。"""
    from langchain_core.messages import HumanMessage
    from model.factory import backup_light_chat_model, chat_model, light_chat_model

    prompt = _react_planner_prompt(**kwargs)
    providers = [light_chat_model, backup_light_chat_model, chat_model]
    last_error: Exception | None = None
    for model in providers:
        if model is None:
            continue
        try:
            result = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt)]), timeout=12.0)
            return getattr(result, "content", result)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"ReAct planner unavailable: {last_error}")


async def execute_tool_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """执行安全工具调用，并把结果作为 observation 交给后续 Agent。"""
    trace = list(plan.get("trace") or [])
    calls: List[Dict[str, Any]] = list(plan.get("tool_calls") or [])
    if plan.get("requires_confirmation"):
        trace.append({"step": "guard", "status": "blocked", "summary": "dangerous tool requires confirmation"})
        return {"guard_blocked": True, "observations": [], "trace": trace}

    tools = _tool_map()
    observations: List[Dict[str, Any]] = []
    for call in calls[:3]:
        name = str(call.get("tool_name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        tool = tools.get(name)
        if tool is None:
            trace.append({"step": "action", "tool_name": name, "status": "failed", "summary": "tool not registered"})
            continue
        try:
            result = await tool.ainvoke(args)
            content = result if isinstance(result, str) else str(result)
            observations.append({"tool_name": name, "args": args, "content": content[:2500]})
            trace.append({"step": "action", "tool_name": name, "status": "success", "summary": f"observation_len={len(content)}"})
            trace.append({"step": "observation", "tool_name": name, "status": "success", "summary": content[:160]})
        except Exception as exc:
            trace.append({"step": "action", "tool_name": name, "status": "failed", "summary": str(exc)[:300]})
    return {"guard_blocked": False, "observations": observations, "trace": trace}


def _legacy_plan_from_task_plan(task_plan: Dict[str, Any], next_agent: str = "") -> Dict[str, Any]:
    """把 TaskPlan 投影成旧 tool_plan，兼容 ops 确认卡和现有测试。"""
    calls: List[Dict[str, Any]] = []
    requires_confirmation = False
    for step in task_plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("action_type") != "tool":
            continue
        tool_name = str(step.get("tool_name") or "").strip()
        if not tool_name:
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        calls.append({"tool_name": tool_name, "args": args})
        if classify_tool(tool_name) == "dangerous_write":
            requires_confirmation = True
    return {
        "needs_tools": bool(calls),
        "next_agent": next_agent or str(task_plan.get("route_hint") or "rag"),
        "requires_confirmation": requires_confirmation,
        "tool_calls": calls,
        "final_answer": "",
        "trace": [],
    }


def _observation_summary(content: Any, limit: int = 180) -> str:
    """压缩 observation，避免前端轨迹过长。"""
    return re.sub(r"\s+", " ", str(content or "")).strip()[:limit]


def _plan_execution_item(step: Dict[str, Any], status: str, summary: str = "") -> Dict[str, Any]:
    """生成前端可展示的计划执行记录。"""
    return {
        "step_id": str(step.get("id") or ""),
        "title": str(step.get("title") or ""),
        "action_type": str(step.get("action_type") or ""),
        "tool_name": str(step.get("tool_name") or ""),
        "status": status,
        "summary": str(summary or "")[:300],
    }


def _is_safety_task_plan(task_plan: Dict[str, Any]) -> bool:
    """判断计划是否包含必须优先拦截的危险写操作。"""
    for step in task_plan.get("steps") or []:
        if isinstance(step, dict) and classify_tool(str(step.get("tool_name") or "")) == "dangerous_write":
            return True
    return False


def _is_exam_format_task_plan(task_plan: Dict[str, Any]) -> bool:
    """判断计划是否为网络题型参考后委派组卷的高确定性计划。"""
    steps = [step for step in (task_plan.get("steps") or []) if isinstance(step, dict)]
    has_format = any(str(step.get("tool_name") or "") == "search_exam_format" for step in steps)
    has_exam_delegate = any(step.get("action_type") == "delegate" and str(step.get("tool_name") or "") == "exam" for step in steps)
    return has_format and has_exam_delegate


async def execute_task_plan(
    task_plan: Dict[str, Any],
    *,
    query: str,
    route: str,
    session_id: str,
    max_steps: int = 6,
    trace: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """执行 TaskPlan 步骤，按工具风险策略自动执行、委派或拦截。"""
    del session_id
    safe_route = str(route or task_plan.get("route_hint") or "rag")
    safe_max_steps = max(1, min(int(max_steps or 6), 6))
    plan = {**task_plan, "steps": [dict(step) for step in (task_plan.get("steps") or []) if isinstance(step, dict)]}
    run_trace: List[Dict[str, Any]] = list(trace or [])
    if not run_trace:
        run_trace.append({"step": "intent", "status": "success", "summary": f"route_hint={safe_route}"})
    run_trace.append({"step": "plan", "status": "success", "summary": f"goal={plan.get('goal', '')[:120]}"})

    observations: List[Dict[str, Any]] = []
    plan_execution: List[Dict[str, Any]] = []
    executed_count = 0

    for step in plan["steps"]:
        if executed_count >= safe_max_steps:
            step["status"] = "partial"
            plan_execution.append(_plan_execution_item(step, "partial", f"max_steps={safe_max_steps} reached"))
            run_trace.append({"step": "final", "status": "partial", "summary": f"max_steps={safe_max_steps} reached"})
            return {
                "next_agent": safe_route,
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, safe_route),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": False,
                "react_status": "partial",
                "delegate": {},
                "final_answer": "",
                "needs_replan": False,
                "executed_count": executed_count,
            }

        action_type = str(step.get("action_type") or "").strip()
        tool_name = str(step.get("tool_name") or "").strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        executed_count += 1

        if action_type == "clarify":
            answer = str(args.get("message") or step.get("fallback") or step.get("title") or "请补充更具体的信息。")
            step["status"] = "completed"
            plan_execution.append(_plan_execution_item(step, "completed", answer))
            run_trace.append({"step": "final", "status": "success", "summary": "clarification requested"})
            return {
                "next_agent": safe_route,
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, safe_route),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": False,
                "react_status": "pass",
                "delegate": {},
                "final_answer": answer,
                "needs_replan": False,
                "executed_count": executed_count,
            }

        if action_type == "final":
            answer = str(args.get("answer") or step.get("expected_observation") or step.get("title") or "")
            step["status"] = "completed"
            plan_execution.append(_plan_execution_item(step, "completed", "final_answer"))
            run_trace.append({"step": "final", "status": "success", "summary": "planner returned final answer"})
            return {
                "next_agent": safe_route,
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, safe_route),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": False,
                "react_status": "pass",
                "delegate": {},
                "final_answer": answer,
                "needs_replan": False,
                "executed_count": executed_count,
            }

        if action_type == "delegate":
            delegate_route = tool_name if tool_name in VALID_DELEGATE_ROUTES else str(args.get("route") or safe_route)
            delegate_args = {key: value for key, value in args.items() if key != "route"}
            step["status"] = "delegated"
            plan_execution.append(_plan_execution_item(step, "delegated", f"route={delegate_route}"))
            run_trace.append({"step": "delegate", "tool_name": tool_name, "status": "success", "summary": f"route={delegate_route}"})
            return {
                "next_agent": delegate_route,
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, delegate_route),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": False,
                "react_status": "pass",
                "delegate": {"route": delegate_route, "route_params": delegate_args},
                "final_answer": "",
                "needs_replan": False,
                "executed_count": executed_count,
            }

        policy = classify_tool(tool_name)
        if policy == "delegate":
            delegate_route = DELEGATE_TOOLS.get(tool_name, safe_route)
            step["status"] = "delegated"
            plan_execution.append(_plan_execution_item(step, "delegated", f"route={delegate_route}"))
            run_trace.append({"step": "delegate", "tool_name": tool_name, "status": "success", "summary": f"route={delegate_route}"})
            return {
                "next_agent": delegate_route,
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, delegate_route),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": False,
                "react_status": "pass",
                "delegate": {"route": delegate_route, "route_params": args},
                "final_answer": "",
                "needs_replan": False,
                "executed_count": executed_count,
            }

        if policy == "dangerous_write":
            step["status"] = "blocked"
            summary = "dangerous tool requires confirmation"
            plan_execution.append(_plan_execution_item(step, "blocked", summary))
            run_trace.append({"step": "guard", "tool_name": tool_name, "status": "blocked", "summary": summary})
            return {
                "next_agent": "ops",
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, "ops"),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": True,
                "react_status": "guarded",
                "delegate": {},
                "final_answer": "",
                "needs_replan": False,
                "executed_count": executed_count,
            }

        export_allowed = bool(re.search(r"(导出|下载|word|docx|答案卷|答题卡)", query, flags=re.IGNORECASE))
        if policy == "managed_write" or (policy == "export_high_cost" and not export_allowed):
            step["status"] = "blocked"
            summary = f"{policy} requires explicit ops handling"
            plan_execution.append(_plan_execution_item(step, "blocked", summary))
            run_trace.append({"step": "guard", "tool_name": tool_name, "status": "blocked", "summary": summary})
            return {
                "next_agent": "ops",
                "task_plan": plan,
                "tool_plan": _legacy_plan_from_task_plan(plan, "ops"),
                "observations": observations,
                "trace": run_trace,
                "plan_execution": plan_execution,
                "guard_blocked": True,
                "react_status": "guarded",
                "delegate": {},
                "final_answer": "",
                "needs_replan": False,
                "executed_count": executed_count,
            }

        tools = _tool_map()
        tool = tools.get(tool_name)
        if tool is None:
            step["status"] = "failed"
            summary = "tool not registered"
            plan_execution.append(_plan_execution_item(step, "failed", summary))
            run_trace.append({"step": "action", "tool_name": tool_name, "status": "failed", "summary": summary})
            continue

        try:
            result = await tool.ainvoke(args)
            content = result if isinstance(result, str) else str(result)
            observations.append({"tool_name": tool_name, "args": args, "content": content[:2500]})
            step["status"] = "completed"
            summary = _observation_summary(content)
            plan_execution.append(_plan_execution_item(step, "completed", summary))
            run_trace.append({"step": "action", "tool_name": tool_name, "status": "success", "summary": f"observation_len={len(content)}"})
            run_trace.append({"step": "observation", "tool_name": tool_name, "status": "success", "summary": summary})
        except Exception as exc:
            step["status"] = "failed"
            summary = str(exc)[:300]
            plan_execution.append(_plan_execution_item(step, "failed", summary))
            run_trace.append({"step": "action", "tool_name": tool_name, "status": "failed", "summary": summary})

    needs_replan = bool(observations)
    status = "pass" if observations or not plan.get("steps") else "partial"
    return {
        "next_agent": safe_route,
        "task_plan": plan,
        "tool_plan": _legacy_plan_from_task_plan(plan, safe_route),
        "observations": observations,
        "trace": run_trace,
        "plan_execution": plan_execution,
        "guard_blocked": False,
        "react_status": status,
        "delegate": {},
        "final_answer": "",
        "needs_replan": needs_replan,
        "executed_count": executed_count,
    }


async def run_react_orchestrator(
    *,
    query: str,
    session_id: str,
    route_hint: str = "",
    recent_history: str = "",
    force_react: bool = False,
    max_steps: int = 6,
) -> Dict[str, Any]:
    """运行本轮 ReAct 编排：先生成 TaskPlan，再执行工具、返工或委派专家。"""
    route = str(route_hint or "").strip() or "rag"
    safe_max_steps = max(1, min(int(max_steps or 6), 6))

    # 兼容旧确定性快路径：尤其保留“搜索词缺失时先澄清”的行为。
    legacy_plan = build_tool_plan(query, session_id=session_id, route_hint=route, recent_history=recent_history)
    if legacy_plan.get("final_answer"):
        task_plan = normalize_task_plan(
            {"final_answer": legacy_plan.get("final_answer"), "stop_reason": "deterministic_clarify"},
            query,
            route,
        )
        trace = list(legacy_plan.get("trace") or []) + [{"step": "final", "status": "success", "summary": "clarification"}]
        return {
            "next_agent": str(legacy_plan.get("next_agent") or route),
            "tool_plan": legacy_plan,
            "task_plan": task_plan,
            "observations": [],
            "trace": trace,
            "plan_execution": [_plan_execution_item(task_plan["steps"][0], "completed", str(legacy_plan.get("final_answer") or ""))] if task_plan.get("steps") else [],
            "guard_blocked": False,
            "react_status": "pass",
            "delegate": {},
            "final_answer": str(legacy_plan.get("final_answer") or ""),
        }

    deterministic_task_plan = build_default_task_plan(query, session_id=session_id, route_hint=route, recent_history=recent_history)
    if legacy_plan.get("needs_tools"):
        deterministic_task_plan = normalize_task_plan(
            {
                "goal": deterministic_task_plan.get("goal") or query,
                "tool_calls": legacy_plan.get("tool_calls", []),
                "stop_reason": "deterministic_tool_plan",
            },
            query,
            route,
        )

    use_deterministic_task = (
        task_plan_has_steps(deterministic_task_plan)
        and (
            not force_react
            or bool(legacy_plan.get("needs_tools"))
            or _is_safety_task_plan(deterministic_task_plan)
            or _is_exam_format_task_plan(deterministic_task_plan)
        )
    )
    if use_deterministic_task:
        return await execute_task_plan(
            deterministic_task_plan,
            query=query,
            route=str(legacy_plan.get("next_agent") or deterministic_task_plan.get("route_hint") or route),
            session_id=session_id,
            max_steps=safe_max_steps,
        )

    if not should_use_react_planner(query, route, force_react=force_react):
        trace = [{"step": "intent", "status": "success", "summary": f"route_hint={route}"}]
        return {
            "next_agent": route,
            "tool_plan": legacy_plan,
            "task_plan": deterministic_task_plan,
            "observations": [],
            "trace": trace,
            "plan_execution": [],
            "guard_blocked": False,
            "react_status": "pass",
            "delegate": {},
            "final_answer": "",
        }

    observations: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = [{"step": "intent", "status": "success", "summary": f"route_hint={route}"}]
    plan_execution: List[Dict[str, Any]] = []
    last_task_plan: Dict[str, Any] = deterministic_task_plan
    executed_total = 0
    replan_count = 0

    while executed_total < safe_max_steps and replan_count <= 1:
        try:
            raw_plan = await _call_react_planner(
                query=query,
                session_id=session_id,
                route_hint=route,
                recent_history=recent_history,
                observations=observations,
            )
            task_plan = normalize_task_plan(raw_plan, query, route)
        except Exception as exc:
            failed_plan = failed_task_plan(query, route, str(exc))
            failure_item = {
                "step_id": "planner",
                "title": "生成任务计划",
                "action_type": "plan",
                "tool_name": "",
                "status": "failed",
                "summary": str(exc)[:300],
            }
            trace.append({"step": "plan", "status": "failed", "summary": str(exc)[:240]})
            return {
                "next_agent": route,
                "tool_plan": _legacy_plan_from_task_plan(failed_plan, route),
                "task_plan": failed_plan,
                "observations": observations,
                "trace": trace,
                "plan_execution": plan_execution + [failure_item],
                "guard_blocked": False,
                "react_status": "partial",
                "delegate": {},
                "final_answer": "",
            }

        last_task_plan = task_plan
        execution = await execute_task_plan(
            task_plan,
            query=query,
            route=route,
            session_id=session_id,
            max_steps=safe_max_steps - executed_total,
            trace=trace,
        )
        observations.extend(execution.get("observations", []))
        trace = list(execution.get("trace") or trace)
        plan_execution.extend(execution.get("plan_execution", []))
        executed_total += int(execution.get("executed_count") or 0)
        last_task_plan = execution.get("task_plan") if isinstance(execution.get("task_plan"), dict) else task_plan

        terminal = bool(execution.get("delegate")) or bool(execution.get("guard_blocked")) or bool(execution.get("final_answer"))
        if terminal:
            return {**execution, "observations": observations, "trace": trace, "plan_execution": plan_execution}

        if not execution.get("needs_replan") or replan_count >= 1:
            break
        replan_count += 1
        trace.append({"step": "plan", "status": "partial", "summary": "observation collected, replan once"})

    trace.append({"step": "final", "status": "partial", "summary": f"max_steps={safe_max_steps} or replan limit reached"})
    return {
        "next_agent": route,
        "tool_plan": _legacy_plan_from_task_plan(last_task_plan, route),
        "task_plan": last_task_plan,
        "observations": observations,
        "trace": trace,
        "plan_execution": plan_execution,
        "guard_blocked": False,
        "react_status": "partial",
        "delegate": {},
        "final_answer": "",
    }
