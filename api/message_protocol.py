import json
import re
import time
import ast
from typing import Any, Dict, List, Optional

from api.routers.exam_export import parse_exam_content


def _parse_quiz_content(content: str) -> List[Dict[str, Any]]:
    """将题目文本解析为结构化题目列表。"""
    questions: List[Dict[str, Any]] = []

    def parse_inline_options(text: str) -> tuple[str, List[str]]:
        match = re.match(r'^(.*?)(?=\s+[A-D][.、]\s*)', text)
        if not match:
            return text.strip(), []
        question = match.group(1).strip()
        option_part = text[len(match.group(0)):].strip()
        options = re.findall(r'[A-D][.、]\s*.*?(?=(?:\s+[A-D][.、]\s*)|$)', option_part, flags=re.DOTALL)
        return question, [opt.strip() for opt in options]

    has_new_format = bool(re.search(r'（选择题，|（填空题，|（判断题，|（简答题，', content))
    has_old_format = bool(re.search(r'【选择题】|【填空题】|【判断题】|【简答题】', content))
    has_inline_options = bool(re.search(r'[A-D][.、]\s*.+[A-D][.、]\s*.+', content, flags=re.DOTALL))
    if not has_new_format and not has_old_format and not has_inline_options:
        return questions

    question_blocks = re.split(r'(?=^\d+\.)', content, flags=re.MULTILINE)
    for block in question_blocks:
        trimmed = block.strip()
        if not trimmed or not re.match(r'^\d+\.', trimmed):
            continue

        lines = trimmed.splitlines()
        question_text = ""
        options: List[str] = []
        answer = ""
        explanation = ""
        score = ""
        difficulty = ""
        current_type = "选择题"
        in_explanation = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            meta_match = re.match(r'^\d+\.\s*（([^）]+)', stripped)
            if meta_match:
                meta = meta_match.group(1)
                if "选择题" in meta:
                    current_type = "选择题"
                elif "填空题" in meta:
                    current_type = "填空题"
                elif "判断题" in meta:
                    current_type = "判断题"
                elif "简答题" in meta:
                    current_type = "简答题"

                score_match = re.search(r'分值[：:]?\s*(\d+)分', meta)
                diff_match = re.search(r'难度[：:]?\s*([^，,]+)', meta)
                if score_match:
                    score = f"{score_match.group(1)}分"
                if diff_match:
                    difficulty = diff_match.group(1).strip()
                in_explanation = False
                continue

            if re.match(r'^[A-D][.、]', stripped):
                options.append(stripped)
                in_explanation = False
            elif stripped.startswith("答案：") or stripped.startswith("答案:"):
                answer = re.sub(r'^答案[：:]\s*', '', stripped)
                in_explanation = True
            elif stripped.startswith("解析：") or stripped.startswith("解析:"):
                explanation = re.sub(r'^解析[：:]\s*', '', stripped)
                in_explanation = True
            elif "【" in stripped and "】" in stripped:
                continue
            elif re.search(r'\s+[A-D][.、]\s*', stripped):
                parsed_question, parsed_options = parse_inline_options(stripped)
                if parsed_question:
                    question_text += (("\n" if question_text else "") + parsed_question)
                if parsed_options:
                    options.extend(parsed_options)
            elif in_explanation:
                explanation += (("\n" if explanation else "") + stripped)
            else:
                question_text += (("\n" if question_text else "") + stripped)

        if question_text:
            questions.append({
                "type": current_type,
                "question": question_text,
                "options": options or None,
                "answer": answer,
                "explanation": explanation,
                "score": score or None,
                "difficulty": difficulty or None,
            })

    if not questions and has_inline_options:
        text = re.sub(r'^\d+[.、]\s*', '', content.strip())
        question_text, options = parse_inline_options(text)
        if question_text and len(options) >= 2:
            questions.append({
                "type": "选择题",
                "question": question_text,
                "options": options,
                "answer": "",
                "explanation": "",
                "score": None,
                "difficulty": None,
            })

    return questions


def _infer_message_kind(route: str, answer: str) -> str:
    """基于最终内容兜底识别消息类型，避免路由误判导致前端降级为纯文本。"""
    if route == "exam":
        return "exam_paper"
    if route == "quiz":
        return "quiz_set"

    parsed_exam = parse_exam_content(answer)
    if parsed_exam.get("questions"):
        return "exam_paper"

    quiz_questions = _parse_quiz_content(answer)
    if quiz_questions:
        return "quiz_set"

    return "chat"


def _parse_serialized_quiz_questions(content: str) -> List[Dict[str, Any]]:
    """兼容 Python/JSON 列表字符串题目输出。"""
    if not content or not str(content).strip():
        return []
    raw = str(content).strip()
    parsed_obj: Any = None
    try:
        parsed_obj = json.loads(raw)
    except Exception:
        try:
            parsed_obj = ast.literal_eval(raw)
        except Exception:
            return []
    if isinstance(parsed_obj, dict):
        parsed_obj = [parsed_obj]
    if not isinstance(parsed_obj, list):
        return []

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed_obj, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or item.get("content") or "").strip()
        if not question:
            continue
        options: List[str] = []
        raw_options = item.get("options")
        if isinstance(raw_options, dict):
            for key in ["A", "B", "C", "D"]:
                val = raw_options.get(key)
                if val is not None and str(val).strip():
                    options.append(f"{key}. {str(val).strip()}")
        elif isinstance(raw_options, list):
            for idx, opt in enumerate(raw_options):
                opt_text = str(opt).strip()
                if not opt_text:
                    continue
                if re.match(r"^[A-D][.、]\s*", opt_text):
                    options.append(opt_text)
                else:
                    options.append(f"{chr(65 + idx)}. {opt_text}")
        out.append({
            "type": str(item.get("type") or "选择题"),
            "question": question,
            "options": options or None,
            "answer": str(item.get("answer") or "").strip(),
            "explanation": str(item.get("explanation") or item.get("analysis") or "").strip(),
            "score": f"{item.get('score')}分" if item.get("score") not in (None, "") else None,
            "difficulty": str(item.get("difficulty") or "").strip() or None,
        })
    return out


def build_assistant_message(
    route: str,
    answer: str,
    session_id: str,
    route_params: Dict[str, Any] | None = None,
    structured_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """构造统一的助手消息协议。"""
    route_params = route_params or {}
    orchestrator = route_params.get("orchestrator") if isinstance(route_params.get("orchestrator"), dict) else {}
    react_trace = orchestrator.get("trace") if isinstance(orchestrator.get("trace"), list) else []
    react_status = str(orchestrator.get("react_status") or "").strip()
    task_plan = orchestrator.get("task_plan") if isinstance(orchestrator.get("task_plan"), dict) else {}
    plan_execution = orchestrator.get("plan_execution") if isinstance(orchestrator.get("plan_execution"), list) else []

    if structured_result:
        content = structured_result.get("text", answer) or answer
        kind = structured_result.get("kind") or _infer_message_kind(route, content)
        render_mode = structured_result.get("render_mode")
        if not render_mode:
            render_mode = (
                "interactive_cards" if kind == "quiz_set"
                else "exam_canvas" if kind == "exam_paper"
                else "markdown"
            )
        payload = structured_result.get("payload") or {}
        if react_trace and isinstance(payload, dict):
            payload = {**payload, "agent_trace": payload.get("agent_trace") or react_trace}
        if isinstance(payload, dict) and task_plan:
            payload = {**payload, "task_plan": payload.get("task_plan") or task_plan}
        if isinstance(payload, dict) and plan_execution:
            payload = {**payload, "plan_execution": payload.get("plan_execution") or plan_execution}
        if kind == "quiz_set":
            if not isinstance(payload, dict):
                payload = {}
            questions = payload.get("questions") if isinstance(payload.get("questions"), list) else []
            if not questions:
                parsed_questions = _parse_quiz_content(content)
                if not parsed_questions:
                    parsed_questions = _parse_serialized_quiz_questions(content)
                if parsed_questions:
                    payload = {
                        **payload,
                        "title": payload.get("title") or (route_params.get("topic") if isinstance(route_params, dict) else "") or "练习题",
                        "show_answers_default": bool(payload.get("show_answers_default", False)),
                        "show_analysis_default": bool(payload.get("show_analysis_default", False)),
                        "questions": parsed_questions,
                    }
        extra_meta = structured_result.get("meta") if isinstance(structured_result.get("meta"), dict) else {}
        merged_meta = {
            "route": route,
            "session_id": session_id,
            "generated_at": int(time.time()),
            "structured": True,
        }
        if react_trace:
            merged_meta.update(
                {
                    "agent_mode": "react",
                    "react_status": react_status or "pass",
                    "agent_trace": extra_meta.get("agent_trace") or react_trace,
                }
            )
        merged_meta.update(extra_meta)
        return {
            "kind": kind,
            "render_mode": render_mode,
            "content": content,
            "payload": payload,
            "meta": merged_meta,
        }

    kind = _infer_message_kind(route, answer)
    render_mode = "markdown"
    payload: Dict[str, Any] = {}

    if kind == "quiz_set":
        render_mode = "interactive_cards"
        parsed_questions = _parse_quiz_content(answer)
        if not parsed_questions:
            parsed_questions = _parse_serialized_quiz_questions(answer)
        payload = {
            "title": route_params.get("topic") or "练习题",
            "show_answers_default": False,
            "show_analysis_default": False,
            "questions": parsed_questions,
        }
    elif kind == "exam_paper":
        kind = "exam_paper"
        render_mode = "exam_canvas"
        payload = _build_exam_payload(answer)

    if task_plan:
        payload = {**payload, "task_plan": task_plan}
    if plan_execution:
        payload = {**payload, "plan_execution": plan_execution}

    return {
        "kind": kind,
        "render_mode": render_mode,
        "content": answer,
        "payload": {**payload, "agent_trace": react_trace} if react_trace else payload,
        "meta": {
            "route": route,
            "session_id": session_id,
            "generated_at": int(time.time()),
            **(
                {
                    "agent_mode": "react",
                    "react_status": react_status or "pass",
                    "agent_trace": react_trace,
                }
                if react_trace
                else {}
            ),
        },
    }


def _build_exam_payload(content: str) -> Dict[str, Any]:
    parsed = parse_exam_content(content)
    questions = []
    total_score = 0
    question_types: Dict[str, Dict[str, int]] = {}

    for q in parsed.get("questions", []):
        difficulty = q.get("difficulty") or "中等"
        score = int(q.get("score") or 0)
        if score <= 0:
            score = 2 if q.get("type") in {"选择题", "填空题", "判断题"} else 10

        stem, options = _split_stem_and_options(q.get("content") or "")

        question = {
            "number": int(q.get("number") or len(questions) + 1),
            "type": q.get("type") or "选择题",
            "content": stem,
            "options": options or None,
            "answer": q.get("answer") or "",
            "analysis": q.get("analysis") or "",
            "score": score,
            "difficulty": difficulty,
            "knowledge_point": q.get("knowledge_point"),
        }
        questions.append(question)
        total_score += score

        qtype = question["type"]
        if qtype not in question_types:
            question_types[qtype] = {"count": 0, "total_score": 0}
        question_types[qtype]["count"] += 1
        question_types[qtype]["total_score"] += score

    return {
        "title": parsed.get("title") or "完整试卷",
        "subtitle": parsed.get("subtitle") or "",
        "show_answers_default": False,
        "show_analysis_default": False,
        "exam_data": {
            "title": parsed.get("title") or "完整试卷",
            "subtitle": parsed.get("subtitle") or "",
            "total_score": total_score,
            "total_questions": len(questions),
            "question_types": question_types,
            "questions": questions,
        },
    }


def _split_stem_and_options(raw_content: str) -> tuple[str, List[str]]:
    content = re.sub(r'^\d+[.、]\s*', '', raw_content.strip())
    if not content:
        return "", []

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    stem_lines: List[str] = []
    options: List[str] = []

    for line in lines:
        if re.match(r'^[A-D][.、]\s*', line):
            options.append(line)
            continue

        if re.search(r'\b[A-D][.、]\s*', line):
            parts = re.split(r'(?=\b[A-D][.、]\s*)', line)
            question_part = parts[0].strip()
            if question_part:
                stem_lines.append(question_part)
            for part in parts[1:]:
                part = part.strip()
                if re.match(r'^[A-D][.、]\s*', part):
                    options.append(part)
            continue

        stem_lines.append(line)

    return "\n".join(stem_lines).strip(), options
