import ast
import json
import re
from typing import Any, List

from agent.multi_agent.quiz_normalization import (
    format_score,
    normalize_int_field,
    normalize_text_field,
    sanitize_question_text,
    sanitize_user_visible_text,
)


def build_quiz_text_from_questions(questions: List[dict]) -> str:
    lines: List[str] = []
    for index, question in enumerate(questions, start=1):
        qtype = question.get("type") or "选择题"
        score = question.get("score") or "5分"
        difficulty = question.get("difficulty") or "中等"
        lines.append(f"{index}.（{qtype}，分值：{score}，难度：{difficulty}）")
        lines.append(str(question.get("question", "")).strip())
        for option in question.get("options") or []:
            lines.append(str(option).strip())
        if question.get("answer"):
            lines.append(f"答案：{question['answer']}")
        if question.get("explanation"):
            lines.append(f"解析：{question['explanation']}")
        lines.append("")
    return "\n".join(lines).strip()


def build_quiz_payload(topic: str, questions: List[dict]) -> dict:
    return {
        "title": topic or "练习题",
        "show_answers_default": False,
        "show_analysis_default": False,
        "questions": questions,
    }


def parse_questions_from_serialized_quiz(text: str) -> List[dict]:
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
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

    normalized: List[dict] = []
    for idx, item in enumerate(parsed_obj, start=1):
        if not isinstance(item, dict):
            continue
        question = normalize_text_field(item.get("question") or item.get("content"), "")
        if not question:
            continue
        raw_options = item.get("options")
        options: List[str] = []
        if isinstance(raw_options, dict):
            for key in ["A", "B", "C", "D"]:
                val = raw_options.get(key)
                if val is not None and str(val).strip():
                    options.append(f"{key}. {str(val).strip()}")
        elif isinstance(raw_options, list):
            for i, opt in enumerate(raw_options):
                opt_text = str(opt).strip()
                if not opt_text:
                    continue
                if re.match(r"^[A-D][.、]\s*", opt_text):
                    options.append(opt_text)
                else:
                    options.append(f"{chr(65 + i)}. {opt_text}")

        normalized.append({
            "id": normalize_int_field(item.get("id") or item.get("question_number"), idx),
            "type": normalize_text_field(item.get("type"), "选择题"),
            "question": sanitize_user_visible_text(sanitize_question_text(question)),
            "options": options or None,
            "answer": sanitize_user_visible_text(normalize_text_field(item.get("answer"), "")),
            "explanation": sanitize_user_visible_text(normalize_text_field(item.get("explanation") or item.get("analysis"), "")),
            "score": format_score(item.get("score")) or "5分",
            "difficulty": normalize_text_field(item.get("difficulty"), "中等"),
        })
    return normalized


def questions_from_quiz_text(text: str, topic: str) -> dict:
    from api.message_protocol import _parse_quiz_content

    questions = _parse_quiz_content(text)
    if not questions:
        questions = parse_questions_from_serialized_quiz(text)
    return build_quiz_payload(topic, questions)

