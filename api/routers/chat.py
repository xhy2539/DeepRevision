import json
import asyncio
import re
import time
import hashlib
import shlex
from contextlib import suppress
from functools import lru_cache
from fastapi import APIRouter, Body, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Set

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent.tools.agent_tools import clear_rag_cache, get_rag_service, tools as registered_tools
from agent.multi_agent.supervisor import supervisor_workflow, supervisor_node, CHITCHAT_PROMPT
from api.message_protocol import build_assistant_message
from utils.logger_handler import logger, update_token_stats, get_system_stats
from utils.memory_service import memory_manager
from utils.session_context import current_session_id
from utils.rag_metrics import rag_get_metrics_snapshot, rag_inc
from utils.config_handler import chroma_conf
from utils.user_profile_service import (
    normalize_username,
    get_user_profile as get_user_profile_data,
    update_user_profile as update_user_profile_data,
    build_persona_instructions,
)
from rag.vector_store import VectorStoreService
from model.factory import light_chat_model, backup_light_chat_model, chat_model

router = APIRouter()
_VALID_SESSION_ID = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$')
_TOOL_CMD_PREFIX = "/tool"
_REGISTERED_TOOL_MAP: Dict[str, Any] = {
    getattr(tool_item, "name", ""): tool_item
    for tool_item in registered_tools
    if getattr(tool_item, "name", "")
}

# 轻量运行指标（进程内）
RUNTIME_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "failed_requests": 0,
    "routes": {"rag": 0, "quiz": 0, "exam": 0, "ops": 0, "planner": 0, "history": 0, "chitchat": 0},
    "quiz_timeout_count": 0,
    "fallback_quality_guard_count": 0,
    "tool_call_count_rag": 0,
    "tool_call_count_web": 0,
    "exam_total": 0,
    "exam_success": 0,
    "revise_timeout_count": 0,
    "degraded_delivery_count": 0,
    "exam_latency_samples_ms": [],
    "quiz_latency_samples_ms": [],
    "exam_stage_timeout_count": 0,
    "exam_stage_retry_count": 0,
    "exam_stage_fail_count": 0,
    "exam_stage_latency_ms": [],
    "exam_choice_missing_options_fix_count": 0,
    "exam_score_rebalance_count": 0,
    "exam_duplicate_rewrite_count": 0,
    "exam_prompt_contract_violation_count": 0,
    "term_style_violation_count": 0,
    "missing_model_retry_count": 0,
    "hard_fail_count": 0,
    "critic_timeout_count": 0,
    "last_error": "",
    "stream_requests_total": 0,
    "stream_success_total": 0,
    "stream_error_total": 0,
    "stream_client_cancel_total": 0,
    "stream_timeout_total": 0,
    "stream_heartbeat_sent_total": 0,
    "stream_ttft_samples_ms": [],
    "stream_first_delta_samples_ms": [],
    "stream_duration_samples_ms": [],
    "stream_termination_reason_counts": {
        "success": 0,
        "error": 0,
        "timeout": 0,
        "client_cancelled": 0,
        "cancelled": 0,
    },
    "stream_v2_requests_total": 0,
    "stream_v2_activated_total": 0,
    "stream_event_counts_by_type": {},
    "mastery_rows_total": 0,
}


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"


class PracticeRecordItem(BaseModel):
    question_id: Optional[str] = None
    question_number: Optional[int] = None
    question_content: str
    knowledge_point: Optional[str] = None
    user_answer: str
    correct_answer: str
    is_correct: bool
    wrong_reason: Optional[str] = None


class PracticeSubmitRequest(BaseModel):
    session_id: str = "default"
    records: List[PracticeRecordItem]


class SimilarBatchItem(BaseModel):
    question_number: int
    question_content: str
    knowledge_point: Optional[str] = None


class SimilarBatchRequest(BaseModel):
    session_id: str = "default"
    wrong_questions: List[SimilarBatchItem]
    limit: int = 3


class PracticeHistoryDeleteRequest(BaseModel):
    session_id: str = "default"
    record_id: int


class PracticeBackfillRequest(BaseModel):
    session_id: str = "default"
    limit: int = 2000
    dry_run: bool = True


class UserProfileUpdateRequest(BaseModel):
    username: str
    profile: Dict[str, Any]


def _clean_answer(answer: str) -> str:
    """清洗回答，移除思考内容和重复"""
    if not answer:
        return ""

    # 阶段0：首先用正则彻底移除所有 think 标签内容（支持嵌套）
    # 循环移除直到没有 think 标签为止
    while '<think>' in answer:
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer)
    answer = answer.strip()

    if not answer:
        return ""

    # 阶段1：去除思考内容（基于行的启发式处理，作为备份）
    lines = answer.split('\n')
    valid_lines = []
    skip_mode = False

    # 思考内容开始的标记模式（多种变体）
    thinking_patterns = [
        r'^用户[说用]',
        r'^The user',
        r'^我认为',
        r'^我应该',
        r'^我需要',
        r'^让我',
        r'^首先',
        r'^其次',
        r'^然后',
        r'^最后',
        r'^考虑',
        r'^分析',
        r'^推理',
        r'^了["""'']',
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 检测是否是思考内容开始
        is_thinking = any(re.match(p, stripped) for p in thinking_patterns)

        # 增强检测：以"了"开头、后面紧跟引号或括号、长度较短
        # 例如：了"你好"，这是一个简单的问候
        if not is_thinking and len(stripped) >= 2:
            # 检查是否以"了"开头，后面跟引号类字符
            if stripped.startswith('了') and len(stripped) >= 4:
                next_char = stripped[1]
                if next_char in '"\'""''【】（）：:':
                    is_thinking = True

        # 注意：不要使用“多逗号/短句”这类弱特征判定思考内容，
        # 否则会误删模型正常回答正文（尤其是闲聊或解释性段落）。

        if is_thinking and not skip_mode:
            # 刚开始进入思考，跳过当前行
            skip_mode = True
            continue
        elif is_thinking and skip_mode:
            # 继续跳过
            continue

        # 处于思考模式，寻找正式回答开始的信号
        if skip_mode:
            # 正式回答以问候语、带emoji的句子、或 Markdown 标题/列表开始
            if len(stripped) > 2 and (
                stripped.startswith('#') or
                stripped.startswith('你好') or
                stripped.startswith('嗨') or
                stripped.startswith('Hi') or
                stripped.startswith('很高兴') or
                stripped.startswith('**') or  # Markdown 标题
                stripped.startswith('- ') or   # 列表项
                stripped.startswith('* ') or
                stripped.startswith('1. ') or
                stripped.startswith('2. ') or
                stripped.startswith('3. ') or
                stripped.startswith('• ') or
                stripped.startswith('1.（') or  # 题目编号（新格式）
                stripped.startswith('2.（') or
                stripped.startswith('3.（') or
                stripped.startswith('【') or  # 旧格式题目标题
                stripped.startswith('### ') or
                re.match(r'^[一二三四五六七八九十]+[、.]\s*', stripped) or
                re.match(r'^##\s*[一二三四五六七八九十0-9]+[、.]?', stripped)
            ):
                skip_mode = False
                valid_lines.append(line)
            continue
        else:
            valid_lines.append(line)

    result = '\n'.join(valid_lines).strip()

    # 如果处理后结果太短，可能是纯思考内容
    if not result or len(result) < 10:
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            return non_empty[-1].strip()
        return answer.strip()

    # 阶段2：去除连续重复的段落
    result = _deduplicate_response(result)

    return result


def _deduplicate_response(text: str) -> str:
    """去除连续重复的段落"""
    if not text or len(text) < 20:
        return text

    lines = text.split('\n')
    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
        else:
            current_para.append(line)

    if current_para:
        paragraphs.append('\n'.join(current_para))

    if len(paragraphs) < 2:
        return text

    # 方法1：去除完全相同的连续段落
    result_paragraphs = []
    for para in paragraphs:
        if not result_paragraphs or para != result_paragraphs[-1]:
            result_paragraphs.append(para)

    if len(result_paragraphs) < len(paragraphs):
        return '\n\n'.join(result_paragraphs)

    return '\n'.join(lines).strip()


def _safe_inc(metric_key: str, delta: int = 1):
    """安全累加运行指标；任何异常都吞掉，避免影响主流程。"""
    try:
        RUNTIME_METRICS[metric_key] = int(RUNTIME_METRICS.get(metric_key, 0)) + delta
    except Exception:
        pass


def _record_exam_latency(latency_ms: int):
    """记录出卷端到端耗时样本（保留最近 200 条）。"""
    try:
        samples = RUNTIME_METRICS.setdefault("exam_latency_samples_ms", [])
        samples.append(int(latency_ms))
        # 保留最近 200 条样本
        if len(samples) > 200:
            del samples[:-200]
    except Exception:
        pass


def _record_quiz_latency(latency_ms: int):
    """记录练习/小测端到端耗时样本（保留最近 200 条）。"""
    try:
        samples = RUNTIME_METRICS.setdefault("quiz_latency_samples_ms", [])
        samples.append(int(latency_ms))
        if len(samples) > 200:
            del samples[:-200]
    except Exception:
        pass


def _record_exam_stage_latency(latency_ms: int):
    """记录分阶段出卷的阶段耗时样本（保留最近 500 条）。"""
    try:
        samples = RUNTIME_METRICS.setdefault("exam_stage_latency_ms", [])
        samples.append(int(latency_ms))
        if len(samples) > 500:
            del samples[:-500]
    except Exception:
        pass


def _record_runtime_sample(metric_key: str, value: int, max_len: int = 500):
    """通用样本记录器，用于 TTFT/首个 delta/流时长等序列指标。"""
    try:
        samples = RUNTIME_METRICS.setdefault(metric_key, [])
        samples.append(int(value))
        if len(samples) > max_len:
            del samples[:-max_len]
    except Exception:
        pass


def _record_stream_termination(reason: str):
    """按终止原因统计流式结束次数（success/error/timeout/...）。"""
    try:
        counts = RUNTIME_METRICS.setdefault("stream_termination_reason_counts", {})
        counts[reason] = int(counts.get(reason, 0)) + 1
    except Exception:
        pass


def _calc_p95(values: List[int]) -> int:
    """对样本序列计算近似 p95（空序列返回 0）。"""
    if not values:
        return 0
    arr = sorted(int(v) for v in values)
    idx = max(0, min(len(arr) - 1, int(len(arr) * 0.95) - 1))
    return arr[idx]


def _validate_session_id(session_id: str) -> str:
    """统一校验 session_id，防止非法字符进入存储或路径逻辑。"""
    if not _VALID_SESSION_ID.match(session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")
    return session_id


def _tool_accepts_argument(tool_obj: Any, arg_name: str) -> bool:
    """检查工具签名是否包含指定参数。"""
    schema = getattr(tool_obj, "args_schema", None)
    if schema is None:
        return False
    model_fields = getattr(schema, "model_fields", None)
    if isinstance(model_fields, dict):
        return arg_name in model_fields
    legacy_fields = getattr(schema, "__fields__", None)
    if isinstance(legacy_fields, dict):
        return arg_name in legacy_fields
    return False


def _coerce_tool_value(raw_value: str) -> Any:
    """把 key=value 中的字符串值做基础类型还原。"""
    text = str(raw_value or "").strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_tool_kv_args(raw_text: str) -> Dict[str, Any]:
    """解析 k=v 形式参数，支持引号。"""
    args: Dict[str, Any] = {}
    tokens = shlex.split(raw_text or "")
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"参数格式错误: {token}（应为 key=value）")
        key, value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"参数名不能为空: {token}")
        args[key] = _coerce_tool_value(value)
    return args


def _parse_explicit_tool_command(text: str) -> Optional[Dict[str, Any]]:
    """
    解析显式工具调用:
    - /tool tool_name {"k":"v"}
    - /tool tool_name k=v k2=v2
    """
    raw = str(text or "").strip()
    if not raw.lower().startswith(_TOOL_CMD_PREFIX):
        return None

    payload = raw[len(_TOOL_CMD_PREFIX):].strip()
    if not payload:
        return {"error": "缺少工具名。示例: /tool get_practice_history {\"limit\":20}"}

    if " " in payload:
        tool_name, raw_args = payload.split(" ", 1)
        raw_args = raw_args.strip()
    else:
        tool_name, raw_args = payload, ""

    if not tool_name:
        return {"error": "工具名不能为空。"}

    args: Dict[str, Any] = {}
    if raw_args:
        if raw_args.startswith("{"):
            try:
                parsed = json.loads(raw_args)
            except Exception as e:
                return {"error": f"JSON 参数解析失败: {e}"}
            if not isinstance(parsed, dict):
                return {"error": "JSON 参数必须是对象，例如 {\"limit\":20}"}
            args = parsed
        else:
            try:
                args = _parse_tool_kv_args(raw_args)
            except ValueError as e:
                return {"error": str(e)}

    return {"tool_name": tool_name.strip(), "args": args}


def _available_tool_names_preview(max_items: int = 18) -> str:
    """返回可用工具名预览文本。"""
    names = sorted(_REGISTERED_TOOL_MAP.keys())
    if len(names) <= max_items:
        return ", ".join(names)
    return ", ".join(names[:max_items]) + f" ...（共 {len(names)} 个）"


async def _execute_registered_tool(tool_name: str, args: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """执行已注册工具并返回统一结果。"""
    tool_obj = _REGISTERED_TOOL_MAP.get(tool_name)
    if tool_obj is None:
        return {
            "ok": False,
            "text": (
                f"未知工具: {tool_name}\n"
                f"可用工具: {_available_tool_names_preview()}"
            ),
        }

    tool_args = dict(args or {})
    # 常见管理工具允许 session 透传，若调用方未给则自动补当前会话。
    if "session_id" not in tool_args and _tool_accepts_argument(tool_obj, "session_id"):
        tool_args["session_id"] = session_id

    try:
        result = await tool_obj.ainvoke(tool_args)
        if isinstance(result, (dict, list)):
            result_text = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            result_text = str(result)
        return {
            "ok": True,
            "text": result_text,
        }
    except Exception as e:
        return {
            "ok": False,
            "text": f"工具执行失败: {e}",
        }


def _extract_kp_rule(raw_kp: str, question_content: str) -> str:
    """基于规则提取考点短语，作为题库统计与检索的统一 key。"""
    s = str(raw_kp or "").strip() or str(question_content or "").strip()
    if not s:
        return "未标注"
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"[*_#>\[\]\(\)✅❌⚠️•·]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    patterns = [
        r"下列关于(.{2,24}?)的",
        r"关于(.{2,24}?)的",
        r"在(.{2,24}?)中",
        r"(.{2,24}?)机制",
        r"(.{2,24}?)管理",
        r"(.{2,24}?)调度",
        r"(.{2,24}?)系统",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            cand = str(m.group(1) or "").strip("：:，,。.;；!?！？ ")
            if 2 <= len(cand) <= 24:
                s = cand
                break

    s = re.split(r"[；;。!?！？\n\r]", s)[0].strip()
    s = re.sub(r"^(下列|以下|请|试|简述|说明|判断|选择|哪个|哪一项|当|在)\s*", "", s).strip()
    s = s.strip("：:，,。.;；!?！？ ")
    if len(s) > 24:
        s = s[:24].rstrip("：:，,。.;；!?！？ ")
    return s or "未标注"


@lru_cache(maxsize=256)
def _kp_polish_prompt_template() -> PromptTemplate:
    """考点润色提示模板（缓存后复用，减少重复构建开销）。"""
    return PromptTemplate.from_template(
        "你是课程考点提炼器。请把输入内容提炼为一个“名词性考点短语”，用于题库检索。"
        "\n要求："
        "\n1) 只输出一个短语，不要句子，不要解释。"
        "\n2) 长度 4-16 字优先。"
        "\n3) 禁止输出“下列关于/哪一项/请说明/如何”等问句模板词。"
        "\n4) 如果输入本身已有合适考点，做轻微润色即可。"
        "\n输入考点: {raw_kp}"
        "\n输入题干: {question}"
    )


async def _polish_knowledge_point(raw_kp: str, question_content: str) -> str:
    """
    先规则提取，再按需调用轻量模型润色考点。
    仅在规则结果不理想时触发 LLM，控制提交练习记录时的额外延迟。
    """
    base = _extract_kp_rule(raw_kp, question_content)
    # 规则已较好时直接返回，避免额外延迟
    if 4 <= len(base) <= 16 and not re.search(r"(下列|以下|哪一项|请|如何|是否)", base):
        return base

    model = light_chat_model or backup_light_chat_model or chat_model
    if model is None:
        return base

    try:
        chain = _kp_polish_prompt_template() | model | StrOutputParser()
        result = await asyncio.wait_for(
            chain.ainvoke({"raw_kp": raw_kp or "", "question": question_content or ""}),
            timeout=3.0,
        )
        polished = _extract_kp_rule(result, question_content)
        return polished or base
    except Exception:
        return base


def _format_sse(payload: Any, event_name: Optional[str] = None) -> str:
    """统一 SSE 输出格式，兼容 event + data 协议字段。"""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if event_name:
        return f"event: {event_name}\ndata: {body}\n\n"
    return f"data: {body}\n\n"


def _parse_stream_v2_routes(raw_value: Any) -> Set[str]:
    """解析 stream_v2 路由配置，支持 list 与逗号分隔字符串。"""
    default_routes = {"rag", "chitchat"}
    if isinstance(raw_value, list):
        routes = {str(item).strip().lower() for item in raw_value if str(item).strip()}
        return routes or default_routes
    if isinstance(raw_value, str):
        parts = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
        return set(parts) or default_routes
    return default_routes


def _stable_bucket_100(session_id: str) -> int:
    """按 session_id 稳定映射到 0-99 桶，用于灰度开关。"""
    sid = str(session_id or "default")
    digest = hashlib.md5(sid.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _stream_v2_config() -> Dict[str, Any]:
    """读取 stream_v2 灰度配置。"""
    enabled = bool(chroma_conf.get("stream_v2_enabled", False))
    routes = _parse_stream_v2_routes(chroma_conf.get("stream_v2_routes", ["rag", "chitchat"]))
    rollout_percent = max(0, min(100, int(chroma_conf.get("stream_v2_rollout_percent", 0))))
    emit_sections = bool(chroma_conf.get("stream_v2_emit_sections", True))
    legacy_delta_mirror = bool(chroma_conf.get("stream_v2_legacy_delta_mirror", True))
    return {
        "enabled": enabled,
        "routes": routes,
        "rollout_percent": rollout_percent,
        "emit_sections": emit_sections,
        "legacy_delta_mirror": legacy_delta_mirror,
    }


def _should_attempt_stream_v2(session_id: str, cfg: Dict[str, Any]) -> bool:
    """是否命中 stream_v2 灰度。"""
    if not bool(cfg.get("enabled", False)):
        return False
    bucket = _stable_bucket_100(session_id)
    return bucket < int(cfg.get("rollout_percent", 0))


def _record_stream_event(event_name: str):
    """统计流式事件类型分布，用于诊断首包/断流。"""
    try:
        counts = RUNTIME_METRICS.setdefault("stream_event_counts_by_type", {})
        counts[event_name] = int(counts.get(event_name, 0)) + 1
    except Exception:
        pass


def _build_v2_event(
    *,
    event: str,
    message_id: str,
    seq: int,
    route: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """构建 stream_v2 统一事件结构。"""
    return {
        "event": event,
        "message_id": message_id,
        "seq": seq,
        "ts": int(time.time() * 1000),
        "route": route,
        "payload": payload,
    }


def _create_stream_sections() -> Dict[str, Any]:
    """创建统一的分区结构，避免前后端字段漂移。"""
    return {"reasoning": "", "actions": [], "citations": [], "progress": [], "tool_calls": []}


def _ops_trace_to_stream_sections(ops_trace: Any) -> Dict[str, Any]:
    """把 ops_trace 映射为可展示的流式分区（动作 + 工具调用）。"""
    sections = _create_stream_sections()
    if not isinstance(ops_trace, list):
        return sections

    for idx, item in enumerate(ops_trace, start=1):
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool") or "").strip()
        guard_blocked = bool(item.get("guard_blocked"))
        has_observation = bool(str(item.get("observation") or "").strip())
        has_error = bool(str(item.get("error") or "").strip())

        if tool_name:
            status = "running"
            if guard_blocked:
                status = "blocked"
            elif has_observation:
                status = "completed"
            elif has_error:
                status = "failed"
            action_item = {
                "name": tool_name,
                "status": status,
                "detail": str(item.get("reason") or item.get("error") or "").strip(),
            }
            sections["actions"].append(action_item)
            sections["tool_calls"].append(
                {
                    "tool_name": tool_name,
                    "status": status,
                    "result_summary": str(item.get("observation") or item.get("reason") or item.get("error") or "").strip()[:160],
                    "source": "ops_react",
                }
            )
            continue

        if guard_blocked:
            sections["progress"].append(str(item.get("reason") or "高风险操作已被安全闸拦截"))
        elif has_error:
            sections["progress"].append(str(item.get("error") or f"步骤 {idx} 执行失败"))

    return sections


def _format_recent_history_for_prompt(chat_history: List[Any], max_chars: int = 600) -> str:
    """将近期对话压缩成 prompt 可用文本，避免上下文过长。"""
    if not chat_history:
        return "（无近期对话）"
    lines: List[str] = []
    used = 0
    for msg in reversed(chat_history):
        role = "学生" if isinstance(msg, HumanMessage) else "助手"
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}"
        if used + len(line) > max_chars:
            remain = max_chars - used - len(f"{role}: ")
            if remain > 32:
                lines.insert(0, f"{role}: {content[:remain]}...")
            break
        lines.insert(0, line)
        used += len(line)
        if used >= max_chars:
            break
    return "\n".join(lines) if lines else "（无近期对话）"


def _extract_citations_from_context(context: str, limit: int = 6) -> List[Dict[str, Any]]:
    """
    从 retrieve_context 结果中提取可展示引用（best-effort）。
    仅用于前端分区展示，不参与业务判定。
    """
    text = str(context or "")
    if not text:
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for match in re.finditer(
        r"\[参考资料(\d+)\]:参考资料:(.*?)\|参考元数据:(\{.*?\})(?:\n|$)",
        text,
        flags=re.DOTALL,
    ):
        idx = int(match.group(1))
        snippet = re.sub(r"\s+", " ", str(match.group(2) or "")).strip()
        snippet = snippet[:120]
        metadata_text = str(match.group(3) or "").strip()
        source = f"参考资料{idx}"
        try:
            parsed_meta = json.loads(metadata_text.replace("'", '"'))
            if isinstance(parsed_meta, dict):
                source = str(parsed_meta.get("source_filename") or parsed_meta.get("source") or source)
        except Exception:
            pass
        key = f"{source}|{snippet}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "index": idx,
                "source": source,
                "quote": snippet,
            }
        )
        if len(out) >= limit:
            break
    return out


def _merge_persona_into_context(memory_context: str, persona_text: str) -> str:
    """把用户级人格指令拼入上下文，供所有路由共享。"""
    persona = str(persona_text or "").strip()
    context = str(memory_context or "").strip()
    if not persona:
        return context
    if context:
        return f"【用户个性化偏好】\n{persona}\n\n{context}"
    return f"【用户个性化偏好】\n{persona}"


class _StreamV2Handled(Exception):
    """内部控制流异常：标记 stream_v2 已完成并跳过 legacy 主流程。"""


@router.post("/stream")
async def chat_stream_endpoint(request: Request):
    """
    流式对话接口：Supervisor 多 Agent 路由 + 记忆组装与入库。
    """
    logger.info("=" * 50)
    logger.info(f"收到对话请求")
    body = await request.json()
    session_id = body.get("session_id", "default")
    username = normalize_username(body.get("username", ""))
    user_profile_payload = get_user_profile_data(username)
    user_profile = user_profile_payload.get("profile", {})
    persona_text = build_persona_instructions(user_profile)
    assistant_name = str(user_profile.get("assistant_name") or "DeepRevision")
    query = body.get("query", "")
    exam_stage_plan = bool(body.get("exam_stage_plan", True))
    exam_rerun_stage = body.get("exam_rerun_stage")
    exam_partial_questions = body.get("exam_partial_questions") if isinstance(body.get("exam_partial_questions"), list) else []
    exam_fast_mode = bool(body.get("exam_fast_mode", True))  # True=快速路径(格式检查), False=完整路径(LLM Critique)
    # 兼容旧前端“full模式”只传 exam_fast_mode=false 的场景：quiz 也强制走 LLM Critic
    quiz_force_llm_critic = bool(body.get("quiz_force_llm_critic", not exam_fast_mode))
    logger.info(f"session_id={session_id}, username={username}, query={query[:50]}...")

    # 安全校验
    _validate_session_id(session_id)

    query = query[:2000]
    current_session_id.set(session_id)

    # 获取历史
    memory_manager._init_session(session_id)
    recent_records = memory_manager.store[session_id]["recent"]
    chat_history = []
    for msg in recent_records:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            meta = msg.get("meta") if isinstance(msg.get("meta"), dict) else {}
            if bool(meta.get("cancelled")):
                # 用户中途取消的半截回答不参与后续上下文，避免污染检索与生成。
                continue
            chat_history.append(AIMessage(content=msg["content"]))

    graph_context = _merge_persona_into_context(memory_manager.get_memory_context(session_id), persona_text)

    def _attach_user_profile_meta(message_obj: Dict[str, Any]) -> Dict[str, Any]:
        """把用户级人格标识注入消息 meta，便于前端展示助手名称。"""
        meta = message_obj.get("meta") if isinstance(message_obj.get("meta"), dict) else {}
        message_obj["meta"] = {
            **meta,
            "assistant_name": assistant_name,
            "username": username,
        }
        return message_obj

    async def event_stream():
        """
        SSE 主循环：
        1) 启动 supervisor 工作流并定期发 heartbeat
        2) 将工作流输出转成 start/delta/complete/done 事件
        3) 记录流式分段指标并按终止状态落库
        """
        _safe_inc("total_requests")
        _safe_inc("stream_requests_total")
        stream_v2_cfg = _stream_v2_config()
        attempt_stream_v2 = _should_attempt_stream_v2(session_id, stream_v2_cfg)
        if attempt_stream_v2:
            _safe_inc("stream_v2_requests_total")
        initial_state = {
            "input": query,
            "chat_history": chat_history,
            "memory_context": graph_context,
            "session_id": session_id,
            "exam_stage_plan": exam_stage_plan,
            "exam_rerun_stage": exam_rerun_stage,
            "exam_partial_questions": exam_partial_questions,
            "exam_fast_mode": exam_fast_mode,
            "quiz_force_llm_critic": quiz_force_llm_critic,
            "route": "",
            "route_reason": "",
            "route_params": {},
            "supervisor_precomputed": False,
            "subagent_result": "",
            "final_answer": "",
        }

        logger.info("开始执行 Supervisor 工作流...")
        stream_start = time.time()
        answer = ""  # 初始化 answer 变量
        sent_content = ""  # 仅累积已真正发送给前端的正文片段
        error_text = ""
        message = None
        client_disconnected = False
        termination_reason = "success"
        stream_ttft_ms: Optional[int] = None
        stream_first_delta_ms: Optional[int] = None
        workflow_task: Optional[asyncio.Task] = None
        route_task: Optional[asyncio.Task] = None
        stream_v2_message_id = f"{session_id}-{int(stream_start * 1000)}"
        stream_v2_seq = 0
        stream_v2_route = "pending"
        stream_v2_enabled_for_this_round = False
        stream_v2_completed = False
        route = "chitchat"
        route_params: Dict[str, Any] = {}
        route_reason = ""
        fallback_used = False
        stream_sections: Dict[str, Any] = _create_stream_sections()
        rag_citations: List[Dict[str, Any]] = []
        v2_text_stream_started = False
        explicit_tool_cmd = _parse_explicit_tool_command(query)

        def _mark_ttft_if_needed():
            """在首次可见有效事件时标记 TTFT（仅记录一次）。"""
            nonlocal stream_ttft_ms
            if stream_ttft_ms is None:
                stream_ttft_ms = int((time.time() - stream_start) * 1000)

        def _next_v2_seq() -> int:
            nonlocal stream_v2_seq
            stream_v2_seq += 1
            return stream_v2_seq

        try:
            # 显式工具调用：跳过 supervisor，直接执行注册工具。
            if explicit_tool_cmd is not None:
                route = "history"
                route_reason = "explicit_tool_command"
                route_params = {"explicit_tool": True}

                parse_error = explicit_tool_cmd.get("error")
                if parse_error:
                    answer = (
                        f"工具调用格式错误：{parse_error}\n"
                        f"示例1: /tool get_practice_history {{\"limit\":20}}\n"
                        f"示例2: /tool get_weak_points_tool top_k=5"
                    )
                else:
                    tool_name = str(explicit_tool_cmd.get("tool_name", "")).strip()
                    route_params["tool_name"] = tool_name
                    exec_result = await _execute_registered_tool(
                        tool_name=tool_name,
                        args=explicit_tool_cmd.get("args") or {},
                        session_id=session_id,
                    )
                    status_text = "成功" if bool(exec_result.get("ok")) else "失败"
                    answer = f"【工具执行{status_text}】{tool_name}\n{exec_result.get('text', '')}"

                answer = _clean_answer(answer)
                message = build_assistant_message(route, answer, session_id, route_params, structured_result=None)
                message = _attach_user_profile_meta(message)

                _mark_ttft_if_needed()
                _record_stream_event("start")
                yield _format_sse(
                    {
                        "event": "start",
                        "message": {
                            "kind": message["kind"],
                            "render_mode": message["render_mode"],
                            "meta": message["meta"],
                        },
                    },
                    event_name="start",
                )
                chunks = answer.splitlines(keepends=True) or [answer]
                for chunk in chunks:
                    if await request.is_disconnected():
                        client_disconnected = True
                        raise asyncio.CancelledError()
                    if stream_first_delta_ms is None:
                        stream_first_delta_ms = int((time.time() - stream_start) * 1000)
                    _record_stream_event("delta")
                    yield _format_sse({"event": "delta", "text": chunk}, event_name="delta")
                    sent_content += chunk
                    await asyncio.sleep(0.01)

                _record_stream_event("complete")
                yield _format_sse({"event": "complete", "message": message}, event_name="complete")
                if route in RUNTIME_METRICS["routes"]:
                    RUNTIME_METRICS["routes"][route] += 1
                raise _StreamV2Handled()

            # 先发送一个占位消息，避免长耗时任务期间前端完全空白
            _record_stream_event("start")
            yield _format_sse(
                {
                    'event': 'start',
                    'message': {'kind': 'chat', 'render_mode': 'markdown', 'meta': {'route': 'pending'}},
                    'message_id': stream_v2_message_id,
                    'seq': 0,
                    'ts': int(time.time() * 1000),
                    'route': 'pending',
                    'payload': {'kind': 'chat', 'render_mode': 'markdown', 'meta': {'route': 'pending'}},
                },
                event_name="start",
            )
            _record_stream_event("delta")
            yield _format_sse(
                {'event': 'delta', 'text': '正在生成内容，请稍候...\n'},
                event_name="delta",
            )

            if attempt_stream_v2:
                supervisor_timeout_s = 45.0
                heartbeat_interval_s = 8.0
                deadline = time.time() + supervisor_timeout_s
                last_heartbeat_at = time.time()
                route_task = asyncio.create_task(
                    supervisor_node(
                        {
                            "input": query,
                            "chat_history": chat_history,
                            "memory_context": graph_context,
                            "session_id": session_id,
                            "route": "",
                            "route_reason": "",
                            "route_params": {},
                            "subagent_result": "",
                            "final_answer": "",
                        }
                    )
                )
                route_state = None
                while True:
                    if await request.is_disconnected():
                        client_disconnected = True
                        route_task.cancel()
                        raise asyncio.CancelledError()
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        route_task.cancel()
                        raise asyncio.TimeoutError()
                    try:
                        route_state = await asyncio.wait_for(asyncio.shield(route_task), timeout=min(1.0, remaining))
                        break
                    except asyncio.TimeoutError:
                        now = time.time()
                        if now - last_heartbeat_at >= heartbeat_interval_s:
                            _safe_inc("stream_heartbeat_sent_total")
                            _record_stream_event("heartbeat")
                            yield _format_sse(
                                {
                                    "event": "heartbeat",
                                    "status": "routing",
                                    "elapsed_ms": int((now - stream_start) * 1000),
                                },
                                event_name="heartbeat",
                            )
                            last_heartbeat_at = now

                route = str((route_state or {}).get("route", "chitchat") or "chitchat")
                route_params = (route_state or {}).get("route_params", {}) or {}
                route_reason = str((route_state or {}).get("route_reason", "") or "")
                rewritten_query = str((route_state or {}).get("input", query) or query).strip() or query
                stream_v2_route = route

                if route in {"rag", "chitchat"} and route in set(stream_v2_cfg.get("routes", set())):
                    stream_v2_enabled_for_this_round = True
                    _safe_inc("stream_v2_activated_total")
                    emit_sections = bool(stream_v2_cfg.get("emit_sections", True))
                    try:
                        _mark_ttft_if_needed()
                        _record_stream_event("start")
                        start_meta = {"route": route, "route_reason": route_reason, "stream_version": "v2"}
                        yield _format_sse(
                            {
                                "event": "start",
                                "message": {"kind": "chat", "render_mode": "markdown", "meta": start_meta},
                                "message_id": stream_v2_message_id,
                                "seq": _next_v2_seq(),
                                "ts": int(time.time() * 1000),
                                "route": route,
                                "payload": {"kind": "chat", "render_mode": "markdown", "meta": start_meta},
                            },
                            event_name="start",
                        )
                        if emit_sections:
                            _record_stream_event("progress")
                            stream_sections["progress"].append("已完成意图识别")
                            yield _format_sse(
                                _build_v2_event(
                                    event="progress",
                                    message_id=stream_v2_message_id,
                                    seq=_next_v2_seq(),
                                    route=route,
                                    payload={"stage": "route", "status": "completed", "message": "已完成意图识别"},
                                ),
                                event_name="progress",
                            )
                            route_summary = f"当前路由：{route}"
                            if route_reason:
                                route_summary += f"（{route_reason[:60]}）"
                            _record_stream_event("progress")
                            stream_sections["progress"].append(route_summary)
                            yield _format_sse(
                                _build_v2_event(
                                    event="progress",
                                    message_id=stream_v2_message_id,
                                    seq=_next_v2_seq(),
                                    route=route,
                                    payload={"stage": "route", "status": "selected", "message": route_summary},
                                ),
                                event_name="progress",
                            )

                        stream_model = chat_model or light_chat_model or backup_light_chat_model
                        if stream_model is None:
                            raise RuntimeError("当前未配置可用于流式输出的模型")

                        if route == "rag":
                            topic = str(route_params.get("topic") or rewritten_query).strip() or rewritten_query
                            if emit_sections:
                                _record_stream_event("progress")
                                stream_sections["progress"].append("开始检索课件上下文")
                                yield _format_sse(
                                    _build_v2_event(
                                        event="progress",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={"stage": "retrieve_context", "status": "running", "message": "开始检索课件上下文"},
                                    ),
                                    event_name="progress",
                                )
                                _record_stream_event("action")
                                stream_sections["actions"].append({"name": "retrieve_context", "status": "running", "detail": topic})
                                stream_sections["tool_calls"].append(
                                    {
                                        "tool_name": "retrieve_context",
                                        "status": "running",
                                        "detail": topic,
                                        "source": "stream_v2_rag",
                                    }
                                )
                                yield _format_sse(
                                    _build_v2_event(
                                        event="action",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={
                                            "name": "retrieve_context",
                                            "tool_name": "retrieve_context",
                                            "kind": "tool_call",
                                            "status": "running",
                                            "detail": topic,
                                            "source": "stream_v2_rag",
                                        },
                                    ),
                                    event_name="action",
                                )
                            retrieve_start = time.time()
                            rag = await get_rag_service()
                            context = await rag.retrieve_context(topic, mode="rag_chat")
                            if not context or not str(context).strip():
                                logger.warning("[StreamV2-RAG] 检索结果为空，注入友好提示上下文")
                                context = "【提示】当前知识库为空，请先上传课件后再提问。你可以通过点击「上传课件」按钮来添加复习资料。"
                            rag_citations = _extract_citations_from_context(context)
                            retrieve_cost_ms = int((time.time() - retrieve_start) * 1000)
                            if emit_sections:
                                _record_stream_event("action_result")
                                retrieve_summary = f"命中引用 {len(rag_citations)} 条"
                                stream_sections["actions"].append(
                                    {
                                        "name": "retrieve_context",
                                        "status": "completed",
                                        "cost_ms": retrieve_cost_ms,
                                        "detail": retrieve_summary,
                                    }
                                )
                                stream_sections["tool_calls"].append(
                                    {
                                        "tool_name": "retrieve_context",
                                        "status": "completed",
                                        "cost_ms": retrieve_cost_ms,
                                        "result_summary": retrieve_summary,
                                        "source": "stream_v2_rag",
                                    }
                                )
                                yield _format_sse(
                                    _build_v2_event(
                                        event="action_result",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={
                                            "name": "retrieve_context",
                                            "tool_name": "retrieve_context",
                                            "kind": "tool_call",
                                            "status": "completed",
                                            "cost_ms": retrieve_cost_ms,
                                            "detail": retrieve_summary,
                                            "result_summary": retrieve_summary,
                                            "source": "stream_v2_rag",
                                        },
                                    ),
                                    event_name="action_result",
                                )
                                _record_stream_event("progress")
                                stream_sections["progress"].append(f"课件检索完成（{retrieve_summary}）")
                                yield _format_sse(
                                    _build_v2_event(
                                        event="progress",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={
                                            "stage": "retrieve_context",
                                            "status": "completed",
                                            "message": f"课件检索完成（{retrieve_summary}）",
                                        },
                                    ),
                                    event_name="progress",
                                )
                                for citation in rag_citations:
                                    _record_stream_event("citation")
                                    stream_sections["citations"].append(citation)
                                    yield _format_sse(
                                        _build_v2_event(
                                            event="citation",
                                            message_id=stream_v2_message_id,
                                            seq=_next_v2_seq(),
                                            route=route,
                                            payload=citation,
                                        ),
                                        event_name="citation",
                                    )
                                reasoning_hint = "已完成课件检索，正在基于引用内容组织答案。"
                                _record_stream_event("reasoning_delta")
                                stream_sections["reasoning"] += reasoning_hint
                                yield _format_sse(
                                    _build_v2_event(
                                        event="reasoning_delta",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={"text": reasoning_hint},
                                    ),
                                    event_name="reasoning_delta",
                                )
                            chain = (
                                PromptTemplate.from_template(
                                    "你是复习助手，请严格基于检索资料回答学生问题。\n"
                                    "要求：直接回答，不要输出思考过程；若资料不足请明确说明“根据当前检索资料”。\n"
                                    "【长期记忆】\n{memory_context}\n\n"
                                    "【近期对话】\n{recent_history}\n\n"
                                    "【检索资料】\n{context}\n\n"
                                    "【学生问题】\n{input}\n"
                                )
                                | stream_model
                                | StrOutputParser()
                            )
                            stream_inputs = {
                                "memory_context": graph_context,
                                "recent_history": _format_recent_history_for_prompt(chat_history),
                                "context": context,
                                "input": rewritten_query,
                            }
                        else:
                            if emit_sections:
                                reasoning_hint = "已加载近期对话上下文，正在生成回复。"
                                _record_stream_event("reasoning_delta")
                                stream_sections["reasoning"] += reasoning_hint
                                yield _format_sse(
                                    _build_v2_event(
                                        event="reasoning_delta",
                                        message_id=stream_v2_message_id,
                                        seq=_next_v2_seq(),
                                        route=route,
                                        payload={"text": reasoning_hint},
                                    ),
                                    event_name="reasoning_delta",
                                )
                            chain = PromptTemplate.from_template(CHITCHAT_PROMPT) | stream_model | StrOutputParser()
                            stream_inputs = {
                                "input": rewritten_query,
                                "context": (
                                    f"【近期对话】\n{_format_recent_history_for_prompt(chat_history)}\n\n"
                                    f"【知识点备忘】\n{graph_context}"
                                ).strip(),
                            }

                        generate_start = time.time()
                        if emit_sections:
                            _record_stream_event("progress")
                            stream_sections["progress"].append("开始流式生成回答")
                            yield _format_sse(
                                _build_v2_event(
                                    event="progress",
                                    message_id=stream_v2_message_id,
                                    seq=_next_v2_seq(),
                                    route=route,
                                    payload={"stage": "generation", "status": "running", "message": "开始流式生成回答"},
                                ),
                                event_name="progress",
                            )

                        async for chunk in chain.astream(stream_inputs):
                            if await request.is_disconnected():
                                client_disconnected = True
                                raise asyncio.CancelledError()
                            chunk_text = str(chunk or "")
                            if not chunk_text:
                                continue
                            v2_text_stream_started = True
                            if stream_first_delta_ms is None:
                                stream_first_delta_ms = int((time.time() - stream_start) * 1000)
                            _mark_ttft_if_needed()
                            answer += chunk_text
                            sent_content += chunk_text
                            _record_stream_event("text_delta")
                            yield _format_sse(
                                {
                                    "event": "text_delta",
                                    "text": chunk_text,
                                    "message_id": stream_v2_message_id,
                                    "seq": _next_v2_seq(),
                                    "ts": int(time.time() * 1000),
                                    "route": route,
                                    "payload": {"text": chunk_text},
                                },
                                event_name="text_delta",
                            )
                            if bool(stream_v2_cfg.get("legacy_delta_mirror", True)):
                                _record_stream_event("delta")
                                yield _format_sse({"event": "delta", "text": chunk_text}, event_name="delta")

                        if emit_sections:
                            generate_cost_ms = int((time.time() - generate_start) * 1000)
                            _record_stream_event("action_result")
                            stream_sections["actions"].append(
                                {"name": "generate_answer", "status": "completed", "cost_ms": generate_cost_ms}
                            )
                            yield _format_sse(
                                _build_v2_event(
                                    event="action_result",
                                    message_id=stream_v2_message_id,
                                    seq=_next_v2_seq(),
                                    route=route,
                                    payload={
                                        "name": "generate_answer",
                                        "kind": "model_generation",
                                        "status": "completed",
                                        "cost_ms": generate_cost_ms,
                                    },
                                ),
                                event_name="action_result",
                            )
                            _record_stream_event("progress")
                            stream_sections["progress"].append("回答生成完成")
                            yield _format_sse(
                                _build_v2_event(
                                    event="progress",
                                    message_id=stream_v2_message_id,
                                    seq=_next_v2_seq(),
                                    route=route,
                                    payload={"stage": "generation", "status": "completed", "message": "回答生成完成"},
                                ),
                                event_name="progress",
                            )

                        answer = _clean_answer(answer)
                        if not answer and sent_content.strip():
                            answer = sent_content.strip()
                        rag_grounded_evidence: List[Dict[str, str]] = []
                        rag_grounding_reason = ""
                        if route == "rag":
                            for citation in rag_citations:
                                source = str(citation.get("source", "参考资料"))
                                quote = str(citation.get("quote", "")).strip()
                                if not quote:
                                    continue
                                if quote in answer:
                                    rag_grounded_evidence.append({"source": source, "quote": quote})
                            if answer and rag_grounded_evidence:
                                rag_inc("rag_grounded_pass_count", 1)
                                rag_grounding_reason = "pass"
                            elif answer:
                                rag_inc("rag_grounded_partial_count", 1)
                                rag_grounding_reason = "partial"
                            else:
                                rag_inc("rag_grounded_fail_count", 1)
                                rag_grounding_reason = "fail"
                                answer = (
                                    "根据当前检索结果，我暂时无法提取到可核验的课件依据。"
                                    "为避免误导，请换个更具体的问题，或补充相关课件后再试。"
                                )
                        if answer:
                            message = build_assistant_message(route, answer, session_id, route_params, structured_result=None)
                            message = _attach_user_profile_meta(message)
                            payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
                            if bool(stream_v2_cfg.get("emit_sections", True)):
                                payload = {**payload, "stream_sections": stream_sections}
                            message["payload"] = payload
                            meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
                            if route == "rag":
                                meta = {
                                    **meta,
                                    "grounding_mode": "stream_v2_citation_match",
                                    "grounded_evidence_count": len(rag_grounded_evidence),
                                    "grounded_reason": rag_grounding_reason or ("pass" if rag_grounded_evidence else "partial"),
                                }
                            message["meta"] = {**meta, "stream_version": "v2", "route_reason": route_reason}
                            _record_stream_event("complete")
                            yield _format_sse(
                                {
                                    "event": "complete",
                                    "message": message,
                                    "message_id": stream_v2_message_id,
                                    "seq": _next_v2_seq(),
                                    "ts": int(time.time() * 1000),
                                    "route": route,
                                    "payload": {"message": message},
                                },
                                event_name="complete",
                            )
                        if route in RUNTIME_METRICS["routes"]:
                            RUNTIME_METRICS["routes"][route] += 1
                        stream_v2_completed = True
                        raise _StreamV2Handled()
                    except asyncio.CancelledError:
                        raise
                    except Exception as stream_v2_error:
                        if v2_text_stream_started:
                            logger.exception("[StreamV2] 流式已开始输出正文，无法无损回退到 V1", exc_info=stream_v2_error)
                            raise
                        logger.exception("[StreamV2] 执行失败，回退到 V1 工作流", exc_info=stream_v2_error)
                        stream_v2_enabled_for_this_round = False
                        stream_v2_route = "pending"
                        answer = ""
                        sent_content = ""
                        message = None
                        rag_citations = []
                        stream_sections = _create_stream_sections()
                        fallback_used = True

                if route_state is not None:
                    # 回退到 workflow 前复用已完成的路由判定，避免 quiz/exam 再次调用 supervisor LLM。
                    initial_state.update(
                        {
                            "input": rewritten_query,
                            "route": route,
                            "route_reason": route_reason,
                            "route_params": route_params,
                            "supervisor_precomputed": True,
                        }
                    )

            workflow_timeout_s = 660.0  # 外层略大于 10 分钟出卷预算，避免前后层超时打架
            heartbeat_interval_s = 8.0
            deadline = time.time() + workflow_timeout_s
            last_heartbeat_at = time.time()
            workflow_task = asyncio.create_task(supervisor_workflow.ainvoke(initial_state))

            while True:
                if await request.is_disconnected():
                    client_disconnected = True
                    logger.info("检测到客户端断开，取消本次 supervisor_workflow")
                    workflow_task.cancel()
                    raise asyncio.CancelledError()

                remaining = deadline - time.time()
                if remaining <= 0:
                    workflow_task.cancel()
                    raise asyncio.TimeoutError()

                try:
                    result = await asyncio.wait_for(asyncio.shield(workflow_task), timeout=min(1.0, remaining))
                    break
                except asyncio.TimeoutError:
                    now = time.time()
                    if now - last_heartbeat_at >= heartbeat_interval_s:
                        _safe_inc("stream_heartbeat_sent_total")
                        _record_stream_event("heartbeat")
                        yield _format_sse(
                            {
                                "event": "heartbeat",
                                "status": "running",
                                "elapsed_ms": int((now - stream_start) * 1000),
                            },
                            event_name="heartbeat",
                        )
                        last_heartbeat_at = now

            workflow_latency_ms = int((time.time() - stream_start) * 1000)
            raw_subagent_result = result.get("subagent_result", "")
            structured_result = raw_subagent_result if isinstance(raw_subagent_result, dict) else None
            answer = result.get("final_answer", "")
            if not answer:
                answer = structured_result.get("text", "") if structured_result else raw_subagent_result
            route = result.get("route", "chitchat")
            route_params = result.get("route_params", {}) or {}
            route_reason = str(result.get("route_reason", "") or "")
            fallback_used = "fallback" in route_reason.lower()
            if route in RUNTIME_METRICS["routes"]:
                RUNTIME_METRICS["routes"][route] += 1
            if route == "exam":
                _safe_inc("exam_total")
            logger.info(f"[Raw Answer] length={len(answer)}, content={answer}")

            # 清洗回答
            answer = _clean_answer(answer)
            logger.info(f"[Final Answer] length={len(answer)}, content={answer}")

            # V1 非真流式也补齐分区：尤其是 ops 的工具轨迹，前端可直接展示“工具调用”分区。
            if structured_result is not None:
                payload_obj = structured_result.get("payload") if isinstance(structured_result.get("payload"), dict) else {}
                ops_trace = payload_obj.get("ops_trace") if isinstance(payload_obj, dict) else None
                if isinstance(ops_trace, list):
                    ops_sections = _ops_trace_to_stream_sections(ops_trace)
                    payload_obj = {**payload_obj, "stream_sections": ops_sections}
                    structured_result = {**structured_result, "payload": payload_obj}

            if answer:
                if structured_result is not None:
                    structured_result = {**structured_result, "text": answer}
                message = build_assistant_message(route, answer, session_id, route_params, structured_result)
                message = _attach_user_profile_meta(message)
                _mark_ttft_if_needed()
                _record_stream_event("start")
                yield _format_sse(
                    {'event': 'start', 'message': {'kind': message['kind'], 'render_mode': message['render_mode'], 'meta': message['meta']}},
                    event_name="start",
                )

                if message["kind"] == "chat":
                    # 普通对话维持伪流式
                    for i, line in enumerate(answer.splitlines(keepends=True)):
                        if await request.is_disconnected():
                            client_disconnected = True
                            logger.info("客户端在回答流式输出阶段断开连接")
                            raise asyncio.CancelledError()
                        logger.info(f"[Stream] line{i}: {line}")
                        if stream_first_delta_ms is None:
                            stream_first_delta_ms = int((time.time() - stream_start) * 1000)
                        _record_stream_event("delta")
                        yield _format_sse({'event': 'delta', 'text': line}, event_name="delta")
                        sent_content += line
                        await asyncio.sleep(0.03)

                _record_stream_event("complete")
                yield _format_sse({'event': 'complete', 'message': message}, event_name="complete")
                if route == "exam" and message.get("kind") == "exam_paper":
                    exam_data = (message.get("payload") or {}).get("exam_data", {})
                    exam_questions = exam_data.get("questions", []) if isinstance(exam_data, dict) else []
                    if isinstance(exam_questions, list) and len(exam_questions) > 0:
                        _safe_inc("exam_success")
                    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
                    if meta.get("degrade_reason") == "revise_timeout":
                        _safe_inc("revise_timeout_count")
                    if meta.get("delivery_mode") == "failed":
                        _safe_inc("hard_fail_count")
                    if meta.get("delivery_mode") == "partial_revised" or meta.get("degrade_reason"):
                        _safe_inc("degraded_delivery_count")
                    if meta.get("critic_timeout_count"):
                        _safe_inc("critic_timeout_count", int(meta.get("critic_timeout_count") or 0))
                    if not bool(meta.get("term_style_ok", True)):
                        _safe_inc("term_style_violation_count")
                    if int(meta.get("missing_after_retries") or 0) > 0:
                        _safe_inc("missing_model_retry_count", int(meta.get("missing_after_retries") or 0))
                    fixed_items = meta.get("fixed_items") if isinstance(meta.get("fixed_items"), list) else []
                    if "missing_options" in fixed_items:
                        _safe_inc("exam_choice_missing_options_fix_count")
                    if "score_rebalanced" in fixed_items:
                        _safe_inc("exam_score_rebalance_count")
                    if "duplicate_rewrite" in fixed_items:
                        _safe_inc("exam_duplicate_rewrite_count")
                    if "prompt_contract_violation" in fixed_items:
                        _safe_inc("exam_prompt_contract_violation_count")
                    if isinstance(meta.get("exam_end_to_end_ms"), int):
                        _record_exam_latency(meta.get("exam_end_to_end_ms"))
                    else:
                        _record_exam_latency(int((time.time() - stream_start) * 1000))
                    tool_calls = meta.get("tool_calls") if isinstance(meta.get("tool_calls"), dict) else {}
                    _safe_inc("tool_call_count_rag", int(tool_calls.get("rag", 0) or 0))
                    _safe_inc("tool_call_count_web", int(tool_calls.get("web", 0) or 0))
                    stage_trace = meta.get("stage_trace") if isinstance(meta.get("stage_trace"), list) else []
                    for st in stage_trace:
                        if not isinstance(st, dict):
                            continue
                        attempt = int(st.get("attempt") or 0)
                        if attempt > 1:
                            _safe_inc("exam_stage_retry_count", attempt - 1)
                        if st.get("stage_status") == "failed":
                            _safe_inc("exam_stage_fail_count")
                        if isinstance(st.get("stage_latency_ms"), int):
                            _record_exam_stage_latency(int(st.get("stage_latency_ms")))
                        if int(st.get("critic_timeout_count") or 0) > 0:
                            _safe_inc("critic_timeout_count", int(st.get("critic_timeout_count") or 0))
                        err_text = str(st.get("error") or "")
                        if "超时" in err_text.lower() or "timeout" in err_text.lower():
                            _safe_inc("exam_stage_timeout_count")
                elif route == "exam":
                    # 严格模式下可能返回失败消息（kind=chat），同样记录段级轨迹
                    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
                    fixed_items = meta.get("fixed_items") if isinstance(meta.get("fixed_items"), list) else []
                    if "missing_options" in fixed_items:
                        _safe_inc("exam_choice_missing_options_fix_count")
                    if "score_rebalanced" in fixed_items:
                        _safe_inc("exam_score_rebalance_count")
                    if "duplicate_rewrite" in fixed_items:
                        _safe_inc("exam_duplicate_rewrite_count")
                    if "prompt_contract_violation" in fixed_items:
                        _safe_inc("exam_prompt_contract_violation_count")
                    if meta.get("delivery_mode") == "failed":
                        _safe_inc("hard_fail_count")
                    if meta.get("critic_timeout_count"):
                        _safe_inc("critic_timeout_count", int(meta.get("critic_timeout_count") or 0))
                    if not bool(meta.get("term_style_ok", True)):
                        _safe_inc("term_style_violation_count")
                    if int(meta.get("missing_after_retries") or 0) > 0:
                        _safe_inc("missing_model_retry_count", int(meta.get("missing_after_retries") or 0))
                    stage_trace = meta.get("stage_trace") if isinstance(meta.get("stage_trace"), list) else []
                    for st in stage_trace:
                        if not isinstance(st, dict):
                            continue
                        attempt = int(st.get("attempt") or 0)
                        if attempt > 1:
                            _safe_inc("exam_stage_retry_count", attempt - 1)
                        if st.get("stage_status") == "failed":
                            _safe_inc("exam_stage_fail_count")
                        if isinstance(st.get("stage_latency_ms"), int):
                            _record_exam_stage_latency(int(st.get("stage_latency_ms")))
                        if int(st.get("critic_timeout_count") or 0) > 0:
                            _safe_inc("critic_timeout_count", int(st.get("critic_timeout_count") or 0))
                        err_text = str(st.get("error") or "")
                        if "超时" in err_text.lower() or "timeout" in err_text.lower():
                            _safe_inc("exam_stage_timeout_count")
                if route == "quiz" and message.get("kind") in {"quiz_set", "chat"}:
                    meta = message.get("meta") if isinstance(message.get("meta"), dict) else {}
                    if meta.get("degrade_reason") in {"budget_exhausted", "revise_timeout"}:
                        _safe_inc("quiz_timeout_count")
                    if meta.get("degrade_reason") == "quality_guard_no_evidence":
                        _safe_inc("fallback_quality_guard_count")
                    if meta.get("delivery_mode") == "partial_revised" or meta.get("degrade_reason"):
                        _safe_inc("degraded_delivery_count")
                    if isinstance(meta.get("quiz_end_to_end_ms"), int):
                        _record_quiz_latency(meta.get("quiz_end_to_end_ms"))
                    else:
                        _record_quiz_latency(int((time.time() - stream_start) * 1000))
                    tool_calls = meta.get("tool_calls") if isinstance(meta.get("tool_calls"), dict) else {}
                    _safe_inc("tool_call_count_rag", int(tool_calls.get("rag", 0) or 0))
                    _safe_inc("tool_call_count_web", int(tool_calls.get("web", 0) or 0))
                logger.info(
                    f"[Observability] session={session_id}, route={route}, reason={route_reason}, "
                    f"fallback={fallback_used}, kind={message['kind']}, latency_ms={int((time.time()-stream_start)*1000)}"
                )
                logger.info(f"[Latency] supervisor_workflow_ms={workflow_latency_ms}")

        except _StreamV2Handled:
            pass
        except asyncio.CancelledError:
            if client_disconnected:
                termination_reason = "client_cancelled"
                logger.info("流式连接已断开，本次请求停止继续推送")
            else:
                termination_reason = "cancelled"
                logger.warning("流式任务被取消")
                _safe_inc("failed_requests")
                error_text = "请求已取消，请稍后重试"
                RUNTIME_METRICS["last_error"] = error_text
                _mark_ttft_if_needed()
                _record_stream_event("error")
                yield _format_sse({'event': 'error', 'error': error_text}, event_name="error")
        except asyncio.TimeoutError:
            termination_reason = "timeout"
            logger.error("流式输出异常: supervisor_workflow timeout")
            _safe_inc("failed_requests")
            error_text = "出题耗时过长已超时，请缩小题量或减少考点后重试"
            RUNTIME_METRICS["last_error"] = error_text
            logger.info(f"[Latency] supervisor_workflow_failed_ms={int((time.time()-stream_start)*1000)}")
            _mark_ttft_if_needed()
            _record_stream_event("error")
            yield _format_sse({'event': 'error', 'error': error_text}, event_name="error")
        except Exception as e:
            termination_reason = "error"
            logger.error(f"流式输出异常: {e}")
            _safe_inc("failed_requests")
            error_text = str(e) or "请求超时，请稍后重试"
            RUNTIME_METRICS["last_error"] = error_text
            logger.info(f"[Latency] supervisor_workflow_failed_ms={int((time.time()-stream_start)*1000)}")
            _mark_ttft_if_needed()
            _record_stream_event("error")
            yield _format_sse({'event': 'error', 'error': error_text}, event_name="error")
        finally:
            if route_task is not None and not route_task.done():
                route_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await route_task
            if workflow_task is not None and not workflow_task.done():
                workflow_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await workflow_task

        stream_duration_ms = int((time.time() - stream_start) * 1000)
        _record_runtime_sample("stream_duration_samples_ms", stream_duration_ms)
        if stream_ttft_ms is not None:
            _record_runtime_sample("stream_ttft_samples_ms", stream_ttft_ms)
        if stream_first_delta_ms is not None:
            _record_runtime_sample("stream_first_delta_samples_ms", stream_first_delta_ms)
        _record_stream_termination(termination_reason)

        if termination_reason == "success":
            _safe_inc("stream_success_total")
        elif termination_reason == "client_cancelled":
            _safe_inc("stream_client_cancel_total")
        elif termination_reason == "timeout":
            _safe_inc("stream_timeout_total")
            _safe_inc("stream_error_total")
        else:
            _safe_inc("stream_error_total")

        if not client_disconnected:
            _record_stream_event("done")
            if stream_v2_enabled_for_this_round:
                yield _format_sse(
                    {
                        "event": "done",
                        "status": termination_reason,
                        "message_id": stream_v2_message_id,
                        "seq": _next_v2_seq(),
                        "ts": int(time.time() * 1000),
                        "route": stream_v2_route or route,
                        "payload": {"status": termination_reason},
                    },
                    event_name="done",
                )
            else:
                yield _format_sse(
                    {"event": "done", "status": termination_reason},
                    event_name="done",
                )
            # 兼容旧前端：保留 [DONE] 终止标记
            yield "data: [DONE]\n\n"

        # 保存对话历史（成功/失败都要记录用户输入，避免历史断层）
        now = int(time.time() * 1000)
        await memory_manager.add_message(
            session_id,
            "user",
            query,
            timestamp=now,
        )

        if client_disconnected:
            if sent_content.strip():
                ai_meta = message.get("meta") if message and isinstance(message.get("meta"), dict) else {}
                ai_meta = {
                    **ai_meta,
                    "cancelled": True,
                    "termination_reason": "client_cancelled",
                    "stream_version": "v2" if stream_v2_enabled_for_this_round else "v1",
                }
                await memory_manager.add_message(
                    session_id,
                    "ai",
                    sent_content,
                    kind="chat",
                    render_mode="markdown",
                    payload=None,
                    meta=ai_meta,
                    timestamp=now + 1,
                )
            return

        if answer:
            ai_meta = message.get("meta") if message else None
            if not isinstance(ai_meta, dict):
                ai_meta = {}
            if error_text:
                ai_meta = {**ai_meta, "error": True, "error_message": error_text}
            ai_meta = {
                **ai_meta,
                "termination_reason": termination_reason,
                "stream_version": "v2" if stream_v2_enabled_for_this_round else "v1",
            }
            await memory_manager.add_message(
                session_id,
                "ai",
                answer,
                kind=message["kind"] if message else "chat",
                render_mode=message["render_mode"] if message else "markdown",
                payload=message.get("payload") if message else None,
                meta=ai_meta,
                timestamp=now + 1,
            )
        elif error_text:
            await memory_manager.add_message(
                session_id,
                "ai",
                f"[系统提示: {error_text}]",
                kind="chat",
                render_mode="markdown",
                payload=None,
                meta={
                    "error": True,
                    "error_message": error_text,
                    "termination_reason": termination_reason,
                    "stream_version": "v2" if stream_v2_enabled_for_this_round else "v1",
                },
                timestamp=now + 1,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/practice/submit")
async def submit_practice_records(req: PracticeSubmitRequest = Body(...)):
    """
    提交练习记录（错题追踪主入口）。
    """
    try:
        session_id = req.session_id or "default"
        _validate_session_id(session_id)

        if not req.records:
            return {"code": 200, "message": "无记录需要提交", "saved": 0}

        current_session_id.set(session_id)
        saved = 0
        # 同批次知识点去重 + 归一化（规则优先，LLM 小规模润色）
        kp_map: Dict[str, str] = {}
        unique_items: List[tuple[str, str, str]] = []
        for record in req.records:
            raw_kp = str(record.knowledge_point or "")
            q_content = str(record.question_content or "")
            key = f"{raw_kp}|||{q_content[:160]}"
            if key not in kp_map:
                kp_map[key] = ""
                unique_items.append((key, raw_kp, q_content))

        llm_budget = 8  # 每次提交最多润色 8 个唯一考点，控制延迟
        sem = asyncio.Semaphore(4)

        async def _norm_one(item: tuple[str, str, str], use_llm: bool):
            key, raw_kp, q_content = item
            if use_llm:
                async with sem:
                    kp_map[key] = await _polish_knowledge_point(raw_kp, q_content)
            else:
                kp_map[key] = _extract_kp_rule(raw_kp, q_content)

        tasks = []
        for idx, item in enumerate(unique_items):
            tasks.append(_norm_one(item, use_llm=(idx < llm_budget)))
        if tasks:
            await asyncio.gather(*tasks)

        for idx, record in enumerate(req.records):
            qid = record.question_id or f"{session_id}_{record.question_number or idx + 1}_{int(time.time() * 1000)}"
            kp_key = f"{str(record.knowledge_point or '')}|||{str(record.question_content or '')[:160]}"
            normalized_kp = kp_map.get(kp_key) or _extract_kp_rule(record.knowledge_point or "", record.question_content or "")
            try:
                memory_manager.add_practice_record_sync(
                    session_id=session_id,
                    question_id=qid,
                    question_content=record.question_content,
                    knowledge_point=normalized_kp,
                    user_answer=record.user_answer,
                    correct_answer=record.correct_answer,
                    is_correct=record.is_correct,
                    wrong_reason=record.wrong_reason,
                )
                # 题库沉淀：用于相似题检索。这里全量入库，便于后续复练。
                try:
                    memory_manager.store_question_to_bank(
                        session_id=session_id,
                        question_id=qid,
                        question_content=record.question_content,
                        knowledge_point=normalized_kp,
                        answer=record.correct_answer,
                    )
                except Exception as e:
                    logger.warning(f"[Practice] 题库写入失败 qid={qid}: {e}")
                saved += 1
            except Exception as e:
                logger.error(f"[Practice] 记录写入失败 qid={qid}: {e}")
                continue

        stats = memory_manager.get_knowledge_point_stats(session_id)
        weak_points = [kp for kp, data in stats.items() if data.get("weak")]
        mastery_rows_total = memory_manager.get_mastery_rows_count(session_id)
        priority_review_points = memory_manager.get_priority_review_points(session_id, limit=10)
        RUNTIME_METRICS["mastery_rows_total"] = mastery_rows_total
        logger.info(f"[Practice] session={session_id}, saved={saved}, weak_points={weak_points}")
        return {
            "code": 200,
            "message": f"已保存 {saved} 条练习记录",
            "saved": saved,
            "weak_points": weak_points,
            "stats": stats,
            "mastery_rows_total": mastery_rows_total,
            "priority_review_points": priority_review_points,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Practice] submit 接口异常: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"练习记录提交失败: {str(e)}", "saved": 0},
        )


@router.post("/practice/backfill-kp")
async def backfill_practice_knowledge_points(req: PracticeBackfillRequest = Body(...)):
    """
    回填历史练习记录的 knowledge_point 字段（历史脏数据治理）。
    dry_run=true 时仅预览，不落库。
    """
    session_id = req.session_id or "default"
    _validate_session_id(session_id)
    summary = memory_manager.backfill_practice_knowledge_points(
        session_id=session_id,
        limit=max(1, min(int(req.limit or 2000), 20000)),
        dry_run=bool(req.dry_run),
    )
    stats = memory_manager.get_knowledge_point_stats(session_id)
    return {
        "code": 200,
        "message": "dry-run 已完成" if bool(req.dry_run) else "回填完成",
        "data": summary,
        "stats_count": len(stats),
    }


@router.post("/practice/similar")
async def get_similar_questions_batch(req: SimilarBatchRequest = Body(...)):
    """
    根据错题批量返回相似题，供前端复练模块直接渲染。
    """
    session_id = req.session_id or "default"
    _validate_session_id(session_id)

    current_session_id.set(session_id)
    result: Dict[int, List[Dict[str, Any]]] = {}
    for item in req.wrong_questions:
        sims = memory_manager.search_similar_questions(
            question_content=item.question_content,
            knowledge_point=item.knowledge_point or "",
            limit=max(1, min(req.limit, 8)),
            session_id=session_id,
        )
        result[item.question_number] = sims

    return {"code": 200, "similar_questions": result}


@router.get("/practice/stats")
async def get_practice_stats(session_id: str):
    """
    获取练习统计（用于错题复练率看板）。
    """
    _validate_session_id(session_id)

    history = memory_manager.get_practice_history(session_id, limit=500)
    kp_stats = memory_manager.get_knowledge_point_stats(session_id)
    total = len(history)
    wrong = sum(1 for h in history if not h.get("is_correct"))
    accuracy = round(((total - wrong) / total) * 100, 2) if total > 0 else 0.0
    retry_rate = round((wrong / total) * 100, 2) if total > 0 else 0.0
    weak_points = [kp for kp, v in kp_stats.items() if v.get("weak")]

    return {
        "code": 200,
        "data": {
            "session_id": session_id,
            "total_attempts": total,
            "wrong_attempts": wrong,
            "accuracy": accuracy,
            "retry_rate": retry_rate,
            "weak_points": weak_points,
            "knowledge_point_stats": kp_stats,
        },
    }


@router.get("/mastery")
async def get_mastery_snapshot(session_id: str, limit: int = 200, priority_limit: int = 10):
    """获取 mastery 快照与优先复习考点。"""
    _validate_session_id(session_id)
    safe_limit = max(1, min(int(limit or 200), 2000))
    safe_priority_limit = max(1, min(int(priority_limit or 10), 100))
    snapshot = memory_manager.get_mastery_snapshot(session_id, limit=safe_limit)
    priority_points = memory_manager.get_priority_review_points(session_id, limit=safe_priority_limit)
    mastery_rows_total = memory_manager.get_mastery_rows_count(session_id)
    RUNTIME_METRICS["mastery_rows_total"] = mastery_rows_total
    return {
        "code": 200,
        "data": {
            "session_id": session_id,
            "count": len(snapshot),
            "mastery_rows_total": mastery_rows_total,
            "mastery": snapshot,
            "priority_review_points": priority_points,
        },
    }


@router.get("/practice/history")
async def get_practice_history(session_id: str, limit: int = 200):
    """
    获取练习历史明细（用于前端练习历史管理面板）。
    """
    _validate_session_id(session_id)
    safe_limit = max(1, min(int(limit or 200), 500))
    history = memory_manager.get_practice_history(session_id, limit=safe_limit)
    return {"code": 200, "data": history, "count": len(history)}


@router.delete("/practice/history/item")
async def delete_practice_history_item(req: PracticeHistoryDeleteRequest = Body(...)):
    """
    删除单条练习历史记录。
    """
    session_id = req.session_id or "default"
    _validate_session_id(session_id)
    deleted = memory_manager.delete_practice_record(session_id=session_id, record_id=req.record_id)
    return {"code": 200 if deleted else 404, "deleted": bool(deleted)}


@router.delete("/practice/history")
async def clear_practice_history(session_id: str):
    """
    清空当前会话全部练习历史记录。
    """
    _validate_session_id(session_id)
    deleted = memory_manager.clear_practice_history(session_id)
    return {"code": 200, "deleted_count": int(deleted)}


@router.get("/metrics")
async def get_runtime_metrics():
    """
    进程内运行指标（出卷成功率、路由分布等）。
    """
    exam_total = int(RUNTIME_METRICS.get("exam_total", 0))
    exam_success = int(RUNTIME_METRICS.get("exam_success", 0))
    exam_success_rate = round((exam_success / exam_total) * 100, 2) if exam_total > 0 else 0.0
    samples = RUNTIME_METRICS.get("exam_latency_samples_ms", []) or []
    avg_ms = int(sum(samples) / len(samples)) if samples else 0
    max_ms = int(max(samples)) if samples else 0
    p95_ms = _calc_p95(samples)
    quiz_samples = RUNTIME_METRICS.get("quiz_latency_samples_ms", []) or []
    quiz_avg_ms = int(sum(quiz_samples) / len(quiz_samples)) if quiz_samples else 0
    quiz_max_ms = int(max(quiz_samples)) if quiz_samples else 0
    quiz_p95_ms = _calc_p95(quiz_samples)
    stage_samples = RUNTIME_METRICS.get("exam_stage_latency_ms", []) or []
    stage_avg_ms = int(sum(stage_samples) / len(stage_samples)) if stage_samples else 0
    stage_max_ms = int(max(stage_samples)) if stage_samples else 0
    stage_p95_ms = _calc_p95(stage_samples)
    stream_ttft_samples = RUNTIME_METRICS.get("stream_ttft_samples_ms", []) or []
    stream_ttft_avg_ms = int(sum(stream_ttft_samples) / len(stream_ttft_samples)) if stream_ttft_samples else 0
    stream_ttft_max_ms = int(max(stream_ttft_samples)) if stream_ttft_samples else 0
    stream_ttft_p95_ms = _calc_p95(stream_ttft_samples)
    stream_first_delta_samples = RUNTIME_METRICS.get("stream_first_delta_samples_ms", []) or []
    stream_first_delta_avg_ms = int(sum(stream_first_delta_samples) / len(stream_first_delta_samples)) if stream_first_delta_samples else 0
    stream_first_delta_max_ms = int(max(stream_first_delta_samples)) if stream_first_delta_samples else 0
    stream_first_delta_p95_ms = _calc_p95(stream_first_delta_samples)
    stream_duration_samples = RUNTIME_METRICS.get("stream_duration_samples_ms", []) or []
    stream_duration_avg_ms = int(sum(stream_duration_samples) / len(stream_duration_samples)) if stream_duration_samples else 0
    stream_duration_max_ms = int(max(stream_duration_samples)) if stream_duration_samples else 0
    stream_duration_p95_ms = _calc_p95(stream_duration_samples)
    stream_termination_reason_counts = RUNTIME_METRICS.get("stream_termination_reason_counts", {}) or {}
    if not isinstance(stream_termination_reason_counts, dict):
        stream_termination_reason_counts = {}
    stream_requests_total = int(RUNTIME_METRICS.get("stream_requests_total", 0))
    stream_success_total = int(RUNTIME_METRICS.get("stream_success_total", 0))
    stream_error_total = int(RUNTIME_METRICS.get("stream_error_total", 0))
    stream_client_cancel_total = int(RUNTIME_METRICS.get("stream_client_cancel_total", 0))
    stream_timeout_total = int(RUNTIME_METRICS.get("stream_timeout_total", 0))
    stream_v2_requests_total = int(RUNTIME_METRICS.get("stream_v2_requests_total", 0))
    stream_v2_activated_total = int(RUNTIME_METRICS.get("stream_v2_activated_total", 0))
    stream_event_counts_by_type = RUNTIME_METRICS.get("stream_event_counts_by_type", {}) or {}
    if not isinstance(stream_event_counts_by_type, dict):
        stream_event_counts_by_type = {}
    rag_metrics = rag_get_metrics_snapshot()
    rag_calls = int(rag_metrics.get("rag_retrieve_calls", 0))
    rag_cache_eligible = int(rag_metrics.get("rag_cache_eligible_calls", 0))
    rag_hits = int(rag_metrics.get("rag_retrieve_cache_hit", 0))
    rag_miss = int(rag_metrics.get("rag_retrieve_cache_miss", 0))
    rag_empty = int(rag_metrics.get("rag_empty_context_count", 0))
    rag_latency_samples = rag_metrics.get("rag_retrieve_latency_samples_ms", []) or []
    rag_latency_avg = int(sum(rag_latency_samples) / len(rag_latency_samples)) if rag_latency_samples else 0
    rag_latency_p95 = _calc_p95(rag_latency_samples)
    comprehensive_samples = rag_metrics.get("comprehensive_parallel_ms_samples", []) or []
    comprehensive_avg = int(sum(comprehensive_samples) / len(comprehensive_samples)) if comprehensive_samples else 0
    comprehensive_p95 = _calc_p95(comprehensive_samples)
    grounded_pass = int(rag_metrics.get("rag_grounded_pass_count", 0))
    grounded_partial = int(rag_metrics.get("rag_grounded_partial_count", 0))
    grounded_fail = int(rag_metrics.get("rag_grounded_fail_count", 0))
    grounded_total = grounded_pass + grounded_partial + grounded_fail
    structured_fail_count_by_reason = {}
    try:
        from agent.multi_agent.quiz_agent import get_structured_fail_stats
        structured_fail_count_by_reason = get_structured_fail_stats()
    except Exception:
        structured_fail_count_by_reason = {}
    return {
        "code": 200,
        "data": {
            **RUNTIME_METRICS,
            **rag_metrics,
            "structured_fail_count_by_reason": structured_fail_count_by_reason,
            "exam_success_rate": exam_success_rate,
            "exam_end_to_end_avg_ms": avg_ms,
            "exam_end_to_end_max_ms": max_ms,
            "exam_end_to_end_p95_ms": p95_ms,
            "quiz_e2e_avg_ms": quiz_avg_ms,
            "quiz_e2e_max_ms": quiz_max_ms,
            "quiz_e2e_p95_ms": quiz_p95_ms,
            "exam_stage_latency_avg_ms": stage_avg_ms,
            "exam_stage_latency_max_ms": stage_max_ms,
            "exam_stage_latency_p95_ms": stage_p95_ms,
            "stream_ttft_avg_ms": stream_ttft_avg_ms,
            "stream_ttft_max_ms": stream_ttft_max_ms,
            "stream_ttft_p95_ms": stream_ttft_p95_ms,
            "stream_first_delta_avg_ms": stream_first_delta_avg_ms,
            "stream_first_delta_max_ms": stream_first_delta_max_ms,
            "stream_first_delta_p95_ms": stream_first_delta_p95_ms,
            "stream_duration_avg_ms": stream_duration_avg_ms,
            "stream_duration_max_ms": stream_duration_max_ms,
            "stream_duration_p95_ms": stream_duration_p95_ms,
            "stream_termination_reason_counts": stream_termination_reason_counts,
            "stream_success_rate": round((stream_success_total / stream_requests_total) * 100, 2) if stream_requests_total > 0 else 0.0,
            "stream_error_rate": round((stream_error_total / stream_requests_total) * 100, 2) if stream_requests_total > 0 else 0.0,
            "stream_client_cancel_rate": round((stream_client_cancel_total / stream_requests_total) * 100, 2) if stream_requests_total > 0 else 0.0,
            "stream_timeout_rate": round((stream_timeout_total / stream_requests_total) * 100, 2) if stream_requests_total > 0 else 0.0,
            "stream_v2_activation_rate": round((stream_v2_activated_total / stream_requests_total) * 100, 2) if stream_requests_total > 0 else 0.0,
            "stream_v2_hit_rate": round((stream_v2_activated_total / stream_v2_requests_total) * 100, 2) if stream_v2_requests_total > 0 else 0.0,
            "stream_event_counts_by_type": stream_event_counts_by_type,
            "rag_cache_hit_rate": round((rag_hits / rag_cache_eligible) * 100, 2) if rag_cache_eligible > 0 else 0.0,
            "rag_empty_context_rate": round((rag_empty / rag_calls) * 100, 2) if rag_calls > 0 else 0.0,
            "rag_retrieve_avg_ms": rag_latency_avg,
            "rag_retrieve_p95_ms": rag_latency_p95,
            "comprehensive_parallel_avg_ms": comprehensive_avg,
            "comprehensive_parallel_p95_ms": comprehensive_p95,
            "rag_grounded_pass_rate": round((grounded_pass / grounded_total) * 100, 2) if grounded_total > 0 else 0.0,
        },
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话及其知识库，并清理对应 RAG 缓存。"""
    _validate_session_id(session_id)
    current_session_id.set(session_id)
    try:
        vs = VectorStoreService()
        await vs.destroy_knowledge_base()
    except Exception as e:
        print(f"知识库清理失败: {e}")
    clear_rag_cache(session_id)
    success = memory_manager.clear_session(session_id)
    if success:
        return {"code": 200, "message": f"科目会话 {session_id} 及其专属知识库已被永久销毁！"}
    return {"code": 404, "message": "该会话不存在"}


class RenameRequest(BaseModel):
    new_name: str


@router.put("/session/{session_id}")
async def rename_session(session_id: str, req: RenameRequest = Body(...)):
    """重命名会话展示名（session_id 本身不变）。"""
    _validate_session_id(session_id)
    success = memory_manager.rename_session(session_id, req.new_name)
    if success:
        return {"code": 200, "message": "重命名成功"}
    return {"code": 404, "message": "该会话不存在"}


class SessionCreateRequest(BaseModel):
    session_id: str
    name: str
    parent_id: Optional[str] = None


class SessionCleanupRequest(BaseModel):
    session_id: str


@router.post("/session")
async def register_session(req: SessionCreateRequest = Body(...)):
    """创建/注册新会话，用于前端会话树管理。"""
    _validate_session_id(req.session_id)
    memory_manager.register_session(req.session_id, req.name, req.parent_id)
    return {"code": 200, "message": "会话注册成功"}


@router.post("/session/cleanup")
async def cleanup_legacy_session(req: SessionCleanupRequest = Body(...)):
    """
    兼容清理历史遗留非法 session_id（例如旧版本产生的 ../x）。
    - 合法 session_id：等价于普通删除（含知识库销毁）
    - 非法 session_id：仅清理记忆与缓存，不触发向量库目录操作
    """
    sid = (req.session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    if _VALID_SESSION_ID.match(sid):
        current_session_id.set(sid)
        try:
            vs = VectorStoreService()
            await vs.destroy_knowledge_base()
        except Exception as e:
            logger.warning(f"[cleanup] 合法会话知识库清理失败 sid={sid}: {e}")
        clear_rag_cache(sid)
        success = memory_manager.clear_session(sid)
        return {"code": 200 if success else 404, "message": "会话已清理" if success else "会话不存在"}

    # 非法 ID 仅从会话存储中移除，避免路径相关安全风险
    clear_rag_cache(sid)
    success = memory_manager.clear_session(sid)
    return {
        "code": 200 if success else 404,
        "message": "已清理历史遗留非法会话" if success else "会话不存在",
    }


@router.get("/sessions")
async def get_all_sessions():
    """返回全部会话列表（含层级关系字段）。"""
    sessions = memory_manager.get_all_sessions()
    return {"code": 200, "data": sessions}


@router.get("/user-profile")
async def get_user_profile_endpoint(username: str = "default_user"):
    """获取用户级人格配置（与会话无关）。"""
    user = normalize_username(username)
    data = get_user_profile_data(user)
    return {"code": 200, "data": data}


@router.put("/user-profile")
async def update_user_profile_endpoint(req: UserProfileUpdateRequest = Body(...)):
    """更新用户级人格配置（白名单字段）。"""
    user = normalize_username(req.username)
    data = update_user_profile_data(user, req.profile or {})
    return {"code": 200, "message": "用户人格配置已更新", "data": data}


# ==================== 消息管理 ====================

@router.get("/messages")
async def get_messages(session_id: str):
    """
    获取指定会话的所有消息，用于历史管理 UI。
    """
    _validate_session_id(session_id)
    messages = memory_manager.get_messages(session_id)
    normalized = []
    for msg in messages:
        normalized.append({
            "id": f"{msg.get('timestamp', 0)}",
            "role": msg.get("role"),
            "content": msg.get("content"),
            "timestamp": msg.get("timestamp", 0),
            "kind": msg.get("kind"),
            "render_mode": msg.get("render_mode"),
            "payload": msg.get("payload"),
            "meta": msg.get("meta"),
        })
    # 兼容前端/旧客户端：同时返回 data 和 messages
    return {"code": 200, "data": normalized, "messages": normalized, "count": len(normalized)}


class DeleteMessageRequest(BaseModel):
    session_id: str
    timestamp: int


@router.delete("/message")
async def delete_message(req: DeleteMessageRequest = Body(...)):
    """
    删除指定 timestamp 的单条消息。
    """
    _validate_session_id(req.session_id)
    success = memory_manager.delete_message(req.session_id, req.timestamp)
    if success:
        return {"code": 200, "message": "消息已删除"}
    return {"code": 404, "message": "消息未找到"}


@router.delete("/messages")
async def clear_messages(session_id: str):
    """
    清空指定会话的所有消息。
    """
    _validate_session_id(session_id)
    success = memory_manager.clear_messages(session_id)
    if success:
        return {"code": 200, "message": "消息已清空"}
    return {"code": 404, "message": "会话不存在"}


@router.get("/tokens")
async def get_tokens():
    """返回全局 token 使用统计，供前端状态栏展示。"""
    try:
        stats = get_system_stats()
        return {"code": 200, "data": stats}
    except Exception as e:
        logger.error(f"[Tokens] 获取统计失败: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 500, "message": f"token统计获取失败: {str(e)}"})
