import json
import asyncio
import re
import time
from fastapi import APIRouter, Body, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from agent.tools.agent_tools import clear_rag_cache
from agent.multi_agent.supervisor import supervisor_workflow
from api.message_protocol import build_assistant_message
from utils.logger_handler import logger, update_token_stats, get_system_stats
from utils.memory_service import memory_manager
from utils.session_context import current_session_id
from rag.vector_store import VectorStoreService

router = APIRouter()

# 轻量运行指标（进程内）
RUNTIME_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "failed_requests": 0,
    "routes": {"rag": 0, "quiz": 0, "exam": 0, "planner": 0, "history": 0, "chitchat": 0},
    "exam_total": 0,
    "exam_success": 0,
    "last_error": "",
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

        # 检测短行思考内容：包含多个逗号/分号、没有emoji、不是完整句子
        # 但列表项（以 -、*、1. 等开头）不视为思考内容
        if not is_thinking and 3 < len(stripped) < 100:
            comma_count = stripped.count('，') + stripped.count('、')
            has_emoji = any(e in stripped for e in ['👋', '😊', '👍', '😄', '🎯', '📚', '💡'])
            ends_with_punct = any(stripped.endswith(c) for c in '。！？.!?…')
            # 列表项格式不视为思考内容（即使有多个逗号）
            is_list_item = stripped.startswith(('- ', '* ', '1. ', '2. ', '3. ', '• ')) or re.match(r'^\d+\.（', stripped) or re.match(r'^[A-D][.、]', stripped)
            # 思考内容的特征：多逗号连接、没有emoji、不以标点结尾、不是列表项
            if comma_count >= 2 and not has_emoji and not ends_with_punct and not is_list_item:
                is_thinking = True

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
    try:
        RUNTIME_METRICS[metric_key] = int(RUNTIME_METRICS.get(metric_key, 0)) + delta
    except Exception:
        pass


@router.post("/stream")
async def chat_stream_endpoint(request: Request):
    """
    流式对话接口：Supervisor 多 Agent 路由 + 记忆组装与入库。
    """
    logger.info("=" * 50)
    logger.info(f"收到对话请求")
    body = await request.json()
    session_id = body.get("session_id", "default")
    query = body.get("query", "")
    logger.info(f"session_id={session_id}, query={query[:50]}...")

    # 安全校验
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$', session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")

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
            chat_history.append(AIMessage(content=msg["content"]))

    graph_context = memory_manager.get_memory_context(session_id)

    async def event_stream():
        _safe_inc("total_requests")
        initial_state = {
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

        logger.info("开始执行 Supervisor 工作流...")
        stream_start = time.time()
        answer = ""  # 初始化 answer 变量
        message = None
        try:
            # 先发送一个占位消息，避免长耗时任务期间前端完全空白
            pending_start = json.dumps(
                {'event': 'start', 'message': {'kind': 'chat', 'render_mode': 'markdown', 'meta': {'route': 'pending'}}},
                ensure_ascii=False,
            )
            pending_delta = json.dumps(
                {'event': 'delta', 'text': '正在生成内容，请稍候...\n'},
                ensure_ascii=False,
            )
            yield f"data: {pending_start}\n\n"
            yield f"data: {pending_delta}\n\n"

            result = await supervisor_workflow.ainvoke(initial_state)
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

            if answer:
                if structured_result is not None:
                    structured_result = {**structured_result, "text": answer}
                message = build_assistant_message(route, answer, session_id, route_params, structured_result)
                yield f"data: {json.dumps({'event': 'start', 'message': {'kind': message['kind'], 'render_mode': message['render_mode'], 'meta': message['meta']}}, ensure_ascii=False)}\n\n"

                if message["kind"] == "chat":
                    # 普通对话维持伪流式
                    for i, line in enumerate(answer.splitlines(keepends=True)):
                        logger.info(f"[Stream] line{i}: {line}")
                        yield f"data: {json.dumps({'event': 'delta', 'text': line}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.03)

                yield f"data: {json.dumps({'event': 'complete', 'message': message}, ensure_ascii=False)}\n\n"
                if route == "exam" and message.get("kind") == "exam_paper":
                    exam_data = (message.get("payload") or {}).get("exam_data", {})
                    exam_questions = exam_data.get("questions", []) if isinstance(exam_data, dict) else []
                    if isinstance(exam_questions, list) and len(exam_questions) > 0:
                        _safe_inc("exam_success")
                logger.info(
                    f"[Observability] session={session_id}, route={route}, reason={route_reason}, "
                    f"fallback={fallback_used}, kind={message['kind']}, latency_ms={int((time.time()-stream_start)*1000)}"
                )

        except Exception as e:
            logger.error(f"流式输出异常: {e}")
            _safe_inc("failed_requests")
            RUNTIME_METRICS["last_error"] = str(e)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        # 保存对话历史
        if answer:
            now = int(time.time() * 1000)
            await memory_manager.add_message(
                session_id,
                "user",
                query,
                timestamp=now,
            )
            await memory_manager.add_message(
                session_id,
                "ai",
                answer,
                kind=message["kind"] if message else "chat",
                render_mode=message["render_mode"] if message else "markdown",
                payload=message.get("payload") if message else None,
                meta=message.get("meta") if message else None,
                timestamp=now + 1,
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/practice/submit")
async def submit_practice_records(req: PracticeSubmitRequest = Body(...)):
    """
    提交练习记录（错题追踪主入口）。
    """
    session_id = req.session_id or "default"
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$', session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")

    if not req.records:
        return {"code": 200, "message": "无记录需要提交", "saved": 0}

    current_session_id.set(session_id)
    saved = 0
    for idx, record in enumerate(req.records):
        qid = record.question_id or f"{session_id}_{record.question_number or idx + 1}_{int(time.time() * 1000)}"
        memory_manager.add_practice_record(
            session_id=session_id,
            question_id=qid,
            question_content=record.question_content,
            knowledge_point=record.knowledge_point or "",
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
                knowledge_point=record.knowledge_point or "",
                answer=record.correct_answer,
            )
        except Exception as e:
            logger.warning(f"[Practice] 题库写入失败 qid={qid}: {e}")
        saved += 1

    stats = memory_manager.get_knowledge_point_stats(session_id)
    weak_points = [kp for kp, data in stats.items() if data.get("weak")]
    logger.info(f"[Practice] session={session_id}, saved={saved}, weak_points={weak_points}")
    return {
        "code": 200,
        "message": f"已保存 {saved} 条练习记录",
        "saved": saved,
        "weak_points": weak_points,
        "stats": stats,
    }


@router.post("/practice/similar")
async def get_similar_questions_batch(req: SimilarBatchRequest = Body(...)):
    """
    根据错题批量返回相似题，供前端复练模块直接渲染。
    """
    session_id = req.session_id or "default"
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$', session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")

    current_session_id.set(session_id)
    result: Dict[int, List[Dict[str, Any]]] = {}
    for item in req.wrong_questions:
        sims = memory_manager.search_similar_questions(
            question_content=item.question_content,
            knowledge_point=item.knowledge_point or "",
            limit=max(1, min(req.limit, 8)),
        )
        result[item.question_number] = sims

    return {"code": 200, "similar_questions": result}


@router.get("/practice/stats")
async def get_practice_stats(session_id: str):
    """
    获取练习统计（用于错题复练率看板）。
    """
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$', session_id):
        raise HTTPException(status_code=400, detail="非法 session_id")

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


@router.get("/metrics")
async def get_runtime_metrics():
    """
    进程内运行指标（出卷成功率、路由分布等）。
    """
    exam_total = int(RUNTIME_METRICS.get("exam_total", 0))
    exam_success = int(RUNTIME_METRICS.get("exam_success", 0))
    exam_success_rate = round((exam_success / exam_total) * 100, 2) if exam_total > 0 else 0.0
    return {
        "code": 200,
        "data": {
            **RUNTIME_METRICS,
            "exam_success_rate": exam_success_rate,
        },
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
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
    success = memory_manager.rename_session(session_id, req.new_name)
    if success:
        return {"code": 200, "message": "重命名成功"}
    return {"code": 404, "message": "该会话不存在"}


class SessionCreateRequest(BaseModel):
    session_id: str
    name: str
    parent_id: Optional[str] = None


@router.post("/session")
async def register_session(req: SessionCreateRequest = Body(...)):
    memory_manager.register_session(req.session_id, req.name, req.parent_id)
    return {"code": 200, "message": "会话注册成功"}


@router.get("/sessions")
async def get_all_sessions():
    sessions = memory_manager.get_all_sessions()
    return {"code": 200, "data": sessions}


@router.get("/messages")
async def get_session_messages(session_id: str):
    memory_manager._init_session(session_id)
    recent_records = memory_manager.store[session_id]["recent"]
    messages = []
    for msg in recent_records:
        messages.append({
            "id": f"{msg.get('timestamp', 0)}",
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", 0),
            "kind": msg.get("kind"),
            "render_mode": msg.get("render_mode"),
            "payload": msg.get("payload"),
            "meta": msg.get("meta"),
        })
    return {"code": 200, "data": messages}


@router.get("/tokens")
async def get_tokens():
    stats = get_system_stats()
    return {"code": 200, "data": stats}
