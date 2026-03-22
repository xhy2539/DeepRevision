import json
import asyncio
import re
from fastapi import APIRouter, Body, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from agent.tools.agent_tools import clear_rag_cache
from agent.multi_agent.supervisor import supervisor_workflow
from utils.logger_handler import logger, update_token_stats, get_system_stats
from utils.memory_service import memory_manager
from utils.session_context import current_session_id
from rag.vector_store import VectorStoreService

router = APIRouter()

# Supervisor 子 Agent 进度提示语
_NODE_LABELS = {
    "rag_agent":     "知识库检索中...",
    "quiz_agent":    "生成题目（Reflexion 优化中）...",
    "exam_agent":    "生成试卷（Reflexion 优化中）...",
    "planner_agent": "制定复习计划...",
}

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"


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

    # 安全校验：session_id 路径穿越防护（允许中文）
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\-]{1,64}$', session_id):
        logger.warning(f"非法 session_id: {session_id}")
        raise HTTPException(status_code=400, detail="非法 session_id")

    # 输入校验：query 长度限制
    query = query[:2000]

    # 注册当前 Session ID，保证 RAG 检索走对应集合
    current_session_id.set(session_id)

    # 1. 近期滑窗记录 → LangChain 消息格式
    memory_manager._init_session(session_id)
    recent_records = memory_manager.store[session_id]["recent"]
    chat_history = []
    for msg in recent_records:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # 2. 长期图谱记忆摘要
    graph_context = memory_manager.get_memory_context(session_id)

    async def event_stream():
        full_response = ""
        shown_nodes: set = set()  # 避免重复显示进度提示

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
        try:
            async for event in supervisor_workflow.astream_events(initial_state, version="v2"):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")
                logger.debug(f"event: kind={kind}, node={node}")

                # Supervisor 节点的 LLM 输出是 JSON 路由决策，不暴露给用户
                if node == "supervisor":
                    continue

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'text': content}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_start" and node and node not in shown_nodes:
                    # 子 Agent 启动时显示一次进度提示
                    label = _NODE_LABELS.get(node, "")
                    if label:
                        shown_nodes.add(node)
                        msg = f"\n**[{label}]**\n"
                        yield f"data: {json.dumps({'text': msg}, ensure_ascii=False)}\n\n"

                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    # 处理 LangChain Message 对象
                    if hasattr(output, 'content'):
                        output = output.content
                    logger.info(f"[Chain End] node={node}, output_type={type(output).__name__}")
                    if node in ("quiz_agent", "exam_agent"):
                        # Quiz/Exam 子 Agent 不逐 token 流式输出，拿到完整结果后假流式输出
                        answer = ""
                        if isinstance(output, dict):
                            answer = output.get("final_answer", "") or output.get("revised_exam", "") or output.get("exam_paper", "") or output.get("subagent_result", "")
                        elif isinstance(output, str):
                            answer = output
                        elif hasattr(output, 'content'):
                            answer = str(output.content)
                        logger.info(f"[Exam Answer] length={len(answer) if answer else 0}")
                        # 直接输出，不管 full_response 是否已有值
                        if answer:
                            full_response = answer
                            logger.info(f"[SSE] 开始流式输出试卷，共 {len(answer.splitlines())} 行")
                            # 按行拆分逐行输出，保留 Markdown 结构
                            for line in answer.splitlines(keepends=True):
                                yield f"data: {json.dumps({'text': line}, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.03)
                            logger.info(f"[SSE] 试卷输出完成")

                elif kind == "on_chat_model_end":
                    # Token 消耗统计
                    try:
                        output = event["data"].get("output")
                        usage = None
                        if hasattr(output, "usage_metadata") and output.usage_metadata:
                            usage = output.usage_metadata
                        elif hasattr(output, "response_metadata"):
                            rm = output.response_metadata
                            if isinstance(rm, dict):
                                usage = rm.get("token_usage") or rm.get("usage") or rm.get("prompt_tokens")

                        if usage:
                            if isinstance(usage, dict):
                                token_info = {
                                    "prompt_tokens":      usage.get("input_tokens")      or usage.get("prompt_tokens")     or usage.get("prompt_token_usage", 0),
                                    "completion_tokens":  usage.get("output_tokens")     or usage.get("completion_tokens") or usage.get("completion_token_usage", 0),
                                    "total_tokens":       usage.get("total_tokens")      or usage.get("total")             or usage.get("total_usage", 0),
                                }
                            elif isinstance(usage, (int, float)):
                                token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": usage}
                            else:
                                token_info = None

                            if token_info and token_info["total_tokens"] > 0:
                                update_token_stats(token_info)
                                logger.info(f"【算力监控】Token 消耗: {token_info}")
                    except Exception as e:
                        logger.error(f"提取 Token 统计失败: {e}")

                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"流式输出异常: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        # 对话结束后入库；超阈值自动触发后台图谱提纯
        await memory_manager.add_message(session_id, "user", query)
        await memory_manager.add_message(session_id, "ai", full_response)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    手动生命周期管理：用户考完试或复习完后删除该课的记忆，同时销毁对应的矢量知识库。
    """
    # 绑定上下文以操作对应集合
    current_session_id.set(session_id)
    
    # 销毁知识库实体与物理文件
    try:
        vs = VectorStoreService()
        await vs.destroy_knowledge_base()  # fix #5：async def 必须 await
    except Exception as e:
        print(f"知识库清理失败: {e}")

    # 清理 RAG 缓存，防止内存泄漏
    clear_rag_cache(session_id)

    success = memory_manager.clear_session(session_id)
    if success:
        return {"code": 200, "message": f"科目会话 {session_id} 及其专属知识库已被永久销毁！"}
    return {"code": 404, "message": "该会话不存在"}

class RenameRequest(BaseModel):
    new_name: str

@router.put("/session/{session_id}")
async def rename_session(session_id: str, req: RenameRequest = Body(...)):
    """
    修改展示分类的名称
    """
    success = memory_manager.rename_session(session_id, req.new_name)
    if success:
        return {"code": 200, "message": "重命名成功"}
    return {"code": 404, "message": "该会话不存在"}

class SessionCreateRequest(BaseModel):
    session_id: str
    name: str

@router.post("/session")
async def register_session(req: SessionCreateRequest = Body(...)):
    """
    显式创建/注册一个新会话，防止切换后因无消息而丢失
    """
    memory_manager.register_session(req.session_id, req.name)
    return {"code": 200, "message": "会话注册成功"}

@router.get("/sessions")
async def get_all_sessions():
    """
    返回当前系统中存在的所有活跃会话。
    """
    sessions = memory_manager.get_all_sessions()
    return {"code": 200, "data": sessions}


@router.get("/messages")
async def get_session_messages(session_id: str):
    """
    获取指定会话的历史消息
    """
    memory_manager._init_session(session_id)
    recent_records = memory_manager.store[session_id]["recent"]

    messages = []
    for msg in recent_records:
        messages.append({
            "id": f"{msg.get('timestamp', 0)}",
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", 0)
        })

    return {"code": 200, "data": messages}


@router.get("/tokens")
async def get_tokens():
    """
    获取系统算力与耗时统计监控板
    """
    stats = get_system_stats()
    return {"code": 200, "data": stats}
