import json
import asyncio
from fastapi import APIRouter, Body, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import tools, clear_rag_cache
from utils.logger_handler import logger, update_token_stats
from utils.memory_service import memory_manager
from utils.session_context import current_session_id
from rag.vector_store import VectorStoreService

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

def init_agent():
    # 获取重写后的主系统提示词
    system_prompt = load_system_prompts()
    
    # 动态插入记忆图谱和近期滑窗插槽
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{memory_context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}")
    ])
    
    agent = create_tool_calling_agent(chat_model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

app_agent = init_agent()

@router.post("/stream")
async def chat_stream_endpoint(request: Request):
    """
    流式对话接口，包含记忆组装与对话截断入库。
    """
    body = await request.json()
    session_id = body.get("session_id", "default")
    query = body.get("query", "")

    # 注册当前的 Session ID 给线程内上下文，保证 RAG 检索时知道该查哪个集合
    current_session_id.set(session_id)
    
    # 1. 向模型拉取近期记录转为LangChain格式
    memory_manager._init_session(session_id)
    recent_records = memory_manager.store[session_id]["recent"]
    chat_history = []
    for msg in recent_records:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
            
    # 2. 提取长期图谱记忆
    graph_context = memory_manager.get_memory_context(session_id)

    async def event_stream():
        full_response = ""

        try:
            # 3. 把用户最新的话和所有记忆投入执行器
            async for event in app_agent.astream_events({
                "input": query,
                "chat_history": chat_history,
                "memory_context": graph_context
            }, version="v1"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'text': content}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_start":
                    tool_name = event['name']
                    msg = f"\n**[系统思考：正在使用 {tool_name} 查阅资料...]**\n"
                    yield f"data: {json.dumps({'text': msg}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_end":
                    tool_name = event['name']
                    # 获取工具输出内容
                    tool_output = event.get('data', {}).get('output', '')
                    if tool_output:
                        # 输出工具返回的内容
                        yield f"data: {json.dumps({'text': tool_output}, ensure_ascii=False)}\n\n"
                    msg = f"\n**[系统思考：{tool_name} 执行完成]**\n"
                    yield f"data: {json.dumps({'text': msg}, ensure_ascii=False)}\n\n"

                elif kind == "on_chat_model_end":
                    # 尝试从模型输出中提取 Token 元数据
                    try:
                        output = event["data"].get("output")
                        usage = None

                        # 尝试多种可能的格式
                        if hasattr(output, "usage_metadata") and output.usage_metadata:
                            usage = output.usage_metadata
                        elif hasattr(output, "response_metadata"):
                            # MiniMax/通义千问格式
                            rm = output.response_metadata
                            if isinstance(rm, dict):
                                usage = rm.get("token_usage") or rm.get("usage") or rm.get("prompt_tokens")
                                # MiniMax 可能直接返回嵌套的 usage 对象
                                if isinstance(usage, dict):
                                    pass  # 保持原样

                        if usage:
                            # 兼容不同格式
                            if isinstance(usage, dict):
                                token_info = {
                                    "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_token_usage", 0),
                                    "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("completion_token_usage", 0),
                                    "total_tokens": usage.get("total_tokens") or usage.get("total") or usage.get("total_usage", 0)
                                }
                            elif isinstance(usage, (int, float)):
                                # 直接返回 total 的情况
                                token_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": usage}
                            else:
                                token_info = None

                            if token_info and token_info["total_tokens"] > 0:
                                update_token_stats(token_info)
                                logger.info(f"【算力监控】拦截到 Token 消耗: {token_info}")
                    except Exception as e:
                        logger.error(f"提取 Token 统计失败: {e}")

                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"流式输出异常: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        try:
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"发送结束信号失败: {e}")
        
        # 4. 对话结束后，把一来一回塞入管理器的管道。
        # 内部超过阈值会自动把老记录打入垃圾桶提取图谱晶体
        memory_manager.add_message(session_id, "user", query)
        memory_manager.add_message(session_id, "ai", full_response)
            
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
        vs.destroy_knowledge_base()
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

from utils.logger_handler import get_system_stats

@router.get("/tokens")
async def get_tokens():
    """
    获取系统算力与耗时统计监控板
    """
    stats = get_system_stats()
    return {"code": 200, "data": stats}
