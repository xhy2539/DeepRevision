import uvicorn
import os
import sqlite3
import asyncio
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from api.routers import knowledge, chat, auth, exam_export
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from utils.memory_service import memory_manager
from utils.logger_handler import logger
from utils.config_handler import chroma_conf

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="期末复习助手 API",
    description="基于高精度RAG与ReAct的期末复习问答系统",
    version="1.0.0"
)

# 挂载路由
app.include_router(auth.router, prefix="/api/auth", tags=["用户认证"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识库管理"])
app.include_router(chat.router, prefix="/api/chat", tags=["聊天问答"])
app.include_router(exam_export.router, prefix="/api/exam", tags=["试卷导出"])

# 挂载静态资源和客户端主页
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/data/temp", StaticFiles(directory=os.path.join(BASE_DIR, "data", "temp")), name="temp")


_ALLOWED_KNOWLEDGE_EXT = {".txt", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _session_has_courseware(session_id: str) -> bool:
    sid = str(session_id or "").strip()
    if not sid:
        return False
    session_dir = os.path.join(BASE_DIR, chroma_conf.get("data_path", "data"), sid)
    if not os.path.isdir(session_dir):
        return False
    for fname in os.listdir(session_dir):
        if fname.startswith("."):
            continue
        fpath = os.path.join(session_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fname)[1].lower() in _ALLOWED_KNOWLEDGE_EXT:
            return True
    return False


def _get_recent_session_ids(limit: int = 20) -> list[str]:
    """
    按最近消息时间排序会话，优先认为最新会话是“当前活跃会话”。
    """
    db_path = os.path.join(BASE_DIR, "data", "sessions.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT session_id, MAX(COALESCE(timestamp, 0)) AS ts
            FROM messages
            GROUP BY session_id
            ORDER BY ts DESC
            LIMIT ?
            """,
            (max(1, int(limit or 20)),),
        ).fetchall()
        return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
    except Exception as e:
        logger.warning(f"[Startup Warmup] 读取最近会话失败: {e}")
        return []
    finally:
        conn.close()


async def _prewarm_sessions_background(session_ids: list[str], delay_seconds: float = 0.6):
    """启动后异步预热其余会话，不阻塞首屏。"""
    if not session_ids:
        return
    try:
        from agent.multi_agent.quiz_agent import prewarm_courseware_summary_for_session
        await asyncio.sleep(max(0.0, float(delay_seconds or 0.0)))
        warmed = 0
        for sid in session_ids:
            if not sid:
                continue
            if sid == "default" and not _session_has_courseware(sid):
                logger.info("[Startup Warmup] 异步预热跳过 default（无课件）")
                continue
            if not _session_has_courseware(sid):
                logger.info(f"[Startup Warmup] 异步预热跳过 {sid}（无课件）")
                continue
            try:
                await prewarm_courseware_summary_for_session(sid, force=False)
                warmed += 1
            except Exception as e:
                logger.warning(f"[Startup Warmup] 异步预热失败 session={sid}: {e}")
        logger.info(f"[Startup Warmup] 异步预热完成 warmed={warmed}, total={len(session_ids)}")
    except Exception as e:
        logger.warning(f"[Startup Warmup] 异步预热任务异常: {e}")


@app.on_event("startup")
async def prewarm_courseware_summary():
    """
    启动预热：
    为已有会话构建课件摘要与考点池缓存，提升随机练习首题命中质量。
    """
    try:
        from agent.multi_agent.quiz_agent import prewarm_courseware_summary_for_session

        sessions = memory_manager.get_all_sessions() or []
        known_ids = [str((item or {}).get("id", "")).strip() for item in sessions]
        known_ids = [sid for sid in known_ids if sid]
        recent_ids = _get_recent_session_ids(limit=24)

        ordered_ids: list[str] = []
        seen = set()
        for sid in recent_ids + known_ids:
            if sid and sid not in seen:
                ordered_ids.append(sid)
                seen.add(sid)
        if "default" not in seen:
            ordered_ids.append("default")

        # 1) 启动时只预热“当前活跃会话”（最近消息且有课件）
        active_sid = ""
        for sid in ordered_ids:
            if sid == "default" and not _session_has_courseware(sid):
                continue
            if _session_has_courseware(sid):
                active_sid = sid
                break

        if active_sid:
            try:
                await prewarm_courseware_summary_for_session(active_sid, force=False)
                logger.info(f"[Startup Warmup] 同步预热完成 active_session={active_sid}")
            except Exception as e:
                logger.warning(f"[Startup Warmup] 同步预热失败 session={active_sid}: {e}")
        else:
            logger.info("[Startup Warmup] 未发现可预热课件会话，同步预热跳过")

        # 2) 其余会话异步预热（不阻塞启动）
        remain_ids = [sid for sid in ordered_ids if sid != active_sid][:12]
        if remain_ids:
            asyncio.create_task(_prewarm_sessions_background(remain_ids, delay_seconds=0.8))
            logger.info(f"[Startup Warmup] 已派发异步预热任务 sessions={len(remain_ids)}")
    except Exception as e:
        logger.warning(f"[Startup Warmup] 跳过: {e}")


@app.get("/", summary="主页 - 跳转到 React 登录页")
async def index_ui():
    # 重定向到 React 开发服务器
    return RedirectResponse(url="http://127.0.0.1:3000", status_code=307)


@app.get("/app", summary="打开复习引擎主界面")
async def app_ui():
    # 统一跳转到 Next 前端入口，避免命中旧 static/index.html 导致 _next 资源 404
    return RedirectResponse(url="http://127.0.0.1:3000/chat", status_code=307)


@app.get("/health", summary="健康检查")
async def health_check():
    try:
        sessions = memory_manager.get_all_sessions()
        vector_status = "unknown"
        vector_collections = 0
        try:
            chroma_path = os.path.join(os.getcwd(), "data", "chroma_db")
            if os.path.exists(chroma_path):
                vector_status = "ready"
                vector_collections = len(os.listdir(chroma_path)) if os.path.isdir(chroma_path) else 0
        except Exception:
            vector_status = "error"

        return {
            "status": "healthy",
            "sessions_count": len(sessions),
            "vector_store": vector_status,
            "vector_collections": vector_collections,
            "model": "dashscope"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "unhealthy",
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    # uvicorn api.main:app --reload
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
