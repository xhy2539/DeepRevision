import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse

from api.routers import chat, exam_export, knowledge
from utils.config_handler import chroma_conf
from utils.memory_service import memory_manager
from utils.path_tool import get_abs_path
from utils.workflow_runtime import durable_workflow
from api.security import IdentityMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with durable_workflow() as workflow:
        app.state.supervisor_workflow = workflow
        yield


app = FastAPI(
    title="DeepRevision API",
    description="面向课程复习的课件检索、练习与组卷服务",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(IdentityMiddleware)

app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识库管理"])
app.include_router(chat.router, prefix="/api/chat", tags=["聊天问答"])
app.include_router(exam_export.router, prefix="/api/exam", tags=["试卷导出"])


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", "http://127.0.0.1:3000").rstrip("/")


@app.get("/", summary="打开前端")
async def index_ui():
    return RedirectResponse(url=f"{_frontend_url()}/chat", status_code=307)


@app.get("/app", summary="打开前端")
async def app_ui():
    return RedirectResponse(url=f"{_frontend_url()}/chat", status_code=307)


@app.get("/health", summary="健康检查")
async def health_check():
    try:
        sessions = memory_manager.get_all_sessions()
        chroma_path = get_abs_path(chroma_conf.get("persist_directory", "chroma_db"))
        vector_status = "ready" if os.path.isdir(chroma_path) else "empty"
        return {
            "status": "healthy",
            "sessions_count": len(sessions),
            "vector_store": vector_status,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(exc)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
