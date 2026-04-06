import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from api.routers import knowledge, chat, auth, exam_export
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from utils.memory_service import memory_manager

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


@app.get("/", summary="主页 - 跳转到 React 登录页")
async def index_ui():
    # 重定向到 React 开发服务器
    return RedirectResponse(url="http://127.0.0.1:3000", status_code=307)


@app.get("/app", summary="打开复习引擎主界面")
async def app_ui():
    return FileResponse("static/index.html")


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
