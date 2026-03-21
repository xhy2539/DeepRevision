import uvicorn
import webbrowser
import threading
import time
import os
import sys
import subprocess

# 设置 API Key
os.environ["MINIMAX_API_KEY"] = "sk-cp-MZK-3ZdmZ4YoWlT5mat91of7svArukPGxTbmXMnVg8OoWlYf8Rjn4Gru_RXROZs8li6kMD_hQC4ZYtkRFh1vdWT5_Oy0zLMFg8b-LlPgCKzRjwrLt9CP9Wk"

# 设置模型缓存路径，避免下载到 C 盘
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["HF_HOME"] = models_path
os.environ["TRANSFORMERS_CACHE"] = models_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_path


def open_browser():
    """等待服务器启动后，自动打开默认浏览器访问系统"""
    time.sleep(6)
    url = "http://127.0.0.1:3000"
    print(f"\n🌐 正在自动打开浏览器: {url}")
    webbrowser.open(url)


def start_frontend():
    """启动 Next.js 前端服务"""
    react_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "react")
    print("\n📦 正在启动前端服务...")
    try:
        subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=react_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ 前端已启动: http://127.0.0.1:3000")
    except Exception as e:
        print(f"⚠️ 前端启动失败: {e}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("="*50)
    print("🚀 正在启动期末复习助手系统...")
    print("="*50)

    # 启动前端服务（后台运行）
    frontend_thread = threading.Thread(target=start_frontend, daemon=True)
    frontend_thread.start()

    # 等待前端启动后打开浏览器
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # 启动后端 Uvicorn 服务器
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
