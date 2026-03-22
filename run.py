import uvicorn
import webbrowser
import threading
import time
import os
import sys
import subprocess
import signal
import atexit

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 设置 API Key
os.environ["MINIMAX_API_KEY"] = "sk-cp-MZK-3ZdmZ4YoWlT5mat91of7svArukPGxTbmXMnVg8OoWlYf8Rjn4Gru_RXROZs8li6kMD_hQC4ZYtkRFh1vdWT5_Oy0zLMFg8b-LlPgCKzRjwrLt9CP9Wk"

# 设置模型缓存路径，避免下载到 C 盘
models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["HF_HOME"] = models_path
os.environ["TRANSFORMERS_CACHE"] = models_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_path

# 保存前端进程对象
_frontend_process = None


def open_browser():
    """等待服务器启动后，自动打开默认浏览器访问系统"""
    time.sleep(6)
    url = "http://127.0.0.1:3000"
    print(f"\n🌐 正在自动打开浏览器: {url}")
    webbrowser.open(url)


def kill_port_3000():
    """杀掉占用 3000 端口的进程"""
    try:
        if sys.platform == "win32":
            # Windows: 使用 netstat 找到占用端口的进程并终止
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                shell=True
            )
            for line in result.stdout.split("\n"):
                if ":3000" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"🔪 终止占用 3000 端口的进程 PID: {pid}")
                        subprocess.run(["taskkill", "/F", "/PID", pid], shell=True)
                        break
        else:
            # Unix: 使用 lsof
            result = subprocess.run(
                ["lsof", "-ti:3000"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pid = result.stdout.strip()
                os.kill(int(pid), signal.SIGTERM)
    except Exception as e:
        pass  # 忽略错误，可能是没有占用端口


def cleanup():
    """程序退出时清理前端进程"""
    global _frontend_process
    if _frontend_process:
        try:
            print("\n🧹 正在终止前端服务...")
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(_frontend_process.pid)], shell=True)
            else:
                _frontend_process.terminate()
            print("✅ 前端服务已终止")
        except Exception as e:
            pass


def start_frontend():
    """启动 Next.js 前端服务"""
    global _frontend_process

    # 先检查并清理可能占用端口的进程
    kill_port_3000()
    time.sleep(1)

    react_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "react")
    print("\n📦 正在启动前端服务...")
    try:
        _frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=react_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        print("✅ 前端已启动: http://127.0.0.1:3000")
    except Exception as e:
        print(f"⚠️ 前端启动失败: {e}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 注册退出清理函数
    atexit.register(cleanup)

    # 处理 Ctrl+C 信号
    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

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
