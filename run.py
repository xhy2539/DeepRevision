"""Start the backend and frontend for a local demo."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import subprocess
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
REACT_DIR = PROJECT_ROOT / "react"
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8001"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

_frontend_process: subprocess.Popen | None = None


def stop_frontend() -> None:
    """Stop only the frontend process started by this script."""
    if _frontend_process is None or _frontend_process.poll() is not None:
        return
    _frontend_process.terminate()
    try:
        _frontend_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _frontend_process.kill()


def start_frontend() -> subprocess.Popen:
    if not (REACT_DIR / "node_modules").is_dir():
        raise RuntimeError("Frontend dependencies are missing; run `cd react && npm install` first.")

    command = "npm.cmd" if sys.platform == "win32" else "npm"
    env = os.environ.copy()
    env.setdefault("BACKEND_API_BASE", f"http://127.0.0.1:{BACKEND_PORT}")
    return subprocess.Popen(
        [command, "run", "dev", "--", "--port", str(FRONTEND_PORT)],
        cwd=REACT_DIR,
        env=env,
    )


def main() -> None:
    global _frontend_process

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("FRONTEND_URL", f"http://127.0.0.1:{FRONTEND_PORT}")
    _frontend_process = start_frontend()
    atexit.register(stop_frontend)
    print(f"Frontend: http://127.0.0.1:{FRONTEND_PORT}/chat")
    print(f"Backend docs: http://127.0.0.1:{BACKEND_PORT}/docs")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
