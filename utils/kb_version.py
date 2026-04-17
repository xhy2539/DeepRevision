import json
import os
import time
from contextlib import contextmanager


_KB_VERSION_FILE = ".kb_version.json"
_KB_VERSION_LOCK_FILE = ".kb_version.lock"


def _version_file_path(session_data_dir: str) -> str:
    """返回会话知识库版本文件路径。"""
    return os.path.join(str(session_data_dir or "").strip(), _KB_VERSION_FILE)


def _lock_file_path(session_data_dir: str) -> str:
    """返回会话知识库版本锁文件路径。"""
    return os.path.join(str(session_data_dir or "").strip(), _KB_VERSION_LOCK_FILE)


def _read_version_file(version_file: str) -> int:
    """读取版本文件的 version 字段。"""
    if not os.path.exists(version_file):
        return 0
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        return max(0, int(payload.get("version", 0) or 0))
    except Exception:
        return 0


def _acquire_file_lock(lock_fp) -> None:
    """加独占锁（兼容 Windows/Linux）。"""
    if os.name == "nt":
        import msvcrt

        lock_fp.seek(0)
        if os.fstat(lock_fp.fileno()).st_size == 0:
            lock_fp.write(b"\0")
            lock_fp.flush()
        lock_fp.seek(0)
        msvcrt.locking(lock_fp.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)


def _release_file_lock(lock_fp) -> None:
    """释放独占锁（兼容 Windows/Linux）。"""
    if os.name == "nt":
        import msvcrt

        lock_fp.seek(0)
        msvcrt.locking(lock_fp.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_version(session_data_dir: str):
    """按会话目录加文件锁，保护版本号读改写。"""
    lock_path = _lock_file_path(session_data_dir)
    with open(lock_path, "a+b") as lock_fp:
        _acquire_file_lock(lock_fp)
        try:
            yield
        finally:
            _release_file_lock(lock_fp)


def read_kb_version(session_data_dir: str) -> int:
    """读取会话知识库版本号；异常或缺失时返回 0。"""
    return _read_version_file(_version_file_path(session_data_dir))


def bump_kb_version(session_data_dir: str, reason: str = "") -> int:
    """递增并持久化会话知识库版本号，返回递增后的版本号。"""
    session_dir = str(session_data_dir or "").strip()
    os.makedirs(session_dir, exist_ok=True)

    with _locked_version(session_dir):
        version_file = _version_file_path(session_dir)
        current = _read_version_file(version_file)
        next_version = current + 1
        payload = {
            "version": next_version,
            "updated_at": int(time.time()),
            "reason": str(reason or "").strip(),
        }
        tmp_file = f"{version_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        # 原子替换，避免写入中断时产生半文件。
        os.replace(tmp_file, version_file)
        return next_version
