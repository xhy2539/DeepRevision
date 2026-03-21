import os
import json
import hashlib
import time
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from utils.path_tool import get_abs_path

router = APIRouter()

# 用户数据存储目录
USERS_DIR = get_abs_path("data/users")

# 确保目录存在
os.makedirs(USERS_DIR, exist_ok=True)


def _hash_password(password: str) -> str:
    """简单哈希存储（生产环境应用 bcrypt 等）"""
    return hashlib.sha256(password.encode()).hexdigest()


def _get_user_file(username: str) -> str:
    """获取用户文件路径"""
    safe_name = username.replace("/", "").replace("\\", "")
    return os.path.join(USERS_DIR, f"{safe_name}.json")


def _user_exists(username: str) -> bool:
    """检查用户是否存在"""
    return os.path.exists(_get_user_file(username))


def _create_user(username: str, password: str) -> bool:
    """创建新用户"""
    if _user_exists(username):
        return False

    user_data = {
        "username": username,
        "password": _hash_password(password),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    user_file = _get_user_file(username)
    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)

    return True


def _verify_user(username: str, password: str) -> bool:
    """验证用户密码"""
    user_file = _get_user_file(username)
    if not os.path.exists(user_file):
        return False

    try:
        with open(user_file, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        return user_data.get("password") == _hash_password(password)
    except:
        return False


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(req: LoginRequest = Body(...)):
    """
    用户登录
    """
    if not _user_exists(req.username):
        return {"code": 401, "message": "用户不存在"}

    if not _verify_user(req.username, req.password):
        return {"code": 401, "message": "密码错误"}

    return {"code": 200, "message": "登录成功", "data": {"username": req.username}}


@router.post("/register")
async def register(req: RegisterRequest = Body(...)):
    """
    用户注册
    """
    if not req.username or not req.password:
        return {"code": 400, "message": "用户名和密码不能为空"}

    if len(req.username) < 2:
        return {"code": 400, "message": "用户名至少2个字符"}

    if len(req.password) < 4:
        return {"code": 400, "message": "密码至少4个字符"}

    if _user_exists(req.username):
        return {"code": 400, "message": "用户名已存在"}

    if _create_user(req.username, req.password):
        return {"code": 200, "message": "注册成功"}

    return {"code": 500, "message": "注册失败"}


@router.get("/check")
async def check_auth():
    """
    检查是否已登录（前端轮询用）
    """
    # 简化版：实际应该用 session/cookie
    return {"code": 200, "authenticated": False}
