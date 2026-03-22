from datetime import datetime
import logging
import os
import threading
import time
import asyncio
from functools import wraps

from utils.path_tool import get_abs_path

# 日志保存根目录
LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式配置
DEFAULT_LOGGING_FORMAT = logging.Formatter(
    "%(asctime)s-%(name)s-%(levelname)s-%(filename)s:%(lineno)d-%(message)s"
)


def get_logger(name: str = "agent",
               console_level: int = logging.INFO,
               file_level: int = logging.DEBUG,
               log_file=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(console_handler)
    # 文件 handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(file_handler)
    return logger


# 快捷获取日志管理器
logger = get_logger()

# ================= AOP 装饰器监控与 Token 统计池 =================

# 全局 Token 消耗黑板
token_stats = {
    "total_calls": 0,
    "total_cost_time": 0.0,
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}

# 保护 token_stats 的线程锁（fix #14）
_stats_lock = threading.Lock()


def get_system_stats():
    return token_stats


def update_token_stats(usage_dict):
    """供外界异步流或回调主动注入 LLM 真实账单"""
    if not usage_dict:
        return
    with _stats_lock:
        token_stats["total_tokens"] += usage_dict.get("total_tokens", 0)
        token_stats["prompt_tokens"] += usage_dict.get("prompt_tokens", 0)
        token_stats["completion_tokens"] += usage_dict.get("completion_tokens", 0)


def timer_and_token_logger(func):
    """
    监控工具执行时间和基本信息的装饰器，同时支持同步和异步函数（fix #3）。
    """
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"【AOP监控】开始执行工具: {func.__name__}，参数: args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                cost_time = time.time() - start_time
                with _stats_lock:
                    token_stats["total_calls"] += 1
                    token_stats["total_cost_time"] += cost_time
                logger.info(f"【AOP监控】工具 {func.__name__} 执行成功。耗时: {cost_time:.2f} 秒。")
                return result
            except Exception as e:
                cost_time = time.time() - start_time
                logger.error(f"【AOP监控】工具 {func.__name__} 执行抛出异常: {e}。终止前耗时: {cost_time:.2f} 秒。")
                raise
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"【AOP监控】开始执行工具: {func.__name__}，参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                cost_time = time.time() - start_time
                with _stats_lock:
                    token_stats["total_calls"] += 1
                    token_stats["total_cost_time"] += cost_time
                logger.info(f"【AOP监控】工具 {func.__name__} 执行成功。耗时: {cost_time:.2f} 秒。")
                return result
            except Exception as e:
                cost_time = time.time() - start_time
                logger.error(f"【AOP监控】工具 {func.__name__} 执行抛出异常: {e}。终止前耗时: {cost_time:.2f} 秒。")
                raise
        return sync_wrapper
