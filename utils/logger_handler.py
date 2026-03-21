from datetime import datetime
import logging
import os

from utils.path_tool import get_abs_path

#日志保存根目录
LOG_ROOT=get_abs_path("logs")

#确保根目录存在
os.makedirs(LOG_ROOT,exist_ok=True)

#日志格式配置
DEFAULT_LOGGING_FORMAT=logging.Formatter(
    "%(asctime)s-%(name)s-%(levelname)s-%(filename)s:%(lineno)d-%(message)s"
)

def get_logger(name:str="agent",
               console_level:int=logging.INFO,
               file_level:int=logging.DEBUG,
               log_file=None,)->logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    #避免重复添加handler
    if logger.handlers:
        return logger
#控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOGGING_FORMAT)

    logger.addHandler(console_handler)
#文件handler
    if not log_file:
        log_file=os.path.join(LOG_ROOT,f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOGGING_FORMAT)
    logger.addHandler(file_handler)

    return logger

#快捷获取日志管理器
logger=get_logger()

# ================= 新增：AOP 装饰器监控与 Token 统计池 =================
import time
from functools import wraps

# 全局 Token 消耗黑板
token_stats = {
    "total_calls": 0,
    "total_cost_time": 0.0,
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0
}

def get_system_stats():
    return token_stats

def update_token_stats(usage_dict):
    """供外界异步流或回调主动注入LLM真实账单"""
    if not usage_dict: return
    token_stats["total_tokens"] += usage_dict.get("total_tokens", 0)
    token_stats["prompt_tokens"] += usage_dict.get("prompt_tokens", 0)
    token_stats["completion_tokens"] += usage_dict.get("completion_tokens", 0)

def timer_and_token_logger(func):
    """
    监控工具执行时间和基本信息的装饰器。
    在真实生产环境中，可以在里面加入 get_openai_callback() 来统计 token。
    这里展示其 AOP(面向切面编程) 分离核心业务和辅助功能的思想。
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"【AOP监控】开始执行工具: {func.__name__}，参数: args={args}, kwargs={kwargs}")
        
        try:
            # 拿到原函数的执行结果
            result = func(*args, **kwargs)
            end_time = time.time()
            cost_time = end_time - start_time
            
            # 更新全局面板数据
            token_stats["total_calls"] += 1
            token_stats["total_cost_time"] += cost_time
            
            logger.info(f"【AOP监控】工具 {func.__name__} 执行成功。耗时: {cost_time:.2f} 秒。")
            return result
        except Exception as e:
            end_time = time.time()
            cost_time = end_time - start_time
            logger.error(f"【AOP监控】工具 {func.__name__} 执行抛出异常: {e}。终止前耗时: {cost_time:.2f} 秒。")
            raise e
            
    return wrapper