import os
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

# 禁用代理，避免 VPN/代理软件干扰
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _load_project_env() -> None:
    """
    读取项目根目录 .env 并注入进程环境变量（仅填充当前未设置的键）。
    避免必须依赖 uvicorn --env-file 或系统级环境变量。
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as e:
        logger.warning(f"[模型工厂] 读取 .env 失败: {e}")


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        max_tokens = rag_conf.get('max_tokens', 8192)
        model = ChatOpenAI(
            model=rag_conf['chat_model_name'],
            api_key=os.environ.get("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=60,
            max_retries=0,  # 关闭内部重试，由上层 call_llm_structured 统一处理
            max_tokens=max_tokens,
        )
        # MiniMax 禁用 extended thinking，避免思考过程混入输出
        try:
            model.extra_body = {"thinking": {"type": "off"}}
        except Exception:
            pass
        # MiniMax 支持 JSON mode
        try:
            model.response_format = {"type": "json_object"}
        except Exception:
            pass
        return model


class LightChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        light_name = rag_conf.get('light_model_name')
        if not light_name:
            light_name = rag_conf.get('chat_model_name')
            logger.warning("[模型工厂] light_model_name 未配置，轻量模型回退到 chat_model_name")
        model = ChatOpenAI(
            model=light_name,
            api_key=os.environ.get("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=60,
            max_retries=0,  # 关闭内部重试，由上层统一处理
            max_tokens=2048,
        )
        # MiniMax 禁用 extended thinking
        try:
            model.extra_body = {"thinking": {"type": "off"}}
        except Exception:
            pass
        # MiniMax 支持 JSON mode
        try:
            model.response_format = {"type": "json_object"}
        except Exception:
            pass
        return model


class BackupChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        model_name = os.environ.get("QWEN_CHAT_MODEL", "qwen-plus")
        max_tokens = rag_conf.get('max_tokens', 8192)
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60,
            max_retries=0,
            max_tokens=max_tokens,
        )


class BackupLightChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        model_name = os.environ.get("QWEN_LIGHT_MODEL", "qwen-turbo")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=30,
            max_retries=0,
            max_tokens=2048,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])


class VisionModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        vision_model_name = rag_conf.get('vision_model_name', 'qwen-vl-max')
        # 千问模型使用 DashScope API (直接用 OpenAI 兼容接口)
        if 'qwen' in vision_model_name.lower():
            api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("MINIMAX_API_KEY")
            return ChatOpenAI(
                model=vision_model_name,
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=120,
                max_retries=3,
            )
        else:
            # MiniMax 模型
            return ChatOpenAI(
                model=vision_model_name,
                api_key=os.environ.get("MINIMAX_API_KEY"),
                base_url="https://api.minimax.chat/v1",
                timeout=120,
                max_retries=3,
            )


def _validate_env():
    """启动前校验必需的环境变量（fix #8）"""
    missing = []
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    qwen_key = os.environ.get("QWEN_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not minimax_key and not qwen_key:
        missing.append("MINIMAX_API_KEY/QWEN_API_KEY")
    if minimax_key:
        # 打印 API key 前缀用于调试（不打印完整 key）
        key_prefix = minimax_key[:8] if len(minimax_key) > 8 else "***"
        print(f"[模型工厂] MiniMax Key 前缀: {key_prefix}..., 长度: {len(minimax_key)}")
    if qwen_key:
        qwen_prefix = qwen_key[:8] if len(qwen_key) > 8 else "***"
        print(f"[模型工厂] Qwen Key 前缀: {qwen_prefix}..., 长度: {len(qwen_key)}")
    if missing:
        raise EnvironmentError(
            f"缺少必需的环境变量：{', '.join(missing)}。"
            f"请在 .env 文件中配置后重启服务。"
        )


try:
    _load_project_env()
    _validate_env()
    minimax_chat_model = ChatModelFactory().generate() if os.environ.get("MINIMAX_API_KEY") else None
    minimax_light_model = LightChatModelFactory().generate() if os.environ.get("MINIMAX_API_KEY") else None
    qwen_chat_model = BackupChatModelFactory().generate()
    qwen_light_model = BackupLightChatModelFactory().generate()

    preferred_provider = str(os.environ.get("PRIMARY_LLM_PROVIDER", "minimax") or "minimax").strip().lower()
    if preferred_provider not in {"minimax", "qwen"}:
        preferred_provider = "minimax"

    if preferred_provider == "qwen":
        chat_model = qwen_chat_model or minimax_chat_model
        light_chat_model = qwen_light_model or minimax_light_model
        backup_chat_model = minimax_chat_model if (chat_model is not minimax_chat_model) else None
        backup_light_chat_model = minimax_light_model if (light_chat_model is not minimax_light_model) else None
    else:
        chat_model = minimax_chat_model or qwen_chat_model
        light_chat_model = minimax_light_model or qwen_light_model
        backup_chat_model = qwen_chat_model if (chat_model is not qwen_chat_model) else None
        backup_light_chat_model = qwen_light_model if (light_chat_model is not qwen_light_model) else None

    logger.info(
        f"[模型工厂] 主模型提供方={preferred_provider}, "
        f"chat_model={'set' if chat_model else 'none'}, backup_chat_model={'set' if backup_chat_model else 'none'}"
    )
    embed_model = EmbeddingsFactory().generate()
    vision_model = VisionModelFactory().generate()
except Exception as _exc:
    import logging as _logging
    _logging.getLogger("agent").critical(f"[模型初始化失败] {_exc}")
    raise
