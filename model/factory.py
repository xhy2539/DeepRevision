import os
from abc import ABC, abstractmethod
from typing import Optional

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
            max_retries=1,
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
            max_retries=2,
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
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        missing.append("MINIMAX_API_KEY")
    else:
        # 打印 API key 前缀用于调试（不打印完整 key）
        key_prefix = api_key[:8] if len(api_key) > 8 else "***"
        print(f"[模型工厂] API Key 前缀: {key_prefix}..., 长度: {len(api_key)}")
    if missing:
        raise EnvironmentError(
            f"缺少必需的环境变量：{', '.join(missing)}。"
            f"请在 .env 文件中配置后重启服务。"
        )


try:
    _validate_env()
    chat_model = ChatModelFactory().generate()
    light_chat_model = LightChatModelFactory().generate()
    embed_model = EmbeddingsFactory().generate()
    vision_model = VisionModelFactory().generate()
except Exception as _exc:
    import logging as _logging
    _logging.getLogger("agent").critical(f"[模型初始化失败] {_exc}")
    raise
