import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        max_tokens = rag_conf.get('max_tokens', 8192)
        return ChatOpenAI(
            model=rag_conf['chat_model_name'],
            api_key=os.environ.get("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=120,
            max_retries=3,
            max_tokens=max_tokens,
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
    if not os.environ.get("MINIMAX_API_KEY"):
        missing.append("MINIMAX_API_KEY")
    if missing:
        raise EnvironmentError(
            f"缺少必需的环境变量：{', '.join(missing)}。"
            f"请在 .env 文件中配置后重启服务。"
        )


try:
    _validate_env()
    chat_model = ChatModelFactory().generate()
    embed_model = EmbeddingsFactory().generate()
    vision_model = VisionModelFactory().generate()
except Exception as _exc:
    import logging as _logging
    _logging.getLogger("agent").critical(f"[模型初始化失败] {_exc}")
    raise
