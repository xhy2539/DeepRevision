from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf
from typing import Optional
import os


class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self)->Optional[Embeddings|BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generate(self)->Optional[Embeddings|BaseChatModel]:
        max_tokens = rag_conf.get('max_tokens', 8192)
        return ChatOpenAI(
            model=rag_conf['chat_model_name'],
            api_key=os.environ.get("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=120,
            max_retries=3,
            max_tokens=max_tokens
        )


class EmbeddingsFactory(BaseModelFactory):
    def generate(self)->Optional[Embeddings|BaseChatModel]:
        # 保持使用千问 Embedding
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])


class VisionModelFactory(BaseModelFactory):
    def generate(self)->Optional[Embeddings|BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf['vision_model_name'],
            api_key=os.environ.get("MINIMAX_API_KEY"),
            base_url="https://api.minimax.chat/v1",
            timeout=120,
            max_retries=3
        )


chat_model=ChatModelFactory().generate()
embed_model=EmbeddingsFactory().generate()
vision_model=VisionModelFactory().generate()