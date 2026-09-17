import os
import httpx
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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


class DeepSeekChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-v4-pro"),
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=60,
            max_retries=0,
            max_tokens=rag_conf.get("max_tokens", 8192),
        )


class DeepSeekLightChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=os.environ.get("DEEPSEEK_LIGHT_MODEL", "deepseek-v4-flash"),
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=30,
            max_retries=0,
            max_tokens=2048,
        )


class TokenPlanChatModelFactory(BaseModelFactory):
    """Alibaba Cloud Model Studio Token Plan (text generation only)."""

    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("TOKEN_PLAN_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=os.environ.get("TOKEN_PLAN_CHAT_MODEL", "qwen3.8-max"),
            api_key=api_key,
            base_url=os.environ.get(
                "TOKEN_PLAN_BASE_URL",
                "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            ),
            timeout=60,
            max_retries=0,
            max_tokens=rag_conf.get("max_tokens", 8192),
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )


class TokenPlanLightChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        api_key = os.environ.get("TOKEN_PLAN_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=os.environ.get("TOKEN_PLAN_LIGHT_MODEL", "qwen3.8-flash"),
            api_key=api_key,
            base_url=os.environ.get(
                "TOKEN_PLAN_BASE_URL",
                "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            ),
            timeout=30,
            max_retries=0,
            max_tokens=2048,
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )


class BackupChatModelFactory(BaseModelFactory):
    def generate(self) -> Optional[BaseChatModel]:
        # QWEN_API_KEY explicitly enables Qwen chat. DASHSCOPE_API_KEY may be an
        # embedding-only workspace key and must not silently alter failover order.
        api_key = os.environ.get("QWEN_API_KEY")
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
        api_key = os.environ.get("QWEN_API_KEY")
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
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
        if not api_key:
            return None
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=api_key,
            base_url=rag_conf.get(
                "embedding_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            dimensions=int(rag_conf.get("embedding_dimensions", 1024)),
            check_embedding_ctx_length=False,
            chunk_size=int(rag_conf.get("embedding_batch_size", 20)),
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )


class VisionModelFactory(BaseModelFactory):
    def generate(self) -> Optional[Embeddings | BaseChatModel]:
        vision_model_name = rag_conf.get('vision_model_name', 'qwen-vl-max')
        # 千问模型使用 DashScope API (直接用 OpenAI 兼容接口)
        if 'qwen' in vision_model_name.lower():
            # Vision is opt-in and uses its own credential. An embedding-only
            # workspace key must never trigger potentially expensive image calls.
            api_key = os.environ.get("VISION_API_KEY")
            if not api_key:
                return None
            return ChatOpenAI(
                model=vision_model_name,
                api_key=api_key,
                base_url=os.environ.get(
                    "VISION_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                timeout=120,
                max_retries=3,
            )
        else:
            # MiniMax 模型
            if not os.environ.get("MINIMAX_API_KEY"):
                return None
            return ChatOpenAI(
                model=vision_model_name,
                api_key=os.environ.get("MINIMAX_API_KEY"),
                base_url="https://api.minimax.chat/v1",
                timeout=120,
                max_retries=3,
            )


def _validate_env():
    """启动前校验至少存在一个可用的聊天模型密钥。"""
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    qwen_key = os.environ.get("QWEN_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    token_plan_key = os.environ.get("TOKEN_PLAN_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not minimax_key and not qwen_key and not token_plan_key and not deepseek_key:
        raise EnvironmentError(
            "缺少聊天模型密钥：请配置 MINIMAX_API_KEY、TOKEN_PLAN_API_KEY、"
            "QWEN_API_KEY/DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY 后重启服务。"
        )


try:
    _load_project_env()
    _validate_env()
    minimax_chat_model = ChatModelFactory().generate() if os.environ.get("MINIMAX_API_KEY") else None
    minimax_light_model = LightChatModelFactory().generate() if os.environ.get("MINIMAX_API_KEY") else None
    token_plan_chat_model = TokenPlanChatModelFactory().generate()
    token_plan_light_model = TokenPlanLightChatModelFactory().generate()
    qwen_chat_model = BackupChatModelFactory().generate()
    qwen_light_model = BackupLightChatModelFactory().generate()
    deepseek_chat_model = DeepSeekChatModelFactory().generate()
    deepseek_light_model = DeepSeekLightChatModelFactory().generate()

    preferred_provider = str(os.environ.get("PRIMARY_LLM_PROVIDER", "minimax") or "minimax").strip().lower()
    if preferred_provider not in {"minimax", "token_plan", "qwen", "deepseek"}:
        preferred_provider = "minimax"

    providers = {
        "minimax": (minimax_chat_model, minimax_light_model),
        "token_plan": (token_plan_chat_model, token_plan_light_model),
        "qwen": (qwen_chat_model, qwen_light_model),
        "deepseek": (deepseek_chat_model, deepseek_light_model),
    }
    provider_order = [preferred_provider] + [name for name in providers if name != preferred_provider]
    chat_candidates = [(name, providers[name][0]) for name in provider_order if providers[name][0] is not None]
    light_candidates = [(name, providers[name][1]) for name in provider_order if providers[name][1] is not None]

    selected_provider, chat_model = chat_candidates[0]
    light_chat_model = light_candidates[0][1] if light_candidates else chat_model
    backup_chat_model = chat_candidates[1][1] if len(chat_candidates) > 1 else None
    backup_light_chat_model = light_candidates[1][1] if len(light_candidates) > 1 else None

    logger.info(
        f"[模型工厂] 主模型提供方={selected_provider}, "
        f"chat_model={'set' if chat_model else 'none'}, backup_chat_model={'set' if backup_chat_model else 'none'}"
    )
    embed_model = EmbeddingsFactory().generate()
    if embed_model is None:
        logger.warning("[模型工厂] 未配置 DashScope Embedding；普通聊天可用，课件上传与 RAG 暂不可用")
    vision_model = VisionModelFactory().generate()
except Exception as _exc:
    import logging as _logging
    _logging.getLogger("agent").critical(f"[模型初始化失败] {_exc}")
    raise
