import asyncio
import random
from typing import Any, Callable, Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


STRUCTURED_FAIL_STATS: Dict[str, int] = {
    "provider_overloaded": 0,
    "request_timeout": 0,
    "invalid_json": 0,
    "schema_type_mismatch": 0,
    "other": 0,
}


def inc_structured_fail(reason: str):
    key = reason if reason in STRUCTURED_FAIL_STATS else "other"
    try:
        STRUCTURED_FAIL_STATS[key] = int(STRUCTURED_FAIL_STATS.get(key, 0)) + 1
    except Exception:
        pass


def classify_structured_error(error: Exception, raw_preview: str = "") -> str:
    if isinstance(error, TimeoutError):
        return "request_timeout"
    text = f"{str(error or '')} {str(raw_preview or '')}".lower()
    if "529" in text or "overloaded" in text or "current service cluster" in text:
        return "provider_overloaded"
    if "timeout" in text or "timed out" in text or "request timed out" in text:
        return "request_timeout"
    if "no valid json found" in text or ("json" in text and "decode" in text):
        return "invalid_json"
    if "validation" in text or "input should be" in text or "pydantic" in text or "string_type" in text:
        return "schema_type_mismatch"
    return "other"


def retry_sleep_seconds(reason: str, attempt: int) -> float:
    if reason == "provider_overloaded":
        base = min(20.0, 4.0 * (attempt + 1))
        return base + random.uniform(1.0, 3.0)
    if reason == "request_timeout":
        base = min(10.0, 2.0 * (attempt + 1))
        return base + random.uniform(0.5, 1.5)
    return 1.2 + random.uniform(0.0, 0.8)


def get_structured_fail_stats() -> Dict[str, int]:
    return {k: int(v) for k, v in STRUCTURED_FAIL_STATS.items()}


def should_retry_structured_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    error_str = str(error).lower()
    retryable_patterns = [
        "connection error",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "overloaded",
        "529",
        "server disconnected",
        "remote protocol error",
        "empty model output",
        "no valid json found",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def build_provider_chain(primary, backup):
    providers: List[Tuple[str, Any]] = []
    seen = set()
    for name, model in (("primary", primary), ("backup", backup)):
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        providers.append((name, model))
    return providers


async def call_llm(
    prompt_template: str,
    kwargs: dict,
    *,
    primary_model: Any,
    backup_model: Any,
    logger: Any,
) -> str:
    prompt = PromptTemplate.from_template(prompt_template)
    llm_timeout = 28.0
    max_retries = 1
    providers = build_provider_chain(primary_model, backup_model)
    if not providers:
        raise RuntimeError("未配置可用的聊天模型")

    last_error: Exception | None = None
    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        for attempt in range(max_retries + 1):
            try:
                return await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
            except Exception as e:
                last_error = e
                logger.warning(f"[call_llm] provider={provider_name} 第{attempt+1}次失败: {e}")
                if attempt >= max_retries or not should_retry_structured_error(e):
                    break
                await asyncio.sleep(1.0 + attempt * 1.0)
        if not should_retry_structured_error(last_error or Exception("unknown")):
            break
        if provider_name == "primary" and backup_model:
            logger.warning("[call_llm] 主模型异常，切换备用模型重试")

    if last_error:
        raise last_error
    raise RuntimeError("call_llm unknown error")


async def call_llm_structured(
    prompt_template: str,
    schema: type[BaseModel],
    kwargs: dict,
    *,
    primary_model: Any,
    backup_model: Any,
    logger: Any,
    extract_json_fn: Callable[[str], dict],
    coerce_payload_fn: Callable[[dict, type[BaseModel]], dict],
    extract_loose_fields_fn: Callable[[Any, type[BaseModel]], dict | None],
    summarize_raw_output_fn: Callable[[Any], str],
) -> BaseModel:
    prompt = PromptTemplate.from_template(prompt_template)
    last_error = None
    schema_name = getattr(schema, "__name__", "")
    timeout_map = {
        "StructuredQuizSetResult": 120.0,
        "QuizGenerateResult": 120.0,
        "ExamPaper": 150.0,
        "ExamPaperText": 150.0,
        "CritiqueResult": 20.0,
        "ReviseResult": 30.0,
        "ExamReviseResult": 45.0,
    }
    retry_map = {
        "StructuredQuizSetResult": 1,
        "QuizGenerateResult": 1,
        "CritiqueResult": 1,
        "ReviseResult": 1,
        "ExamPaper": 1,
        "ExamPaperText": 1,
        "ExamReviseResult": 1,
    }
    llm_timeout = timeout_map.get(schema_name, 35.0)
    max_retries = retry_map.get(schema_name, 1)
    overload_bonus_retry = 1
    providers = build_provider_chain(primary_model, backup_model)
    if not providers:
        raise RuntimeError("未配置可用的聊天模型")

    for provider_name, provider_model in providers:
        chain = prompt | provider_model | StrOutputParser()
        attempt = 0
        local_max_retries = max_retries
        local_overload_bonus_used = False
        while attempt <= local_max_retries:
            result = None
            try:
                result = await asyncio.wait_for(chain.ainvoke(kwargs), timeout=llm_timeout)
                parsed = extract_json_fn(result)
                parsed = coerce_payload_fn(parsed, schema)
                return schema(**parsed)
            except asyncio.TimeoutError:
                fail_reason = "request_timeout"
                last_error = asyncio.TimeoutError(f"LLM 调用超时（{llm_timeout}s）")
                inc_structured_fail(fail_reason)
                logger.warning(f"[call_llm_structured] provider={provider_name} 第{attempt+1}次失败: {last_error}")
                if attempt >= local_max_retries:
                    break
                await asyncio.sleep(retry_sleep_seconds(fail_reason, attempt))
                attempt += 1
            except Exception as e:
                last_error = e
                raw_preview = summarize_raw_output_fn(result)
                fail_reason = classify_structured_error(e, raw_preview)
                inc_structured_fail(fail_reason)
                recovered = extract_loose_fields_fn(result, schema) if result else None
                if recovered is not None:
                    logger.warning(f"[call_llm_structured] 宽松解析兜底成功(schema={getattr(schema, '__name__', '')})")
                    recovered = coerce_payload_fn(recovered, schema)
                    return schema(**recovered)
                logger.warning(f"[call_llm_structured] provider={provider_name} 第{attempt+1}次失败: {e}; raw='{raw_preview}...'")
                should_retry = should_retry_structured_error(e)
                if fail_reason == "provider_overloaded" and overload_bonus_retry > 0 and not local_overload_bonus_used:
                    local_overload_bonus_used = True
                    local_max_retries += overload_bonus_retry
                    logger.warning("[call_llm_structured] 检测到 provider_overloaded，已启用额外重试机会")
                if attempt >= local_max_retries or not should_retry:
                    break
                await asyncio.sleep(retry_sleep_seconds(fail_reason, attempt))
                attempt += 1
        if provider_name == "primary" and backup_model and should_retry_structured_error(last_error or Exception("unknown")):
            logger.warning("[call_llm_structured] 主模型异常，切换备用模型继续结构化生成")

    raise last_error

