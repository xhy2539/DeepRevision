from typing import Optional

from utils.logger_handler import logger
from utils.prompt_loader import (
    load_exam_contract,
    load_exam_critique,
    load_exam_generate_reasoning,
    load_exam_generate_single_type,
    load_exam_generate_structured,
    load_exam_revise,
    load_format_examples,
    load_quiz_critique,
    load_quiz_generate_fallback,
    load_quiz_generate_reasoning,
    load_quiz_generate_structured,
    load_quiz_revise,
)

EXAM_GENERATE_SINGLE_TYPE_PROMPT: Optional[str] = None
EXAM_GENERATE_STRUCTURED_PROMPT: Optional[str] = None
GENERATE_STRUCTURED_QUIZ_PROMPT: Optional[str] = None
GENERATE_WITH_REASONING_PROMPT: Optional[str] = None
CRITIQUE_PROMPT: Optional[str] = None
REVISE_WITH_REFLECTION_PROMPT: Optional[str] = None
GENERATE_TEXT_FALLBACK_PROMPT: Optional[str] = None
EXAM_PROMPT_CONTRACT: Optional[str] = None
EXAM_GENERATE_WITH_REASONING_PROMPT: Optional[str] = None
EXAM_CRITIQUE_PROMPT: Optional[str] = None
EXAM_REVISE_WITH_REFLECTION_PROMPT: Optional[str] = None
FORMAT_EXAMPLES_PROMPT: Optional[str] = None
_PROMPTS_FROM_CONFIG_LOADED = False


def _load_prompts_from_config_once(force: bool = False) -> None:
    global EXAM_GENERATE_SINGLE_TYPE_PROMPT, EXAM_GENERATE_STRUCTURED_PROMPT
    global GENERATE_STRUCTURED_QUIZ_PROMPT, GENERATE_WITH_REASONING_PROMPT
    global CRITIQUE_PROMPT, REVISE_WITH_REFLECTION_PROMPT
    global GENERATE_TEXT_FALLBACK_PROMPT, EXAM_PROMPT_CONTRACT
    global EXAM_GENERATE_WITH_REASONING_PROMPT, EXAM_CRITIQUE_PROMPT
    global EXAM_REVISE_WITH_REFLECTION_PROMPT, FORMAT_EXAMPLES_PROMPT
    global _PROMPTS_FROM_CONFIG_LOADED

    if _PROMPTS_FROM_CONFIG_LOADED and not force:
        return

    try:
        EXAM_GENERATE_SINGLE_TYPE_PROMPT = load_exam_generate_single_type()
        EXAM_GENERATE_STRUCTURED_PROMPT = load_exam_generate_structured()
        GENERATE_STRUCTURED_QUIZ_PROMPT = load_quiz_generate_structured()
        GENERATE_WITH_REASONING_PROMPT = load_quiz_generate_reasoning()
        CRITIQUE_PROMPT = load_quiz_critique()
        REVISE_WITH_REFLECTION_PROMPT = load_quiz_revise()
        GENERATE_TEXT_FALLBACK_PROMPT = load_quiz_generate_fallback()
        EXAM_PROMPT_CONTRACT = load_exam_contract()
        EXAM_GENERATE_WITH_REASONING_PROMPT = load_exam_generate_reasoning()
        EXAM_CRITIQUE_PROMPT = load_exam_critique()
        EXAM_REVISE_WITH_REFLECTION_PROMPT = load_exam_revise()
        FORMAT_EXAMPLES_PROMPT = load_format_examples()
        _PROMPTS_FROM_CONFIG_LOADED = True
    except Exception as e:
        logger.error(f"[PromptConfig] 从配置加载提示词失败: {e}")
        raise RuntimeError("提示词加载失败：请检查 prompts.yml 与 prompts/*.txt 配置") from e


def get_exam_generate_single_type_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_GENERATE_SINGLE_TYPE_PROMPT or ""


def get_exam_generate_structured_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_GENERATE_STRUCTURED_PROMPT or ""


def get_quiz_generate_structured_prompt() -> str:
    _load_prompts_from_config_once()
    return GENERATE_STRUCTURED_QUIZ_PROMPT or ""


def get_quiz_generate_reasoning_prompt() -> str:
    _load_prompts_from_config_once()
    return GENERATE_WITH_REASONING_PROMPT or ""


def get_quiz_critique_prompt() -> str:
    _load_prompts_from_config_once()
    return CRITIQUE_PROMPT or ""


def get_quiz_revise_prompt() -> str:
    _load_prompts_from_config_once()
    return REVISE_WITH_REFLECTION_PROMPT or ""


def get_quiz_generate_fallback_prompt() -> str:
    _load_prompts_from_config_once()
    return GENERATE_TEXT_FALLBACK_PROMPT or ""


def get_exam_contract_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_PROMPT_CONTRACT or ""


def get_exam_generate_reasoning_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_GENERATE_WITH_REASONING_PROMPT or ""


def get_exam_critique_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_CRITIQUE_PROMPT or ""


def get_exam_revise_prompt() -> str:
    _load_prompts_from_config_once()
    return EXAM_REVISE_WITH_REFLECTION_PROMPT or ""


def get_format_examples_prompt() -> str:
    _load_prompts_from_config_once()
    return FORMAT_EXAMPLES_PROMPT or ""

