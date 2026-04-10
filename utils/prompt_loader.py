from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def _load_prompt_from_config(key: str) -> str:
    """通用的提示词加载函数"""
    try:
        path = get_abs_path(prompts_conf[key])
        return open(path, "r", encoding="utf-8").read()
    except KeyError:
        logger.error(f"[prompt_loader] 配置中缺少 {key}")
        raise
    except Exception as e:
        logger.error(f"[prompt_loader] 加载提示词 {key} 出错: {e}")
        raise


# ============ 系统提示词 ============

def load_system_prompts():
    """加载主系统提示词"""
    return _load_prompt_from_config("main_prompt_path")


# ============ RAG提示词 ============

def load_rag_prompts():
    """加载RAG总结提示词"""
    return _load_prompt_from_config("rag_summarize_prompt_path")


# ============ 出题提示词（单题）============

def load_quiz_generate_structured():
    """加载单题结构化生成提示词"""
    return _load_prompt_from_config("quiz_generate_structured_path")


def load_quiz_generate_reasoning():
    """加载单题带推理链生成提示词"""
    return _load_prompt_from_config("quiz_generate_reasoning_path")


def load_quiz_generate_fallback():
    """加载文本回退生成提示词"""
    return _load_prompt_from_config("quiz_generate_fallback_path")


def load_quiz_critique():
    """加载单题评审提示词"""
    return _load_prompt_from_config("quiz_critique_path")


def load_quiz_revise():
    """加载单题修订提示词"""
    return _load_prompt_from_config("quiz_revise_path")


# ============ 试卷提示词（多题型）============

def load_exam_generate_single_type():
    """加载分批生成单题型试卷提示词"""
    return _load_prompt_from_config("exam_generate_single_type_path")


def load_exam_generate_structured():
    """加载结构化试卷生成提示词"""
    return _load_prompt_from_config("exam_generate_structured_path")


def load_exam_generate_reasoning():
    """加载带推理链试卷生成提示词"""
    return _load_prompt_from_config("exam_generate_reasoning_path")


def load_exam_critique():
    """加载试卷评审提示词"""
    return _load_prompt_from_config("exam_critique_path")


def load_exam_revise():
    """加载试卷修订提示词"""
    return _load_prompt_from_config("exam_revise_path")


def load_exam_contract():
    """加载统一主契约"""
    return _load_prompt_from_config("exam_contract_path")


# ============ 格式示例 ============

def load_format_examples():
    """加载格式示例"""
    return _load_prompt_from_config("format_examples_path")


# ============ 兼容旧版 ============

def load_report_prompts():
    """加载报告生成提示词（兼容旧版）"""
    return _load_prompt_from_config("report_prompt_path")


def load_quiz_prompts():
    """加载出题提示词（兼容旧版）"""
    return _load_prompt_from_config("quiz_prompt_path")


# ============ 便捷加载函数 ============

def load_all_exam_prompts() -> dict:
    """一次性加载所有试卷相关提示词"""
    return {
        "single_type": load_exam_generate_single_type(),
        "structured": load_exam_generate_structured(),
        "reasoning": load_exam_generate_reasoning(),
        "critique": load_exam_critique(),
        "revise": load_exam_revise(),
        "contract": load_exam_contract(),
    }


def load_all_quiz_prompts() -> dict:
    """一次性加载所有单题相关提示词"""
    return {
        "generate_structured": load_quiz_generate_structured(),
        "generate_reasoning": load_quiz_generate_reasoning(),
        "generate_fallback": load_quiz_generate_fallback(),
        "critique": load_quiz_critique(),
        "revise": load_quiz_revise(),
    }


if __name__ == '__main__':
    # 测试加载
    print("=== 加载所有提示词 ===")
    print("Exam prompts:", list(load_all_exam_prompts().keys()))
    print("Quiz prompts:", list(load_all_quiz_prompts().keys()))
    print("Contract:", load_exam_contract()[:100], "...")

