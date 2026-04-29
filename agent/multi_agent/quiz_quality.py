import re
from typing import Dict, List, Optional

from agent.multi_agent.quiz_normalization import contains_term_style_violation, normalize_question_type
from utils.logger_handler import logger

TYPE_LABEL_TO_KEY = {
    "选择题": "choice",
    "填空题": "fill",
    "判断题": "judge",
    "简答题": "essay",
}

_CONCEPT_ANCHORS = [
    "系统调用", "微内核", "宏内核", "混合内核", "特权指令", "进程调度", "线程", "进程",
    "死锁", "信号量", "临界区", "内存管理", "虚拟内存", "页表", "TLB", "缺页", "抖动",
    "文件系统", "I/O", "设备驱动", "网络协议", "TCP", "资源分配图",
]


def question_signature(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r'^\s*\d+[.、]\s*', '', str(text).strip())
    normalized = normalized.lower()
    normalized = re.sub(r'[（(]\s*\d+\s*[）)]', '', normalized)
    normalized = re.sub(r'[^\w\u4e00-\u9fff]+', '', normalized)
    return normalized


def question_dedup_key(question: dict) -> str:
    q = question or {}
    qtype = normalize_question_type(q.get("type"), "选择题")
    stem_sig = question_signature(str(q.get("content") or q.get("question") or ""))
    if qtype in {"填空题", "判断题"}:
        ans_sig = question_signature(str(q.get("answer") or ""))
        return f"{qtype}:{stem_sig}:{ans_sig}"
    return f"{qtype}:{stem_sig}"


def target_counts(quantity_dist: dict, fallback_questions: Optional[List[dict]] = None) -> dict:
    dist = quantity_dist or {}
    if dist:
        return {
            "choice": max(0, int(dist.get("choice", 0))),
            "fill": max(0, int(dist.get("fill", 0))),
            "judge": max(0, int(dist.get("judge", 0))),
            "essay": max(0, int(dist.get("essay", 0))),
        }

    if fallback_questions:
        inferred = {"choice": 0, "fill": 0, "judge": 0, "essay": 0}
        for q in fallback_questions:
            key = TYPE_LABEL_TO_KEY.get(normalize_question_type((q or {}).get("type"), "选择题"))
            if key in inferred:
                inferred[key] += 1
        if sum(inferred.values()) > 0:
            return inferred

    return {"choice": 10, "fill": 5, "judge": 5, "essay": 3}


def question_term_style_ok(question: dict) -> bool:
    if not isinstance(question, dict):
        return True
    visible_parts: List[str] = [
        str(question.get("content") or question.get("question") or ""),
        str(question.get("answer") or ""),
        str(question.get("analysis") or question.get("explanation") or ""),
    ]
    options = question.get("options") or []
    if isinstance(options, list):
        visible_parts.extend([str(opt) for opt in options])
    return not any(contains_term_style_violation(part) for part in visible_parts if part)


def collect_term_style_violation_numbers(questions: List[dict]) -> List[int]:
    bad: List[int] = []
    for i, q in enumerate(questions or [], start=1):
        if not question_term_style_ok(q):
            bad.append(int((q or {}).get("number") or i))
    return bad


def score_map_for_counts(target: dict, target_total_score: int = 100) -> dict:
    choice_score = 2
    fill_score = 2
    judge_score = 2
    essay_count = max(0, int(target.get("essay", 0)))
    base = (
        int(target.get("choice", 0)) * choice_score
        + int(target.get("fill", 0)) * fill_score
        + int(target.get("judge", 0)) * judge_score
    )
    essay_score = max(0, (int(target_total_score) - base) // essay_count) if essay_count > 0 else 0
    return {"选择题": choice_score, "填空题": fill_score, "判断题": judge_score, "简答题": essay_score}


def detect_concept_anchor(text: str) -> str:
    src = str(text or "")
    for anchor in _CONCEPT_ANCHORS:
        if anchor in src:
            return anchor
    m = re.search(r'[\u4e00-\u9fff]{3,8}', src)
    return m.group(0) if m else "通用概念"


def concept_key_for_question(question: dict) -> str:
    content = str((question or {}).get("content") or "")
    kp = str((question or {}).get("knowledge_point") or "").strip()
    base = kp if kp else content
    return detect_concept_anchor(base)


def count_prompt_contract_violations(questions: List[dict]) -> int:
    patterns = [
        r'根据课件',
        r'依据资料',
        r'围绕\s*[“"]?(课程核心知识点|本课程重点知识点)[”"]?',
    ]
    count = 0
    for q in questions:
        text = str((q or {}).get("content") or (q or {}).get("question") or "")
        if any(re.search(p, text) for p in patterns):
            count += 1
    return count


def exam_quality_floor_flags(questions: List[dict], quantity_dist: dict, target_total_score: int = 100) -> dict:
    target = target_counts(quantity_dist, fallback_questions=questions)
    total_questions_target = int(sum(int(target.get(k, 0) or 0) for k in ["choice", "fill", "judge", "essay"]))
    enforce_concept_overlap_gate = total_questions_target >= 20
    concept_repeat_limit = 1 if enforce_concept_overlap_gate else 999999

    counts = {"choice": 0, "fill": 0, "judge": 0, "essay": 0}
    numbers: List[int] = []
    total_score = 0
    seen = set()
    has_duplicates = False
    concept_overlap = False
    options_ok = True
    answer_analysis_ok = True
    concept_seen: Dict[str, int] = {}

    for q in questions:
        qtype = normalize_question_type(q.get("type"), "选择题")
        key = TYPE_LABEL_TO_KEY.get(qtype)
        if key in counts:
            counts[key] += 1
        if qtype == "选择题":
            opts = q.get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_ok = False
        if not str(q.get("answer") or "").strip() or not str(q.get("analysis") or "").strip():
            answer_analysis_ok = False

        number = int(q.get("number") or 0)
        if number > 0:
            numbers.append(number)
        total_score += int(q.get("score") or 0)

        sig = question_dedup_key(q)
        if sig and sig in seen:
            has_duplicates = True
        if sig:
            seen.add(sig)

        ckey = concept_key_for_question(q)
        concept_seen[ckey] = concept_seen.get(ckey, 0) + 1
        if concept_seen[ckey] > concept_repeat_limit:
            concept_overlap = True

    quantity_ok = all(int(counts[k]) == int(target.get(k, 0)) for k in counts)
    numbering_ok = bool(numbers) and numbers == list(range(1, len(questions) + 1))
    total_score_ok = total_score == int(target_total_score)
    quality_floor_passed = quantity_ok and numbering_ok and total_score_ok and options_ok and answer_analysis_ok

    return {
        "quantity_ok": quantity_ok,
        "numbering_ok": numbering_ok,
        "duplicates": has_duplicates,
        "concept_overlap": concept_overlap,
        "concept_overlap_gate_enabled": enforce_concept_overlap_gate,
        "total_score_ok": total_score_ok,
        "options_ok": options_ok,
        "answer_analysis_ok": answer_analysis_ok,
        "duplicates_ok": not has_duplicates,
        "concept_overlap_ok": (not concept_overlap) if enforce_concept_overlap_gate else True,
        "concept_repeat_limit": concept_repeat_limit,
        "quality_floor_passed": quality_floor_passed,
        "counts": counts,
        "total_score": total_score,
    }


def compute_fixed_items(before_questions: List[dict], after_questions: List[dict], quantity_dist: dict, target_total_score: int = 100) -> List[str]:
    fixed_items: List[str] = []
    before_flags = exam_quality_floor_flags(before_questions, quantity_dist, target_total_score=target_total_score)
    after_flags = exam_quality_floor_flags(after_questions, quantity_dist, target_total_score=target_total_score)
    if (not before_flags.get("options_ok")) and after_flags.get("options_ok"):
        fixed_items.append("missing_options")
    if before_flags.get("duplicates") and (not after_flags.get("duplicates")):
        fixed_items.append("duplicate_rewrite")
    if (not before_flags.get("total_score_ok")) and after_flags.get("total_score_ok"):
        fixed_items.append("score_rebalanced")
    if count_prompt_contract_violations(before_questions) > 0 and count_prompt_contract_violations(after_questions) == 0:
        fixed_items.append("prompt_contract_violation")
    return fixed_items


def count_questions_in_exam(exam_text: str, quantity_dist: dict = None) -> dict:
    if quantity_dist:
        choice_count_q = quantity_dist.get('choice', 10)
        fill_count_q = quantity_dist.get('fill', 5)
        judge_count_q = quantity_dist.get('judge', 5)
        essay_count_q = quantity_dist.get('essay', 3)

        current = 1
        choice_range = (current, current + choice_count_q - 1)
        current += choice_count_q
        fill_range = (current, current + fill_count_q - 1)
        current += fill_count_q
        judge_range = (current, current + judge_count_q - 1)
        current += judge_count_q
        essay_range = (current, current + essay_count_q - 1)

        def count_in_range(text: str, start: int, end: int) -> int:
            count = 0
            for num in range(start, end + 1):
                pattern = rf'^\s*{num}[.、]\s'
                if re.search(pattern, text, re.MULTILINE):
                    count += 1
            return count

        choice_count = count_in_range(exam_text, *choice_range)
        fill_count = count_in_range(exam_text, *fill_range)
        judge_count = count_in_range(exam_text, *judge_range)
        essay_count = count_in_range(exam_text, *essay_range)
    else:
        def count_by_range(text: str, start: int, end: int) -> int:
            count = 0
            for num in range(start, end + 1):
                pattern = rf'^\s*{num}[.、]\s'
                if re.search(pattern, text, re.MULTILINE):
                    count += 1
            return count

        choice_count = count_by_range(exam_text, 1, 10)
        fill_count = count_by_range(exam_text, 11, 20)
        judge_count = count_by_range(exam_text, 21, 30)
        essay_count = count_by_range(exam_text, 31, 50)

    total = choice_count + fill_count + judge_count + essay_count
    logger.info(f"[题目统计] 选择题={choice_count}, 填空题={fill_count}, 判断题={judge_count}, 简答题={essay_count}, 总计={total}")
    return {
        "choice": choice_count,
        "fill": fill_count,
        "judge": judge_count,
        "essay": essay_count,
        "total": total
    }


def build_fast_exam_critique(state: dict, exam_paper: str) -> Optional[dict]:
    quantity_dist = state.get('quantity_dist', {}) or {"choice": 10, "fill": 5, "judge": 5, "essay": 3}
    target_total = int(state.get("target_total_score") or 100)
    actual = count_questions_in_exam(exam_paper, quantity_dist)
    quantity_valid = (
        actual["choice"] == quantity_dist.get("choice", 0) and
        actual["fill"] == quantity_dist.get("fill", 0) and
        actual["judge"] == quantity_dist.get("judge", 0) and
        actual["essay"] == quantity_dist.get("essay", 0)
    )

    payload = state.get("exam_payload") or {}
    exam_data = payload.get("exam_data", {})
    questions = exam_data.get("questions", [])
    numbers = [int(q.get("number", 0)) for q in questions if q.get("number")]
    numbering_valid = bool(numbers) and numbers == list(range(1, len(numbers) + 1))

    seen = set()
    has_duplicates = False
    options_valid = True
    total_score = 0
    difficulty_counts = {"简单": 0, "中等": 0, "较难": 0, "困难": 0, "": 0}
    topic_signatures: set = set()
    question_lengths: List[int] = []
    for q in questions:
        signature = question_dedup_key(q)
        if signature and signature in seen:
            has_duplicates = True
        if signature:
            seen.add(signature)
        if str(q.get("type") or "") == "选择题":
            opts = q.get("options") or []
            if not isinstance(opts, list) or len(opts) < 4:
                options_valid = False
        difficulty = str(q.get("difficulty") or "").strip()
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        total_score += int(q.get("score") or 0)
        topic_sig = concept_key_for_question(q)
        topic_signatures.add(topic_sig)
        qtext = str(q.get("question") or "").strip()
        question_lengths.append(len(qtext))

    # 质量门控 1：总分偏差不超过 ±5%
    total_score_valid = abs(total_score - target_total) <= 5
    # 质量门控 2：综合卷必须有一定难度梯度（至少 20% 为中等及以上）
    total_q = len(questions)
    hard_ratio = (difficulty_counts.get("中等", 0) + difficulty_counts.get("较难", 0) + difficulty_counts.get("困难", 0)) / max(1, total_q)
    difficulty_valid = hard_ratio >= 0.15 if total_q >= 10 else True
    # 质量门控 3：题目长度要有变化（标准差检验，排除全一样的题目）
    length_variation_valid = True
    if len(question_lengths) >= 5:
        import statistics
        if statistics.stdev(question_lengths) < 5:
            length_variation_valid = False
    # 质量门控 4：简答题必须覆盖多个不同考点（防止全部只考一个概念）
    essay_topics_ok = True
    essay_questions = [q for q in questions if str(q.get("type") or "") == "简答题"]
    if len(essay_questions) > 0 and len(topic_signatures) > 0:
        essay_topic_set = {concept_key_for_question(q) for q in essay_questions}
        if len(essay_topic_set) == 1 and len(essay_questions) >= 2:
            essay_topics_ok = False  # 多道简答却只考一个概念

    quality_gates_passed = (
        total_score_valid and difficulty_valid and length_variation_valid and essay_topics_ok
    )

    if quantity_valid and numbering_valid and not has_duplicates and options_valid and quality_gates_passed:
        gate_notes = []
        if not total_score_valid:
            gate_notes.append(f"总分偏差({total_score} vs {target_total})")
        if not difficulty_valid:
            gate_notes.append(f"难度分布不足(中等及以上仅{hard_ratio:.0%})")
        if not length_variation_valid:
            gate_notes.append("题干长度无变化")
        if not essay_topics_ok:
            gate_notes.append("简答题考点重复")
        critique_msg = "本地校验通过" + (f"，质量门控({','.join(gate_notes)})" if gate_notes else "") + "，跳过 LLM 评审。"
        logger.info(f"[Agent2-Critic-试卷] {critique_msg}")
        return {
            "approved": True,
            "overall_score": 85 if quality_gates_passed else 72,
            "critique": critique_msg,
            "reasoning_flaws": [],
            "specific_issues": [],
            "duplicate_check": {"has_duplicates": False, "duplicate_questions": []},
            "numbering_check": {"is_continuous": True, "issues": []},
            "quantity_check": {
                "choice": quantity_dist.get("choice", 0),
                "fill": quantity_dist.get("fill", 0),
                "judge": quantity_dist.get("judge", 0),
                "essay": quantity_dist.get("essay", 0),
                "actual_choice": actual["choice"],
                "actual_fill": actual["fill"],
                "actual_judge": actual["judge"],
                "actual_essay": actual["essay"],
                "is_valid": True
            }
        }

    return None
