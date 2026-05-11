import pytest

from agent.multi_agent.question_type_resolver import resolve_question_type_plan


@pytest.mark.asyncio
async def test_user_explicit_question_distribution_wins():
    plan = await resolve_question_type_plan(
        query="请出选择题2道、填空题2道",
        route="quiz",
        session_id="操作系统",
    )

    assert plan["source"] == "user_explicit"
    assert plan["quantity_dist"] == {"choice": 2, "fill": 2, "judge": 0, "essay": 0}
    assert plan["total_questions"] == 4
    assert plan["quiz_types"] == ["选择题", "填空题"]


@pytest.mark.asyncio
async def test_sample_paper_beats_web_reference():
    sample = "一、判断题（每题2分，共5题）\n二、简答题 3 道"
    web_ref = "网络题型参考：选择题10道、判断题5道、填空题5道、简答题3道"

    plan = await resolve_question_type_plan(
        query="生成综合测试卷，题型你决定",
        route="exam",
        session_id="操作系统",
        sample_paper_context=sample,
        search_observation=web_ref,
    )

    assert plan["source"] == "sample_paper"
    assert plan["quantity_dist"] == {"choice": 0, "fill": 0, "judge": 5, "essay": 3}


@pytest.mark.asyncio
async def test_web_reference_used_when_no_sample_or_explicit_types():
    plan = await resolve_question_type_plan(
        query="生成综合测试卷，题型你参考网络决定",
        route="exam",
        session_id="操作系统",
        search_observation="网络题型参考：选择题8道、判断题4道、填空题4道、简答题2道",
    )

    assert plan["source"] == "web_search"
    assert plan["quantity_dist"] == {"choice": 8, "fill": 4, "judge": 4, "essay": 2}
    assert plan["total_questions"] == 18


@pytest.mark.asyncio
async def test_resolver_can_fetch_web_reference_when_needed():
    calls = []

    async def fake_search(course_name):
        calls.append(course_name)
        return "题型结构：选择题6道、填空题3道、判断题3道、简答题2道"

    plan = await resolve_question_type_plan(
        query="生成综合测试卷，题型参考网络",
        route="exam",
        session_id="操作系统",
        search_exam_format_func=fake_search,
    )

    assert calls == ["操作系统"]
    assert plan["source"] == "web_search"
    assert plan["quantity_dist"] == {"choice": 6, "fill": 3, "judge": 3, "essay": 2}
