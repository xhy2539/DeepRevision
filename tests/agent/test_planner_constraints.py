from agent.multi_agent.supervisor import _extract_planner_constraints, _normalize_planner_payload


def test_tonight_hours_are_one_day_and_converted_to_minutes():
    constraints = _extract_planner_constraints("我明天考操作系统，今晚只有3小时，请帮我安排复习计划")
    assert constraints["cycle_days"] == 1
    assert constraints["daily_minutes"] == 180


def test_half_hour_expression_is_supported():
    constraints = _extract_planner_constraints("明天考试，今晚只有2个半小时")
    assert constraints["cycle_days"] == 1
    assert constraints["daily_minutes"] == 150


def test_explicit_constraints_override_model_payload():
    model_payload = {
        "cycle_days": 7,
        "daily_plan": [
            {"day": day, "focus": f"主题{day}", "tasks": ["复习"], "duration_min": 60}
            for day in range(1, 8)
        ],
    }
    plan = _normalize_planner_payload(
        model_payload,
        "",
        cycle_days_hint=1,
        daily_minutes_hint=180,
    )
    assert plan["cycle_days"] == 1
    assert len(plan["daily_plan"]) == 1
    assert plan["daily_plan"][0]["duration_min"] == 180
