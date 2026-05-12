import pytest


@pytest.fixture
def base_state():
    return {
        "messages":           [],
        "learning_goals":     [],
        "active_goal":        "",
        "mood":               "",
        "today_achievements": "",
        "streak":             0,
        "search_results":     "",
        "diary_content":      "",
        "next_action":        "",
        "curriculum":         {},
        "current_week":       0,
        "current_topic":      "",
        "tutor_persona":      "",
        "quiz_history":       [],
        "progress_pct":       0.0,
        "user_level":         "",
    }


@pytest.fixture
def state_with_goal(base_state):
    return {
        **base_state,
        "learning_goals": ["LangGraph 공부"],
        "active_goal":    "LangGraph 공부",
        "streak":         5,
        "tutor_persona":  "당신은 LangGraph 전문 강사입니다.",
        "user_level":     "intermediate",
    }


@pytest.fixture
def state_with_diary(state_with_goal):
    return {
        **state_with_goal,
        "mood":               "설레고 에너지 넘치는 하루",
        "today_achievements": "interrupt()와 Command(resume)을 공부했어요",
        "diary_content":      "오늘은 LangGraph interrupt를 배웠다...",
        "streak":             6,
    }


@pytest.fixture
def state_with_search(state_with_goal):
    return {
        **state_with_goal,
        "today_achievements": "LangGraph StateGraph를 실습했어요",
        "search_results":     "📌 LangGraph 공식 문서\n   🔗 https://...",
        "current_topic":      "LangGraph 기초",
    }
