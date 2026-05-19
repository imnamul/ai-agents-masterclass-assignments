import pytest


@pytest.fixture
def base_state():
    return {
        "messages":            [],
        "learning_goals":      [],
        "active_goal":         "",
        "today_achievements":  "",
        "streak":              0,
        "search_results":      "",
        "diary_content":       "",
        "next_action":         "",
        "curriculum":          {},
        "current_week":        0,
        "current_topic":       "",
        "tutor_persona":       "",
        "quiz_history":        [],
        "quiz_questions":      "",
        "user_level":          "",
        "entry_mode":          "",
        "session_date":        "",
        "curriculum_feedback": "",
        "tutor_qa_index":      0,
        "ready_for_quiz":      False,
        "tutor_qa_question":   "",
        "tutor_qa_answer":     "",
        "tutor_qa_history":    [],
    }


@pytest.fixture
def state_with_goal(base_state):
    return {
        **base_state,
        "learning_goals": ["LangGraph 공부"],
        "active_goal":    "LangGraph 공부",
        "streak":         5,
        "current_week":   1,
        "current_topic":  "LangGraph 기초",
        "tutor_persona":  "당신은 LangGraph 전문 강사입니다.",
        "user_level":     "intermediate",
    }


@pytest.fixture
def state_with_diary(state_with_goal):
    return {
        **state_with_goal,
        "today_achievements": "interrupt()와 Command(resume)을 공부했어요",
        "diary_content":      "오늘은 LangGraph interrupt를 배웠다...",
        "streak":             6,
    }


@pytest.fixture
def state_with_search(state_with_goal):
    curriculum = {
        "total_weeks": 4,
        "daily_minutes": 60,
        "subject": "LangGraph",
        "domain": "programming",
        "level": "intermediate",
        "phases": [
            {
                "week": 1,
                "theme": "LangGraph 기초",
                "checkpoint": "기본 그래프 구성 이해",
                "days": [{"day": 1, "topic": "StateGraph 소개"}],
                "resources": [
                    {"title": "LangGraph 공식 문서",
                     "url": "https://langchain-ai.github.io/langgraph/"}
                ],
            }
        ],
    }
    return {
        **state_with_goal,
        "today_achievements": "LangGraph StateGraph를 실습했어요",
        "search_results":     "📌 LangGraph 공식 문서\n   🔗 https://langchain-ai.github.io/langgraph/",
        "current_topic":      "LangGraph 기초",
        "current_week":       1,
        "curriculum":         curriculum,
    }
