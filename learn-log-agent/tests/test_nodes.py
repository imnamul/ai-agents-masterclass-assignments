"""
LearnLog 노드 단위 테스트
Mock LLM 사용 — API 비용 없음

실행:
    uv run pytest tests/test_nodes.py -v
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph import (
    get_active_goal,
    mood_node,
    goal_check_node,
    goal_detail_node,
    domain_analysis_node,
    curriculum_build_node,
    persona_build_node,
    checkin_node,
    quiz_generate_node,
    quiz_node,
    diary_writer_node,
    notion_post_node,
    _get_review_topics,
)


# ── Helper ──────────────────────────────────────────────────────

def make_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


def make_structured_mock(return_value):
    """llm.with_structured_output(...).invoke(...) 체인을 모킹"""
    structured = MagicMock()
    structured.invoke.return_value = return_value
    return structured


# ── get_active_goal ─────────────────────────────────────────────

def test_get_active_goal_returns_active_goal(state_with_goal):
    assert get_active_goal(state_with_goal) == "LangGraph 공부"


def test_get_active_goal_fallback_to_list(base_state):
    state = {**base_state, "learning_goals": ["Python 기초"], "active_goal": ""}
    assert get_active_goal(state) == "Python 기초"


def test_get_active_goal_empty(base_state):
    assert get_active_goal(base_state) == ""


# ── mood_node ───────────────────────────────────────────────────

def test_mood_node_returns_mood(base_state):
    """기분 입력이 요약되어 mood에 저장되는지"""
    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="오늘 너무 피곤해요"):
        mock_llm.invoke.return_value = make_llm_response("지친 하루였지만 잘 버텼어요.")
        result = mood_node(base_state)

    assert result["mood"] == "지친 하루였지만 잘 버텼어요."
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "오늘 너무 피곤해요"


def test_mood_node_greeting_with_streak(base_state):
    """streak이 있으면 연속달성 메시지가 포함되는지"""
    state = {**base_state, "streak": 7}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "좋아요"
        mock_llm.invoke.return_value = make_llm_response("활기찬 하루!")
        mood_node(state)

    prompt_shown = mock_interrupt.call_args[0][0]
    assert "7일" in prompt_shown


def test_mood_node_greeting_new_user(base_state):
    """streak이 0이면 환영 메시지가 표시되는지"""
    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "좋아요"
        mock_llm.invoke.return_value = make_llm_response("활기찬 하루!")
        mood_node(base_state)

    prompt_shown = mock_interrupt.call_args[0][0]
    assert "환영" in prompt_shown


# ── goal_check_node ─────────────────────────────────────────────

def test_goal_check_new_user_wants_new_goal(base_state):
    """신규 사용자 → wants_new_goal=True, goal_text 추출"""
    goal_json = '{"wants_new_goal": true, "goal_text": "Python 기초"}'

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="Python 공부하고 싶어요"):
        mock_llm.invoke.return_value = make_llm_response(goal_json)
        result = goal_check_node(base_state)

    assert result["next_action"] == "setup"
    assert result["active_goal"] == "Python 기초"


def test_goal_check_existing_user_keeps_goal(state_with_goal):
    """기존 목표 있는 사용자가 계속 선택 → skip"""
    goal_json = '{"wants_new_goal": false, "goal_text": null}'

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="그대로 유지할게요"):
        mock_llm.invoke.return_value = make_llm_response(goal_json)
        result = goal_check_node(state_with_goal)

    assert result["next_action"] == "skip"
    assert result["active_goal"] == ""


def test_goal_check_fallback_on_json_error(base_state):
    """LLM이 잘못된 JSON 반환 시 user input을 goal로 사용"""
    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="LangGraph 마스터하기"):
        mock_llm.invoke.return_value = make_llm_response("잘못된 응답")
        result = goal_check_node(base_state)

    assert result["next_action"] == "setup"
    assert result["active_goal"] == "LangGraph 마스터하기"


# ── goal_detail_node ────────────────────────────────────────────

def test_goal_detail_extracts_level(state_with_goal):
    """사용자 응답에서 레벨이 추출되는지"""
    detail_json = '{"level": "intermediate", "focus": null}'

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="중급 수준이에요. 집중하고 싶은 부분은 없어요"):
        mock_llm.invoke.return_value = make_llm_response(detail_json)
        result = goal_detail_node(state_with_goal)

    assert result["user_level"] == "intermediate"
    assert result["active_goal"] == "LangGraph 공부"  # focus 없으면 그대로


def test_goal_detail_appends_focus_to_goal(state_with_goal):
    """focus가 있으면 active_goal에 추가되는지"""
    detail_json = '{"level": "beginner", "focus": "StateGraph 위주"}'

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="입문이고 StateGraph 위주로 배우고 싶어요"):
        mock_llm.invoke.return_value = make_llm_response(detail_json)
        result = goal_detail_node(state_with_goal)

    assert result["user_level"] == "beginner"
    assert "StateGraph 위주" in result["active_goal"]


def test_goal_detail_fallback_on_json_error(state_with_goal):
    """LLM JSON 오류 시 beginner로 fallback"""
    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="잘 모르겠어요"):
        mock_llm.invoke.return_value = make_llm_response("잘못된 응답")
        result = goal_detail_node(state_with_goal)

    assert result["user_level"] == "beginner"


# ── domain_analysis_node ────────────────────────────────────────

def test_domain_analysis_returns_domain_info(state_with_goal):
    """domain_analysis_node가 domain_info dict를 반환하는지"""
    mock_domain = MagicMock()
    mock_domain.model_dump.return_value = {
        "domain": "programming", "subject": "LangGraph",
        "level": "intermediate", "learning_style": "hands-on",
        "estimated_weeks": 4, "prerequisites": [],
        "is_framework": True, "official_name": "LangGraph by LangChain",
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = make_structured_mock(mock_domain)
        result = domain_analysis_node(state_with_goal)

    assert "domain_info" in result
    assert result["domain_info"]["domain"] == "programming"
    assert result["domain_info"]["is_framework"] is True


def test_domain_analysis_prompt_contains_goal(state_with_goal):
    """user_level이 LLM 프롬프트에 포함되는지"""
    state = {**state_with_goal, "user_level": "advanced"}
    mock_domain = MagicMock()
    mock_domain.model_dump.return_value = {
        "domain": "programming", "subject": "LangGraph", "level": "advanced",
        "learning_style": "hands-on", "estimated_weeks": 4,
        "prerequisites": [], "is_framework": False, "official_name": "",
    }

    with patch("graph.llm") as mock_llm:
        structured = make_structured_mock(mock_domain)
        mock_llm.with_structured_output.return_value = structured
        domain_analysis_node(state)

    prompt_content = structured.invoke.call_args[0][0][0].content
    assert "advanced" in prompt_content


# ── curriculum_build_node ───────────────────────────────────────

def test_curriculum_build_returns_curriculum(state_with_goal):
    """curriculum_build_node가 curriculum dict를 반환하는지"""
    from graph import DayPlan, WeekPhase, Curriculum, ResourceLink
    mock_curriculum = Curriculum(
        domain="programming", subject="LangGraph",
        level="intermediate", total_weeks=4, daily_minutes=60,
        phases=[
            WeekPhase(
                week=1, theme="LangGraph 기초", checkpoint="기본 그래프 구성",
                days=[DayPlan(day=1, topic="StateGraph 소개")],
                resources=[ResourceLink(title="LangGraph 공식 문서",
                                        url="https://langchain-ai.github.io/langgraph/")],
            )
        ],
    )

    state = {
        **state_with_goal,
        "domain_info": {
            "subject": "LangGraph", "level": "intermediate",
            "learning_style": "hands-on", "estimated_weeks": 4,
            "domain": "programming", "is_framework": True,
            "official_name": "LangGraph by LangChain",
        },
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = make_structured_mock(mock_curriculum)
        result = curriculum_build_node(state)

    assert "curriculum" in result
    assert result["curriculum"]["total_weeks"] == 4
    assert result["curriculum"]["phases"][0]["resources"][0]["url"].startswith("https://")


# ── persona_build_node ──────────────────────────────────────────

def test_persona_build_returns_tutor_persona(state_with_goal):
    """persona_build_node가 tutor_persona와 summary 메시지를 반환하는지"""
    curriculum = {
        "subject": "LangGraph", "domain": "programming",
        "level": "intermediate", "total_weeks": 4, "daily_minutes": 60,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "기본 이해",
             "days": [{"day": 1, "topic": "StateGraph 소개"}],
             "resources": [{"title": "LangGraph Docs",
                            "url": "https://langchain-ai.github.io/langgraph/"}]},
        ],
    }
    state = {
        **state_with_goal,
        "curriculum": curriculum,
        "domain_info": {"learning_style": "hands-on"},
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            make_llm_response("1. 매일 30분\n2. 실습 위주\n3. 복습"),
            make_llm_response("당신은 LangGraph 전문 강사입니다."),
        ]
        result = persona_build_node(state)

    assert result["tutor_persona"] == "당신은 LangGraph 전문 강사입니다."
    assert result["current_week"] == 1
    assert result["current_topic"] == "StateGraph 소개"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "LangGraph 공부" in result["learning_goals"]


def test_persona_build_summary_includes_resources(state_with_goal):
    """1주차 자료 링크가 summary 메시지에 포함되는지"""
    curriculum = {
        "subject": "LangGraph", "domain": "programming",
        "level": "intermediate", "total_weeks": 4, "daily_minutes": 60,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "이해",
             "days": [{"day": 1, "topic": "StateGraph"}],
             "resources": [{"title": "LangGraph Docs",
                            "url": "https://langchain-ai.github.io/langgraph/"}]},
        ],
    }
    state = {**state_with_goal, "curriculum": curriculum, "domain_info": {}}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.side_effect = [
            make_llm_response("1. 실습\n2. 복습\n3. 정리"),
            make_llm_response("당신은 LangGraph 튜터입니다."),
        ]
        result = persona_build_node(state)

    summary_content = result["messages"][0].content
    assert "langchain-ai.github.io" in summary_content


# ── checkin_node ────────────────────────────────────────────────

def test_checkin_streak_zero_message(base_state):
    """streak=0이면 '오늘부터 시작' 메시지"""
    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(base_state)

    prompt = mock_interrupt.call_args[0][0]
    assert "오늘부터 시작" in prompt


def test_checkin_streak_positive_message(state_with_goal):
    """streak>0이면 연속달성 메시지"""
    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(state_with_goal)

    prompt = mock_interrupt.call_args[0][0]
    assert "연속 달성" in prompt


def test_checkin_increments_streak(base_state):
    state = {**base_state, "streak": 4}

    with patch("graph.interrupt", return_value="오늘 공부했어요"):
        result = checkin_node(state)

    assert result["streak"] == 5


def test_checkin_saves_achievements(base_state):
    user_msg = "LangGraph StateGraph를 실습했어요"

    with patch("graph.interrupt", return_value=user_msg):
        result = checkin_node(base_state)

    assert result["today_achievements"] == user_msg


def test_checkin_prompt_includes_resources(state_with_goal):
    """curriculum에 resources가 있으면 체크인 prompt에 포함되는지"""
    curriculum = {
        "total_weeks": 4,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "이해",
             "days": [{"day": 1, "topic": "StateGraph"}],
             "resources": [{"title": "LangGraph Docs",
                            "url": "https://langchain-ai.github.io/langgraph/"}]},
        ],
    }
    state = {**state_with_goal, "curriculum": curriculum, "streak": 0}

    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(state)

    prompt = mock_interrupt.call_args[0][0]
    assert "langchain-ai.github.io" in prompt


# ── _get_review_topics ──────────────────────────────────────────

def test_get_review_topics_empty_history():
    """quiz_history가 비어있으면 빈 리스트 반환"""
    assert _get_review_topics([], "오늘 주제") == []


def test_get_review_topics_detects_weak_topic():
    """피드백에 부정 키워드가 있으면 약점 토픽으로 감지"""
    history = [{"date": "2026-05-01", "topic": "API 활용",
                "feedback": "2번 아쉬워요. 점수: 1/3", "questions": "", "answers": ""}]
    result = _get_review_topics(history, "오늘 주제")
    assert len(result) == 1
    assert result[0]["topic"] == "API 활용"
    assert result[0]["is_weak"] is True


def test_get_review_topics_detects_old_topic():
    """3일 이상 지난 토픽은 복습 대상"""
    history = [{"date": "2020-01-01", "topic": "기초 개념",
                "feedback": "완벽해요!", "questions": "", "answers": ""}]
    result = _get_review_topics(history, "오늘 주제")
    assert len(result) == 1
    assert result[0]["topic"] == "기초 개념"
    assert result[0]["is_weak"] is False


def test_get_review_topics_excludes_current_topic():
    """오늘 주제와 같은 항목은 제외"""
    history = [{"date": "2020-01-01", "topic": "오늘 주제",
                "feedback": "아쉬워요", "questions": "", "answers": ""}]
    result = _get_review_topics(history, "오늘 주제")
    assert result == []


def test_get_review_topics_max_two():
    """최대 2개까지만 반환"""
    history = [
        {"date": "2020-01-01", "topic": f"토픽{i}",
         "feedback": "아쉬워요", "questions": "", "answers": ""}
        for i in range(5)
    ]
    result = _get_review_topics(history, "오늘 주제")
    assert len(result) <= 2


# ── quiz_generate_node ──────────────────────────────────────────

def test_quiz_generate_node_returns_questions(state_with_search):
    """quiz_generate_node가 quiz_questions를 state에 저장하는지"""
    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("1번 문제: StateGraph란?\n2번 문제: ...\n3번 문제: ...")
        result = quiz_generate_node(state_with_search)

    assert "quiz_questions" in result
    assert "StateGraph" in result["quiz_questions"]


def test_quiz_generate_node_uses_tutor_persona(state_with_search):
    """quiz_generate_node가 tutor_persona를 SystemMessage로 사용하는지"""
    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("퀴즈 문제들...")
        quiz_generate_node(state_with_search)

    first_call_messages = mock_llm.invoke.call_args_list[0][0][0]
    assert isinstance(first_call_messages[0], SystemMessage)
    assert "LangGraph 전문 강사" in first_call_messages[0].content


# ── quiz_node ───────────────────────────────────────────────────

def test_quiz_node_returns_feedback_message(state_with_search):
    """quiz_node가 state의 quiz_questions를 사용해 평가하는지"""
    state = {**state_with_search, "quiz_questions": "1번 문제: StateGraph란?\n2번 문제: ...\n3번 문제: ..."}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="1. StateGraph  2. 모르겠어요  3. interrupt"):
        mock_llm.invoke.return_value = make_llm_response("1번 정답! 2번 아쉬워요. 전체 점수: 2/3")
        result = quiz_node(state)

    assert "quiz_history" in result
    assert len(result["quiz_history"]) == 1
    assert result["quiz_history"][0]["topic"] == "LangGraph 기초"
    assert result["quiz_questions"] == ""  # 사용 후 초기화 확인


def test_quiz_node_appends_to_existing_history(state_with_search):
    existing = [{"date": "2026-01-01", "topic": "이전 토픽",
                 "questions": "", "answers": "", "feedback": ""}]
    state = {**state_with_search, "quiz_history": existing, "quiz_questions": "퀴즈 문제들..."}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="답변입니다"):
        mock_llm.invoke.return_value = make_llm_response("피드백: 잘했어요! 점수: 3/3")
        result = quiz_node(state)

    assert len(result["quiz_history"]) == 2


def test_quiz_node_uses_state_quiz_questions(state_with_search):
    """quiz_node가 LLM을 재호출하지 않고 state의 quiz_questions를 사용하는지"""
    state = {**state_with_search, "quiz_questions": "미리 생성된 퀴즈 문제"}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="답변"):
        mock_llm.invoke.return_value = make_llm_response("피드백")
        quiz_node(state)

    # quiz_node는 LLM을 1번만 호출 (평가용)
    assert mock_llm.invoke.call_count == 1


# ── diary_writer_node ───────────────────────────────────────────

def test_diary_writer_mood_fallback(state_with_goal):
    """mood가 비어있으면 '평온한 하루' 사용"""
    state = {**state_with_goal, "mood": ""}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("오늘의 일기...")
        diary_writer_node(state)

    system_content = mock_llm.invoke.call_args[0][0][0].content
    assert "평온한 하루" in system_content


def test_diary_writer_achievements_fallback(state_with_goal):
    """today_achievements가 비어있으면 fallback 메시지 사용"""
    state = {**state_with_goal, "today_achievements": ""}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("오늘의 일기...")
        diary_writer_node(state)

    system_content = mock_llm.invoke.call_args[0][0][0].content
    assert "오늘의 학습을 기록하지 않았어요" in system_content


def test_diary_writer_returns_diary_content(state_with_goal):
    state = {**state_with_goal, "mood": "좋아요", "today_achievements": "공부했어요"}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("완성된 일기 내용입니다.")
        result = diary_writer_node(state)

    assert result["diary_content"] == "완성된 일기 내용입니다."


# ── notion_post_node ────────────────────────────────────────────

def test_notion_post_skips_when_diary_empty(state_with_goal):
    state  = {**state_with_goal, "diary_content": ""}
    result = notion_post_node(state)

    assert "건너뛰었어요" in result["messages"][0].content


def test_notion_post_success_message(state_with_diary):
    with patch("graph.post_to_notion") as mock_tool:
        mock_tool.invoke.return_value = "✅ Notion 포스팅 성공! https://notion.so/..."
        result = notion_post_node(state_with_diary)

    assert "저장됐어요" in result["messages"][0].content


def test_notion_post_failure_message(state_with_diary):
    with patch("graph.post_to_notion") as mock_tool:
        mock_tool.invoke.return_value = "⚠️ 오류 (400): 연결 실패"
        result = notion_post_node(state_with_diary)

    assert "실패" in result["messages"][0].content
