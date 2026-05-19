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
    goal_check_node,
    goal_detail_node,
    curriculum_build_node,
    persona_build_node,
    checkin_topic_node,
    checkin_node,
    tutor_qa_generate_node,
    tutor_qa_feedback_node,
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
         patch("graph.interrupt", return_value="중급 수준이에요"):
        mock_llm.invoke.return_value = make_llm_response(detail_json)
        result = goal_detail_node(state_with_goal)

    assert result["user_level"] == "intermediate"
    assert result["active_goal"] == "LangGraph 공부"


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


# ── curriculum_build_node ───────────────────────────────────────

def test_curriculum_build_returns_curriculum(state_with_goal):
    """curriculum_build_node가 curriculum dict를 반환하는지"""
    from graph import DayPlan, WeekPhase, Curriculum, DomainAnalysis

    mock_domain = DomainAnalysis(
        domain="programming", subject="LangGraph",
        level="intermediate", learning_style="hands-on",
        estimated_weeks=4, prerequisites=[],
        is_framework=True, official_name="LangGraph by LangChain",
    )
    mock_curriculum = Curriculum(
        domain="programming", subject="LangGraph",
        level="intermediate", total_weeks=4, daily_minutes=60,
        phases=[
            WeekPhase(
                week=1, theme="LangGraph 기초", checkpoint="기본 그래프 구성",
                days=[DayPlan(day=1, topic="StateGraph 소개")],
                resources=[],
            )
        ],
    )

    with patch("graph.llm") as mock_llm:
        # curriculum_build_node는 내부에서 DomainAnalysis → Curriculum 순으로
        # with_structured_output을 두 번 호출
        mock_llm.with_structured_output.side_effect = [
            make_structured_mock(mock_domain),
            make_structured_mock(mock_curriculum),
        ]
        result = curriculum_build_node(state_with_goal)

    assert "curriculum" in result
    assert result["curriculum"]["total_weeks"] == 4
    assert result["curriculum"]["phases"][0]["theme"] == "LangGraph 기초"


def test_curriculum_build_includes_framework_hint(state_with_goal):
    """is_framework=True일 때 framework hint가 커리큘럼 프롬프트에 포함되는지"""
    from graph import DayPlan, WeekPhase, Curriculum, DomainAnalysis

    mock_domain = DomainAnalysis(
        domain="programming", subject="LangGraph",
        level="intermediate", learning_style="hands-on",
        estimated_weeks=4, prerequisites=[],
        is_framework=True, official_name="LangGraph by LangChain",
    )
    mock_curriculum = Curriculum(
        domain="programming", subject="LangGraph",
        level="intermediate", total_weeks=4, daily_minutes=60,
        phases=[
            WeekPhase(
                week=1, theme="기초", checkpoint="이해",
                days=[DayPlan(day=1, topic="소개")],
                resources=[],
            )
        ],
    )

    with patch("graph.llm") as mock_llm:
        curriculum_mock = make_structured_mock(mock_curriculum)
        mock_llm.with_structured_output.side_effect = [
            make_structured_mock(mock_domain),
            curriculum_mock,
        ]
        curriculum_build_node(state_with_goal)

    # 두 번째 with_structured_output 호출(curriculum)의 프롬프트에 framework hint 포함 확인
    curriculum_prompt = curriculum_mock.invoke.call_args[0][0][0].content
    assert "LangGraph by LangChain" in curriculum_prompt


# ── persona_build_node ──────────────────────────────────────────

def test_persona_build_returns_tutor_persona(state_with_goal):
    """persona_build_node가 tutor_persona와 summary 메시지를 반환하는지"""
    curriculum = {
        "subject": "LangGraph", "domain": "programming",
        "level": "intermediate", "total_weeks": 4, "daily_minutes": 60,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "기본 이해",
             "days": [{"day": 1, "topic": "StateGraph 소개"}],
             "resources": []},
        ],
    }
    state = {**state_with_goal, "curriculum": curriculum}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("당신은 LangGraph 전문 강사입니다.")
        result = persona_build_node(state)

    assert result["tutor_persona"] == "당신은 LangGraph 전문 강사입니다."
    assert result["current_week"] == 1
    assert result["current_topic"] == "StateGraph 소개"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "LangGraph 공부" in result["learning_goals"]


def test_persona_build_single_llm_call(state_with_goal):
    """persona_build_node는 LLM을 1번만 호출하는지 (habit plan 제거 후)"""
    curriculum = {
        "subject": "LangGraph", "domain": "programming",
        "level": "intermediate", "total_weeks": 4, "daily_minutes": 60,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "이해",
             "days": [{"day": 1, "topic": "StateGraph"}],
             "resources": []},
        ],
    }
    state = {**state_with_goal, "curriculum": curriculum}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("튜터 페르소나.")
        persona_build_node(state)

    assert mock_llm.invoke.call_count == 1


# ── checkin_topic_node ──────────────────────────────────────────

def test_checkin_topic_pre_increments_streak(base_state):
    """checkin_topic_node가 streak을 미리 증가시키는지"""
    state = {**base_state, "streak": 3}

    with patch("graph.TavilySearch", None):
        result = checkin_topic_node(state)

    assert result["streak"] == 4


def test_checkin_topic_sets_week_and_topic(base_state):
    """checkin_topic_node가 current_week, current_topic을 state에 저장하는지"""
    curriculum = {
        "total_weeks": 4,
        "phases": [
            {"week": 1, "theme": "기초", "checkpoint": "이해",
             "days": [
                 {"day": 1, "topic": "StateGraph 소개"},
                 {"day": 2, "topic": "Nodes and Edges"},
             ],
             "resources": []},
        ],
    }
    state = {**base_state, "streak": 0, "curriculum": curriculum}

    with patch("graph.tavily_search", None):
        result = checkin_topic_node(state)

    assert result["current_week"] == 1
    assert result["current_topic"] == "StateGraph 소개"
    assert result["streak"] == 1


def test_checkin_topic_resets_tutor_qa(base_state):
    """checkin_topic_node가 tutor_qa 관련 state를 초기화하는지"""
    state = {
        **base_state,
        "tutor_qa_index": 3,
        "tutor_qa_history": [{"question": "q", "answer": "a"}],
        "ready_for_quiz": True,
    }

    with patch("graph.tavily_search", None):
        result = checkin_topic_node(state)

    assert result["tutor_qa_index"] == 0
    assert result["tutor_qa_history"] == []
    assert result["ready_for_quiz"] is False


# ── checkin_node ────────────────────────────────────────────────

def test_checkin_first_day_message(base_state):
    """streak=1 (Day 1, pre-incremented)이면 '첫 날 시작' 메시지"""
    state = {**base_state, "streak": 1, "current_topic": "StateGraph"}

    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(state)

    prompt = mock_interrupt.call_args[0][0]
    assert "first day" in prompt.lower()


def test_checkin_streak_message(state_with_goal):
    """streak>1이면 연속달성 streak 메시지 포함"""
    state = {**state_with_goal, "streak": 6}  # already pre-incremented

    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(state)

    prompt = mock_interrupt.call_args[0][0]
    assert "streak" in prompt.lower()


def test_checkin_saves_achievements(base_state):
    """사용자 입력이 today_achievements에 저장되는지"""
    state = {**base_state, "streak": 1, "current_topic": "StateGraph"}
    user_msg = "LangGraph StateGraph를 실습했어요"

    with patch("graph.interrupt", return_value=user_msg):
        result = checkin_node(state)

    assert result["today_achievements"] == user_msg


def test_checkin_does_not_modify_streak(base_state):
    """checkin_node는 streak을 변경하지 않음 (checkin_topic_node에서 처리)"""
    state = {**base_state, "streak": 4, "current_topic": "test"}

    with patch("graph.interrupt", return_value="공부"):
        result = checkin_node(state)

    assert "streak" not in result


def test_checkin_prompt_includes_resources(state_with_goal):
    """search_results가 있으면 체크인 prompt에 포함되는지"""
    state = {
        **state_with_goal,
        "streak": 1,
        "search_results": "https://langchain-ai.github.io/langgraph/",
    }

    with patch("graph.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "공부했어요"
        checkin_node(state)

    prompt = mock_interrupt.call_args[0][0]
    assert "langchain-ai.github.io" in prompt


# ── tutor_qa_generate_node ──────────────────────────────────────

def test_tutor_qa_turn1_uses_curriculum_context(state_with_search):
    """Turn 1에서 week theme과 checkpoint이 프롬프트에 포함되는지"""
    state = {**state_with_search, "tutor_qa_index": 0, "streak": 1}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("오늘 배울 내용입니다.")
        result = tutor_qa_generate_node(state)

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert result["ready_for_quiz"] is False

    prompt = mock_llm.invoke.call_args[0][0][1].content
    assert "LangGraph 기초" in prompt  # week theme


def test_tutor_qa_turn2_returns_json_message(state_with_search):
    """Turn 2+에서 JSON 응답을 파싱하여 message를 추출하는지"""
    history = [{"question": "StateGraph란?", "answer": "그래프 구조입니다."}]
    state = {
        **state_with_search,
        "tutor_qa_index": 1,
        "tutor_qa_history": history,
        "streak": 2,
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(
            '{"message": "좋은 답변이에요! 다음으로 넘어가겠습니다.", "ready_for_quiz": false}'
        )
        result = tutor_qa_generate_node(state)

    assert result["messages"][0].content == "좋은 답변이에요! 다음으로 넘어가겠습니다."
    assert result["ready_for_quiz"] is False


def test_tutor_qa_ready_for_quiz_signal(state_with_search):
    """LLM이 ready_for_quiz=true를 반환하면 state에 반영되는지"""
    history = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]
    state = {
        **state_with_search,
        "tutor_qa_index": 2,
        "tutor_qa_history": history,
        "streak": 3,
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(
            '{"message": "수고했어요!", "ready_for_quiz": true}'
        )
        result = tutor_qa_generate_node(state)

    assert result["ready_for_quiz"] is True


def test_tutor_qa_max_turns_forces_quiz(state_with_search):
    """MAX_TURNS(5) 도달 시 ready_for_quiz가 강제로 True가 되는지"""
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
    state = {
        **state_with_search,
        "tutor_qa_index": 5,
        "tutor_qa_history": history,
        "streak": 6,
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(
            '{"message": "계속 합시다.", "ready_for_quiz": false}'
        )
        result = tutor_qa_generate_node(state)

    assert result["ready_for_quiz"] is True


# ── tutor_qa_feedback_node ──────────────────────────────────────

def test_tutor_qa_feedback_appends_history(base_state):
    """tutor_qa_feedback_node가 history에 Q&A를 추가하는지"""
    state = {
        **base_state,
        "tutor_qa_index":   1,
        "tutor_qa_question": "StateGraph란?",
        "tutor_qa_answer":   "노드와 엣지로 구성됩니다.",
        "tutor_qa_history":  [],
    }
    result = tutor_qa_feedback_node(state)

    assert len(result["tutor_qa_history"]) == 1
    assert result["tutor_qa_history"][0]["question"] == "StateGraph란?"
    assert result["tutor_qa_index"] == 2


# ── _get_review_topics ──────────────────────────────────────────

def test_get_review_topics_empty_history():
    assert _get_review_topics([], "오늘 주제") == []


def test_get_review_topics_detects_weak_topic():
    history = [{"date": "2026-05-01", "topic": "API 활용",
                "feedback": "2번 incorrect. 점수: 1/3", "questions": "", "answers": ""}]
    result = _get_review_topics(history, "오늘 주제")
    assert len(result) == 1
    assert result[0]["topic"] == "API 활용"
    assert result[0]["is_weak"] is True


def test_get_review_topics_detects_old_topic():
    history = [{"date": "2020-01-01", "topic": "기초 개념",
                "feedback": "완벽해요!", "questions": "", "answers": ""}]
    result = _get_review_topics(history, "오늘 주제")
    assert len(result) == 1
    assert result[0]["is_weak"] is False


def test_get_review_topics_excludes_current_topic():
    history = [{"date": "2020-01-01", "topic": "오늘 주제",
                "feedback": "아쉬워요", "questions": "", "answers": ""}]
    assert _get_review_topics(history, "오늘 주제") == []


def test_get_review_topics_max_two():
    history = [
        {"date": "2020-01-01", "topic": f"토픽{i}",
         "feedback": "아쉬워요", "questions": "", "answers": ""}
        for i in range(5)
    ]
    assert len(_get_review_topics(history, "오늘 주제")) <= 2


# ── quiz_generate_node ──────────────────────────────────────────

def test_quiz_generate_node_returns_questions(state_with_search):
    """quiz_generate_node가 quiz_questions를 state에 저장하는지"""
    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response(
            "1번 문제: StateGraph란?\n2번 문제: ...\n3번 문제: ..."
        )
        result = quiz_generate_node(state_with_search)

    assert "quiz_questions" in result
    assert "StateGraph" in result["quiz_questions"]


def test_quiz_generate_uses_tutor_persona(state_with_search):
    """quiz_generate_node가 tutor_persona를 SystemMessage로 사용하는지"""
    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("퀴즈 문제들...")
        quiz_generate_node(state_with_search)

    first_call_messages = mock_llm.invoke.call_args_list[0][0][0]
    assert isinstance(first_call_messages[0], SystemMessage)
    assert "LangGraph 전문 강사" in first_call_messages[0].content


# ── quiz_node ───────────────────────────────────────────────────

def test_quiz_node_returns_feedback(state_with_search):
    """quiz_node가 quiz_questions를 사용해 평가하는지"""
    state = {
        **state_with_search,
        "quiz_questions": "1번: StateGraph란?\n2번: ...\n3번: ...",
    }

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="1. StateGraph  2. 모르겠어요  3. interrupt"):
        mock_llm.invoke.return_value = make_llm_response("1번 정답! 2번 아쉬워요. 점수: 2/3")
        result = quiz_node(state)

    assert "quiz_history" in result
    assert len(result["quiz_history"]) == 1
    assert result["quiz_questions"] == ""


def test_quiz_node_appends_to_existing_history(state_with_search):
    existing = [{"date": "2026-01-01", "topic": "이전 토픽",
                 "questions": "", "answers": "", "feedback": ""}]
    state = {**state_with_search, "quiz_history": existing, "quiz_questions": "퀴즈..."}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="답변"):
        mock_llm.invoke.return_value = make_llm_response("피드백: 잘했어요! 점수: 3/3")
        result = quiz_node(state)

    assert len(result["quiz_history"]) == 2


def test_quiz_node_single_llm_call(state_with_search):
    """quiz_node는 LLM을 1번만 호출 (평가용)"""
    state = {**state_with_search, "quiz_questions": "미리 생성된 퀴즈"}

    with patch("graph.llm") as mock_llm, \
         patch("graph.interrupt", return_value="답변"):
        mock_llm.invoke.return_value = make_llm_response("피드백")
        quiz_node(state)

    assert mock_llm.invoke.call_count == 1


# ── diary_writer_node ───────────────────────────────────────────

def test_diary_writer_achievements_fallback(state_with_goal):
    """today_achievements가 비어있으면 fallback 메시지 사용"""
    state = {**state_with_goal, "today_achievements": ""}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("오늘의 일기...")
        diary_writer_node(state)

    system_content = mock_llm.invoke.call_args[0][0][0].content
    assert "No learning recorded" in system_content


def test_diary_writer_uses_tutor_history(state_with_goal):
    """tutor_qa_history가 있으면 achievements 대신 사용되는지"""
    state = {
        **state_with_goal,
        "tutor_qa_history": [
            {"question": "q1", "answer": "StateGraph는 노드와 엣지로 구성됩니다."},
        ],
        "today_achievements": "이 내용은 사용되지 않아야 함",
    }

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("일기 내용...")
        diary_writer_node(state)

    system_content = mock_llm.invoke.call_args[0][0][0].content
    assert "StateGraph는 노드와 엣지로 구성됩니다." in system_content


def test_diary_writer_returns_diary_content(state_with_goal):
    state = {**state_with_goal, "today_achievements": "공부했어요"}

    with patch("graph.llm") as mock_llm:
        mock_llm.invoke.return_value = make_llm_response("완성된 일기 내용입니다.")
        result = diary_writer_node(state)

    assert result["diary_content"] == "완성된 일기 내용입니다."


# ── notion_post_node ────────────────────────────────────────────

def test_notion_post_skips_when_diary_empty(state_with_goal):
    state = {**state_with_goal, "diary_content": ""}
    result = notion_post_node(state)

    assert "No diary content" in result["messages"][0].content


def test_notion_post_success_message(state_with_diary):
    with patch("graph.post_to_notion") as mock_tool:
        mock_tool.invoke.return_value = "✅ Notion 포스팅 성공! https://notion.so/..."
        result = notion_post_node(state_with_diary)

    assert "saved to Notion" in result["messages"][0].content


def test_notion_post_failure_message(state_with_diary):
    with patch("graph.post_to_notion") as mock_tool:
        mock_tool.invoke.return_value = "⚠️ 오류 (400): 연결 실패"
        result = notion_post_node(state_with_diary)

    assert "Couldn't save" in result["messages"][0].content
