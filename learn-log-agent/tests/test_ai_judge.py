"""
AI-as-judge 품질 테스트 — 실제 LLM 호출 (비용 발생)

실행:
    uv run pytest tests/test_ai_judge.py -v -m ai_judge
"""
import json
import pytest
from unittest.mock import patch
from langchain_core.messages import HumanMessage
from graph import (
    llm,
    parse_json,
    goal_setup_node,
    checkin_node,
    quiz_generate_node,
    diary_writer_node,
)

pytestmark = pytest.mark.ai_judge


# ── Helper ──────────────────────────────────────────────────────

def ai_judge(criteria: str, content: str) -> dict:
    prompt = f"""아래 내용을 평가해주세요.

평가 기준: {criteria}
평가 대상:
{content}

JSON으로 응답해주세요:
{{
  "score": 1~5,
  "passed": true 또는 false,
  "reason": "판단 근거"
}}

JSON만 응답해주세요."""

    res = llm.invoke([HumanMessage(content=prompt)])
    result = parse_json(res.content)
    if result is None:
        pytest.fail(f"judge 응답 파싱 실패: {res.content}")
    return result


# ── goal_setup_node 품질 테스트 ────────────────────────────────

def test_curriculum_covers_topic(state_with_goal):
    """생성된 커리큘럼이 학습 목표를 충분히 커버하는지"""
    result         = goal_setup_node(state_with_goal)
    curriculum_str = json.dumps(result["curriculum"], ensure_ascii=False)

    verdict = ai_judge(
        criteria=(
            "이 커리큘럼이 'LangGraph 공부'를 체계적으로 다루는가? "
            "(주차별 구성, 토픽 깊이, 체크포인트 적절성 기준)"
        ),
        content=curriculum_str,
    )
    assert verdict["passed"], f"커리큘럼 품질 미달: {verdict['reason']}"
    assert verdict["score"] >= 3


def test_tutor_persona_is_domain_specific(state_with_goal):
    """생성된 튜터 페르소나가 도메인에 맞게 전문적인지"""
    result = goal_setup_node(state_with_goal)

    verdict = ai_judge(
        criteria=(
            "이 시스템 프롬프트가 LangGraph/AI Agent 전문 강사로서 "
            "구체적이고 도메인에 맞는 교육 원칙을 담고 있는가?"
        ),
        content=result["tutor_persona"],
    )
    assert verdict["passed"], f"페르소나 품질 미달: {verdict['reason']}"
    assert verdict["score"] >= 3


# ── checkin_node 품질 테스트 ───────────────────────────────────

def test_checkin_response_is_encouraging(state_with_diary):
    """체크인 튜터 응답이 학습 주제와 관련 있고 격려하는가"""
    with patch("graph.interrupt", return_value="interrupt()와 Command(resume)을 공부했어요"):
        result = checkin_node(state_with_diary)
    response_text = result["messages"][0].content if result.get("messages") else ""

    verdict = ai_judge(
        criteria=(
            "이 튜터 응답이 학습자의 오늘 성취('interrupt()와 Command(resume) 공부')를 "
            "인정하고, 다음 학습을 격려하며, LangGraph 주제에 맞는 구체적인 피드백을 주는가?"
        ),
        content=response_text,
    )
    assert verdict["passed"], f"체크인 응답 품질 미달: {verdict['reason']}"
    assert verdict["score"] >= 3


# ── quiz_generate_node 품질 테스트 ────────────────────────────

def test_quiz_questions_are_relevant(state_with_search):
    """퀴즈 문제가 학습 주제에 맞고 명확한가"""
    result = quiz_generate_node(state_with_search)
    questions = result.get("quiz_questions", "")

    verdict = ai_judge(
        criteria=(
            "이 퀴즈 문제들이 'LangGraph 기초' 주제에 적합하고, "
            "질문이 명확하며, 학습자가 답변할 수 있는 수준인가?"
        ),
        content=questions,
    )
    assert verdict["passed"], f"퀴즈 품질 미달: {verdict['reason']}"
    assert verdict["score"] >= 3


# ── diary_writer_node 품질 테스트 ─────────────────────────────

def test_diary_reflects_mood_and_achievements(state_with_diary):
    """일기가 기분과 성취를 자연스럽게 반영하는지"""
    result = diary_writer_node(state_with_diary)

    verdict = ai_judge(
        criteria=(
            "이 일기가 '설레고 에너지 넘치는 하루'라는 기분과 "
            "'interrupt() 공부'라는 성취를 자연스럽게 반영하는가?"
        ),
        content=result["diary_content"],
    )
    assert verdict["passed"], f"일기 품질 미달: {verdict['reason']}"
    assert verdict["score"] >= 4
