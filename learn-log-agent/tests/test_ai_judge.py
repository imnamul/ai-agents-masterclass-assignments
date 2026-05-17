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
    domain_analysis_node,
    curriculum_build_node,
    persona_build_node,
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


# ── 커리큘럼 생성 품질 테스트 (domain_analysis → curriculum_build → persona_build) ──

def _run_full_goal_setup(state_with_goal):
    """3개 노드를 순서대로 실행해 커리큘럼 + 튜터 페르소나를 생성하는 헬퍼"""
    domain_result = domain_analysis_node(state_with_goal)
    state1 = {**state_with_goal, **domain_result}

    curriculum_result = curriculum_build_node(state1)
    state2 = {**state1, **curriculum_result}

    persona_result = persona_build_node(state2)
    return {**state2, **persona_result}


def test_curriculum_covers_topic(state_with_goal):
    """생성된 커리큘럼이 학습 목표를 충분히 커버하는지"""
    result         = _run_full_goal_setup(state_with_goal)
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


def test_curriculum_resources_are_included(state_with_goal):
    """커리큘럼 각 주차에 자료 링크가 포함되는지"""
    result = _run_full_goal_setup(state_with_goal)
    phases = result["curriculum"].get("phases", [])

    has_resources = any(
        len(p.get("resources", [])) > 0
        for p in phases
    )
    assert has_resources, "최소 한 주차 이상 resources가 있어야 합니다"

    for p in phases:
        for r in p.get("resources", []):
            assert r.get("url", "").startswith("https://"), \
                f"유효하지 않은 URL: {r.get('url')}"


def test_tutor_persona_is_domain_specific(state_with_goal):
    """생성된 튜터 페르소나가 도메인에 맞게 전문적인지"""
    result = _run_full_goal_setup(state_with_goal)

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

def test_checkin_prompt_includes_resources(state_with_goal):
    """checkin interrupt 질문에 학습 자료 링크가 포함되는지"""
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

    captured = {}
    def capture_interrupt(question):
        captured["question"] = question
        return "오늘 StateGraph를 공부했어요"

    with patch("graph.interrupt", side_effect=capture_interrupt):
        checkin_node(state)

    assert "langchain-ai.github.io" in captured["question"], \
        "체크인 질문에 자료 링크가 포함되어야 합니다"


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
