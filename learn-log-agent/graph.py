"""
LearnLog — 학습 습관 트래커 에이전트
langgraph dev 서버용 그래프 정의 파일

실행 방법:
    langgraph dev
"""

import json
import os
import re
import sqlite3
import requests
from typing import TypedDict, Annotated, List
from datetime import date

from dotenv import load_dotenv
load_dotenv()

# ── LangGraph ──────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

# ── LangChain ──────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

# ── 환경변수 ────────────────────────────────────────────────────
NOTION_TOKEN       = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

# ── LLM & Tool 초기화 ───────────────────────────────────────────
llm           = init_chat_model("openai:gpt-4o-mini")
tavily_search = TavilySearch(max_results=3)


# ══════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════

class LearnLogState(TypedDict):
    messages:           Annotated[list, add_messages]
    learning_goals:     List[str]   # 전체 목표 목록 (복수 지원)
    active_goal:        str         # 오늘 활성 목표
    mood:               str         # 오늘 기분
    today_achievements: str         # 오늘 학습 성취
    streak:             int         # 연속 달성 일수
    search_results:     str
    diary_content:      str
    next_action:        str         # 조건 분기용
    # ── 학습 커리큘럼 ────────────────────────────────────────────
    curriculum:         dict        # 주차별 커리큘럼 구조
    current_week:       int         # 현재 몇 주차
    current_topic:      str         # 오늘 학습 토픽
    tutor_persona:      str         # 동적 생성 튜터 System Prompt
    quiz_history:       List[dict]  # 퀴즈 기록 (스페이스드 리피티션용)
    progress_pct:       float       # 전체 진도율 (0.0 ~ 1.0)
    user_level:         str         # 사용자 학습 수준 (beginner/intermediate/advanced)


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════

def get_active_goal(state: LearnLogState) -> str:
    """active_goal이 없으면 learning_goals[0]으로 폴백"""
    return state.get("active_goal") or (state.get("learning_goals") or [""])[0]


def parse_json(text: str) -> dict | None:
    """LLM 응답에서 JSON을 안정적으로 추출 (마크다운 펜스, 대소문자, 앞뒤 설명 텍스트 모두 처리)"""
    text = text.strip()
    # 1) 직접 파싱
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) ```json ... ``` 또는 ``` ... ``` 안에서 추출
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) 텍스트 안의 첫 번째 { ... } 블록 추출
    m = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


# ══════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════

@tool
def post_to_notion(diary_content: str, learning_goal: str, mood: str, streak: int) -> str:
    """오늘의 학습 일기를 Notion 데이터베이스에 포스팅합니다.

    Args:
        diary_content: 일기 본문
        learning_goal: 오늘 활성 목표
        mood: 오늘 기분
        streak: 연속 달성 일수
    """
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    today = date.today().isoformat()

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title":         {"title":     [{"text": {"content": f"📔 {today} Learning Diary"}}]},
            "Date":          {"date":      {"start": today}},
            "Learning Goal": {"rich_text": [{"text": {"content": learning_goal[:200]}}]},
            "Mood":          {"rich_text": [{"text": {"content": mood[:100]}}]},
            "Streak":        {"number":    streak},
        },
        "children": [{
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": diary_content[:2000]}}]},
        }],
    }

    try:
        res = requests.post(
            "https://api.notion.com/v1/pages",
            headers=headers, json=payload, timeout=10,
        )
        if res.status_code == 200:
            return f"✅ Notion 포스팅 성공! {res.json().get('url', '')}"
        return f"⚠️ 오류 ({res.status_code}): {res.text[:200]}"
    except Exception as e:
        return f"❌ 연결 오류: {e}"


# ══════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════

def mood_node(state: LearnLogState) -> dict:
    """노드 0: 기분 체크 (interrupt)"""
    streak = state.get("streak", 0)

    greeting = (
        f"🎉 {streak}일 연속 달성 중이에요!\n\n오늘 기분이 어떠세요?"
        if streak else
        "LearnLog에 오신 걸 환영해요! 📚\n\n오늘 기분이 어떠세요?"
    )
    mood_input = interrupt(greeting)

    mood_res = llm.invoke([HumanMessage(content=
        f'사용자 기분: "{mood_input}"\n'
        f"기분을 따뜻한 한 문장으로 요약해주세요. 문장만 출력하세요."
    )])

    return {
        "messages": [HumanMessage(content=mood_input)],
        "mood":     mood_res.content.strip(),
    }


def goal_check_node(state: LearnLogState) -> dict:
    """노드 1: 목표 유지/변경 의향 파악 (interrupt)"""
    existing_goals = state.get("learning_goals", [])

    if existing_goals:
        goals_str     = ", ".join(f"'{g}'" for g in existing_goals)
        goal_question = (
            f"현재 목표: [{goals_str}]\n\n"
            f"기존 목표를 계속할까요, 아니면 새로 추가/변경할까요?"
        )
    else:
        goal_question = "어떤 학습 목표가 있으신가요? 🎯"

    goal_input = interrupt(goal_question)

    goal_res = llm.invoke([HumanMessage(content=f"""사용자 응답: "{goal_input}"
기존 목표: {existing_goals if existing_goals else '없음'}

JSON으로 응답해주세요:
{{
  "wants_new_goal": true 또는 false,
  "goal_text": "학습 주제명만 간결하게 추출 (예: 'LangGraph', 'Python 기초', '영어 회화'). 없으면 null"
}}

판단 기준:
- '새로운', '추가', '바꾸', '변경', '다른 목표' → wants_new_goal: true
- '계속', '유지', '그대로', 변경 요청 없음      → wants_new_goal: false
- 기존 목표가 없는 경우                          → wants_new_goal: true

JSON만 응답해주세요.""")])

    parsed    = parse_json(goal_res.content)
    if parsed:
        wants_new = parsed.get("wants_new_goal", not bool(existing_goals))
        goal_text = parsed.get("goal_text") or ""
    else:
        change_kw = ["새로", "추가", "변경", "바꾸", "다른"]
        wants_new = any(kw in goal_input for kw in change_kw) or not existing_goals
        goal_text = goal_input

    return {
        "messages":   [HumanMessage(content=goal_input)],
        "active_goal": goal_text,
        "next_action": "setup" if wants_new else "skip",
    }


def goal_detail_node(state: LearnLogState) -> dict:
    """노드 1-2: 학습 수준 및 집중 영역 수집 (interrupt)"""
    goal = state.get("active_goal", "")

    question = (
        f"🎯 목표: {goal}\n\n"
        f"맞춤 학습 계획을 만들기 위해 조금 더 알고 싶어요!\n\n"
        f"① 현재 수준이 어느 정도인가요?\n"
        f"   (입문 / 초급 / 중급 / 고급)\n\n"
        f"② 특별히 집중하고 싶은 부분이 있나요?\n"
        f"   (없으면 '없어요'라고 해주세요)"
    )

    user_detail = interrupt(question)

    detail_res = llm.invoke([HumanMessage(content=f"""사용자 응답: "{user_detail}"

JSON으로 추출해주세요:
{{
  "level": "beginner / intermediate / advanced",
  "focus": "집중하고 싶은 부분 요약 (없으면 null)"
}}

레벨 판단 기준:
- 입문, 처음, 모름, 아무것도 모름 → beginner
- 기초는 있어, 조금 알아, 초급   → intermediate
- 많이 알아, 심화, 고급           → advanced

JSON만 응답해주세요.""")])

    parsed = parse_json(detail_res.content)
    if parsed:
        level = parsed.get("level", "beginner")
        focus = parsed.get("focus") or ""
    else:
        level = "beginner"
        focus = ""

    # focus가 있으면 active_goal에 context로 추가
    refined_goal = f"{goal} ({focus})" if focus else goal

    return {
        "messages":   [HumanMessage(content=user_detail)],
        "user_level": level,
        "active_goal": refined_goal,
    }


def goal_setup_node(state: LearnLogState) -> dict:
    """노드 2: 학습 계획 수립 (습관 + 커리큘럼 + 튜터 페르소나 통합)"""
    existing_goals = state.get("learning_goals", [])
    new_goal       = state.get("active_goal", "")
    user_level     = state.get("user_level", "beginner")
    updated_goals  = (existing_goals + [new_goal]
                      if new_goal and new_goal not in existing_goals
                      else existing_goals)

    # ── Step 1: 도메인 분석 ─────────────────────────────────────
    domain_res = llm.invoke([HumanMessage(content=f"""학습 목표: "{new_goal}"
사용자 수준: "{user_level}"

JSON으로 분석해주세요:
{{
  "domain": "programming / language / science / art / ...",
  "subject": "구체적인 주제명",
  "level": "{user_level}",
  "learning_style": "conceptual / hands-on / mixed",
  "estimated_weeks": 숫자,
  "prerequisites": []
}}

JSON만 응답해주세요.""")])

    domain = parse_json(domain_res.content)
    if not domain:
        domain = {
            "domain": "general", "subject": new_goal,
            "level": user_level, "learning_style": "mixed",
            "estimated_weeks": 4, "prerequisites": []
        }

    # ── Step 2: 커리큘럼 생성 ───────────────────────────────────
    curriculum_res = llm.invoke([HumanMessage(content=f"""주제: {domain['subject']}
레벨: {domain['level']} / 학습 방식: {domain['learning_style']} / 총 기간: {domain['estimated_weeks']}주

아래 JSON 형식으로 실제 주차별 학습 커리큘럼을 만들어주세요.
theme, topics, checkpoint는 실제 학습 내용으로 채워주세요 (예시 placeholder가 아닌 실제 값).

{{
  "domain": "{domain['domain']}",
  "subject": "{domain['subject']}",
  "level": "{domain['level']}",
  "total_weeks": {domain['estimated_weeks']},
  "daily_minutes": 60,
  "phases": [
    {{
      "week": 1,
      "theme": "1주차 핵심 학습 주제 (예: 기초 개념과 환경 설정)",
      "topics": ["첫 번째 세부 토픽", "두 번째 세부 토픽", "세 번째 세부 토픽"],
      "checkpoint": "1주차 완료 기준 (예: 간단한 예제 실행 성공)"
    }}
  ]
}}

total_weeks 수만큼 phases 배열을 채워주세요. JSON만 응답해주세요.""")])

    curriculum = parse_json(curriculum_res.content)
    if not curriculum or "phases" not in curriculum:
        curriculum = {
            "domain": domain["domain"], "subject": new_goal,
            "level": domain["level"], "total_weeks": domain["estimated_weeks"],
            "daily_minutes": 60,
            "phases": [{"week": 1, "theme": new_goal, "topics": [], "checkpoint": ""}]
        }

    # ── Step 3: 일일 습관 분해 ──────────────────────────────────
    habits_res = llm.invoke([HumanMessage(content=
        f"학습 목표: {new_goal}\n"
        f"수준: {user_level}\n"
        f"하루 학습 시간: {curriculum['daily_minutes']}분\n\n"
        f"매일 실천 가능한 습관 3가지를 간결하게 번호 목록으로만 출력해주세요."
    )])
    habits_text = habits_res.content.strip()

    # ── Step 4: 튜터 페르소나 생성 ──────────────────────────────
    first_topic = curriculum["phases"][0]["theme"] if curriculum.get("phases") else new_goal

    persona_res = llm.invoke([HumanMessage(content=f"""당신은 최고의 교육 설계자입니다.
아래 정보를 바탕으로 AI 튜터의 시스템 프롬프트를 작성해주세요.

- 주제: {curriculum['subject']} / 도메인: {curriculum['domain']}
- 수준: {curriculum['level']} / 학습 방식: {domain['learning_style']}
- 총 기간: {curriculum['total_weeks']}주 / 오늘 주제: {first_topic}

요구사항:
- 해당 분야 전문가 AI 튜터 페르소나
- 도메인 특성에 맞는 교육 원칙 3~5가지
- 수준({curriculum['level']})에 맞는 접근 방식 명시
- 시스템 프롬프트만 작성, 다른 설명 없이""")])
    tutor_persona = persona_res.content.strip()

    # ── 통합 메시지 (질문 없이 정보만) ─────────────────────────
    phases_summary = "\n\n".join([
        f"**{p['week']}주차** {p['theme']}\n"
        + (f" — {' / '.join(p['topics'])}" if p.get("topics") else "")
        for p in curriculum.get("phases", [])
    ])

    summary_msg = (
        f"📚 **{curriculum['subject']}** 학습 플랜을 준비했어요!\n\n"
        f"📋 **일일 습관:**\n{habits_text}\n\n"
        f"📅 **{curriculum['total_weeks']}주 커리큘럼** (하루 {curriculum['daily_minutes']}분)\n\n\n"
        f"{phases_summary}"
    )

    return {
        "messages":       [AIMessage(content=summary_msg)],
        "learning_goals":  updated_goals,
        "active_goal":    new_goal,
        "curriculum":     curriculum,
        "tutor_persona":  tutor_persona,
        "current_week":   1,
        "current_topic":  first_topic,
        "progress_pct":   0.0,
        "quiz_history":   [],
    }


def checkin_node(state: LearnLogState) -> dict:
    """노드 3: 오늘 학습 체크인 (interrupt)"""
    streak      = state.get("streak", 0)
    active_goal = get_active_goal(state)

    streak_msg = f"🔥 {streak}일 연속 달성 중!" if streak > 0 else "🌱 오늘부터 시작이에요!"

    question = (
        f"{streak_msg}\n\n"
        f"오늘 목표: {active_goal}\n\n"
        f"오늘 어떤 학습을 하셨나요? 😊\n\n"
        f"(관련 자료가 필요하면 '검색해줘', 퀴즈로 복습하고 싶으면 '퀴즈'를 포함해주세요)"
    )

    user_input = interrupt(question)

    search_keywords = ["검색", "자료", "찾아", "추천", "알려",
                       "search", "find", "resource", "recommend", "look up"]
    quiz_keywords   = ["퀴즈", "문제", "테스트", "확인", "복습",
                       "quiz", "test", "question", "review", "check"]

    needs_search = any(kw in user_input.lower() for kw in search_keywords)
    needs_quiz   = any(kw in user_input.lower() for kw in quiz_keywords)

    if needs_search:
        next_action = "search"
    elif needs_quiz:
        next_action = "quiz"
    else:
        next_action = "write"

    return {
        "messages":          [HumanMessage(content=user_input)],
        "today_achievements": user_input,
        "streak":            streak + 1,
        "next_action":       next_action,
    }


def resource_search_node(state: LearnLogState) -> dict:
    """노드 4: Tavily로 학습 자료 검색 + 퀴즈 여부 확인 (interrupt)"""
    goal         = get_active_goal(state)
    achievements = state.get("today_achievements", "")
    query        = f"{goal} {achievements} 학습 자료".strip()

    try:
        results = tavily_search.invoke(query)
        if isinstance(results, str):
            formatted = results
        elif isinstance(results, list):
            lines = []
            for r in results:
                if isinstance(r, dict):
                    title   = r.get("title", "")
                    url     = r.get("url", "")
                    content = r.get("content", "")[:120]
                    lines.append(f"📌 {title}\n   🔗 {url}\n   {content}...")
                else:
                    lines.append(str(r))
            formatted = "\n".join(lines)
        else:
            formatted = str(results)
    except Exception as e:
        error_msg = f"검색 중 오류가 발생했어요: {e}\n직접 검색해보시는 걸 추천드려요 🙏"
        return {"messages": [AIMessage(content=error_msg)], "search_results": "", "next_action": "write"}

    response = llm.invoke([SystemMessage(
        content=f"학습 목표: {goal}\n오늘 학습 내용: {achievements}\n\n검색 결과:\n{formatted}\n\n자료를 친절하게 소개해주세요."
    )])

    # ── 퀴즈 제안 interrupt ─────────────────────────────────────
    quiz_offer = interrupt(
        f"{response.content}\n\n"
        f"---\n"
        f"📝 방금 찾은 자료를 바탕으로 퀴즈를 풀어볼까요? (예 / 아니요)"
    )

    wants_quiz = any(kw in quiz_offer.lower() for kw in
                     ["예", "네", "응", "ㅇ", "좋아", "yes", "y", "퀴즈", "해줘"])

    return {
        "messages":       [response],
        "search_results": formatted,
        "next_action":    "quiz" if wants_quiz else "write",
    }


def quiz_node(state: LearnLogState) -> dict:
    """노드 4-2: 오늘 학습 내용 기반 퀴즈 (interrupt)"""
    tutor_persona = state.get("tutor_persona", "")
    goal          = get_active_goal(state)
    achievements  = state.get("today_achievements", "") or goal
    current_topic = state.get("current_topic", goal)
    quiz_history  = state.get("quiz_history", [])

    system = (SystemMessage(content=tutor_persona) if tutor_persona
              else SystemMessage(content=f"당신은 {goal} 전문 튜터입니다."))

    # ── Step 1: 퀴즈 문제 생성 ─────────────────────────────────
    question_prompt = HumanMessage(content=f"""오늘 학습한 내용을 바탕으로 퀴즈 3문제를 만들어주세요.

오늘 학습 주제: {current_topic}
오늘 학습 내용: {achievements}

요구사항:
- 핵심 개념을 확인할 수 있는 질문
- 단계적 난이도 (쉬움 → 보통 → 어려움)
- 번호를 붙여 명확하게 구분

퀴즈만 제시하고 답은 포함하지 마세요.""")

    quiz_questions = llm.invoke([system, question_prompt]).content

    # ── Step 2: interrupt — 사용자 답변 대기 ─────────────────
    user_answers = interrupt(
        f"📝 오늘의 퀴즈입니다!\n\n{quiz_questions}\n\n"
        f"위 문제들에 답해주세요. 모르는 건 '모르겠어요'라고 써도 괜찮아요 😊"
    )

    # ── Step 3: 답변 평가 + 피드백 ────────────────────────────
    eval_prompt = HumanMessage(content=f"""퀴즈 문제:
{quiz_questions}

학생 답변:
{user_answers}

각 문제에 대해 피드백을 주세요:
- 정답 여부
- 핵심 개념 보충 설명
- 격려 메시지

마지막에 전체 점수(X/3)와 다음 복습 토픽을 알려주세요.""")

    eval_res = llm.invoke([system, eval_prompt])

    # ── Step 4: quiz_history 업데이트 ─────────────────────────
    new_record = {
        "date":      date.today().isoformat(),
        "topic":     current_topic,
        "questions": quiz_questions,
        "answers":   user_answers,
        "feedback":  eval_res.content,
    }

    return {
        "messages":    [AIMessage(content=eval_res.content)],
        "quiz_history": quiz_history + [new_record],
    }


def diary_writer_node(state: LearnLogState) -> dict:
    """노드 5: 기분 + 성취를 묶어 일기 작성"""
    today        = date.today().strftime("%Y년 %m월 %d일")
    mood         = state.get("mood", "") or "평온한 하루"
    goal         = get_active_goal(state)
    achievements = state.get("today_achievements", "") or "오늘의 학습을 기록하지 않았어요"
    streak       = state.get("streak", 0)

    prompt = f"""오늘의 학습 일기를 작성해주세요.

날짜: {today}
오늘 기분: {mood}
학습 목표: {goal}
오늘 성취: {achievements}
연속 달성: {streak}일

형식:
---
📅 {today}
😊 오늘의 기분: [기분을 감성적으로 한 문장]
🎯 목표: [목표 요약]

✅ 오늘의 학습:
[오늘 한 내용]

💭 회고:
[배운 점, 느낀 점 2-3문장]

🔥 스트릭: {streak}일 연속 달성!

🌱 내일 계획:
[내일 할 일 1-2가지]
---

따뜻한 격려 메시지로 마무리해주세요."""

    response = llm.invoke([SystemMessage(content=prompt)])
    return {"messages": [response], "diary_content": response.content}


def notion_post_node(state: LearnLogState) -> dict:
    """노드 6: 일기를 Notion에 포스팅"""
    diary  = state.get("diary_content", "")
    goal   = get_active_goal(state)
    mood   = state.get("mood", "") or "평온한 하루"
    streak = state.get("streak", 0)

    if not diary:
        return {"messages": [AIMessage(content="일기 내용이 없어서 Notion 저장을 건너뛰었어요.")]}

    result = post_to_notion.invoke({
        "diary_content": diary,
        "learning_goal": goal,
        "mood":          mood,
        "streak":        streak,
    })

    if result.startswith("✅"):
        final = (
            f"📔 오늘의 학습 일기가 Notion에 저장됐어요!\n\n"
            f"{result}\n\n"
            f"오늘 기분 [{mood}] 이었는데도 공부하셨군요! 🌟\n\n"
            f"{streak}일 연속 달성 중이에요. 내일도 화이팅! 💪"
        )
    else:
        final = (
            f"⚠️ Notion 저장에 실패했어요.\n\n"
            f"{result}\n\n"
            f"오늘 학습은 정말 수고하셨어요! 🌟\n"
            f"내일 다시 시도해봐요. 화이팅! 💪"
        )
    return {"messages": [AIMessage(content=final)]}


# ══════════════════════════════════════════════════════════════
# Conditional Edge 함수
# ══════════════════════════════════════════════════════════════

def route_after_goal_check(state: LearnLogState) -> str:
    """CE1: 'setup' → goal_detail / 'skip' → checkin"""
    return state.get("next_action", "skip")


def route_after_checkin(state: LearnLogState) -> str:
    """CE2: 'search' → resource_search / 'quiz' → quiz / 'write' → diary_writer"""
    action = state.get("next_action", "write")
    if action == "search":
        return "resource_search"
    if action == "quiz":
        return "quiz"
    return "diary_writer"


def route_after_resource_search(state: LearnLogState) -> str:
    """CE3: 검색 후 퀴즈 제안 결과에 따라 분기"""
    return "quiz" if state.get("next_action") == "quiz" else "diary_writer"


# ══════════════════════════════════════════════════════════════
# Graph 빌드 & 컴파일
# ══════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(LearnLogState)

    # 노드 등록
    builder.add_node("mood_check",      mood_node)
    builder.add_node("goal_check",      goal_check_node)
    builder.add_node("goal_detail",     goal_detail_node)
    builder.add_node("goal_setup",      goal_setup_node)
    builder.add_node("checkin",         checkin_node)
    builder.add_node("resource_search", resource_search_node)
    builder.add_node("quiz",            quiz_node)
    builder.add_node("diary_writer",    diary_writer_node)
    builder.add_node("notion_post",     notion_post_node)

    # 엣지
    builder.add_edge(START,        "mood_check")
    builder.add_edge("mood_check", "goal_check")

    builder.add_conditional_edges(
        "goal_check",
        route_after_goal_check,
        {"setup": "goal_detail", "skip": "checkin"},
    )

    builder.add_edge("goal_detail", "goal_setup")
    builder.add_edge("goal_setup",  "checkin")

    builder.add_conditional_edges(
        "checkin",
        route_after_checkin,
        {
            "resource_search": "resource_search",
            "quiz":            "quiz",
            "diary_writer":    "diary_writer",
        },
    )

    builder.add_conditional_edges(
        "resource_search",
        route_after_resource_search,
        {"quiz": "quiz", "diary_writer": "diary_writer"},
    )

    builder.add_edge("quiz",         "diary_writer")
    builder.add_edge("diary_writer", "notion_post")
    builder.add_edge("notion_post",  END)

    conn   = sqlite3.connect("learnlog.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


# langgraph dev 가 이 변수를 import 해서 사용합니다
learnlog_graph = build_graph()
