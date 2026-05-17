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
from pydantic import BaseModel


def _get_secret(key: str, default: str = "") -> str:
    """Streamlit secrets 우선, 없으면 환경변수 폴백 (로컬 .env 포함)"""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# ── 환경변수 ────────────────────────────────────────────────────
NOTION_TOKEN       = _get_secret("NOTION_TOKEN")
NOTION_DATABASE_ID = _get_secret("NOTION_DATABASE_ID")

# ── LLM & Tool 초기화 ───────────────────────────────────────────
# OPENAI_API_KEY, TAVILY_API_KEY는 환경변수로 자동 탐지됨
# Streamlit Cloud에서는 secrets에서 읽어서 환경변수에 설정
_openai_key  = _get_secret("OPENAI_API_KEY")
_tavily_key  = _get_secret("TAVILY_API_KEY")
if _openai_key:
    os.environ.setdefault("OPENAI_API_KEY", _openai_key)
if _tavily_key:
    os.environ.setdefault("TAVILY_API_KEY", _tavily_key)

llm           = init_chat_model("openai:gpt-4o-mini")
tavily_search = TavilySearch(max_results=3)


# ══════════════════════════════════════════════════════════════
# Pydantic Models (Structured Output)
# ══════════════════════════════════════════════════════════════

class DomainAnalysis(BaseModel):
    domain: str
    subject: str
    level: str
    learning_style: str
    estimated_weeks: int
    prerequisites: list[str] = []
    is_framework: bool = False   # True이면 특정 라이브러리/프레임워크
    official_name: str = ""      # 프레임워크/라이브러리 공식 명칭

class DayPlan(BaseModel):
    day: int
    topic: str

class ResourceLink(BaseModel):
    title: str
    url: str

class WeekPhase(BaseModel):
    week: int
    theme: str
    checkpoint: str
    days: list[DayPlan]
    resources: list[ResourceLink] = []

class Curriculum(BaseModel):
    domain: str
    subject: str
    level: str
    total_weeks: int
    daily_minutes: int
    phases: list[WeekPhase]


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
    quiz_questions:     str         # 생성된 퀴즈 문제 (interrupt 재실행 방지용)
    progress_pct:       float       # 전체 진도율 (0.0 ~ 1.0)
    user_level:         str         # 사용자 학습 수준 (beginner/intermediate/advanced)
    entry_mode:         str         # "checkin" | "" — 진입 모드
    session_date:       str         # 오늘 세션 완료 날짜 (YYYY-MM-DD)
    domain_info:        dict        # domain_analysis_node → curriculum_build_node 중간값


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


def domain_analysis_node(state: LearnLogState) -> dict:
    """노드 2a: 도메인 분석 (DomainAnalysis Structured Output)"""
    new_goal   = state.get("active_goal", "")
    user_level = state.get("user_level", "beginner")

    domain_llm = llm.with_structured_output(DomainAnalysis)
    domain = domain_llm.invoke([HumanMessage(content=f"""학습 목표: "{new_goal}"
사용자 수준: "{user_level}"

위 학습 목표를 분석해주세요.
- domain: programming / language / science / art / other 중 하나
- subject: 구체적인 주제명
- level: 사용자 수준 그대로
- learning_style: conceptual / hands-on / mixed 중 하나
- estimated_weeks: 적정 학습 기간 (주 단위, 숫자)
- prerequisites: 사전 지식 목록 (없으면 빈 배열)
- is_framework: 학습 목표가 특정 라이브러리/프레임워크/툴 이름이면 true (예: LangGraph, React, FastAPI, PyTorch 등)
- official_name: is_framework가 true이면 공식 명칭 (예: "LangGraph by LangChain"), 아니면 빈 문자열

중요: "LangGraph"는 Python 기반 AI 에이전트 워크플로우 프레임워크입니다. Graph 자료구조 라이브러리가 아닙니다.""")])

    return {"domain_info": domain.model_dump()}


def curriculum_build_node(state: LearnLogState) -> dict:
    """노드 2b: 커리큘럼 생성 (Curriculum Structured Output, resources 포함)"""
    domain_info    = state.get("domain_info", {})
    d_subject      = domain_info.get("subject", state.get("active_goal", ""))
    d_level        = domain_info.get("level", "beginner")
    d_style        = domain_info.get("learning_style", "mixed")
    d_weeks        = domain_info.get("estimated_weeks", 4)
    d_domain       = domain_info.get("domain", "programming")
    is_framework   = domain_info.get("is_framework", False)
    official_name  = domain_info.get("official_name", "")

    framework_hint = (
        f"\n⚠️ 중요: {official_name}은 특정 프레임워크/라이브러리입니다. "
        f"일반 {d_domain} 이론이 아닌, 해당 프레임워크의 실제 API·개념·사용법 중심으로 커리큘럼을 작성하세요. "
        f"공식 문서 기반의 실습 위주 내용으로 구성하세요."
        if is_framework else ""
    )

    curriculum_llm = llm.with_structured_output(Curriculum)
    curriculum_obj = curriculum_llm.invoke([HumanMessage(content=f"""주제: {d_subject}
레벨: {d_level} / 학습 방식: {d_style} / 총 기간: {d_weeks}주
{framework_hint}

실제 학습 내용으로 채운 주차별 + 일별 커리큘럼을 만들어주세요.
- total_weeks: {d_weeks}
- daily_minutes: 60
- {d_weeks}주차 분량의 phases 생성
- 각 주차마다 5일치 days를 실제 학습 내용으로 채울 것
- 예시 placeholder 사용 금지, 실제 내용만 작성
- domain: {d_domain} / subject: {d_subject} / level: {d_level}

각 주차(phase)마다 resources 필드에 학습 자료 2~3개를 포함하세요:
- 공식 문서 루트 URL, GitHub 저장소, 잘 알려진 튜토리얼 사이트만
- 확실하지 않은 URL은 포함하지 마세요 (없으면 빈 배열)
- URL은 반드시 https://로 시작하는 실제 주소만
- 예시: LangGraph → https://langchain-ai.github.io/langgraph/""")])

    return {"curriculum": curriculum_obj.model_dump()}


def persona_build_node(state: LearnLogState) -> dict:
    """노드 2c: 습관 + 튜터 페르소나 생성 + 요약 메시지"""
    existing_goals = state.get("learning_goals", [])
    new_goal       = state.get("active_goal", "")
    user_level     = state.get("user_level", "beginner")
    curriculum     = state.get("curriculum", {})
    domain_info    = state.get("domain_info", {})

    updated_goals = (existing_goals + [new_goal]
                     if new_goal and new_goal not in existing_goals
                     else existing_goals)

    # ── Step 3: 일일 습관 분해 ──────────────────────────────────
    habits_res = llm.invoke([HumanMessage(content=
        f"학습 목표: {new_goal}\n"
        f"수준: {user_level}\n"
        f"하루 학습 시간: {curriculum.get('daily_minutes', 60)}분\n\n"
        f"매일 실천 가능한 습관 3가지를 간결하게 번호 목록으로만 출력해주세요."
    )])
    habits_text = habits_res.content.strip()

    # ── Step 4: 튜터 페르소나 생성 ──────────────────────────────
    first_phase = curriculum.get("phases", [{}])[0] if curriculum.get("phases") else {}
    first_topic = (first_phase.get("days", [{}])[0].get("topic")
                   or first_phase.get("theme", new_goal))
    learning_style = domain_info.get("learning_style", "mixed")

    persona_res = llm.invoke([HumanMessage(content=f"""당신은 최고의 교육 설계자입니다.
아래 정보를 바탕으로 AI 튜터의 시스템 프롬프트를 작성해주세요.

- 주제: {curriculum.get('subject', new_goal)} / 도메인: {curriculum.get('domain', '')}
- 수준: {curriculum.get('level', user_level)} / 학습 방식: {learning_style}
- 총 기간: {curriculum.get('total_weeks', 4)}주 / 오늘 주제: {first_topic}

요구사항:
- 해당 분야 전문가 AI 튜터 페르소나
- 도메인 특성에 맞는 교육 원칙 3~5가지
- 수준({curriculum.get('level', user_level)})에 맞는 접근 방식 명시
- 시스템 프롬프트만 작성, 다른 설명 없이""")])
    tutor_persona = persona_res.content.strip()

    # ── 통합 메시지 — 마크다운 테이블 (학습 자료 컬럼 포함) ────
    table  = "| 주차 | 주제 | 일별 학습 내용 | 완료 기준 | 학습 자료 |\n"
    table += "|------|------|----------------|-----------|----------|\n"
    for p in curriculum.get("phases", []):
        days_str = "<br>".join(
            f"• Day{d['day']}: {d['topic']}" for d in p.get("days", [])
        )
        week_res = p.get("resources", [])
        res_str  = "<br>".join(f"[{r['title']}]({r['url']})" for r in week_res) if week_res else "-"
        table += f"| {p['week']}주차 | {p['theme']} | {days_str} | {p.get('checkpoint', '')} | {res_str} |\n"

    summary_msg = (
        f"📚 **{curriculum.get('subject', new_goal)}** 학습 플랜을 준비했어요!\n\n"
        f"📋 **일일 습관:**\n{habits_text}\n\n"
        f"📅 **{curriculum.get('total_weeks', 4)}주 커리큘럼** (하루 {curriculum.get('daily_minutes', 60)}분)\n\n"
        f"{table}"
    )

    return {
        "messages":      [AIMessage(content=summary_msg)],
        "learning_goals": updated_goals,
        "active_goal":   new_goal,
        "tutor_persona": tutor_persona,
        "current_week":  1,
        "current_topic": first_topic,
        "progress_pct":  0.0,
        "quiz_history":  [],
    }


def checkin_node(state: LearnLogState) -> dict:
    """노드 3a: 오늘 학습 내용 입력 (interrupt)"""
    streak      = state.get("streak", 0)
    active_goal = get_active_goal(state)
    curriculum  = state.get("curriculum", {})
    prev_week   = state.get("current_week", 1)
    new_streak  = streak + 1

    # ── 진도 계산 ──────────────────────────────────────────────
    if curriculum.get("phases"):
        total_weeks  = curriculum.get("total_weeks", 1)
        new_week     = min((new_streak - 1) // 7 + 1, total_weeks)
        phases       = curriculum["phases"]
        phase        = next((p for p in phases if p["week"] == new_week), phases[-1])
        day_in_week  = ((new_streak - 1) % 7) + 1
        days         = phase.get("days", [])
        day_entry    = next((d for d in days if d["day"] == day_in_week), None)
        new_topic    = day_entry["topic"] if day_entry else phase["theme"]
        new_progress = round(min(new_streak / (total_weeks * 7), 1.0), 2)
    else:
        new_week     = prev_week
        new_topic    = state.get("current_topic", "")
        new_progress = state.get("progress_pct", 0.0)

    # ── 주차 변경 알림 ─────────────────────────────────────────
    week_up_msg = (
        f"\n\n🎉 {new_week}주차 시작! 오늘부터 [{new_topic}]을 배워요!"
        if new_week > prev_week else ""
    )

    # ── 오늘 주차 학습 자료 ────────────────────────────────────
    if curriculum.get("phases"):
        cur_phase   = next((p for p in curriculum["phases"] if p["week"] == new_week),
                           curriculum["phases"][-1])
        resources   = cur_phase.get("resources", [])
        resource_lines = "\n".join(f"🔗 [{r['title']}]({r['url']})" for r in resources)
        resource_section = f"\n\n📚 학습 자료:\n{resource_lines}\n" if resource_lines else ""
    else:
        resource_section = ""

    streak_msg = f"🔥 {streak}일 연속 달성 중!" if streak > 0 else "🌱 오늘부터 시작이에요!"
    topic_line = f"📖 오늘의 주제: **{new_topic}**\n" if new_topic else ""
    topic_name = new_topic if new_topic else active_goal

    question = (
        f"{streak_msg}{week_up_msg}\n\n"
        f"{topic_line}"
        f"{resource_section}\n"
        f"자료를 학습하신 후, 오늘 [{topic_name}]에서 배운 내용을 자유롭게 적어주세요. 😊"
    )

    user_input = interrupt(question)

    return {
        "messages":           [HumanMessage(content=user_input)],
        "today_achievements":  user_input,
        "streak":              new_streak,
        "current_week":        new_week,
        "current_topic":       new_topic,
        "progress_pct":        new_progress,
    }


def checkin_response_node(state: LearnLogState) -> dict:
    """노드 3b: 튜터 응답 생성 (LLM only, no interrupt)"""
    tutor_persona    = state.get("tutor_persona", "당신은 친절한 학습 튜터입니다.")
    today_achievements = state.get("today_achievements", "")
    current_topic    = state.get("current_topic", "")
    active_goal      = get_active_goal(state)

    response = llm.invoke([
        SystemMessage(content=tutor_persona),
        HumanMessage(content=(
            f"학습 주제: {current_topic or active_goal}\n"
            f"오늘 배운 내용: {today_achievements}\n\n"
            f"학습자의 오늘 성취를 격려하고, 핵심 개념을 짚어주는 튜터 응답을 해주세요. (3~5문장)"
        )),
    ])

    return {"messages": [AIMessage(content=response.content)]}


def checkin_action_node(state: LearnLogState) -> dict:
    """노드 3c: 다음 액션 선택 (interrupt — app.py에서 버튼으로 렌더링)"""
    action = interrupt("__ACTION_SELECT__")
    # action: "search" | "quiz" | "diary"
    next_action = {
        "search": "search",
        "quiz":   "quiz",
        "diary":  "write",
    }.get(action, "write")
    return {"next_action": next_action}


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
    current_topic = state.get("current_topic", "") or goal

    quiz_offer = interrupt(
        f"{response.content}\n\n"
        f"---\n"
        f"📝 오늘 주제 [{current_topic}]에 대한 퀴즈를 풀어볼까요? (예 / 아니요)"
    )

    wants_quiz = any(kw in quiz_offer.lower() for kw in
                     ["예", "네", "응", "ㅇ", "좋아", "yes", "y", "퀴즈", "해줘"])

    return {
        "messages":       [],
        "search_results": formatted,
        "next_action":    "quiz" if wants_quiz else "write",
    }


def _get_review_topics(quiz_history: list, current_topic: str) -> list[dict]:
    """quiz_history에서 복습이 필요한 토픽을 추출 (스페이스드 리피티션)

    복습 조건:
    - 약점: 피드백에 부정 키워드가 포함된 항목
    - 오래됨: 마지막 퀴즈로부터 3일 이상 지난 항목
    오늘 주제와 중복되는 항목은 제외.
    """
    if not quiz_history:
        return []

    weak_keywords = ["아쉬워요", "틀렸", "모르겠", "incorrect", "wrong", "오답", "아쉽"]
    today         = date.today()
    seen_topics   = set()
    review_topics = []

    for record in reversed(quiz_history):  # 최근 기록부터
        topic = record.get("topic", "")
        if topic == current_topic or topic in seen_topics:
            continue
        seen_topics.add(topic)

        feedback     = record.get("feedback", "").lower()
        record_date  = record.get("date", "")
        is_weak      = any(kw in feedback for kw in weak_keywords)
        days_ago     = 999
        try:
            days_ago = (today - date.fromisoformat(record_date)).days
        except Exception:
            pass

        if is_weak or days_ago >= 3:
            review_topics.append({
                "topic":    topic,
                "days_ago": days_ago,
                "is_weak":  is_weak,
            })
            if len(review_topics) >= 2:  # 최대 2개까지만
                break

    return review_topics


def quiz_generate_node(state: LearnLogState) -> dict:
    """노드 4-2a: 퀴즈 문제 생성 (LLM만, interrupt 없음)
    interrupt 재실행 시 LLM이 재호출되어 문제가 바뀌는 문제를 방지하기 위해
    quiz_node에서 분리. 생성된 문제는 state에 저장.
    스페이스드 리피티션: quiz_history 기반 복습 토픽을 함께 포함.
    """
    tutor_persona = state.get("tutor_persona", "")
    goal          = get_active_goal(state)
    achievements  = state.get("today_achievements", "") or goal
    current_topic = state.get("current_topic", goal)
    quiz_history  = state.get("quiz_history", [])

    system = (SystemMessage(content=tutor_persona) if tutor_persona
              else SystemMessage(content=f"당신은 {goal} 전문 튜터입니다."))

    # ── 복습 토픽 추출 ──────────────────────────────────────────
    review_topics = _get_review_topics(quiz_history, current_topic)

    if review_topics:
        def _label(r):
            return "약점 토픽" if r["is_weak"] else f"{r['days_ago']}일 전 학습"
        review_lines = "\n".join([
            f"- {r['topic']} ({_label(r)})"
            for r in review_topics
        ])
        review_section = f"\n\n[복습이 필요한 이전 토픽]\n{review_lines}"
        distribution   = "- 2문제: 오늘 주제\n- 1문제: 복습 토픽 중 하나"
    else:
        review_section = ""
        distribution   = "- 3문제: 오늘 주제"

    question_prompt = HumanMessage(content=f"""오늘 학습한 내용을 바탕으로 퀴즈 3문제를 만들어주세요.

오늘 학습 주제: {current_topic}
오늘 학습 내용: {achievements}{review_section}

문제 구성:
{distribution}

요구사항:
- 핵심 개념을 확인할 수 있는 질문
- 단계적 난이도 (쉬움 → 보통 → 어려움)
- 번호를 붙여 명확하게 구분하고 난이도는 질문 맨 뒤에 이어붙여주세요.
- 복습 문제는 끝에 (복습) 표시를 붙여주세요.

퀴즈만 제시하고 답은 포함하지 마세요.""")

    quiz_questions = llm.invoke([system, question_prompt]).content

    return {"quiz_questions": quiz_questions}


def quiz_node(state: LearnLogState) -> dict:
    """노드 4-2b: 퀴즈 제시 (interrupt) + 답변 평가"""
    tutor_persona  = state.get("tutor_persona", "")
    goal           = get_active_goal(state)
    current_topic  = state.get("current_topic", goal)
    quiz_history   = state.get("quiz_history", [])
    quiz_questions = state.get("quiz_questions", "")  # quiz_generate_node에서 저장한 문제

    system = (SystemMessage(content=tutor_persona) if tutor_persona
              else SystemMessage(content=f"당신은 {goal} 전문 튜터입니다."))

    # ── Step 1: interrupt — 사용자 답변 대기 (LLM 호출 없음) ──
    user_answers = interrupt(
        f"📝 오늘의 퀴즈입니다!\n\n{quiz_questions}\n\n"
        f"위 문제들에 답해주세요. 모르는 건 '모르겠어요'라고 써도 괜찮아요 😊"
    )

    # ── Step 2: 답변 평가 + 피드백 ────────────────────────────
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

    # ── Step 3: quiz_history 업데이트 ─────────────────────────
    new_record = {
        "date":      date.today().isoformat(),
        "topic":     current_topic,
        "questions": quiz_questions,
        "answers":   user_answers,
        "feedback":  eval_res.content,
    }

    return {
        "messages":     [AIMessage(content=eval_res.content)],
        "quiz_history": quiz_history + [new_record],
        "quiz_questions": "",  # 사용 후 초기화
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
        return {
                "messages": [AIMessage(content="일기 내용이 없어서 Notion 저장을 건너뛰었어요.")],
                "session_date": date.today().isoformat(),   
                }

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
            f"{streak}일 연속 달성 중이에요. 내일도 화이팅! 💪"
        )
    else:
        final = (
            f"⚠️ Notion 저장에 실패했어요.\n\n"
            f"{result}\n\n"
            f"오늘 학습은 정말 수고하셨어요! 🌟\n"
            f"내일 다시 시도해봐요. 화이팅! 💪"
        )
    return {
        "messages":     [AIMessage(content=final)],
        "session_date": date.today().isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# Conditional Edge 함수
# ══════════════════════════════════════════════════════════════

def route_entry(state: LearnLogState) -> str:
    """START 진입 모드 분기: 체크인 직행 vs 전체 flow"""
    if state.get("entry_mode") == "checkin":
        return "checkin"
    return "full"


def route_after_goal_check(state: LearnLogState) -> str:
    """CE1: 'setup' → goal_detail / 'skip' → checkin"""
    return state.get("next_action", "skip")


def route_after_checkin(state: LearnLogState) -> str:
    """CE2: 'search' → resource_search / 'quiz' → quiz_generate / 'write' → diary_writer"""
    action = state.get("next_action", "write")
    if action == "search":
        return "resource_search"
    if action == "quiz":
        return "quiz_generate"
    return "diary_writer"


def route_after_resource_search(state: LearnLogState) -> str:
    """CE3: 검색 후 퀴즈 제안 결과에 따라 분기"""
    return "quiz_generate" if state.get("next_action") == "quiz" else "diary_writer"


# ══════════════════════════════════════════════════════════════
# Graph 빌드 & 컴파일
# ════════════════════════════════════════════════════════
def build_graph():
    builder = StateGraph(LearnLogState)

    # 노드 등록
    builder.add_node("mood_check",       mood_node)
    builder.add_node("goal_check",       goal_check_node)
    builder.add_node("goal_detail",      goal_detail_node)
    builder.add_node("domain_analysis",  domain_analysis_node)
    builder.add_node("curriculum_build", curriculum_build_node)
    builder.add_node("persona_build",    persona_build_node)
    builder.add_node("checkin",          checkin_node)
    builder.add_node("checkin_response", checkin_response_node)
    builder.add_node("checkin_action",   checkin_action_node)
    builder.add_node("resource_search",  resource_search_node)
    builder.add_node("quiz_generate",    quiz_generate_node)
    builder.add_node("quiz",             quiz_node)
    builder.add_node("diary_writer",     diary_writer_node)
    builder.add_node("notion_post",      notion_post_node)

    # 엣지
    builder.add_conditional_edges(
        START,
        route_entry,
        {"full": "mood_check", "checkin": "checkin"},
    )
    builder.add_edge("mood_check", "goal_check")

    builder.add_conditional_edges(
        "goal_check",
        route_after_goal_check,
        {"setup": "goal_detail", "skip": "checkin"},
    )

    builder.add_edge("goal_detail",      "domain_analysis")
    builder.add_edge("domain_analysis",  "curriculum_build")
    builder.add_edge("curriculum_build", "persona_build")
    builder.add_edge("persona_build",    "checkin")
    builder.add_edge("checkin",          "checkin_response")
    builder.add_edge("checkin_response", "checkin_action")

    builder.add_conditional_edges(
        "checkin_action",
        route_after_checkin,
        {
            "resource_search": "resource_search",
            "quiz_generate":   "quiz_generate",
            "diary_writer":    "diary_writer",
        },
    )

    builder.add_conditional_edges(
        "resource_search",
        route_after_resource_search,
        {"quiz_generate": "quiz_generate", "diary_writer": "diary_writer"},
    )

    builder.add_edge("quiz_generate", "quiz")
    builder.add_edge("quiz",          "diary_writer")
    builder.add_edge("diary_writer",  "notion_post")
    builder.add_edge("notion_post",   END)

    conn   = sqlite3.connect("learnlog.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


# langgraph dev 가 이 변수를 import 해서 사용합니다
learnlog_graph = build_graph()
