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
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False

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

try:
    llm = init_chat_model("openai:gpt-4o-mini")
except Exception:
    llm = None  # type: ignore  # 테스트 환경에서 mock으로 대체

try:
    tavily_search = TavilySearch(max_results=3)
except Exception:
    tavily_search = None  # type: ignore


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

class WeekPhase(BaseModel):
    week: int
    theme: str
    checkpoint: str
    days: list[DayPlan]

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
    user_level:         str         # 사용자 학습 수준 (beginner/intermediate/advanced)
    entry_mode:         str         # "checkin" | "" — 진입 모드
    session_date:       str         # 오늘 세션 완료 날짜 (YYYY-MM-DD)
    curriculum_feedback: str        # 사용자 커리큘럼 피드백 (재생성 시 반영)
    # ── 튜터 Q&A 세션 ────────────────────────────────────────────
    tutor_qa_index:     int         # 현재 질문 번호 (0-based)
    ready_for_quiz:     bool        # 튜터 퀴즈 진입 신호 (LLM 판단)
    tutor_qa_question:  str         # 현재 질문 (interrupt 재실행 방지)
    tutor_qa_answer:    str         # 현재 답변
    tutor_qa_history:   List[dict]  # [{question, answer, feedback}, ...]


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
            return f"✅ Posted to Notion! {res.json().get('url', '')}"
        return f"⚠️ Error ({res.status_code}): {res.text[:200]}"
    except Exception as e:
        return f"❌ Connection error: {e}"


# ══════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════

def mood_node(state: LearnLogState) -> dict:
    """Node 0: Mood check (interrupt)"""
    streak = state.get("streak", 0)

    greeting = (
        f"🎉 {streak} day streak going strong!\n\nHow are you feeling today?"
        if streak else
        "Welcome to LearnLog! 📚\n\nHow are you feeling today?"
    )
    mood_input = interrupt(greeting)

    mood_res = llm.invoke([HumanMessage(content=
        f'User mood: "{mood_input}"\n\n'
        f'Write a warm, empathetic 1-2 sentence response to their mood in English.\n'
        f'Then on a new line write "MOOD_SUMMARY:" followed by a concise mood phrase.\n\n'
        f'Example:\n'
        f'That sounds exhausting! Rest is part of the process too. 💙\n'
        f'MOOD_SUMMARY: tired but resilient'
    )])

    content = mood_res.content.strip()
    if "MOOD_SUMMARY:" in content:
        parts         = content.split("MOOD_SUMMARY:", 1)
        warm_response = parts[0].strip()
        mood_summary  = parts[1].strip()
    else:
        warm_response = content
        mood_summary  = content

    return {
        "messages": [
            HumanMessage(content=mood_input),
            AIMessage(content=warm_response),
        ],
        "mood": mood_summary,
    }


def goal_check_node(state: LearnLogState) -> dict:
    """Node 1: Check goal — keep or change (interrupt)"""
    existing_goals = state.get("learning_goals", [])

    if existing_goals:
        goals_str     = ", ".join(f"'{g}'" for g in existing_goals)
        goal_question = (
            f"Current goal(s): [{goals_str}]\n\n"
            f"Would you like to continue with your existing goal, or set a new one?"
        )
    else:
        goal_question = "What would you like to learn? 🎯"

    goal_input = interrupt(goal_question)

    goal_res = llm.invoke([HumanMessage(content=f"""User response: "{goal_input}"
Existing goals: {existing_goals if existing_goals else 'none'}

Respond in JSON:
{{
  "wants_new_goal": true or false,
  "goal_text": "concise topic name only (e.g. 'LangGraph', 'Python basics', 'Spanish'). null if none"
}}

Rules:
- 'new', 'add', 'change', 'switch', 'different goal' → wants_new_goal: true
- 'continue', 'keep', 'same', no change requested    → wants_new_goal: false
- no existing goals                                   → wants_new_goal: true

Respond with JSON only.""")])

    parsed    = parse_json(goal_res.content)
    if parsed:
        wants_new = parsed.get("wants_new_goal", not bool(existing_goals))
        goal_text = parsed.get("goal_text") or ""
    else:
        change_kw = ["new", "add", "change", "switch", "different"]
        wants_new = any(kw in goal_input.lower() for kw in change_kw) or not existing_goals
        goal_text = goal_input

    return {
        "messages":    [HumanMessage(content=goal_input)],
        "active_goal": goal_text,
        "next_action": "setup" if wants_new else "skip",
    }


def goal_detail_node(state: LearnLogState) -> dict:
    """Node 1-2: Collect learning level and focus area (interrupt)"""
    goal = state.get("active_goal", "")

    question = (
        f"🎯 Goal: {goal}\n\n"
        f"To build your personalized learning plan, I'd love to know a bit more!\n\n"
        f"① What's your current level?\n"
        f"   (Beginner / Intermediate / Advanced)\n\n"
        f"② Any specific area you'd like to focus on?\n"
        f"   (Type 'none' if not sure)"
    )

    user_detail = interrupt(question)

    detail_res = llm.invoke([HumanMessage(content=f"""User response: "{user_detail}"

Extract as JSON:
{{
  "level": "beginner / intermediate / advanced",
  "focus": "brief summary of focus area (null if none)"
}}

Level rules:
- total beginner, never tried, no idea → beginner
- some basics, a little experience     → intermediate
- experienced, want deep dive          → advanced

Respond with JSON only.""")])

    parsed = parse_json(detail_res.content)
    if parsed:
        level = parsed.get("level", "beginner")
        focus = parsed.get("focus") or ""
    else:
        level = "beginner"
        focus = ""

    refined_goal = f"{goal} ({focus})" if focus else goal

    return {
        "messages":    [HumanMessage(content=user_detail)],
        "user_level":  level,
        "active_goal": refined_goal,
    }


def curriculum_build_node(state: LearnLogState) -> dict:
    """노드 2a+b: 도메인 분석 + 커리큘럼 생성 (LLM only, no interrupt)"""
    new_goal   = state.get("active_goal", "")
    user_level = state.get("user_level", "beginner")
    feedback   = state.get("curriculum_feedback", "")

    # ── 도메인 분석 ──────────────────────────────────────────────
    domain_llm = llm.with_structured_output(DomainAnalysis)
    domain = domain_llm.invoke([HumanMessage(content=f"""Learning goal: "{new_goal}"
User level: "{user_level}"

Analyze the learning goal above.
- domain: one of programming / language / science / art / other
- subject: specific topic name
- level: same as user level
- learning_style: one of conceptual / hands-on / mixed
- estimated_weeks: recommended study duration in weeks (number)
- prerequisites: list of prior knowledge needed (empty array if none)
- is_framework: true if the goal is a specific library/framework/tool (e.g. LangGraph, React, FastAPI, PyTorch)
- official_name: if is_framework is true, the official name (e.g. "LangGraph by LangChain"), otherwise empty string

Important: "LangGraph" is a Python-based AI agent workflow framework, NOT a graph data structure library.""")])

    d_subject     = domain.subject
    d_level       = domain.level
    d_style       = domain.learning_style
    d_weeks       = domain.estimated_weeks
    d_domain      = domain.domain
    is_framework  = domain.is_framework
    official_name = domain.official_name

    framework_hint = (
        f"\n⚠️ Important: {official_name} is a specific framework/library. "
        f"Focus the curriculum on its actual APIs, concepts, and usage — not general {d_domain} theory. "
        f"Structure content around hands-on practice with official documentation."
        if is_framework else ""
    )
    feedback_hint = (
        f"\n\n⚠️ The user reviewed the previous curriculum and requested the following changes:\n"
        f"{feedback}\nPlease revise the curriculum accordingly."
        if feedback else ""
    )

    # ── 커리큘럼 생성 ────────────────────────────────────────────
    curriculum_llm = llm.with_structured_output(Curriculum)
    curriculum_obj = curriculum_llm.invoke([HumanMessage(content=f"""Topic: {d_subject}
Level: {d_level} / Learning style: {d_style} / Duration: {d_weeks} weeks
{framework_hint}{feedback_hint}

Create a detailed week-by-week, day-by-day curriculum with real content.
- total_weeks: {d_weeks}
- daily_minutes: 60
- Generate {d_weeks} phases (one per week)
- Each week must have 5 days filled with actual learning content
- No placeholder text — write real, specific topics only
- domain: {d_domain} / subject: {d_subject} / level: {d_level}
- Write all content in English""")])

    return {"curriculum": curriculum_obj.model_dump()}


def curriculum_confirm_node(state: LearnLogState) -> dict:
    """Node 2d: Show curriculum to user and collect confirmation or feedback"""
    curriculum = state.get("curriculum", {})

    table  = "Here's your personalized curriculum! 🎓\n\n"
    table += "| Week | Theme | Daily Topics | Checkpoint |\n"
    table += "|------|-------|--------------|------------|\n"
    for p in curriculum.get("phases", []):
        days_str = "<br>".join(f"• Day{d['day']}: {d['topic']}" for d in p.get("days", []))
        table  += f"| Week {p['week']} | {p['theme']} | {days_str} | {p.get('checkpoint', '')} |\n"

    table += (
        "\n\nDoes this look good? Feel free to let me know if you'd like any changes "
        "— or just say **yes** to get started! 🚀"
    )

    response = interrupt(table)
    lower    = response.strip().lower()

    confirmed = any(w in lower for w in [
        "yes", "ok", "good", "great", "perfect", "looks good",
        "fine", "proceed", "go", "start", "sure", "sounds good",
    ])

    if confirmed:
        return {"next_action": "confirm", "curriculum_feedback": ""}
    else:
        return {"next_action": "revise", "curriculum_feedback": response}


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

    # ── Tutor persona ───────────────────────────────────────────
    first_phase = curriculum.get("phases", [{}])[0] if curriculum.get("phases") else {}
    first_topic = (first_phase.get("days", [{}])[0].get("topic")
                   or first_phase.get("theme", new_goal))
    learning_style = domain_info.get("learning_style", "mixed")

    persona_res = llm.invoke([HumanMessage(content=f"""You are a world-class instructional designer.
Write a system prompt for an AI tutor based on the information below.

- Subject: {curriculum.get('subject', new_goal)} / Domain: {curriculum.get('domain', '')}
- Level: {curriculum.get('level', user_level)} / Learning style: {learning_style}
- Duration: {curriculum.get('total_weeks', 4)} weeks / Today's topic: {first_topic}

Requirements:
- Expert AI tutor persona in this field
- 3–5 teaching principles suited to this domain
- Approach tailored to {curriculum.get('level', user_level)} learners
- Always respond in English
- Output the system prompt only, no extra explanation""")])
    tutor_persona = persona_res.content.strip()

    # ── Summary message — markdown table ─────────────────────────
    table  = "| Week | Theme | Daily Topics | Checkpoint |\n"
    table += "|------|-------|--------------|------------|\n"
    for p in curriculum.get("phases", []):
        days_str = "<br>".join(
            f"• Day{d['day']}: {d['topic']}" for d in p.get("days", [])
        )
        table += f"| Week {p['week']} | {p['theme']} | {days_str} | {p.get('checkpoint', '')} |\n"

    summary_msg = (
        f"✅ Great! Your **{curriculum.get('subject', new_goal)}** learning plan is all set.\n\n"
        f"📅 {curriculum.get('total_weeks', 4)} weeks · {curriculum.get('daily_minutes', 60)} min/day\n\n"
        f"Let's kick off **Week 1** with today's topic: **{first_topic}** 🚀"
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


def checkin_topic_node(state: LearnLogState) -> dict:
    """노드 3a: 오늘 토픽 계산 + Tavily 검색 (interrupt 없음)"""
    streak     = state.get("streak", 0)
    curriculum = state.get("curriculum", {})
    new_streak = streak + 1

    if curriculum.get("phases"):
        total_weeks = curriculum.get("total_weeks", 1)
        new_week    = min((new_streak - 1) // 7 + 1, total_weeks)
        phases      = curriculum["phases"]
        phase       = next((p for p in phases if p["week"] == new_week), phases[-1])
        day_in_week = ((new_streak - 1) % 7) + 1
        days        = phase.get("days", [])
        day_entry   = next((d for d in days if d["day"] == day_in_week), None)
        new_topic   = day_entry["topic"] if day_entry else phase["theme"]
    else:
        new_week  = state.get("current_week", 1)
        new_topic = state.get("current_topic", "")

    active_goal    = get_active_goal(state)
    search_results = ""
    if new_topic and tavily_search:
        try:
            query   = f"{active_goal} {new_topic} tutorial guide"
            results = tavily_search.invoke(query)
            # TavilySearch returns Dict{"query":..., "results":[...]}
            if isinstance(results, dict) and "results" in results:
                items = results["results"]
            elif isinstance(results, list):
                items = results
            else:
                items = []
            lines = [
                f"[{r.get('title', '')}]({r.get('url', '')})"
                for r in items
                if isinstance(r, dict) and r.get("title") and r.get("url")
            ]
            search_results = "\n\n".join(lines[:3])
        except Exception:
            pass

    return {
        "search_results":    search_results,
        "tutor_qa_index":    0,
        "ready_for_quiz":    False,
        "tutor_qa_history":  [],
        "tutor_qa_question": "",
        "tutor_qa_answer":   "",
        "streak":            new_streak,   # 사이드바 즉시 반영
        "current_week":      new_week,
        "current_topic":     new_topic,
    }


def checkin_node(state: LearnLogState) -> dict:
    """노드 3b: 오늘 학습 시작 체크인 (interrupt) — 진도는 checkin_topic_node에서 미리 계산"""
    streak       = state.get("streak", 0)        # checkin_topic_node에서 이미 증가된 값
    active_goal  = get_active_goal(state)
    curriculum   = state.get("curriculum", {})
    new_week     = state.get("current_week", 1)  # checkin_topic_node에서 저장
    new_topic    = state.get("current_topic", "") # checkin_topic_node에서 저장

    # ── New week notification (streak 기준: 7의 배수+1이면 새 주 시작) ──
    total_weeks = curriculum.get("total_weeks", 1)
    prev_week   = min((streak - 2) // 7 + 1, total_weeks) if streak > 1 else 0
    week_up_msg = (
        f"\n\n🎉 Week {new_week} starts today! You'll be learning [{new_topic}]!"
        if new_week > prev_week else ""
    )

    streak_msg = f"🔥 {streak - 1} day streak!" if streak > 1 else "🌱 Let's start your first day!"
    topic_line = f"📖 Today's topic: **{new_topic}**\n" if new_topic else ""

    search_results = state.get("search_results", "")
    resources_section = ""
    if search_results:
        resources_section = f"\n\n📚 **Today's Resources:**\n\n{search_results}"

    question = (
        f"{streak_msg}{week_up_msg}\n\n"
        f"{topic_line}"
        f"{resources_section}\n\n"
        f"Ready to dive into today's lesson? 🚀"
    )

    user_input = interrupt(question)

    return {
        "messages":           [HumanMessage(content=user_input)],
        "today_achievements":  user_input,
        # streak / current_week / current_topic: checkin_topic_node에서 이미 저장
    }





def tutor_qa_generate_node(state: LearnLogState) -> dict:
    """Node 3b: Turn1=수준파악+수업시작, Turn2+=답변응답+진도추적+ready판단 (LLM, no interrupt)"""
    MIN_TURNS = 2
    MAX_TURNS = 5

    tutor_persona = state.get("tutor_persona", "You are a friendly learning tutor.")
    topic         = state.get("current_topic", "") or get_active_goal(state)
    active_goal   = get_active_goal(state)
    current_week  = state.get("current_week", 1)
    streak        = state.get("streak", 0)
    index         = state.get("tutor_qa_index", 0)
    history       = state.get("tutor_qa_history", [])

    # 커리큘럼 컨텍스트 추출
    curriculum  = state.get("curriculum", {})
    total_weeks = curriculum.get("total_weeks", 4)
    phases      = curriculum.get("phases", [])
    phase       = next((p for p in phases if p["week"] == current_week), {})
    week_theme  = phase.get("theme", "")
    checkpoint  = phase.get("checkpoint", "")
    all_days    = phase.get("days", [])

    # 오늘이 이번 주 몇 번째 날인지 (1~7)
    day_in_week     = ((streak - 1) % 7) + 1 if streak > 0 else 1
    remaining_days  = [d["topic"] for d in all_days if d.get("day", 0) > day_in_week]

    if index == 0:
        # ── Turn 1: 수준 파악 → 오늘 범위 소개 → 첫 개념 진입 ──────
        remaining_str = (f"- Remaining days this week: {', '.join(remaining_days)}"
                         if remaining_days else "- Last day of this week")
        prompt = (
            f"You are teaching {topic} to a student working toward: {active_goal}.\n\n"
            f"[Lesson Context]\n"
            f"- Week {current_week} of {total_weeks}: {week_theme}\n"
            f"- Today (Day {day_in_week}/7): {topic}\n"
            f"- Week mastery goal: {checkpoint}\n"
            f"{remaining_str}\n\n"
            f"[Your task — conversational, NOT lecture-style]\n"
            f"1. Start with ONE short diagnostic question to gauge what the student "
            f"already knows about '{topic}'\n"
            f"2. Give a concise 2-3 sentence intro: what it is and why it matters today\n"
            f"3. Introduce the first core concept with a real, concrete example\n"
            f"4. End with ONE open question to check understanding\n\n"
            f"Keep it warm and conversational. In English."
        )
        res = llm.invoke([SystemMessage(content=tutor_persona),
                          HumanMessage(content=prompt)])
        return {
            "messages":          [AIMessage(content=res.content)],
            "tutor_qa_question": res.content,
            "ready_for_quiz":    False,
        }
    else:
        # ── Turn 2+: 답변 응답 + 진도 추적 + ready_for_quiz 판단 ──────
        history_text = "\n".join([
            f"Tutor: {h['question']}\nStudent: {h['answer']}"
            for h in history
        ])
        prompt = (
            f"You are teaching {topic} (Week {current_week}: {week_theme}) "
            f"to a student.\n\n"
            f"[Today's target]\n"
            f"- Core topic: {topic}\n"
            f"- Mastery goal: {checkpoint}\n"
            f"- Exchange {index} of max {MAX_TURNS}\n\n"
            f"[Conversation so far]\n"
            f"{history_text}\n\n"
            f"[Your task]\n"
            f"1. Respond to the student's last answer:\n"
            f"   - Correct → affirm briefly, then go deeper or move to next concept\n"
            f"   - Partial → acknowledge what's right, then fill the gap\n"
            f"   - Wrong → reteach with a different angle or simpler example\n"
            f"2. Keep advancing — don't re-explain what's already understood\n"
            f"3. End with ONE focused question or hands-on prompt\n\n"
            f"Set ready_for_quiz: true ONLY when:\n"
            f"- The core concept of '{topic}' has been sufficiently covered\n"
            f"- Student has demonstrated basic understanding\n"
            f"- At least {MIN_TURNS} exchanges done (current: {index})\n\n"
            f"Respond in JSON only:\n"
            f'{{"message": "...", "ready_for_quiz": false}}'
        )
        res    = llm.invoke([SystemMessage(content=tutor_persona),
                             HumanMessage(content=prompt)])
        parsed = parse_json(res.content) or {}
        msg    = parsed.get("message") or res.content
        ready  = bool(parsed.get("ready_for_quiz", False))
        if index >= MAX_TURNS:
            ready = True
        return {
            "messages":          [AIMessage(content=msg)],
            "tutor_qa_question": msg,
            "ready_for_quiz":    ready,
        }


def tutor_qa_answer_node(state: LearnLogState) -> dict:
    """Node 3c: 사용자 답변 대기 (interrupt)"""
    user_answer = interrupt("__TUTOR_QA__")
    return {
        "messages":        [HumanMessage(content=user_answer)],
        "tutor_qa_answer": user_answer,
    }


def tutor_qa_feedback_node(state: LearnLogState) -> dict:
    """Node 3d: 답변을 history에 저장 + index 증가 (LLM 없음 — generate_node가 응답 담당)"""
    index    = state.get("tutor_qa_index", 0)
    question = state.get("tutor_qa_question", "")
    answer   = state.get("tutor_qa_answer", "")
    history  = state.get("tutor_qa_history", [])

    new_record = {"question": question, "answer": answer}
    return {
        "tutor_qa_history": history + [new_record],
        "tutor_qa_index":   index + 1,
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

    weak_keywords = ["incorrect", "wrong", "not sure", "don't know", "mistake", "error", "missed"]
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
              else SystemMessage(content=f"You are an expert tutor for {goal}."))

    # ── 복습 토픽 추출 ──────────────────────────────────────────
    review_topics = _get_review_topics(quiz_history, current_topic)

    if review_topics:
        def _label(r):
            return "weak topic" if r["is_weak"] else f"studied {r['days_ago']} days ago"
        review_lines = "\n".join([
            f"- {r['topic']} ({_label(r)})"
            for r in review_topics
        ])
        review_section = f"\n\n[Topics needing review]\n{review_lines}"
        distribution   = "- 2 questions: today's topic\n- 1 question: a review topic"
    else:
        review_section = ""
        distribution   = "- 3 questions: today's topic"

    question_prompt = HumanMessage(content=f"""Create 3 quiz questions based on today's study session.

Today's topic: {current_topic}
What was studied: {achievements}{review_section}

Question distribution:
{distribution}

Requirements:
- Questions that test understanding of key concepts
- Progressive difficulty (easy → medium → hard)
- Number each question clearly; add difficulty in parentheses at the end
- Mark review questions with (Review) at the end
- Write all questions in English

Present questions only — do not include answers.""")

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
              else SystemMessage(content=f"You are an expert tutor for {goal}."))

    # ── Step 1: interrupt — wait for user answers (no LLM call) ─
    user_answers = interrupt(
        f"📝 Here's your quiz!\n\n{quiz_questions}\n\n"
        f"Answer each question below. It's okay to write 'I don't know' if you're unsure 😊"
    )

    # ── Step 2: Evaluate answers + feedback ───────────────────
    eval_prompt = HumanMessage(content=f"""Quiz questions:
{quiz_questions}

Student answers:
{user_answers}

For each question, provide feedback in English:
- Whether the answer is correct
- A brief explanation of the key concept
- An encouraging message

At the end, give the total score (X/3) and suggest a topic to review next.""")

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
    today        = date.today().strftime("%B %d, %Y")
    goal         = get_active_goal(state)
    tutor_history = state.get("tutor_qa_history", [])
    if tutor_history:
        covered      = "\n".join([f"- {h['answer'][:120]}" for h in tutor_history])
        achievements = f"Topics covered in tutoring session:\n{covered}"
    else:
        achievements = state.get("today_achievements", "") or "No learning recorded for today"
    streak       = state.get("streak", 0)

    prompt = f"""Write today's learning journal entry in English.

Date: {today}
Learning goal: {goal}
Today's achievements: {achievements}
Streak: {streak} days

Format:
---
📅 {today}
🎯 Goal: [brief goal summary]

✅ Today's Learning:
[what was studied today]

💭 Reflection:
[2–3 sentences on what was learned and how it felt]

🔥 Streak: {streak} days in a row!

🌱 Tomorrow's Plan:
[1–2 things to do tomorrow]
---

Close with a warm, encouraging message in English."""

    response = llm.invoke([SystemMessage(content=prompt)])
    return {
        "messages":     [AIMessage(content="✍️ Your diary is ready! Review it below.")],
        "diary_content": response.content,
    }


def diary_confirm_node(state: LearnLogState) -> dict:
    """Node 5b: Show diary draft to user — allow editing before posting"""
    diary    = state.get("diary_content", "")
    response = interrupt("__DIARY_CONFIRM__")

    # Short confirmations ("post", "ok", etc.) → keep existing diary
    # Longer text → treat as user-edited diary
    final_diary = response if len(response.strip()) > 30 else diary
    return {"diary_content": final_diary}


def notion_post_node(state: LearnLogState) -> dict:
    """노드 6: 일기를 Notion에 포스팅"""
    diary  = state.get("diary_content", "")
    goal   = get_active_goal(state)
    streak = state.get("streak", 0)

    if not diary:
        return {
                "messages": [AIMessage(content="No diary content found, skipping Notion save.")],
                "session_date": date.today().isoformat(),   
                }

    result = post_to_notion.invoke({
        "diary_content": diary,
        "learning_goal": goal,
        "mood":          "a calm day",
        "streak":        streak,
    })

    if result.startswith("✅"):
        final = (
            f"📔 Today's learning journal has been saved to Notion!\n\n"
            f"{result}\n\n"
            f"{streak} day streak and counting. Keep it up tomorrow! 💪"
        )
    else:
        final = (
            f"⚠️ Couldn't save to Notion.\n\n"
            f"{result}\n\n"
            f"Great work today regardless! 🌟\n"
            f"We'll try again tomorrow. Keep going! 💪"
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


def route_after_curriculum_confirm(state: LearnLogState) -> str:
    """CE_curriculum: 'confirm' → persona_build / else → curriculum_build (재생성)"""
    return "persona_build" if state.get("next_action") == "confirm" else "curriculum_build"


def route_tutor_qa(state: LearnLogState) -> str:
    """CE_tutor_qa: ready_for_quiz(LLM 판단) + min/max 턴 안전장치"""
    MIN_TURNS, MAX_TURNS = 2, 5
    index          = state.get("tutor_qa_index", 0)
    ready_for_quiz = state.get("ready_for_quiz", False)
    if index >= MAX_TURNS:
        return "quiz_generate"
    if index >= MIN_TURNS and ready_for_quiz:
        return "quiz_generate"
    return "tutor_qa_generate"


# ══════════════════════════════════════════════════════════════
# Graph 빌드 & 컴파일
# ════════════════════════════════════════════════════════
def build_graph():
    builder = StateGraph(LearnLogState)

    # 노드 등록
    builder.add_node("goal_check",         goal_check_node)
    builder.add_node("goal_detail",        goal_detail_node)
    builder.add_node("curriculum_build",   curriculum_build_node)
    builder.add_node("curriculum_confirm", curriculum_confirm_node)
    builder.add_node("persona_build",      persona_build_node)
    builder.add_node("checkin_topic",      checkin_topic_node)
    builder.add_node("checkin",            checkin_node)
    builder.add_node("tutor_qa_generate",  tutor_qa_generate_node)
    builder.add_node("tutor_qa_answer",    tutor_qa_answer_node)
    builder.add_node("tutor_qa_feedback",  tutor_qa_feedback_node)
    builder.add_node("quiz_generate",      quiz_generate_node)
    builder.add_node("quiz",               quiz_node)
    builder.add_node("diary_writer",       diary_writer_node)
    builder.add_node("diary_confirm",      diary_confirm_node)
    builder.add_node("notion_post",        notion_post_node)

    # 진입점 분기 엣지
    builder.add_conditional_edges(
        START,
        route_entry,
        {"checkin": "checkin_topic", "full": "goal_check"},
    )

    # 목표 설정 경로
    builder.add_conditional_edges(
        "goal_check",
        route_after_goal_check,
        {"setup": "goal_detail", "skip": "checkin_topic"},
    )
    builder.add_edge("goal_detail",      "curriculum_build")
    builder.add_edge("curriculum_build", "curriculum_confirm")
    builder.add_conditional_edges(
        "curriculum_confirm",
        route_after_curriculum_confirm,
        {"persona_build": "persona_build", "curriculum_build": "curriculum_build"},
    )
    builder.add_edge("persona_build",  "checkin_topic")

    # 체크인 → 튜터 Q&A → 퀴즈 → 일기
    builder.add_edge("checkin_topic",     "checkin")
    builder.add_edge("checkin",           "tutor_qa_generate")
    builder.add_edge("tutor_qa_generate", "tutor_qa_answer")
    builder.add_edge("tutor_qa_answer",   "tutor_qa_feedback")
    builder.add_conditional_edges(
        "tutor_qa_feedback",
        route_tutor_qa,
        {"tutor_qa_generate": "tutor_qa_generate", "quiz_generate": "quiz_generate"},
    )
    builder.add_edge("quiz_generate", "quiz")
    builder.add_edge("quiz",          "diary_writer")
    builder.add_edge("diary_writer",  "diary_confirm")
    builder.add_edge("diary_confirm", "notion_post")
    builder.add_edge("notion_post",   END)

    postgres_url = None
    try:
        import streamlit as _st
        postgres_url = _st.secrets.get("POSTGRES_URL")
    except Exception:
        pass
    if not postgres_url:
        import os
        postgres_url = os.getenv("POSTGRES_URL")

    if postgres_url and _POSTGRES_AVAILABLE:
        memory = PostgresSaver.from_conn_string(postgres_url)
        memory.setup()
    else:
        conn   = sqlite3.connect("learnlog.db", check_same_thread=False)
        memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


learnlog_graph = build_graph()
