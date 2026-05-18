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
    curriculum_feedback: str       # 사용자 커리큘럼 피드백 (재생성 시 반영)


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
    """Node 0: Mood check (interrupt)"""
    streak = state.get("streak", 0)

    greeting = (
        f"🎉 {streak} day streak going strong!\n\nHow are you feeling today?"
        if streak else
        "Welcome to LearnLog! 📚\n\nHow are you feeling today?"
    )
    mood_input = interrupt(greeting)

    mood_res = llm.invoke([HumanMessage(content=
        f'User mood: "{mood_input}"\n'
        f"Summarize their mood in one warm sentence in English. Output only the sentence."
    )])

    return {
        "messages": [HumanMessage(content=mood_input)],
        "mood":     mood_res.content.strip(),
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


def domain_analysis_node(state: LearnLogState) -> dict:
    """노드 2a: 도메인 분석 (DomainAnalysis Structured Output)"""
    new_goal   = state.get("active_goal", "")
    user_level = state.get("user_level", "beginner")

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
    feedback       = state.get("curriculum_feedback", "")

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
- Write all content in English

For each phase, include 2–3 resources in the resources field:
- Official documentation URLs, GitHub repos, or well-known tutorial sites only
- Omit any URL you are not certain about (use empty array if unsure)
- All URLs must start with https:// and be real addresses
- Example: LangGraph → https://langchain-ai.github.io/langgraph/""")])

    return {"curriculum": curriculum_obj.model_dump()}


def resource_verify_node(state: LearnLogState) -> dict:
    """Node 2c: Verify resource URLs — Tavily Extract + Search fallback (Option C)"""
    curriculum = state.get("curriculum", {})
    phases     = curriculum.get("phases", [])
    tavily_key = _get_secret("TAVILY_API_KEY")

    if not tavily_key or not phases:
        return {}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
    except Exception:
        return {}

    verified_phases = []
    for phase in phases:
        resources          = phase.get("resources", [])
        verified_resources = []

        for resource in resources:
            url   = resource.get("url", "")
            title = resource.get("title", "")
            if not url:
                continue

            # Step 1: Tavily Extract — verify URL is accessible
            try:
                extract = client.extract(urls=[url])
                if extract.get("results"):
                    verified_resources.append(resource)
                    continue
            except Exception:
                pass

            # Step 2: Tavily Search fallback — find replacement URL
            try:
                query          = f"{phase.get('theme', '')} {title} official documentation"
                search_results = tavily_search.invoke(query)
                if isinstance(search_results, list) and search_results:
                    top = search_results[0]
                    if isinstance(top, dict) and top.get("url"):
                        verified_resources.append({
                            "title": top.get("title", title),
                            "url":   top["url"],
                        })
            except Exception:
                pass  # Both failed — drop this resource

        phase_copy             = dict(phase)
        phase_copy["resources"] = verified_resources
        verified_phases.append(phase_copy)

    curriculum_copy           = dict(curriculum)
    curriculum_copy["phases"] = verified_phases
    return {"curriculum": curriculum_copy}


def curriculum_confirm_node(state: LearnLogState) -> dict:
    """Node 2d: Show curriculum to user and collect confirmation or feedback"""
    curriculum = state.get("curriculum", {})

    table  = "Here's your personalized curriculum! 🎓\n\n"
    table += "| Week | Theme | Daily Topics | Checkpoint | Study Resources |\n"
    table += "|------|-------|--------------|------------|----------------|\n"
    for p in curriculum.get("phases", []):
        days_str = "<br>".join(f"• Day{d['day']}: {d['topic']}" for d in p.get("days", []))
        res_str  = "<br>".join(f"[{r['title']}]({r['url']})" for r in p.get("resources", [])) or "-"
        table  += f"| Week {p['week']} | {p['theme']} | {days_str} | {p.get('checkpoint', '')} | {res_str} |\n"

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

    # ── Step 3: Daily habits ────────────────────────────────────
    habits_res = llm.invoke([HumanMessage(content=
        f"Learning goal: {new_goal}\n"
        f"Level: {user_level}\n"
        f"Daily study time: {curriculum.get('daily_minutes', 60)} minutes\n\n"
        f"List 3 simple daily habits to build this skill. Output a numbered list only, in English."
    )])
    habits_text = habits_res.content.strip()

    # ── Step 4: Tutor persona ───────────────────────────────────
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

    # ── Summary message — markdown table with resources ─────────
    table  = "| Week | Theme | Daily Topics | Checkpoint | Study Resources |\n"
    table += "|------|-------|--------------|------------|----------------|\n"
    for p in curriculum.get("phases", []):
        days_str = "<br>".join(
            f"• Day{d['day']}: {d['topic']}" for d in p.get("days", [])
        )
        week_res = p.get("resources", [])
        res_str  = "<br>".join(f"[{r['title']}]({r['url']})" for r in week_res) if week_res else "-"
        table += f"| Week {p['week']} | {p['theme']} | {days_str} | {p.get('checkpoint', '')} | {res_str} |\n"

    summary_msg = (
        f"📚 Your **{curriculum.get('subject', new_goal)}** learning plan is ready!\n\n"
        f"📋 **Daily Habits:**\n{habits_text}\n\n"
        f"📅 **{curriculum.get('total_weeks', 4)}-Week Curriculum** ({curriculum.get('daily_minutes', 60)} min/day)\n\n"
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

    # ── New week notification ──────────────────────────────────
    week_up_msg = (
        f"\n\n🎉 Week {new_week} starts today! You'll be learning [{new_topic}]!"
        if new_week > prev_week else ""
    )

    # ── Current week study resources ───────────────────────────
    if curriculum.get("phases"):
        cur_phase   = next((p for p in curriculum["phases"] if p["week"] == new_week),
                           curriculum["phases"][-1])
        resources   = cur_phase.get("resources", [])
        resource_lines = "\n".join(f"🔗 [{r['title']}]({r['url']})" for r in resources)
        resource_section = f"\n\n📚 Study Resources:\n{resource_lines}\n" if resource_lines else ""
    else:
        resource_section = ""

    streak_msg = f"🔥 {streak} day streak!" if streak > 0 else "🌱 Let's start your first day!"
    topic_line = f"📖 Today's topic: **{new_topic}**\n" if new_topic else ""
    topic_name = new_topic if new_topic else active_goal

    question = (
        f"{streak_msg}{week_up_msg}\n\n"
        f"{topic_line}"
        f"{resource_section}\n"
        f"After studying the resources, share what you learned about [{topic_name}] today. 😊"
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
            f"Topic: {current_topic or active_goal}\n"
            f"What the learner studied today: {today_achievements}\n\n"
            f"Encourage the learner and highlight the key concepts from today's session. (3–5 sentences, in English)"
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
        content=f"Learning goal: {goal}\nToday's study: {achievements}\n\nSearch results:\n{formatted}\n\nIntroduce these resources in a helpful and friendly way in English."
    )])

    # ── Quiz offer interrupt ────────────────────────────────────
    current_topic = state.get("current_topic", "") or goal

    quiz_offer = interrupt(
        f"{response.content}\n\n"
        f"---\n"
        f"📝 Would you like to take a quiz on today's topic [{current_topic}]? (yes / no)"
    )

    wants_quiz = any(kw in quiz_offer.lower() for kw in
                     ["yes", "y", "sure", "ok", "yep", "quiz", "let's"])

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
              else SystemMessage(content=f"당신은 {goal} 전문 튜터입니다."))

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
    today        = date.today().strftime("%Y년 %m월 %d일")
    mood         = state.get("mood", "") or "평온한 하루"
    goal         = get_active_goal(state)
    achievements = state.get("today_achievements", "") or "오늘의 학습을 기록하지 않았어요"
    streak       = state.get("streak", 0)

    prompt = f"""Write today's learning journal entry in English.

Date: {today}
Mood: {mood}
Learning goal: {goal}
Today's achievements: {achievements}
Streak: {streak} days

Format:
---
📅 {today}
😊 Mood: [one expressive sentence about today's mood]
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


def route_after_checkin(state: LearnLogState) -> str:
    """CE2: 'search' → resource_search / 'quiz' → quiz_generate / 'write' → diary_writer"""
    action = state.get("next_action", "write")
    if action == "search":
        return "resource_search"
    if action == "quiz":
        return "quiz_generate"
    return "diary_writer"


def route_after_curriculum_confirm(state: LearnLogState) -> str:
    """CE_curriculum: 'confirm' → persona_build / else → curriculum_build (재생성)"""
    return "persona_build" if state.get("next_action") == "confirm" else "curriculum_build"


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
    builder.add_node("resource_verify",    resource_verify_node)
    builder.add_node("curriculum_confirm", curriculum_confirm_node)
    builder.add_node("persona_build",      persona_build_node)
    builder.add_node("checkin",          checkin_node)
    builder.add_node("checkin_response", checkin_response_node)
    builder.add_node("checkin_action",   checkin_action_node)
    builder.add_node("resource_search",  resource_search_node)
    builder.add_node("quiz_generate",    quiz_generate_node)
    builder.add_node("quiz",             quiz_node)
    builder.add_node("diary_writer",     diary_writer_node)
    builder.add_node("notion_post",      notion_post_node)

    # 진입점 분기 엣지
    builder.add_conditional_edges(
        START,
        route_entry,
        {"checkin": "checkin", "full": "mood_check"},
    )

    # 목표 설정 경로
    builder.add_edge("mood_check", "goal_check")
    builder.add_conditional_edges(
        "goal_check",
        route_after_goal_check,
        {"setup": "goal_detail", "skip": "checkin"},
    )
    builder.add_edge("goal_detail",     "domain_analysis")
    builder.add_edge("domain_analysis", "curriculum_build")
    builder.add_edge("curriculum_build",  "resource_verify")
    builder.add_edge("resource_verify",   "curriculum_confirm")
    builder.add_conditional_edges(
        "curriculum_confirm",
        route_after_curriculum_confirm,
        {"persona_build": "persona_build", "curriculum_build": "curriculum_build"},
    )
    builder.add_edge("persona_build", "checkin")

    # 체크인 경로
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


learnlog_graph = build_graph()
