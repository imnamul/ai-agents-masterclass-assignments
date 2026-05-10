"""
LearnLog — 학습 습관 트래커 에이전트
langgraph dev 서버용 그래프 정의 파일

실행 방법:
    langgraph dev
"""

import json
import os
import requests
from typing import TypedDict, Annotated, List
from datetime import date

from dotenv import load_dotenv
load_dotenv()

# ── LangGraph ──────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

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


# ══════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════

def get_active_goal(state: LearnLogState) -> str:
    """active_goal이 없으면 learning_goals[0]으로 폴백"""
    return state.get("active_goal") or (state.get("learning_goals") or [""])[0]


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

def goal_check_node(state: LearnLogState) -> dict:
    """노드 1: 기분 체크 + 목표 유지/변경 의향 파악 (interrupt)"""
    existing_goals = state.get("learning_goals", [])
    streak         = state.get("streak", 0)

    if existing_goals:
        goals_str = ", ".join(f"'{g}'" for g in existing_goals)
        question = (
            f"안녕하세요! 🎉 {streak}일 연속 달성 중이에요!\n"
            f"현재 목표: [{goals_str}]\n\n"
            f"① 오늘 기분이 어떠세요?\n"
            f"② 기존 목표를 계속할까요, 아니면 새로 추가/변경할까요?"
        )
    else:
        question = (
            f"LearnLog에 오신 걸 환영해요! 📚\n\n"
            f"① 오늘 기분이 어떠세요?\n"
            f"② 어떤 학습 목표가 있으신가요?"
        )

    # ── interrupt: 그래프 일시정지 → 사용자 입력 대기 ──
    user_response = interrupt(question)

    # LLM으로 기분과 의향 분석
    analysis_prompt = f"""사용자 응답: "{user_response}"

아래 JSON 형식으로만 응답해주세요:
{{
  "mood": "오늘 기분을 따뜻한 한 문장으로 요약",
  "wants_new_goal": true 또는 false,
  "goal_text": "사용자가 언급한 학습 목표만 추출 (없으면 null)"
}}

판단 기준:
- '새로운', '추가', '바꾸', '변경', '다른 목표' → wants_new_goal: true
- '계속', '유지', '그대로', 변경 요청 없음      → wants_new_goal: false
- 기존 목표가 없는 경우                          → wants_new_goal: true"""

    analysis = llm.invoke([HumanMessage(content=analysis_prompt)])

    try:
        raw    = analysis.content.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(raw)
        mood      = parsed.get("mood", user_response[:50])
        wants_new = parsed.get("wants_new_goal", not bool(existing_goals))
        goal_text = parsed.get("goal_text") or ""
    except Exception:
        change_kw = ["새로", "추가", "변경", "바꾸", "다른"]
        wants_new = any(kw in user_response for kw in change_kw) or not existing_goals
        mood      = user_response[:50]
        goal_text = ""

    return {
        "messages":    [HumanMessage(content=user_response)],
        "mood":         mood,
        "active_goal":  goal_text,
        "next_action":  "setup" if wants_new else "skip",
    }


def goal_setup_node(state: LearnLogState) -> dict:
    """노드 2: 새 학습 목표 설정 or 기존 목표에 추가"""
    existing_goals = state.get("learning_goals", [])
    new_goal       = state.get("active_goal", "")

    system_prompt = f"""당신은 학습 코치 LearnLog입니다.
사용자의 새 학습 목표를 확인하고, 매일 실천 가능한 습관 3가지로 분해해주세요.

새 학습 목표: {new_goal}
기존 목표: {existing_goals if existing_goals else '없음'}

출력 형식:
🎯 목표 목록: [목표1, 목표2, ...]
📋 일일 습관:
  1. [습관] (XX분)
  2. [습관] (XX분)
  3. [습관] (XX분)

마지막에 체크인 준비가 됐는지 물어보세요."""

    response      = llm.invoke([SystemMessage(content=system_prompt)])
    updated_goals = existing_goals + [new_goal] if new_goal else existing_goals

    return {
        "messages":      [response],
        "learning_goals": updated_goals,
        "active_goal":   new_goal or (existing_goals[0] if existing_goals else ""),
    }


def checkin_node(state: LearnLogState) -> dict:
    """노드 3: 오늘 학습 체크인 (interrupt)"""
    streak      = state.get("streak", 0)
    active_goal = get_active_goal(state)

    question = (
        f"🔥 {streak}일 연속 달성 중!\n"
        f"오늘 목표: {active_goal}\n\n"
        f"오늘 어떤 학습을 하셨나요? 😊\n"
        f"(관련 자료가 필요하면 '자료 찾아줘' 또는 '검색해줘'를 포함해주세요)"
    )

    # ── interrupt: 두 번째 일시정지 ──
    user_input = interrupt(question)

    search_keywords = ["검색", "자료", "찾아", "추천", "알려",
                       "search", "find", "resource", "recommend", "look up"]
    needs_search    = any(kw in user_input.lower() for kw in search_keywords)

    return {
        "messages":          [HumanMessage(content=user_input)],
        "today_achievements": user_input,
        "streak":            streak + 1,
        "next_action":       "search" if needs_search else "write",
    }


def resource_search_node(state: LearnLogState) -> dict:
    """노드 4: Tavily로 학습 자료 검색"""
    goal         = get_active_goal(state)
    achievements = state.get("today_achievements", "")
    query        = f"{goal} {achievements} 학습 자료".strip()

    try:
        results = tavily_search.invoke(query)
        if isinstance(results, str):
            formatted = results
        else:
            formatted = "\n".join([
                f"📌 {r.get('title', '')}\n   🔗 {r.get('url', '')}\n   {r.get('content', '')[:120]}..."
                for r in results
            ])
    except Exception as e:
        error_msg = f"검색 중 오류가 발생했어요: {e}\n직접 검색해보시는 걸 추천드려요 🙏"
        return {"messages": [AIMessage(content=error_msg)], "search_results": ""}

    response = llm.invoke([SystemMessage(
        content=f"학습 목표: {goal}\n오늘 학습 내용: {achievements}\n\n검색 결과:\n{formatted}\n\n자료를 친절하게 소개해주세요."
    )])

    return {"messages": [response], "search_results": formatted}


def diary_writer_node(state: LearnLogState) -> dict:
    """노드 5: 기분 + 성취를 묶어 일기 작성 (검색 자료는 채팅 메시지로만 제공)"""
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

    # 포인트 1: diary_content 빈 값 guard
    if not diary:
        return {"messages": [AIMessage(content="일기 내용이 없어서 Notion 저장을 건너뛰었어요.")]}

    result = post_to_notion.invoke({
        "diary_content": diary,
        "learning_goal": goal,
        "mood":          mood,
        "streak":        streak,
    })

    # 포인트 2: 성공/실패에 따라 메시지 분기
    if result.startswith("✅"):
        final = (
            f"📔 오늘의 학습 일기가 Notion에 저장됐어요!\n\n"
            f"{result}\n\n"
            f"오늘 기분 [{mood}] 이었는데도 공부하셨군요! 🌟\n"
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
    """CE1: 'setup' → goal_setup / 'skip' → checkin"""
    return state.get("next_action", "skip")


def route_after_checkin(state: LearnLogState) -> str:
    """CE2: 'search' → resource_search / 'write' → diary_writer"""
    return state.get("next_action", "write")


# ══════════════════════════════════════════════════════════════
# Graph 빌드 & 컴파일
# ══════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(LearnLogState)

    # 노드 등록
    builder.add_node("goal_check",      goal_check_node)
    builder.add_node("goal_setup",      goal_setup_node)
    builder.add_node("checkin",         checkin_node)
    builder.add_node("resource_search", resource_search_node)
    builder.add_node("diary_writer",    diary_writer_node)
    builder.add_node("notion_post",     notion_post_node)

    # 엣지
    builder.add_edge(START, "goal_check")

    builder.add_conditional_edges(
        "goal_check",
        route_after_goal_check,
        {"setup": "goal_setup", "skip": "checkin"},
    )

    builder.add_edge("goal_setup", "checkin")

    builder.add_conditional_edges(
        "checkin",
        route_after_checkin,
        {"search": "resource_search", "write": "diary_writer"},
    )

    builder.add_edge("resource_search", "diary_writer")
    builder.add_edge("diary_writer",    "notion_post")
    builder.add_edge("notion_post",     END)

    return builder.compile()


# langgraph dev 가 이 변수를 import 해서 사용합니다
learnlog_graph = build_graph()
