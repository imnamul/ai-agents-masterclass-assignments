"""
LearnLog — Streamlit UI
실행: uv run streamlit run app.py
"""

import os
import time
import streamlit as st
from datetime import date
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import learnlog_graph

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

st.set_page_config(page_title="LearnLog", page_icon="📚", layout="wide")
st.title("📚 LearnLog — 학습 습관 트래커")

# ── Session State 초기화 ────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-main"   # 고정 thread — 날짜 간 상태 유지
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []          # {"role": "assistant"|"user", "content": "..."}
if "synced_msg_count" not in st.session_state:
    st.session_state.synced_msg_count = 0   # graph state에서 sync한 메시지 수

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Graph State 확인 ────────────────────────────────────────────
graph_state  = learnlog_graph.get_state(config)
graph_msgs   = graph_state.values.get("messages", []) if graph_state.values else []
is_running   = bool(graph_state.next)
session_date = (graph_state.values or {}).get("session_date", "")
is_done      = not is_running and session_date == date.today().isoformat()

# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 학습 현황")

    vals          = graph_state.values if graph_state.values else {}
    active_goal   = vals.get("active_goal", "")
    learning_goals = vals.get("learning_goals", [])
    streak        = vals.get("streak", 0)
    current_week  = vals.get("current_week", 0)
    current_topic = vals.get("current_topic", "")
    progress_pct  = vals.get("progress_pct", 0.0)
    total_weeks   = vals.get("curriculum", {}).get("total_weeks", 0)

    if active_goal:
        st.subheader("🎯 현재 목표")
        st.write(active_goal)

        if total_weeks:
            st.subheader("📈 진도")
            st.progress(progress_pct)
            st.caption(f"{int(progress_pct * 100)}% 완료 · {current_week}주차 / 전체 {total_weeks}주")
            if current_topic:
                st.info(f"📖 **오늘의 주제**\n\n{current_topic}")
            # 오늘 주차 자료 링크
            phases = vals.get("curriculum", {}).get("phases", [])
            cur_phase = next((p for p in phases if p.get("week") == current_week), None)
            if cur_phase:
                resources = cur_phase.get("resources", [])
                if resources:
                    st.caption("📚 학습 자료")
                    for r in resources:
                        st.markdown(f"🔗 [{r['title']}]({r['url']})")

    if streak:
        st.metric("🔥 스트릭", f"{streak}일 연속")

    if learning_goals:
        st.subheader("📋 전체 목표")
        for g in learning_goals:
            marker = "✅" if g == active_goal else "•"
            st.write(f"{marker} {g}")

    st.divider()
    st.caption("LearnLog · LangGraph 기반 학습 트래커")

    # ── 개발자 모드 ─────────────────────────────────────────────
    if DEBUG:
        st.divider()
        with st.expander("🔧 개발자 모드", expanded=False):
            st.caption("그래프 state를 직접 수정합니다")
            dbg_streak  = st.number_input("streak (일)",         min_value=0, value=int(streak),        step=1)
            dbg_week    = st.number_input("current_week (주차)", min_value=0, value=int(current_week),  step=1)
            dbg_topic   = st.text_input("current_topic",         value=current_topic)
            dbg_goal    = st.text_input("active_goal",           value=active_goal)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💉 상태 주입", use_container_width=True):
                    learnlog_graph.update_state(config, {
                        "streak":        dbg_streak,
                        "current_week":  dbg_week,
                        "current_topic": dbg_topic,
                        "active_goal":   dbg_goal,
                    })
                    st.session_state.chat_log         = []
                    st.session_state.synced_msg_count = 0
                    st.success("주입 완료!")
                    st.rerun()
            with col_b:
                if st.button("🗑️ 상태 초기화", use_container_width=True):
                    learnlog_graph.update_state(config, {
                        "streak":             0,
                        "current_week":       0,
                        "current_topic":      "",
                        "active_goal":        "",
                        "learning_goals":     [],
                        "curriculum":         {},
                        "today_achievements": "",
                        "diary_content":      "",
                        "messages":           [],
                        "progress_pct":       0.0,
                        "session_date":       "",
                    })
                    st.session_state.chat_log         = []
                    st.session_state.synced_msg_count = 0
                    st.success("초기화 완료!")
                    st.rerun()

            st.divider()
            st.caption(f"현재 session_date: `{session_date or '없음'}`")
            if st.button("📅 다음날 시뮬레이션", use_container_width=True):
                # 일별 데이터만 리셋, 영구 데이터(goal/curriculum/streak 등)는 유지
                learnlog_graph.update_state(config, {
                    "session_date":      "2000-01-01",
                    "mood":              "",
                    "today_achievements": "",
                    "diary_content":     "",
                    "next_action":       "",
                    "search_results":    "",
                    "quiz_questions":    "",
                    "messages":          [],
                })
                st.session_state.chat_log         = [{"role": "divider", "content": "📅 새로운 날 시작"}]
                st.session_state.synced_msg_count = 0
                st.success("다음날 시뮬레이션 적용!")
                st.rerun()

# ── Graph AI 메시지 → chat_log sync ─────────────────────────────
# 노드가 반환한 AI 메시지만 sync (HumanMessage는 사용자 입력 시 이미 추가됨)
new_graph_msgs = graph_msgs[st.session_state.synced_msg_count:]
for msg in new_graph_msgs:
    if not isinstance(msg, HumanMessage):
        st.session_state.chat_log.append({"role": "assistant", "content": msg.content})
st.session_state.synced_msg_count = len(graph_msgs)

# ── 현재 Interrupt 질문 감지 → chat_log에 추가 ──────────────────
current_interrupt = None
if is_running and graph_state.tasks:
    for task in graph_state.tasks:
        for intr in task.interrupts:
            current_interrupt = intr.value

if current_interrupt:
    last = st.session_state.chat_log[-1] if st.session_state.chat_log else None
    if not last or last["content"] != current_interrupt:
        st.session_state.chat_log.append({"role": "assistant", "content": current_interrupt})

# ── 채팅 표시 ───────────────────────────────────────────────────
for msg in st.session_state.chat_log:
    if msg["role"] == "divider":
        st.divider()
        st.caption(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# ── 입력 / 버튼 처리 ────────────────────────────────────────────
if is_done:
    st.success("✅ 오늘 학습 기록 완료! 내일 또 만나요 🌱")
    if st.button("🔄 채팅 비우기"):
        st.session_state.chat_log         = []
        st.session_state.synced_msg_count = 0
        st.rerun()

elif is_running:
    def _resume(value: str):
        """그래프 재개 헬퍼 — 커리큘럼 생성 노드 실행 중 진행 상황 표시"""
        _progress_labels = {
            "domain_analysis":  "📊 도메인 분석 중...",
            "curriculum_build": "📅 커리큘럼 생성 중...",
            "persona_build":    "🎓 AI 튜터 설정 중...",
        }
        with st.spinner("생각 중..."):
            _placeholder = st.empty()
            for _event in learnlog_graph.stream(
                Command(resume=value),
                config=config,
                stream_mode="updates",
            ):
                _node = next(iter(_event), "")
                if _node in _progress_labels:
                    _placeholder.info(_progress_labels[_node])
            _placeholder.empty()
        st.rerun()

    if current_interrupt == "__ACTION_SELECT__":
        # 버튼 기반 액션 선택
        st.markdown("**다음 단계를 선택하세요:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📝 퀴즈로 복습", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "📝 퀴즈로 복습"})
                _resume("quiz")
        with col2:
            if st.button("✍️ 바로 일기 쓰기", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "✍️ 바로 일기 쓰기"})
                _resume("diary")
        with col3:
            if st.button("🔍 검색해보기", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "🔍 검색해보기"})
                _resume("search")
    else:
        user_input = st.chat_input("답변을 입력하세요...")
        if user_input:
            st.session_state.chat_log.append({"role": "user", "content": user_input})
            _resume(user_input)

else:
    # 그래프 미시작 상태
    st.info("오늘의 학습을 시작해볼까요? 🚀")

    if active_goal:
        # 목표가 있을 때 — 체크인 강조, 새 목표 보조
        if st.button("📝 오늘 체크인", type="primary", use_container_width=True):
            with st.spinner("시작 중..."):
                for _ in learnlog_graph.stream(
                    {"messages": [], "entry_mode": "checkin"},
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

        if st.button("🎯 새 목표 설정", use_container_width=True):
            initial_state = {
                "messages":           [],
                "learning_goals":     [],
                "active_goal":        "",
                "mood":               "",
                "today_achievements": "",
                "streak":             0,
                "search_results":     "",
                "diary_content":      "",
                "next_action":        "",
                "curriculum":         {},
                "current_week":       0,
                "current_topic":      "",
                "tutor_persona":      "",
                "quiz_history":       [],
                "progress_pct":       0.0,
                "entry_mode":         "",
            }
            with st.spinner("시작 중..."):
                for _ in learnlog_graph.stream(
                    initial_state,
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

    else:
        # 목표가 없을 때 — 목표 설정만
        if st.button("🎯 목표 설정하기", type="primary", use_container_width=True):

            initial_state = {
                "messages":           [],
                "learning_goals":     [],
                "active_goal":        "",
                "mood":               "",
                "today_achievements": "",
                "streak":             0,
                "search_results":     "",
                "diary_content":      "",
                "next_action":        "",
                "curriculum":         {},
                "current_week":       0,
                "current_topic":      "",
                "tutor_persona":      "",
                "quiz_history":       [],
                "progress_pct":       0.0,
                "entry_mode":         "",
                "session_date":       "",
            }
            with st.spinner("시작 중..."):
                for _ in learnlog_graph.stream(
                    initial_state,
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()
