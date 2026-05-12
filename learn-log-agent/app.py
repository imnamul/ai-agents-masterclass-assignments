"""
LearnLog — Streamlit UI
실행: uv run streamlit run app.py
"""

import time
import streamlit as st
from datetime import date
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import learnlog_graph

st.set_page_config(page_title="LearnLog", page_icon="📚", layout="centered")
st.title("📚 LearnLog — 학습 습관 트래커")

# ── Session State 초기화 ────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user-{date.today().isoformat()}"
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []          # {"role": "assistant"|"user", "content": "..."}
if "synced_msg_count" not in st.session_state:
    st.session_state.synced_msg_count = 0   # graph state에서 sync한 메시지 수

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Graph State 확인 ────────────────────────────────────────────
graph_state = learnlog_graph.get_state(config)
graph_msgs  = graph_state.values.get("messages", []) if graph_state.values else []
is_running  = bool(graph_state.next)
is_done     = not is_running and bool(graph_msgs)

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
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 입력 / 버튼 처리 ────────────────────────────────────────────
if is_done:
    st.success("✅ 오늘 학습 기록 완료! 내일 또 만나요 🌱")
    if st.button("🔄 새 세션 시작"):
        st.session_state.thread_id       = f"user-{date.today().isoformat()}-{int(time.time())}"
        st.session_state.chat_log        = []
        st.session_state.synced_msg_count = 0
        st.rerun()

elif is_running:
    user_input = st.chat_input("답변을 입력하세요...")
    if user_input:
        # 사용자 메시지를 즉시 chat_log에 추가 (rerun 전에 표시)
        st.session_state.chat_log.append({"role": "user", "content": user_input})
        with st.spinner("생각 중..."):
            for _ in learnlog_graph.stream(
                Command(resume=user_input),
                config=config,
                stream_mode="values",
            ):
                pass
        st.rerun()

else:
    # 그래프 미시작 상태
    st.info("오늘의 학습을 시작해볼까요? 🚀")
    if st.button("오늘 학습 시작!", type="primary", use_container_width=True):
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
        }
        with st.spinner("시작 중..."):
            for _ in learnlog_graph.stream(
                initial_state,
                config=config,
                stream_mode="values",
            ):
                pass
        st.rerun()
