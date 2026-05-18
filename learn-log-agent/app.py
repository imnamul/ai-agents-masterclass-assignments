"""
LearnLog — Streamlit UI
Run: uv run streamlit run app.py
"""

import os
import time
import uuid
import streamlit as st
from datetime import date
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import learnlog_graph

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

st.set_page_config(page_title="LearnLog", page_icon="📚", layout="wide")
st.title("📚 LearnLog — Learning Habit Tracker")

# ── Session State ───────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "synced_msg_count" not in st.session_state:
    st.session_state.synced_msg_count = 0

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── Graph State ─────────────────────────────────────────────────
graph_state  = learnlog_graph.get_state(config)
graph_msgs   = graph_state.values.get("messages", []) if graph_state.values else []
is_running   = bool(graph_state.next)
session_date = (graph_state.values or {}).get("session_date", "")
is_done      = not is_running and session_date == date.today().isoformat()

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Learning Status")

    vals           = graph_state.values if graph_state.values else {}
    active_goal    = vals.get("active_goal", "")
    learning_goals = vals.get("learning_goals", [])
    streak         = vals.get("streak", 0)
    current_week   = vals.get("current_week", 0)
    current_topic  = vals.get("current_topic", "")
    progress_pct   = vals.get("progress_pct", 0.0)
    total_weeks    = vals.get("curriculum", {}).get("total_weeks", 0)

    if active_goal:
        st.subheader("🎯 Current Goal")
        st.write(active_goal)

        if total_weeks:
            st.subheader("📈 Progress")
            st.progress(progress_pct)
            st.caption(f"{int(progress_pct * 100)}% complete · Week {current_week} / {total_weeks} weeks")
            if current_topic:
                st.info(f"📖 **Today's Topic**\n\n{current_topic}")
            # Current week study resources
            phases = vals.get("curriculum", {}).get("phases", [])
            cur_phase = next((p for p in phases if p.get("week") == current_week), None)
            if cur_phase:
                resources = cur_phase.get("resources", [])
                if resources:
                    st.caption("📚 Study Resources")
                    for r in resources:
                        st.markdown(f"🔗 [{r['title']}]({r['url']})")

    if streak:
        st.metric("🔥 Streak", f"{streak} day streak")

    if learning_goals:
        st.subheader("📋 All Goals")
        for g in learning_goals:
            marker = "✅" if g == active_goal else "•"
            st.write(f"{marker} {g}")

    st.divider()
    if active_goal or is_running:
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.thread_id        = str(uuid.uuid4())
            st.session_state.chat_log         = []
            st.session_state.synced_msg_count = 0
            st.rerun()
    st.caption("LearnLog · Powered by LangGraph")

    # ── Developer Mode ──────────────────────────────────────────
    if DEBUG:
        st.divider()
        with st.expander("🔧 Developer Mode", expanded=False):
            st.caption("Directly modify graph state")
            dbg_streak = st.number_input("streak (days)",      min_value=0, value=int(streak),       step=1)
            dbg_week   = st.number_input("current_week",       min_value=0, value=int(current_week), step=1)
            dbg_topic  = st.text_input("current_topic",        value=current_topic)
            dbg_goal   = st.text_input("active_goal",          value=active_goal)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💉 Inject State", use_container_width=True):
                    learnlog_graph.update_state(config, {
                        "streak":        dbg_streak,
                        "current_week":  dbg_week,
                        "current_topic": dbg_topic,
                        "active_goal":   dbg_goal,
                    })
                    st.session_state.chat_log         = []
                    st.session_state.synced_msg_count = 0
                    st.success("State injected!")
                    st.rerun()
            with col_b:
                if st.button("🗑️ Reset State", use_container_width=True):
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
                    st.success("State reset!")
                    st.rerun()

            st.divider()
            st.caption(f"Current session_date: `{session_date or 'None'}`")
            if st.button("📅 Simulate Next Day", use_container_width=True):
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
                st.session_state.chat_log         = [{"role": "divider", "content": "📅 New Day Started"}]
                st.session_state.synced_msg_count = 0
                st.success("Next day simulation applied!")
                st.rerun()

# ── Sync AI messages from graph state → chat_log ────────────────
new_graph_msgs = graph_msgs[st.session_state.synced_msg_count:]
for msg in new_graph_msgs:
    if not isinstance(msg, HumanMessage):
        st.session_state.chat_log.append({"role": "assistant", "content": msg.content})
st.session_state.synced_msg_count = len(graph_msgs)

# ── Detect current interrupt → append to chat_log ───────────────
current_interrupt = None
if is_running and graph_state.tasks:
    for task in graph_state.tasks:
        for intr in task.interrupts:
            current_interrupt = intr.value

if current_interrupt:
    last = st.session_state.chat_log[-1] if st.session_state.chat_log else None
    if not last or last["content"] != current_interrupt:
        st.session_state.chat_log.append({"role": "assistant", "content": current_interrupt})

# ── Render chat ─────────────────────────────────────────────────
for msg in st.session_state.chat_log:
    if msg["role"] == "divider":
        st.divider()
        st.caption(msg["content"])
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# ── Input / Button handling ─────────────────────────────────────
if is_done:
    st.success("✅ All done for today! See you tomorrow 🌱")
    if st.button("🔄 Clear Chat"):
        st.session_state.chat_log         = []
        st.session_state.synced_msg_count = 0
        st.rerun()

elif is_running:
    def _resume(value: str):
        """Resume graph — show per-node progress during curriculum generation"""
        _progress_labels = {
            "domain_analysis":  "📊 Analyzing domain...",
            "curriculum_build": "📅 Building curriculum...",
            "resource_verify":  "🔗 Verifying resource links...",
            "persona_build":    "🎓 Setting up AI tutor...",
        }
        with st.spinner("Thinking..."):
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
        st.markdown("**What would you like to do next?**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📝 Take a Quiz", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "📝 Take a Quiz"})
                _resume("quiz")
        with col2:
            if st.button("✍️ Write Journal", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "✍️ Write Journal"})
                _resume("diary")
        with col3:
            if st.button("🔍 Search Resources", use_container_width=True):
                st.session_state.chat_log.append({"role": "user", "content": "🔍 Search Resources"})
                _resume("search")
    else:
        user_input = st.chat_input("Type your response...")
        if user_input:
            st.session_state.chat_log.append({"role": "user", "content": user_input})
            _resume(user_input)

else:
    st.info("Ready to start today's learning? 🚀")

    if active_goal:
        if st.button("📝 Today's Check-in", type="primary", use_container_width=True):
            with st.spinner("Starting..."):
                for _ in learnlog_graph.stream(
                    {"messages": [], "entry_mode": "checkin"},
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

        if st.button("🎯 Set New Goal", use_container_width=True):
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
            with st.spinner("Starting..."):
                for _ in learnlog_graph.stream(
                    initial_state,
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()

    else:
        if st.button("🎯 Set Your Goal", type="primary", use_container_width=True):
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
            with st.spinner("Starting..."):
                for _ in learnlog_graph.stream(
                    initial_state,
                    config=config,
                    stream_mode="values",
                ):
                    pass
            st.rerun()
