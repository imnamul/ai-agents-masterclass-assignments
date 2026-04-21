import dotenv

from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import Runner, SQLiteSession, RunContextWrapper, InputGuardrailTripwireTriggered
from models import RestaurantContext
from my_agents.triage_agent import triage_agent


dotenv.load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="나물이네",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ Namul's Restaurant")
st.caption("메뉴 문의, 주문, 예약 — 무엇이든 도와드립니다!")

# ── Session state ──────────────────────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "restaurant-chat",
        "restaurant-memory.db",
    )
session = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent

restaurant_ctx = RestaurantContext(
    restaurant_name="Namul's Restaurant",
    customer_name=None,
    opening_hours="09:00 – 20:00",
    avg_prep_minutes=20,
    max_party_size=10,
)


# ── Paint chat history ─────────────────────────────────────────────────────────
async def paint_history():
    messages = await session.get_items()
    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message.get("role") == "user":
                    st.write(message["content"])
                else:
                    if message.get("type") == "message":
                        content = message.get("content")
                        text = ""
                        if isinstance(content, list) and len(content) > 0:
                            first_part = content[0]
                            if isinstance(first_part, dict) and "text" in first_part:
                                text = first_part["text"]
                            else:
                                text = str(first_part)
                        elif isinstance(content, str):
                            text = content
                        if text:
                            st.write(text.replace("$", "\\$"))

asyncio.run(paint_history())

# ── Run agent (streaming) ──────────────────────────────────────────────────────
async def run_agent(message):
    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""        

        st.session_state["text_placeholder"] = text_placeholder

        try:
            # keep current agent object for handoff comparison
            current_agent = st.session_state["agent"]

            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=restaurant_ctx,
            )
            
            async for event in stream.stream_events():

                # ── Stream text delta ──────────────────────────────────────────
                if event.type == "raw_response_event":
                    
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\\$"))

                # ── Detect agent handoff ───────────────────────────────────────
                if event.type == "agent_updated_stream_event":
                    new_agent = event.new_agent
                    if current_agent.name != new_agent.name:
                        st.write(f"🤖 Transferred from {current_agent.name} to {new_agent.name}")

                        st.session_state["agent"] = new_agent
                        current_agent = new_agent

                        # Reset response buffer for new agent
                        response =""
                        text_placeholder.empty() 
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder


        except InputGuardrailTripwireTriggered:
            text_placeholder.empty()  
            st.warning("죄송합니다, 해당 요청은 처리할 수 없습니다.")


# ── Chat input ─────────────────────────────────────────────────────────────────
message = st.chat_input(
    "메시지를 입력하세요...",
)

if message:
    with st.chat_message("human"):
        st.write(message)

    asyncio.run(run_agent(message))


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:

    st.subheader("🎛️ 현재 상태")
    
    reset = st.button("🗑️ 대화 초기화")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["agent"] = triage_agent
        st.rerun()

    with st.expander("💬 메시지 히스토리 보기"):        
        st.write(asyncio.run(session.get_items()))
