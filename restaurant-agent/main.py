import dotenv

from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import OutputGuardrailTripwireTriggered, Runner, SQLiteSession, RunContextWrapper, InputGuardrailTripwireTriggered
from models import RestaurantContext
from my_agents.triage_agent import triage_agent

dotenv.load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="나물이네",
    page_icon="🍽️",
    layout="centered",
)

st.title("🍽️ 나물이네 키친")
st.caption("메뉴 문의, 주문, 예약 — 무엇이든 도와드립니다!")

# ── Session state ──────────────────────────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "restaurant-chat",
        "restaurant-memory.db",
    )
session = st.session_state["session"]

if "initialized" not in st.session_state:
    asyncio.run(session.clear_session())
    st.session_state["agent"] = triage_agent
    st.session_state["initialized"] = True

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent

restaurant_ctx = RestaurantContext(
    restaurant_name="나물이네 키친",
    customer_name=None,
    opening_hours="11:00 – 22:00",
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
        st.session_state["text_placeholder"] = text_placeholder

        response = ""        
        current_agent = triage_agent  # session_state 대신 triage_agent로 고정
        last_responding_agent = triage_agent  # ← 응답을 실제로 생성한 에이전트

        try:
            stream = Runner.run_streamed(
                triage_agent,
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
                        last_responding_agent = current_agent  # 실제 텍스트 생성 시점에 기록

                # ── Detect agent handoff ───────────────────────────────────────
                if event.type == "agent_updated_stream_event":
                    new_agent = event.new_agent
                    if current_agent.name != new_agent.name:
                        st.write(f"🤖 Transferred from {current_agent.name} to {new_agent.name}")
                        current_agent = new_agent  # 로컬 변수만 업데이트
                        # Reset response buffer for new agent
                        response =""
                        text_placeholder.empty() 
                        text_placeholder = st.empty()
                        st.session_state["text_placeholder"] = text_placeholder
            
                # current_agent 대신 last_responding_agent 사용
                st.session_state["last_agent"] = last_responding_agent.name

        except InputGuardrailTripwireTriggered:
            text_placeholder.empty()  
            st.warning("죄송합니다, 해당 요청은 처리할 수 없습니다.")

        except OutputGuardrailTripwireTriggered as e:
            text_placeholder.empty()  
            output_info = {}
            try:
                output_info = e.guardrail_result.output.output_info or {}
            except Exception:
                pass
            
            #agent_name = output_info.get("agent", current_agent)
           
            reasoning = output_info.get("reasoning", "")
            
            # 화면에 디버그 정보 표시
            # import json
            # with st.expander("🛡️ Output Guardrail Debug", expanded=True):
            #     st.write("raw output_info:")
            #     st.json(output_info)
            #     st.write("reasoning:")
            #     st.code(reasoning if reasoning else "(empty)", language="text")
            
            # # 콘솔 로그에도 출력 (streamlit run 터미널에서 확인)
            # print("[OutputGuardrailTripwireTriggered] output_info=", json.dumps(output_info, ensure_ascii=False))
            # print("[OutputGuardrailTripwireTriggered] reasoning=", reasoning)
            
            # Determine specific block reason
            block_reason = "응답 품질 기준을 충족하지 못했습니다."
            if not output_info.get("is_professional_and_polite", True):
                block_reason = "전문적이고 정중한 응답 기준을 충족하지 못해 차단되었습니다."
            elif output_info.get("exposes_internal_info", False):
                block_reason = "내부 정보 노출 가능성이 있어 응답이 차단되었습니다."
            
            # Render guardrail block notice
            st.write(f"— 응답 차단됨 ({block_reason})")
            st.caption("다시 질문해 주시거나 다른 방식으로 요청해 보세요.")

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

    st.subheader("🤖 마지막 응답 에이전트")
    st.write(st.session_state.get("last_agent", "triage_agent"))
 
    st.divider()
    
    reset = st.button("🗑️ 대화 초기화")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["agent"] = triage_agent
        st.rerun()

    with st.expander("💬 메시지 히스토리 보기"):        
        st.write(asyncio.run(session.get_items()))
