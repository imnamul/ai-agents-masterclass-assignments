import dotenv

from my_agents import triage_agent

dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import Runner, SQLiteSession, RunContextWrapper, InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from models import UserAccountContext
from my_agents.triage_agent import triage_agent



client = OpenAI()

user_account_ctx = UserAccountContext(
    customer_id=1,
    name="nico",
    tier="basic",
)

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "customer-support-memory.db",
    )
session = st.session_state["session"]

if "agent" not in st.session_state:
    st.session_state["agent"] = triage_agent

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


async def run_agent(message):

    with st.chat_message("ai"):
        text_placeholder = st.empty()
        response = ""

        st.session_state["text_placeholder"] = text_placeholder

        try:
            stream = Runner.run_streamed(
                st.session_state["agent"],
                message,
                session=session,
                context=user_account_ctx,
            )
            
            async for event in stream.stream_events():

                if event.type == "raw_response_event":
                    
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\\$"))

                elif event.type == "agent_updated_stream_event":

                    if st.session_state["agent"].name != event.new_agent.name:
                        st.write(f"🤖 Transfered from {st.session_state["agent"].name} to {event.new_agent.name}")
                        st.session_state["agent"] = event.new_agent
                        text_placeholder.empty() 
                        st.session_state["text_placeholder"] = text_placeholder
                        response =""


        except InputGuardrailTripwireTriggered:
            st.write("I can't help you with that.")
        
        except OutputGuardrailTripwireTriggered:
            st.write("Cant show you that answer.")
            st.session_state["text_placeholder"].empty()


message = st.chat_input(
    "Write a message for your assistant",
)

if message:
    with st.chat_message("human"):
        st.write(message)
    asyncio.run(run_agent(message))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.rerun()
    st.write(asyncio.run(session.get_items()))
