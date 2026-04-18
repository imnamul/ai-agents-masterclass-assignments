import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64 
import streamlit as st
from agents import (
    Agent, 
    Runner, 
    SQLiteSession, 
    WebSearchTool, 
    FileSearchTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    HostedMCPTool,
)

from agents.mcp.server import MCPServerStdio

client = OpenAI()

VECTOR_STORE_ID = "vs_69de7b79381c8191a7e28b77fa8005e9"


if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )

session = st.session_state["session"]

async def paint_history():
    messages = await session.get_items()

    for message in messages:

        # 1. 메시지 역할(role) 확인
        role = message.get("role")

        # 2. 메시지 출력 영역
        if role:
            with st.chat_message(role):

                if role == "user":
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                                
                elif role == "ai" or role == "assistant":
                    # --- 이 부분을 안전하게 수정합니다 ---
                    if message.get("type") == "message":
                        content = message.get("content")
                        
                        # 데이터 구조에 따른 텍스트 추출 (방어적 코드)
                        text = ""
                        # 1. content가 리스트이고 첫 번째 요소가 딕셔너리인 경우 (기존 예상 구조)
                        if isinstance(content, list) and len(content) > 0:
                            first_part = content[0]
                            if isinstance(first_part, dict) and "text" in first_part:
                                text = first_part["text"]
                            else:
                                text = str(first_part) # 문자열일 경우 그대로 사용
                        # 2. content 자체가 그냥 문자열인 경우
                        elif isinstance(content, str):
                            text = content
                        
                        # 텍스트가 있을 때만 출력
                        if text:
                            st.write(text.replace("$", "\\$"))
                    # ----------------------------------
        
        # 3. 도구 실행 상태 표시            
        if "type" in message:    
            message_type = message.get("type")

            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)    
            elif message_type == "code_interpreter_call":
                with st.chat_message("ai"):
                    st.code(message["code"])
            elif message_type == "mcp_list_tools":
                with st.chat_message("ai"):
                    st.write(f"Listed {message["server_label"]}'s tools")
            elif message_type == "mcp_call":
                with st.chat_message("ai"):
                    st.write(f"Called {message["server_label"]}'s {message["name"]} with args {message["arguments"]}")        


asyncio.run(paint_history())


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": (
            "✅ Web search completed.",
            "complete"
        ),
        "response.web_search_call.in_progress": (
            "🔍 Starting web search...", 
            "running",
        ),
        "response.web_search_call.searching": (
            "🔍 Web search in progress...", 
            "running",
        ),
        "response.file_search_call.completed": (
            "✅ File search completed.", 
            "complete",
        ),
        "response.file_search_call.in_progress": (
            "🗂️ Starting file search...", 
            "running",
        ),
        "response.file_search_call.searching": (
            "🗂️ File search in progress...",
            "running",
        ),
        "response.image_generation_call.generating": (
            "🎨 Drawing image...", 
            "running",
        ),
        "response.image_generation_call.in_progress": (
            "🎨 Drawing image...", 
            "running",
        ),
        "response.code_interpreter_call_code.done": (
            "🤖 Ran code", 
            "complete",
        ),
        "response.code_interpreter_call.completed": (
            "🤖 Ran code", 
            "complete",
        ),
        "response.code_interpreter_call.in_progress": (
            "🤖 Running code", 
            "complete",
        ),
        "response.mcp_call.completed": (
            "⚒️ Called MCP tool",
            "complete",
        ),
        "response.mcp_call.failed": (
            "⚒️ Error calling MCP tool",
            "complete",
        ),
        "response.mcp_call.in_progress": (
            "⚒️ Calling MCP tool...",
            "running",
        ),
        "response.mcp_list_tools.completed": (
            "⚒️ Listed MCP tools",
            "complete",
        ),
        "response.mcp_list_tools.failed": (
            "⚒️ Error listing MCP tools",
            "complete",
        ),
        "response.mcp_list_tools.in_progress": (
            "⚒️ Listing MCP tools",
            "running",
        ),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):

    yfinance_server = MCPServerStdio(
        params={
            "command": "uvx",
            "args": ["mcp-yahoo-finance"] 
        },
        cache_tools_list=True,
        client_session_timeout_seconds = 60, 
    )

    async with yfinance_server:
        agent = Agent(
            mcp_servers=[
                yfinance_server,
            ],
            name="ChatGPT Clone",
            instructions="""
            You are a helpful assistant.

            You have access to the followign tools:
                - Web Search Tool: Use this when the user asks a questions that isn't in your training data. Use this tool when the users asks about current or future events, when you think you don't know the answer, try searching for it in the web first.
                - File Search Tool: Use this tool when the user asks a question about facts related to themselves. Or when they ask questions about specific files.
                - Code Interpreter Tool: Use this tool when you need to write and run code to answer the user's question.

                """,
                tools=[
                    WebSearchTool(),
                    FileSearchTool(
                        vector_store_ids=[VECTOR_STORE_ID],
                        max_num_results=3,
                    ),
                    ImageGenerationTool(
                        tool_config={
                            "type": "image_generation",
                            "quality": "high",
                            "output_format": "jpeg",
                            "moderation": "low",
                            "partial_images": 1,
                        }
                    ),
                    CodeInterpreterTool(
                        tool_config={
                            "type": "code_interpreter",
                            "container": {
                                "type": "auto",
                            },
                        }
                    ),
                    HostedMCPTool(
                        tool_config={
                            "server_url" : "https://mcp.context7.com/mcp",
                            "type": "mcp",
                            "server_label": "Context7",
                            "server_description": "Use this to get the docs from software projects.",
                            "require_approval": "never",
                        }
                    )
                ],
            )

        with st.chat_message("ai"):
            status_container = st.status("⏳", expanded=False)
            image_placeholder = st.empty()
            code_placeholder = st.empty()
            text_placeholder = st.empty()
            response = ""
            code_response = ""

            st.session_state["code_placeholder"] = code_placeholder
            st.session_state["image_placeholder"] = image_placeholder
            st.session_state["text_placeholder"] = text_placeholder

            stream = Runner.run_streamed(
                agent, 
                message, 
                session=session,)

            async for event in stream.stream_events():
                if event.type == "raw_response_event":

                    update_status(status_container, event.data.type)
                    
                    if event.data.type == "response.output_text.delta":
                        response += event.data.delta
                        text_placeholder.write(response.replace("$", "\\$"))

                    if event.data.type == "response.code_interpreter_call_code.delta":
                        code_response += event.data.delta
                        code_placeholder.code(code_response)

                    elif event.data.type == "response.image_generation_call.partial_image": 
                        image = base64.b64decode(event.data.partial_image_b64)
                        image_placeholder.image(image)
                

prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
    )

if prompt:

    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching file...")
                    client.vector_stores.files.create(
                        vector_store_id = VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded", state="complete")

        elif file.type.startswith("image/"):
            with st.status("⏳ Uploading image=...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")              
                data_uri = f"data:{file.type};base64,{base64_data}"
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role":"user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")
            with st.chat_message("human"):
                st.image(data_uri)

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))
