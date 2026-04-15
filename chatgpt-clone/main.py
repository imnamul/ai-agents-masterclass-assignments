import dotenv
from numpy.char import center
from openai.types.responses.response_output_message import Content
from streamlit.elements.lib.layout_utils import TextAlignment
dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool
import base64 

client = OpenAI()

VECTOR_STORE_ID = "vs_69de7b79381c8191a7e28b77fa8005e9"


if "agent" not in st.session_state:

    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions="""
            You are a strategic Life Coach AI. Your goal is to help the user achieve the specific goals outlined in their uploaded documents.

            Core Responsibilities:
            1. Contextual Awareness: Always check the user's uploaded "Personal Goal" files using the File Search Tool to ensure your advice aligns with their long-term vision.
            2. Continuity: Reference previous conversations (Chat History) to track progress. If the user mentioned a challenge yesterday, follow up on it today.
            3. Progress Tracking: Periodically summarize the user's journey. Use phrases like "Since you achieved X last week, let's aim for Y today."
            4. Proactive Coaching: If the user's current input contradicts their uploaded goals, gently point it out and ask for clarification.

            Tools:
            - File Search Tool: Use this to retrieve details from the "Personal Goal" documents.
            - Web Search Tool: Use this for external research, habit science, or motivational content.
            """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[
                    VECTOR_STORE_ID
                ],
                max_num_results=3,
            )
        ],
    )


agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )

session = st.session_state["session"]

# 1. 페이지 설정
st.set_page_config(page_title="♣️My Life Coach", layout="centered", initial_sidebar_state="collapsed")


# 2. CSS를 이용한 상단 고정 레이아웃 설정
# 이 코드는 header라는 클래스를 가진 div를 화면 상단에 고정시킵니다.
st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #f0f2f6;
        padding: 10px 0;
    }
    .main-content {
        margin-top: 180px; /* 상단 바 높이만큼 여백을 주어 내용이 겹치지 않게 함 */
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 고정 상단 영역 (Header)
# 위에서 정의한 스타일을 적용합니다.
header_container = st.container()
with header_container:
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    
    # 타이틀 (중앙)
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>♣️ My Life Coach</h1>", unsafe_allow_html=True)
    
    # 파일 업로드 UI (우측 배치)
    col1, col2 = st.columns([1, 1])
    with col2:
        # 1. 이미 파일이 처리되었는지 확인
        if st.session_state.get("goal_file_processed"):
            # 업로드된 파일 정보를 표시
            st.markdown(f"""
                <div style='background-color: #e8f0fe; padding: 10px; border-radius: 5px; border: 1px solid #1a73e8; margin-bottom: 5px;'>
                    <span style='font-size: 1.2rem; color: #1a73e8;'>🎯 Goal Set:</span>
                    <strong style='font-size: 1rem;'>{st.session_state.get("goal_file_name", "Uploaded file")}</strong>
                </div>
            """, unsafe_allow_html=True)

            if st.button("Replace the file", key="replace_file"):
                st.session_state.goal_file_processed = False
                if "goal_file_id" in st.session_state:
                    del st.session_state.goal_file_id
                st.rerun()
        else :
            # 2. 파일이 업로드되지 않았을 때만 업로더 표시
            goal_file = st.file_uploader("Upload Personal Goal (PDF, TXT)", type=["pdf", "txt"], label_visibility="visible")

            if goal_file: 
                with st.status("🎯 Analyzing Personal Goal...") as status:
                    uploaded_goal = client.files.create(
                        file=(goal_file.name, goal_file.getvalue()),
                        purpose="user_data",
                    )
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_goal.id,
                    )

                    # 파일 이름과 처리 상태를 세션에 저장
                    st.session_state.goal_file_processed = True
                    st.session_state.goal_file_id = uploaded_goal.id
                    st.session_state.goal_file_name = goal_file.name 
                    status.update(label="✅ Goal Setting Complete!", state="complete")
                
                    # [추가] 파일 업로드 사실을 대화 역사(Session)에 기록하여 AI가 인지하게 함
                    asyncio.run(session.add_items([{
                        "role": "system",
                        "content": f"The user has uploaded a personal goal document: {goal_file.name}. Please refer to this for future coaching.",
                        "type": "message"
                    }]))

                    st.rerun() # 화면 갱신을 위해 재실행    

st.markdown('</div>', unsafe_allow_html=True)

# 4. 스크롤되는 대화 영역 (Main Content)
# 위 CSS에서 설정한 margin-top을 반영하기 위해 div로 감쌉니다.
st.markdown('<div class="main-content">', unsafe_allow_html=True)

async def paint_history():
    messages = await session.get_items()

    for message in messages:

        # 1. 메시지 역할(role) 확인
        role = message["role"]

        # [필터링] 'system' 역할은 AI 내부 지침용이므로 화면에 그리지 않음
        if role == "system":
            continue

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
            if message["type"] == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message["type"] == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")

asyncio.run(paint_history())


def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.completed": (" ", "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        response = ""

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


st.markdown('</div>', unsafe_allow_html=True)

# 5. 하단 입력창 (Streamlit 기본 chat_input은 화면 하단에 자동 고정됨)

prompt = st.chat_input(
    "Shall we talk about your goals for today?",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
    )

if prompt:

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
