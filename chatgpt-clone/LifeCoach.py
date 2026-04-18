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
)

client = OpenAI()

VECTOR_STORE_ID = "vs_69de7b79381c8191a7e28b77fa8005e9"

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(page_title="🌟 Life Coach Agent", page_icon="🌟")
st.title("🌟 Life Coach Agent")
st.caption("당신의 목표 달성을 함께하는 AI 코치")

# ── 세션 초기화 ──────────────────────────────────────────
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "life-coach-memory.db",
    )

session = st.session_state["session"]


# ── 대화 기록 렌더링 ─────────────────────────────────────
async def paint_history():
    messages = await session.get_items()

    for message in messages:
        role = message.get("role")

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

                elif role in ("ai", "assistant"):
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

        if "type" in message:
            message_type = message.get("type")

            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 웹에서 정보를 검색했어요...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ 개인 목표 파일을 확인했어요...")
            elif message_type == "image_generation_call":
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)


asyncio.run(paint_history())


# ── 상태 메시지 업데이트 ─────────────────────────────────
def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.in_progress":       ("🔍 웹 검색 중...",              "running"),
        "response.web_search_call.searching":         ("🔍 정보를 찾고 있어요...",        "running"),
        "response.web_search_call.completed":         ("✅ 웹 검색 완료!",               "complete"),
        "response.file_search_call.in_progress":      ("🗂️ 목표 파일 검색 중...",        "running"),
        "response.file_search_call.searching":        ("🗂️ 개인 목표 확인 중...",        "running"),
        "response.file_search_call.completed":        ("✅ 파일 검색 완료!",             "complete"),
        "response.image_generation_call.in_progress": ("🎨 이미지를 그리는 중...",        "running"),
        "response.image_generation_call.generating":  ("🎨 비전 보드 생성 중...",         "running"),
        "response.image_generation_call.completed":   ("✅ 이미지 생성 완료!",           "complete"),
        "response.completed":                         (" ",                            "complete"),
    }

    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


# ── Life Coach Agent 실행 ────────────────────────────────
async def run_agent(message):

    agent = Agent(
        name="Life Coach",
        instructions="""
        당신은 따뜻하고 동기부여가 넘치는 AI 라이프 코치입니다. 🌟

        ## 당신의 역할
        - 사용자의 목표 달성을 돕고 진심으로 응원합니다
        - 개인화된 조언과 실행 가능한 계획을 제공합니다
        - 긍정적이고 격려하는 톤을 유지합니다
        - 한국어로 대화합니다

        ## 도구 사용 지침

        ### 🔍 웹 검색 도구 (WebSearchTool)
        다음 상황에서 반드시 사용하세요:
        - 동기부여 콘텐츠, 명언, 성공 사례 검색
        - 목표 달성 방법론, 습관 형성 팁 검색
        - 특정 분야(운동, 공부, 재정 등)의 최신 조언 검색
        - 사용자가 언급한 주제에 대한 전문 정보 검색

        ### 🗂️ 파일 검색 도구 (FileSearchTool)
        다음 상황에서 반드시 사용하세요:
        - 사용자의 개인 목표나 계획에 대해 질문할 때
        - "내 목표", "나의 계획", "내 일기"를 언급할 때
        - 비전 보드나 개인화된 이미지 생성 전 목표 파악
        - 과거 진행 상황이나 일기 내용 참조 시

        ### 🎨 이미지 생성 도구 (ImageGenerationTool)
        다음 상황에서 반드시 사용하세요:

        **목표 달성 축하 이미지:**
        - 사용자가 목표를 달성했다고 말할 때
        - "해냈어", "완료", "성공", "달성" 키워드 감지 시
        - 예: "🎉 [달성한 목표] 축하 이미지" 생성

        **비전 보드:**
        - "비전 보드 만들어줘" 요청 시
        - 새해/새 학기 목표 설정 시
        - 파일 검색으로 목표를 파악한 후 해당 목표들을 시각화
        - 예: "[목표1], [목표2], [목표3] 테마의 밝고 영감을 주는 비전 보드"

        **동기부여 포스터:**
        - 사용자가 의욕을 잃거나 힘들어할 때
        - "포스터", "이미지", "그림" 요청 시
        - 명언이나 특별한 메시지와 함께 이미지 생성
        - 예: "밝고 활기찬 동기부여 포스터, 텍스트: '[메시지]'"

        **진행 상황 시각화:**
        - 진행 상황을 공유할 때 (예: "30% 완료")
        - 마일스톤 달성 시
        - 예: "진행률 [X]%를 나타내는 인포그래픽"

        ## 이미지 생성 프롬프트 원칙
        - 항상 영어로 이미지 프롬프트 작성 (품질 향상)
        - 밝고 긍정적인 색감 지정
        - 구체적인 시각 요소 포함
        - 동기부여적이고 영감을 주는 스타일 명시

        ## 대화 흐름 예시
        1. 목표 달성 → 축하 메시지 + 축하 이미지 자동 생성
        2. 비전 보드 요청 → 파일 검색(목표 파악) → 이미지 생성
        3. 조언 요청 → 웹 검색(최신 정보) → 개인화된 조언 제공
        4. 새 목표 설정 → 웹 검색(관련 팁) + 이미지(동기부여 포스터)

        항상 사용자를 진심으로 응원하고, 가능하면 시각적인 요소로 동기를 높여주세요! 💪
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
        ],
    )

    with st.chat_message("ai"):
        status_container = st.status("⏳ 코치가 생각하는 중...", expanded=False)
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""

        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":

                update_status(status_container, event.data.type)

                # 텍스트 스트리밍
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\\$"))

                # 이미지 부분 생성 미리보기
                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)


# ── 채팅 입력 처리 ───────────────────────────────────────
prompt = st.chat_input(
    "코치에게 목표나 고민을 이야기해보세요 💬",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
)

if prompt:
    # 이전 플레이스홀더 초기화
    for key in ("image_placeholder", "text_placeholder"):
        if key in st.session_state:
            st.session_state[key].empty()

    # 파일 업로드 처리
    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ 파일 업로드 중...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ 목표 파일 저장 중...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ 파일 업로드 완료!", state="complete")

        elif file.type.startswith("image/"):
            with st.status("⏳ 이미지 업로드 중...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
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
                status.update(label="✅ 이미지 업로드 완료!", state="complete")
            with st.chat_message("human"):
                st.image(data_uri)

    # 텍스트 메시지 처리
    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


# ── 사이드바 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    if st.button("🗑️ 대화 기록 초기화"):
        asyncio.run(session.clear_session())
        st.success("대화 기록이 초기화되었습니다!")
        st.rerun()

    st.divider()
    st.subheader("📋 사용 가이드")
    st.markdown("""
    **이런 말을 해보세요:**
    
    🎯 *"올해 목표로 비전 보드 만들어줘"*
    
    🏆 *"책 10권 읽기 목표 달성했어!"*
    
    💪 *"운동 습관 만드는 법 알려줘"*
    
    📁 *"내 목표 파일 업로드할게"*
    
    🖼️ *"동기부여 포스터 만들어줘"*
    """)

    st.divider()
    with st.expander("🔍 대화 기록 보기"):
        st.write(asyncio.run(session.get_items()))