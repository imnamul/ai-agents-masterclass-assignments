from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from .story_writer.agent import story_writer_agent
from .illustrator.agent import illustrator_agent
from .book_assembler.agent import book_assembler_agent
from .callbacks import on_pipeline_done

MODEL = LiteLlm(model="openai/gpt-4o")


def _status_agent(name: str, text: str) -> Agent:
    """
    Creates a no-op agent whose sole job is to emit one status line in ADK Web.

    How it works:
      - before_agent_callback returns types.Content → ADK emits the event in the
        Web UI, then sets end_invocation=True and skips this agent's body.
      - That is harmless because the agent has no real work to do.
      - SequentialAgent does NOT check end_invocation between sub-agents, so the
        next real sub-agent runs normally.
    """
    def _announce(callback_context: CallbackContext) -> types.Content:
        return types.Content(role="model", parts=[types.Part(text=text)])

    return Agent(
        name=name,
        model=MODEL,
        description=f"Status announcer: {text}",
        before_agent_callback=_announce,
    )


# Step 1~3 pipeline — executes after theme is confirmed by the user
# Note: current_run_artifacts state is reset by story_writer_agent's before_agent_callback,
# which fires reliably on every pipeline invocation (including same-session re-runs).
pipeline_agent = SequentialAgent(
    name="story_book_maker_pipeline",
    description="Given a theme, writes a story, illustrates all pages, then assembles the final book",
    sub_agents=[
        _status_agent("announce_story_writing",  "📖 Writing the story…"),
        story_writer_agent,           # Step 1: write the story
        _status_agent("announce_illustrating",   "🎨 Generating illustrations for all 5 pages…"),
        illustrator_agent,            # Step 2: generate all 5 images
        _status_agent("announce_assembly",       "📚 Assembling the final book…"),
        book_assembler_agent,         # Step 3: overlay text + page numbers
    ],
    after_agent_callback=on_pipeline_done,
)

# root_agent: the only agent that talks to the user
# Once theme is confirmed, delegates everything to pipeline_agent
story_book_maker_agent = Agent(
    name="story_book_maker_agent",
    model=MODEL,
    description="Collects a theme from the user then runs the full picture book pipeline",
    instruction="""
    당신은 따뜻하고 창의적인 어린이 그림책 감독입니다.

    다음 단계를 정확히 따르세요:

    Step 1. 사용자가 테마를 제공하지 않은 경우, 인사 후 테마를 요청하세요.
            예시: "안녕하세요! 오늘 어떤 테마로 그림책을 만들어 드릴까요?
                  (예: '용감한 아기 고양이', '불을 무서워하는 아기 용')"

    Step 2. 테마가 정해지면 시작 전에 사용자에게 확인하세요.
            예시: "'용감한 아기 고양이' 테마로 그림책을 만들겠습니다.
                  잠시만 기다려 주세요! 📖✨"

    Step 3. 확정된 테마를 메시지에 포함하여 story_book_maker_pipeline으로 전달하세요.

    Step 4. 파이프라인이 완료되면 최종 그림책을 사용자에게 이렇게 제시하세요:
            - 제목
            - 표지 및 각 페이지의 텍스트
            - 각 페이지의 최종 이미지 파일명 (final_page_00.jpeg ~ final_page_05.jpeg)
            - 따뜻한 마무리 메시지
            예시: "그림책이 완성되었습니다! 🎉
                  제목: 용감한 아기 고양이의 모험
                  - 표지 (final_page_00.jpeg)
                  - 1페이지: ... (final_page_01.jpeg)
                  - 2페이지: ... (final_page_02.jpeg)
                  ..."

    항상 한국어로 소통하세요. 사용자가 다른 언어를 사용하는 경우 해당 언어로 소통하세요.
    Step 1을 절대 건너뛰지 마세요 — 파이프라인 시작 전에 반드시 테마를 확인하세요.
    """,
    sub_agents=[pipeline_agent],
)

root_agent = story_book_maker_agent
