import asyncio
import base64
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from openai import AsyncOpenAI

from ..callbacks import make_page_illustration_done  # noqa: F401 (kept for import compat)

_STYLE = "watercolor illustration, children's book style"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(obj) -> dict:
    """Pydantic 모델이든 dict든 항상 순수 dict로 반환."""
    return obj if isinstance(obj, dict) else obj.model_dump()


def _build_prompt(characters: list[dict], scene_description: str) -> str:
    """
    이미지 생성 프롬프트 조립:
    1. Style anchor  — 모든 페이지에 공통 적용되는 아트 스타일
    2. Character desc — 해당 씬에 등장하는 캐릭터의 외형 설명
    3. Scene desc    — 페이지별 배경, 행동, 분위기

    scene_description이 이미 스타일 prefix로 시작하면 중복을 제거합니다.
    """
    clean_scene = scene_description
    if clean_scene.lower().startswith(_STYLE.lower()):
        clean_scene = clean_scene[len(_STYLE):].lstrip(",").strip()

    if not characters:
        return f"{_STYLE}, {clean_scene}"

    appearances = [c.get("appearance", "") for c in characters]
    appearances = [a for a in appearances if a]

    if not appearances:
        return f"{_STYLE}, {clean_scene}"

    if len(appearances) == 1:
        char_desc = appearances[0]
    elif len(appearances) == 2:
        char_desc = f"{appearances[0]} and {appearances[1]}"
    else:
        char_desc = ", ".join(appearances[:-1]) + f", and {appearances[-1]}"

    return f"{_STYLE}, featuring {char_desc}, {clean_scene}, consistent character design"


def _get_characters(story: dict, names: list[str] | None = None) -> list[dict]:
    """
    story dict에서 캐릭터 목록을 반환합니다.
    names가 주어지면 해당 이름의 캐릭터만 필터링해서 반환합니다.
    """
    characters = story.get("characters", [])
    if not characters:
        return []

    if names is None:
        return list(characters)

    by_name = {c["name"]: c for c in characters}
    return [by_name[n] for n in names if n in by_name]


# ── Per-page generation ────────────────────────────────────────────────────────

async def _generate_one(
    tool_ctx: ToolContext,
    story: dict,
    page_number: int,
    run_artifacts: set[str],
) -> dict:
    """
    Generates and saves the illustration for one page.

    Safe to run concurrently with other calls to this function because:
    - Each page writes to a uniquely-named artifact (page_00.jpeg … page_05.jpeg)
    - Uses AsyncOpenAI so the HTTP request yields the event loop to siblings
    - Does not touch shared state (current_run_artifacts is read-only here;
      the caller updates it once after asyncio.gather returns)
    """
    filename = f"page_{page_number:02d}.jpeg"

    if filename in run_artifacts:
        return {"status": "skipped", "page_number": page_number, "filename": filename}

    if page_number == 0:
        visual_description = story.get("title_visual_description", "")
        if not visual_description:
            return {"status": "error", "page_number": page_number,
                    "message": "title_visual_description not found in story_output"}
        characters = _get_characters(story)
    else:
        pages = story.get("pages", [])
        page = next((p for p in pages if p["page_number"] == page_number), None)
        if not page:
            return {"status": "error", "page_number": page_number,
                    "message": f"page_number {page_number} not found in story_output"}
        visual_description = page["visual_description"]
        characters = _get_characters(story, names=page.get("characters_present"))

    prompt = _build_prompt(characters, visual_description)

    client = AsyncOpenAI()
    response = await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        quality="high",
        moderation="low",
        output_format="jpeg",
        background="opaque",
        size="1024x1536",
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)

    await tool_ctx.save_artifact(
        filename=filename,
        artifact=types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)
        ),
    )
    return {"status": "complete", "page_number": page_number, "filename": filename}


# ── Agent ──────────────────────────────────────────────────────────────────────

class IllustratorAgent(BaseAgent):
    """
    Generates all page illustrations concurrently with asyncio.gather.

    Why this replaces ParallelAgent + 6 LLM-based sub-agents:

    1. No LLM needed — _generate_one() is a deterministic async function.
       The old LLM-agent approach let gpt-4o skip the tool call and return
       plain text, leaving page_XX.jpeg at version 0.

    2. True async concurrency — AsyncOpenAI releases the event loop while
       waiting for each HTTP response, so all 6 image requests fly in
       parallel.  The old OpenAI() (sync) client blocked the loop.

    3. No race condition — current_run_artifacts is written once after
       asyncio.gather returns, not from 6 agents simultaneously.

    4. Dynamic page count — page_numbers is built from story_output at
       runtime, so the agent handles any number of pages without code
       changes.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        tool_ctx = ToolContext(invocation_context=ctx)

        raw = tool_ctx.state.get("story_output")
        if not raw:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="❌ story_output not found in state.")],
                ),
            )
            return

        story = _normalize(raw)
        run_artifacts = set(tool_ctx.state.get("current_run_artifacts", []))

        # Cover page (0) + however many story pages the writer produced
        page_numbers = [0] + [p["page_number"] for p in story.get("pages", [])]

        # Announce start — one message before any image request goes out
        total = len(page_numbers)
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"🎨 Generating {total} illustrations in parallel…")],
            ),
        )

        # Launch all tasks simultaneously, then yield a status event as each
        # one finishes — gives the same progressive feedback as the old
        # per-agent after_agent_callback, but without waiting for all pages.
        tasks = [
            asyncio.create_task(_generate_one(tool_ctx, story, n, run_artifacts))
            for n in page_numbers
        ]

        completed_filenames = []
        for future in asyncio.as_completed(tasks):
            try:
                r = await future
            except Exception as e:
                r = {"status": "error", "page_number": "?", "message": str(e)}

            n = r.get("page_number", "?")
            label = "Title page" if n == 0 else f"Image {n}/5"
            if r.get("status") == "complete":
                completed_filenames.append(r["filename"])
                msg = f"✅ {label} saved."
            elif r.get("status") == "skipped":
                msg = f"⏭ {label} skipped (already generated this run)."
            else:
                msg = f"❌ {label} error: {r.get('message', 'unknown')}"

            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=msg)],
                ),
            )

        # Update current_run_artifacts once, after all tasks finish
        tool_ctx.state["current_run_artifacts"] = list(run_artifacts) + completed_filenames


illustrator_agent = IllustratorAgent(
    name="illustrator_agent",
    description="Generates illustrations for the title page and all 5 story pages simultaneously",
)
