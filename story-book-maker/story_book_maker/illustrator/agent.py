import base64
from openai import OpenAI
from google.adk.agents import Agent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from ..callbacks import make_page_illustration_done

MODEL = LiteLlm(model="openai/gpt-4o")

_STYLE = "watercolor illustration, children's book style"


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


def _make_title_page_agent() -> Agent:
    async def _generate_title(tool_context: ToolContext) -> dict:
        raw = tool_context.state.get("story_output")
        if not raw:
            return {"status": "error", "message": "'story_output' not found in state."}

        # ── 정규화: 이후 코드는 항상 dict 접근 ──────────────────────────────
        story = _normalize(raw)

        filename = "page_00.jpeg"
        run_artifacts = tool_context.state.get("current_run_artifacts", [])
        if filename in run_artifacts:
            return {"status": "skipped", "filename": filename}

        visual_description = story.get("title_visual_description", "")
        if not visual_description:
            return {"status": "error", "message": "title_visual_description not found in story_output."}

        characters = _get_characters(story)
        full_prompt = _build_prompt(characters, visual_description)

        client = OpenAI()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        await tool_context.save_artifact(
            filename=filename,
            artifact=types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)
            ),
        )
        fresh = tool_context.state.get("current_run_artifacts", [])
        tool_context.state["current_run_artifacts"] = fresh + [filename]
        return {"status": "complete", "page_number": 0, "filename": filename}

    _generate_title.__name__ = "generate_title_page_image"

    return Agent(
        name="title_page_illustrator",
        model=MODEL,
        description="Generates the illustration for the title/cover page",
        instruction=(
            "Call generate_title_page_image to create and save the cover illustration. "
            "Report the result."
        ),
        tools=[FunctionTool(func=_generate_title)],
        after_agent_callback=make_page_illustration_done(0),
    )


def _make_page_agent(page_number: int) -> Agent:
    async def _generate(tool_context: ToolContext) -> dict:
        raw = tool_context.state.get("story_output")
        if not raw:
            return {"status": "error", "message": "'story_output' not found in state."}

        # ── 정규화: 이후 코드는 항상 dict 접근 ──────────────────────────────
        story = _normalize(raw)

        pages = story.get("pages", [])
        page = next((p for p in pages if p["page_number"] == page_number), None)
        if not page:
            return {"status": "error", "message": f"Page {page_number} not found in story output."}

        filename = f"page_{page_number:02d}.jpeg"
        run_artifacts = tool_context.state.get("current_run_artifacts", [])
        if filename in run_artifacts:
            return {"status": "skipped", "filename": filename}

        visual_description = page["visual_description"]
        characters_present = page.get("characters_present", [])
        page_characters = _get_characters(story, names=characters_present)
        full_prompt = _build_prompt(page_characters, visual_description)

        client = OpenAI()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",
        )
        image_bytes = base64.b64decode(response.data[0].b64_json)
        await tool_context.save_artifact(
            filename=filename,
            artifact=types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)
            ),
        )
        fresh = tool_context.state.get("current_run_artifacts", [])
        tool_context.state["current_run_artifacts"] = fresh + [filename]
        return {"status": "complete", "page_number": page_number, "filename": filename}

    _generate.__name__ = f"generate_page_{page_number}_image"

    return Agent(
        name=f"page_illustrator_{page_number}",
        model=MODEL,
        description=f"Generates the illustration for page {page_number}",
        instruction=(
            f"Call generate_page_{page_number}_image to create and save the illustration "
            f"for page {page_number}. Report the result."
        ),
        tools=[FunctionTool(func=_generate)],
        after_agent_callback=make_page_illustration_done(page_number),
    )


illustrator_agent = ParallelAgent(
    name="illustrator_agent",
    description="Generates illustrations for the title page and all 5 story pages simultaneously",
    sub_agents=[_make_title_page_agent()] + [_make_page_agent(i) for i in range(1, 6)],
)
