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


def _build_prompt(characters: list, scene_description: str) -> str:
    """
    Assembles a full image generation prompt by combining:
    1. Style anchor   — consistent art style across all pages
    2. Character desc — fixed appearance tokens for each character present in the scene
    3. Scene desc     — page-specific background, action, and mood

    characters: list of CharacterSheet objects or dicts with 'appearance' key.
    scene_description: may already start with the style prefix; it will be stripped
                       to avoid duplication.
    """
    # Strip the style prefix if the scene_description already contains it
    clean_scene = scene_description
    if clean_scene.lower().startswith(_STYLE.lower()):
        clean_scene = clean_scene[len(_STYLE):].lstrip(",").strip()

    if not characters:
        return f"{_STYLE}, {clean_scene}"

    appearances = [
        c.get("appearance", "") if isinstance(c, dict) else getattr(c, "appearance", "")
        for c in characters
    ]
    appearances = [a for a in appearances if a]  # drop empty strings

    if not appearances:
        return f"{_STYLE}, {clean_scene}"

    if len(appearances) == 1:
        char_desc = appearances[0]
    elif len(appearances) == 2:
        char_desc = f"{appearances[0]} and {appearances[1]}"
    else:
        char_desc = ", ".join(appearances[:-1]) + f", and {appearances[-1]}"

    return f"{_STYLE}, featuring {char_desc}, {clean_scene}, consistent character design"


def _get_characters(story_output, names: list = None) -> list:
    """
    Returns character objects from story_output.characters.
    If names is provided, returns only characters whose name is in the list,
    preserving the order of names (main character first).
    Handles both dict and ADK object forms of story_output.
    """
    characters = (
        story_output.get("characters") if isinstance(story_output, dict)
        else getattr(story_output, "characters", None)
    )
    if not characters:
        return []

    if names is None:
        return list(characters)

    # Build a lookup by name for O(1) access
    by_name = {}
    for c in characters:
        c_name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "")
        by_name[c_name] = c

    # Return in the order specified by names (preserves main-character-first ordering)
    return [by_name[n] for n in names if n in by_name]


def _make_title_page_agent() -> Agent:
    async def _generate_title(tool_context: ToolContext) -> dict:
        client = OpenAI()
        story_output = tool_context.state.get("story_output")
        if not story_output:
            return {"status": "error", "message": "'story_output' not found in state."}

        filename = "page_00.jpeg"
        run_artifacts = tool_context.state.get("current_run_artifacts", [])
        if filename in run_artifacts:
            return {"status": "skipped", "filename": filename}

        visual_description = (
            story_output.get("title_visual_description", "")
            if isinstance(story_output, dict)
            else getattr(story_output, "title_visual_description", "")
        )
        if not visual_description:
            return {"status": "error", "message": "title_visual_description not found in story_output."}

        # Cover page shows all characters (or at least the main one)
        characters = _get_characters(story_output)
        full_prompt = _build_prompt(characters, visual_description)

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
        tool_context.state["current_run_artifacts"] = run_artifacts + [filename]
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
        client = OpenAI()
        story_output = tool_context.state.get("story_output")
        if not story_output:
            return {"status": "error", "message": "'story_output' not found in state."}

        pages = story_output.get("pages") if isinstance(story_output, dict) else story_output.pages
        page = next(
            (p for p in pages if (p["page_number"] if isinstance(p, dict) else p.page_number) == page_number),
            None,
        )
        if not page:
            return {"status": "error", "message": f"Page {page_number} not found in story output."}

        filename = f"page_{page_number:02d}.jpeg"
        run_artifacts = tool_context.state.get("current_run_artifacts", [])
        if filename in run_artifacts:
            return {"status": "skipped", "filename": filename}

        visual_description = page["visual_description"] if isinstance(page, dict) else page.visual_description

        # ── Inject only the characters actually present in this scene ────────
        characters_present = (
            page.get("characters_present", []) if isinstance(page, dict)
            else getattr(page, "characters_present", [])
        )
        page_characters = _get_characters(story_output, names=characters_present)
        full_prompt = _build_prompt(page_characters, visual_description)

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
        tool_context.state["current_run_artifacts"] = run_artifacts + [filename]
        return {"status": "complete", "page_number": page_number, "filename": filename}

    # ADK registers the tool by function name — give each one a unique name
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
