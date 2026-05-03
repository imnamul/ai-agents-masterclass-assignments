from typing import Optional, List
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from pydantic import BaseModel, Field
from ..callbacks import on_story_writing_done


def _reset_run_artifacts(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """
    Clears the per-run artifact tracker at the start of every new story.
    Attached to story_writer_agent (the first real step in the pipeline)
    so it fires reliably on each pipeline invocation, including re-runs
    in the same session.
    """
    callback_context.state["current_run_artifacts"] = []
    return None


MODEL = LiteLlm(model="openai/gpt-4o")


class CharacterSheet(BaseModel):
    name: str = Field(
        description="Short identifier for this character (e.g. 'bunny', 'wizard', 'fox'). "
                    "Used in characters_present to reference who appears on each page."
    )
    appearance: str = Field(
        description=(
            "Detailed visual description used verbatim in every illustration prompt where "
            "this character appears. Include: species/type, body size, fur/skin/scale color "
            "and markings, eye color, clothing with exact colors and patterns, distinctive "
            "accessories or features. "
            "Example: 'a small orange tabby kitten with a white chest patch and bright green "
            "eyes, wearing a tiny red scarf with white polka dots and a yellow sun hat, "
            "fluffy tail with a dark tip'"
        )
    )


# Pydantic model for a single page — ADK validates the LLM output against this schema
class PageOutput(BaseModel):
    page_number: int = Field(description="Page number (1-5)")
    text: str = Field(description="Page body text in Korean, 2-3 sentences")
    characters_present: List[str] = Field(
        description=(
            "List of character name identifiers (matching CharacterSheet.name) who are "
            "visually present in this page's scene. Only include characters who actually "
            "appear in the illustration — do not list off-screen characters."
        )
    )
    visual_description: str = Field(
        description=(
            "Image generation prompt in English, starting with "
            "'watercolor illustration, children's book style,' "
            "describing ONLY the scene, background, mood, action, and colors. "
            "Do NOT describe character appearances here — those come from CharacterSheet."
        )
    )


# Pydantic model for the full story output
class StoryOutput(BaseModel):
    theme: str = Field(description="The theme of the story")
    title: str = Field(description="A short, catchy title for the picture book")
    total_pages: int = Field(description="Total number of pages including the title/cover page. Always 6 (1 cover + 5 story pages).")
    characters: List[CharacterSheet] = Field(
        description=(
            "ALL characters who appear anywhere in the story, each with a detailed visual "
            "description. Define EVERY character here before referencing them in any page. "
            "The first entry is the main character. Secondary characters (friends, antagonists, "
            "helpers, etc.) must also be listed here if they appear in any illustration."
        )
    )
    title_visual_description: str = Field(
        description=(
            "Image generation prompt for the title/cover page illustration. "
            "Start with 'watercolor illustration, children's book style,' "
            "Show the main character in a full portrait pose that captures "
            "the overall mood and atmosphere of the story. "
            "Do NOT include any text or lettering in the image."
        )
    )
    pages: List[PageOutput] = Field(description="List of 5 story pages (pages 1-5)")


story_writer_agent = Agent(
    name="story_writer_agent",
    model=MODEL,
    description="An agent that receives a theme and writes a 5-page children's story",
    instruction="""
    You are a creative children's story writer.
    Write a 6-page picture book (1 title/cover page + 5 story pages) based on the theme provided.

    ── title ────────────────────────────────────────────────────────────────
    A short, memorable title for the book in Korean.
    Example: "용감한 아기 고양이의 모험"

    ── characters (CRITICAL — define ALL characters upfront) ─────────────────
    List EVERY character who appears in ANY page illustration before writing pages.
    Each entry needs:
    - name: short English identifier (e.g. "bunny", "wizard", "fox"). Used in characters_present.
    - appearance: extremely detailed ENGLISH visual description injected verbatim into every
      illustration where this character appears:
        * Species and body size (e.g. "small orange tabby kitten")
        * Exact fur / skin / scale color and markings
        * Eye color
        * Clothing: item + exact color + pattern
          (e.g. "tiny red scarf with white polka dots", "yellow rain boots")
        * Distinctive accessories (e.g. "round gold glasses", "silver wand")

    RULE: ANY character visible in a visual_description MUST be listed in characters.
    Do NOT introduce unnamed or undescribed characters inside visual_descriptions.

    ── title_visual_description (cover page) ────────────────────────────────
    - Must be written in English
    - Start exactly with "watercolor illustration, children's book style,"
    - Show the main character in a full portrait pose that captures the story's mood
    - Include a rich, atmospheric background that hints at the story world
    - Do NOT include any text or lettering in the image

    ── pages (5 story pages) ────────────────────────────────────────────────
    For each page provide:

    text:
    - Write in Korean, 2-3 sentences
    - Simple, warm language suitable for young children

    characters_present:
    - List only the character name identifiers (from characters) who are VISUALLY
      IN the scene on this page.
    - Omit characters who are mentioned in text but not shown in the image.

    visual_description:
    - Must be written in English
    - Start exactly with "watercolor illustration, children's book style,"
    - Describe ONLY the scene: background, setting, action, mood, lighting, colors
    - Do NOT describe character appearances — those come from the characters list
    - Example: "watercolor illustration, children's book style, standing at the edge of a dark enchanted forest, golden afternoon light filtering through the trees, soft pastel greens and yellows"
    """,
    output_schema=StoryOutput,
    output_key="story_output",
    before_agent_callback=_reset_run_artifacts,
    after_agent_callback=on_story_writing_done,
)
