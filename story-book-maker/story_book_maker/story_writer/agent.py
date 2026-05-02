from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from typing import List
from ..callbacks import on_story_writing_done


MODEL = LiteLlm(model="openai/gpt-4o")


# Pydantic model for a single page — ADK validates the LLM output against this schema
class PageOutput(BaseModel):
    page_number: int = Field(description="Page number (1-5)")
    text: str = Field(description="Page body text in English, 2-3 sentences")
    visual_description: str = Field(
        description=(
            "Image generation prompt in English, starting with "
            "'watercolor illustration, children's book style,' "
            "describing ONLY the scene, background, mood, action, and colors. "
            "Do NOT repeat the character appearance here — that is in character_sheet."
        )
    )


# Pydantic model for the full story output
class StoryOutput(BaseModel):
    theme: str = Field(description="The theme of the story")
    title: str = Field(description="A short, catchy title for the picture book")
    total_pages: int = Field(description="Total number of pages including title page (always 6)")
    character_sheet: str = Field(
        description=(
            "A single, detailed visual description of the main character(s) "
            "that will be used verbatim in EVERY page illustration to keep "
            "their appearance perfectly consistent. "
            "Include: species, body size, fur/skin/scale color, eye color, "
            "clothing with exact colors and patterns, any distinctive features "
            "(spots, stripes, accessories, etc.). "
            "Example: 'a small orange tabby kitten with a white chest patch and "
            "bright green eyes, wearing a tiny red scarf with white polka dots "
            "and a yellow sun hat, fluffy tail with a dark tip'"
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
    A short, memorable title for the book (e.g. "The Brave Baby Kitten's Adventure").

    ── character_sheet (CRITICAL for illustration consistency) ──────────────
    Define the main character(s) ONCE with extreme visual specificity:
    - Species and body size (e.g. "small orange tabby kitten")
    - Exact fur / skin / scale color and any markings
    - Eye color
    - Clothing: item name + exact color + pattern
      (e.g. "tiny red scarf with white polka dots", "yellow rain boots")
    - Distinctive accessories or features (e.g. "round gold glasses")
    This string will be prepended to EVERY page illustration prompt.

    ── title_visual_description (cover page) ────────────────────────────────
    - Must be written in English
    - Start with "watercolor illustration, children's book style,"
    - Show the main character in a full portrait pose that captures the story's mood
    - Include a rich, atmospheric background that hints at the story world
    - Do NOT include any text or lettering in the image

    ── visual_description (per story page) ──────────────────────────────────
    - Must be written in English
    - Start with "watercolor illustration, children's book style,"
    - Describe ONLY the scene: background, setting, action, mood, lighting, colors
    - Do NOT describe the character's appearance here — that comes from character_sheet
    - Example: "watercolor illustration, children's book style,
      standing at the edge of a dark enchanted forest, golden afternoon light
      filtering through the trees, soft pastel greens and yellows"
    """,
    output_schema=StoryOutput,
    output_key="story_output",
    after_agent_callback=on_story_writing_done,
)


