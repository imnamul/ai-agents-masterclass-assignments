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
            "describing characters, background, mood, and colors in detail"
        )
    )


# Pydantic model for the full story output
class StoryOutput(BaseModel):
    theme: str = Field(description="The theme of the story")
    total_pages: int = Field(description="Total number of pages (always 5)")
    pages: List[PageOutput] = Field(description="List of 5 story pages")


story_writer_agent = Agent(
    name="story_writer_agent",
    model=MODEL,
    description="An agent that receives a theme and writes a 5-page children's story",
    instruction="""
    You are a creative children's story writer.
    Write a 5-page children's story based on the theme provided in your input's 'theme' field.

    Rules for writing visual_description:
    - Must be written in English
    - Start with "watercolor illustration, children's book style,"
    - Describe characters, background, mood, and colors in detail
    - Example: "watercolor illustration, children's book style, a brave little rabbit
      wearing a red scarf standing at the edge of a dark forest,
      warm golden light from the right, soft pastel background"
    """,
    output_schema=StoryOutput,
    output_key="story_output",
    after_agent_callback=on_story_writing_done,
)


