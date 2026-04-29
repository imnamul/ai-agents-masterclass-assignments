from google.genai import types
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from .story_writer.agent import story_writer_agent
from .illustrator.agent import illustrator_agent

MODEL = LiteLlm(model="openai/gpt-4o")


story_book_maker_agent = Agent(
    name="StoryBookMaker",
    model=MODEL,
    description="Root orchestrator agent for children's book creation",
    instruction="""
        You are a children's book production orchestrator agent.

        When you receive a theme from the user, follow these steps in order:

        Step 1. Call the story_writer_agent tool with the theme to generate the story.
                - If it fails, stop immediately and report the error to the user.

        Step 2. Only if Step 1 succeeds, call the illustrator_agent tool to generate images.
                - The illustrator will read the story data from the shared session state.
                - If it fails, report which pages could not be generated.

        Step 3. Summarize the final results and report back to the user,
                including the story text and a list of generated image URLs per page.

        Never run the two tools in parallel. Always wait for each step to complete
        before proceeding to the next.
        """,
    tools=[
        AgentTool(agent=story_writer_agent),
        AgentTool(agent=illustrator_agent),
    ],
)

root_agent = story_book_maker_agent