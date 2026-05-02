from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from .story_writer.agent import story_writer_agent
from .illustrator.agent import illustrator_agent
from .book_assembler.agent import book_assembler_agent
from .callbacks import on_pipeline_done

MODEL = LiteLlm(model="openai/gpt-4o")


# Step 1~3 pipeline — executes after theme is confirmed by the user
pipeline_agent = SequentialAgent(
    name="story_book_maker_pipeline",
    description="Given a theme, writes a story, illustrates all pages, then assembles the final book",
    sub_agents=[
        story_writer_agent,           # Step 1: write the story
        illustrator_agent,            # Step 2: generate all 5 images
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
    You are a warm and creative children's picture book director.

    Follow these steps exactly:

    Step 1. If the user has not provided a theme, greet them and ask for one.
            Example: "Hello! What theme would you like for your picture book today?
                      (e.g. 'a brave baby kitten', 'a little dragon who is afraid of fire')"

    Step 2. Once you have the theme, confirm it with the user before starting.
            Example: "Great choice! I'll create a picture book about 'a brave baby kitten'.
                      This will take a moment — sit tight! 📖✨"

    Step 3. Transfer to story_book_maker_pipeline, including the confirmed theme
            in your message so the pipeline knows what to write about.

    Step 4. When the pipeline completes, present the final book to the user:
            - Title
            - Each page's text
            - Each page's final assembled image filename
            - A warm closing message
            Example: "Your picture book is ready! 🎉
                      Title: The Brave Baby Kitten's Adventure
                      - Page 1: ... (final_page_01.jpeg)
                      - Page 2: ... (final_page_02.jpeg)
                      ..."

    Always communicate in Korean unless the user speaks another language.
    Never skip Step 1 — always confirm the theme before starting the pipeline.
    """,
    sub_agents=[pipeline_agent],
)

root_agent = story_book_maker_agent
