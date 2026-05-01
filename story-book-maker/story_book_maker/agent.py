from google.adk.agents import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from .story_writer.agent import story_writer_agent
from .illustrator.agent import illustrator_agent
from .book_assembler.agent import book_assembler_agent
from .callbacks import before_agent_callback, after_agent_callback

MODEL = LiteLlm(model="openai/gpt-4o")


story_book_maker_agent = SequentialAgent(
    name="StoryBookMaker",
    description="Writes a story, illustrates each page in parallel, then assembles the final picture book",
    sub_agents=[
        story_writer_agent,     # Step 1: write 5-page story → state["story_output"]
        illustrator_agent,      # Step 2: generate page_01.jpeg ~ page_05.jpeg in parallel
        book_assembler_agent,   # Step 3: overlay text + page numbers → final_page_01.jpeg ~ final_page_05.jpeg
    ],
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
)

root_agent = story_book_maker_agent
