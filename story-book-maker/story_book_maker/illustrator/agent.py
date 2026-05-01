import base64
from openai import OpenAI
from google.adk.agents import Agent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")


def _make_page_agent(page_number: int) -> Agent:
    async def _generate(tool_context: ToolContext) -> dict:
        client = OpenAI()
        story_output = tool_context.state.get("story_output")
        if not story_output:
            return {"status": "error", "message": "'story_output' not found in state."}

        pages = story_output.get("pages") if isinstance(story_output, dict) else story_output.pages
        page = next((p for p in pages if p["page_number"] == page_number), None)
        if not page:
            return {"status": "error", "message": f"Page {page_number} not found in story output."}

        filename = f"page_{page_number:02d}.jpeg"
        existing = await tool_context.list_artifacts()
        if filename in existing:
            return {"status": "skipped", "filename": filename}

        response = client.images.generate(
            model="gpt-image-1",
            prompt=page["visual_description"],
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
    )


illustrator_agent = ParallelAgent(
    name="illustrator_agent",
    description="Generates illustrations for all 5 story pages simultaneously",
    sub_agents=[_make_page_agent(i) for i in range(1, 6)],
)
