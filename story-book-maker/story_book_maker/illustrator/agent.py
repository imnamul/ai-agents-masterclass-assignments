import base64
from openai import OpenAI
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")

# Initialize the OpenAI client at module level to reuse across calls
client = OpenAI()


async def generate_image(tool_context: ToolContext) -> dict:
    """
    Reads all story pages from Session State, generates an image for each page,
    and saves each image as an ADK artifact.

    Skips pages that already have a saved artifact (idempotent — safe to retry).
    Returns a summary dict of all generated images.
    """
    # Read the structured story output saved by story_writer_agent
    story_output = tool_context.state.get("story_output")

    if not story_output:
        return {
            "status": "error",
            "message": "'story_output' not found in state. Please run story_writer_agent first.",
        }

    # story_output may be a dict (ADK serializes Pydantic models to dict in state)
    pages = story_output.get("pages") if isinstance(story_output, dict) else story_output.pages

    if not pages:
        return {
            "status": "error",
            "message": "'story_pages' not found in state. Please run story_writer_agent first.",
        }

    # Fetch the list of already-saved artifacts to avoid regenerating existing images
    existing_artifacts = await tool_context.list_artifacts()

    generated_images = []

    for page in pages:
        page_number = page["page_number"]
        prompt = page["visual_description"]
        filename = f"page_{page_number:02d}.jpeg"

        # Skip this page if the artifact already exists (e.g. on retry)
        if filename in existing_artifacts:
            generated_images.append({
                "page_number": page_number,
                "filename": filename,
                "prompt": prompt[:100],
                "skipped": True,
            })
            continue

        # Call OpenAI image generation API with the visual description prompt
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            quality="low",
            moderation="low",
            output_format="jpeg",
            background="opaque",
            size="1024x1536",   # Portrait orientation — suits children's book pages
        )

        # Decode the base64-encoded image bytes returned by the API
        image_bytes = base64.b64decode(response.data[0].b64_json)

        # Wrap the image bytes in a genai Part for ADK artifact storage
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="image/jpeg",
                data=image_bytes,
            )
        )

        # Save the image as a named artifact in the ADK artifact service
        await tool_context.save_artifact(
            filename=filename,
            artifact=artifact,
        )

        generated_images.append({
            "page_number": page_number,
            "filename": filename,
            "prompt": prompt[:100],
            "skipped": False,
        })

    # Persist the generated image metadata to State for downstream access
    tool_context.state["generated_images"] = {
        str(item["page_number"]): {
            "artifact_filename": item["filename"],
            "prompt": item["prompt"],
        }
        for item in generated_images
    }

    return {
        "total_images": len(generated_images),
        "generated_images": generated_images,
        "status": "complete",
    }


# Register the async function as an ADK tool
generate_image_tool = FunctionTool(func=generate_image)

illustrator_agent = Agent(
    name="illustrator_agent",
    model=MODEL,
    description="An agent that reads story data from State and generates an image for each page",
    instruction="""
    You are a children's book illustrator.
    Call the generate_image tool once — it will handle all pages automatically.

    After the tool returns, report a summary to the user listing:
    - Total number of images generated
    - Each page number and its saved artifact filename
    - Any pages that were skipped (already existed)
    """,
    tools=[generate_image_tool],
)
