from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types


def _make_content(text: str) -> Optional[LlmResponse]:
    """
    Helper that wraps a text message into an LlmResponse,
    which ADK renders as a chat message in the Web UI.
    """
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        )
    )


def on_pipeline_start(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the full pipeline starts."""
    theme = callback_context.state.get("theme", "")
    print(f"🚀 Starting the pipeline! Theme: **{theme}**")
    return None


def on_pipeline_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the full pipeline completes."""
    return _make_content("🎉 Pipeline complete! Your picture book is ready.")


def on_story_writing_start(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the story writer agent starts."""
    print("📖 Writing the story...")
    return None


def on_story_writing_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the story writer agent finishes."""
    return _make_content("✅ Story writing complete!")


def make_page_illustration_start(page_number: int):
    """Returns a callback that fires when a specific page illustrator starts."""
    def callback(callback_context: CallbackContext) -> Optional[LlmResponse]:
        print(f"🎨 Generating image {page_number}/5...")
        return None
    return callback


def make_page_illustration_done(page_number: int):
    """Returns a callback that fires when a specific page illustrator finishes."""
    def callback(callback_context: CallbackContext) -> Optional[LlmResponse]:
        return _make_content(f"✅ Image {page_number}/5 saved.")
    return callback


def on_assembly_start(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the book assembler agent starts."""
    print("📚 Assembling the final book...")
    return None

def on_assembly_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the book assembler agent finishes."""
    return _make_content("🎉 Picture book assembly complete!")