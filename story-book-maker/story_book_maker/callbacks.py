from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types


def _make_content(text: str) -> LlmResponse:
    """
    Wraps a text message into an LlmResponse.
    ADK renders this as a chat message in the Web UI when returned from
    after_agent_callback.
    """
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        )
    )


# ── Pipeline-level ────────────────────────────────────────────────────────────

def on_pipeline_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the full pipeline completes (after_agent_callback)."""
    return _make_content("🎉 Pipeline complete! Your picture book is ready.")


# ── Story writer ──────────────────────────────────────────────────────────────

def on_story_writing_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the story writer agent finishes (after_agent_callback)."""
    return _make_content("✅ Story writing complete!")


# ── Illustrator ───────────────────────────────────────────────────────────────

def make_page_illustration_done(page_number: int):
    """Returns an after_agent_callback that fires when a page illustrator finishes."""
    def callback(callback_context: CallbackContext) -> Optional[LlmResponse]:
        label = "Title page" if page_number == 0 else f"Image {page_number}/5"
        return _make_content(f"✅ {label} saved.")
    return callback


# ── Book assembler ────────────────────────────────────────────────────────────

def on_assembly_done(callback_context: CallbackContext) -> Optional[LlmResponse]:
    """Fires when the book assembler agent finishes (after_agent_callback)."""
    return _make_content("📚 Picture book assembly complete!")