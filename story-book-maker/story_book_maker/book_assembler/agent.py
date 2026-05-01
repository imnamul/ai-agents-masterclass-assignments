import io
from PIL import Image, ImageDraw, ImageFont
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = LiteLlm(model="openai/gpt-4o")

_BANNER_HEIGHT = 220
_PADDING = 16
_PAGE_NUM_AREA = 32
_TEXT_SIZE = 28
_NUM_SIZE = 22


def _get_font(size: int):
    for path in [
        "C:/Windows/Fonts/malgun.ttf",       # Windows — Malgun Gothic (Korean)
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


async def assemble_book(tool_context: ToolContext) -> dict:
    """
    Loads each page_XX.jpeg artifact, overlays the narration text and page number,
    and saves the result as final_page_XX.jpeg.
    """
    story_output = tool_context.state.get("story_output")
    if not story_output:
        return {"status": "error", "message": "'story_output' not found in state."}

    pages = story_output.get("pages") if isinstance(story_output, dict) else story_output.pages
    if not pages:
        return {"status": "error", "message": "No pages found in story_output."}

    existing = await tool_context.list_artifacts()
    text_font = _get_font(_TEXT_SIZE)
    num_font = _get_font(_NUM_SIZE)
    results = []

    for page in pages:
        page_number = int(page["page_number"])
        src = f"page_{page_number:02d}.jpeg"
        dst = f"final_page_{page_number:02d}.jpeg"

        if dst in existing:
            results.append({"page_number": page_number, "filename": dst, "skipped": True})
            continue

        if src not in existing:
            results.append({"page_number": page_number, "error": f"{src} not found"})
            continue

        src_artifact = await tool_context.load_artifact(src)
        img = Image.open(io.BytesIO(src_artifact.inline_data.data)).convert("RGBA")
        w, h = img.size

        # Semi-transparent dark banner at the bottom
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(
            [(0, h - _BANNER_HEIGHT), (w, h)], fill=(0, 0, 0, 185)
        )
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Wrap and center the narration text inside the banner
        lines = _wrap(draw, page["text"], text_font, w - _PADDING * 2)
        line_h = _TEXT_SIZE + 6
        text_block_h = len(lines) * line_h
        usable_h = _BANNER_HEIGHT - _PAGE_NUM_AREA - _PADDING
        ty = h - _BANNER_HEIGHT + (usable_h - text_block_h) // 2

        for line in lines:
            lw = draw.textbbox((0, 0), line, font=text_font)[2]
            x = (w - lw) // 2
            # thin shadow for legibility
            draw.text((x + 1, ty + 1), line, font=text_font, fill=(0, 0, 0, 180))
            draw.text((x, ty), line, font=text_font, fill=(255, 255, 255, 255))
            ty += line_h

        # Page number centered at the very bottom
        label = str(page_number)
        lb = draw.textbbox((0, 0), label, font=num_font)
        px = (w - (lb[2] - lb[0])) // 2
        py = h - _PAGE_NUM_AREA + (_PAGE_NUM_AREA - (lb[3] - lb[1])) // 2
        draw.text((px + 1, py + 1), label, font=num_font, fill=(0, 0, 0, 180))
        draw.text((px, py), label, font=num_font, fill=(255, 255, 220, 255))

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        await tool_context.save_artifact(
            filename=dst,
            artifact=types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue())
            ),
        )
        results.append({"page_number": page_number, "filename": dst, "skipped": False})

    return {"status": "complete", "results": results}


assemble_book_tool = FunctionTool(func=assemble_book)

book_assembler_agent = Agent(
    name="book_assembler_agent",
    model=MODEL,
    description="Overlays narration text and page numbers onto each illustrated page and saves the final book",
    instruction="""
    You are a book layout artist.
    Call assemble_book once — it overlays the story text and page number onto every page image
    and saves the results as final_page_01.jpeg through final_page_05.jpeg.

    After the tool returns, report:
    - Total pages assembled
    - Filename of each final page
    - Any pages that were skipped or failed
    """,
    tools=[assemble_book_tool],
)
