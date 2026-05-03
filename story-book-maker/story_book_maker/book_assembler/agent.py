import io
from PIL import Image, ImageDraw, ImageFont
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from ..callbacks import on_assembly_done

MODEL = LiteLlm(model="openai/gpt-4o")

_BANNER_HEIGHT = 240   # height of the bottom text banner on story pages
_PADDING = 16
_PAGE_NUM_AREA = 50   # vertical strip reserved for the page number (moves it up from edge)
_TEXT_SIZE = 32       # story narration font size
_NUM_SIZE = 22        # page number font size
_TITLE_BANNER_HEIGHT = 240  # height of the top banner on the title/cover page
_TITLE_SIZE = 54      # book title font size


def _normalize(obj) -> dict:
    """Pydantic 모델이든 dict든 항상 순수 dict로 반환."""
    return obj if isinstance(obj, dict) else obj.model_dump()


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
    raw = tool_context.state.get("story_output")
    if not raw:
        return {"status": "error", "message": "'story_output' not found in state."}

    # ── 정규화: 이후 코드는 항상 dict 접근 ──────────────────────────────────
    story = _normalize(raw)

    pages = story.get("pages", [])
    if not pages:
        return {"status": "error", "message": "No pages found in story_output."}

    existing = await tool_context.list_artifacts()   # 소스 이미지 존재 여부 확인용
    run_artifacts = tool_context.state.get("current_run_artifacts", [])
    text_font  = _get_font(_TEXT_SIZE)
    num_font   = _get_font(_NUM_SIZE)
    title_font = _get_font(_TITLE_SIZE)
    results = []

    # ── 표지 (page_00) ────────────────────────────────────────────────────
    title = story.get("title", "")
    src_title = "page_00.jpeg"
    dst_title = "final_page_00.jpeg"

    if dst_title in run_artifacts:
        results.append({"page_number": 0, "filename": dst_title, "skipped": True})
    elif src_title not in existing:
        results.append({"page_number": 0, "error": f"{src_title} not found"})
    else:
        src_artifact = await tool_context.load_artifact(src_title)
        img = Image.open(io.BytesIO(src_artifact.inline_data.data)).convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(
            [(0, 0), (w, _TITLE_BANNER_HEIGHT)], fill=(0, 0, 0, 185)
        )
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        if title:
            lines = _wrap(draw, title, title_font, w - _PADDING * 2)
            line_h = _TITLE_SIZE + 8
            text_block_h = len(lines) * line_h
            ty = (_TITLE_BANNER_HEIGHT - text_block_h) // 2
            for line in lines:
                lw = draw.textbbox((0, 0), line, font=title_font)[2]
                x = (w - lw) // 2
                draw.text((x + 2, ty + 2), line, font=title_font, fill=(0, 0, 0, 180))
                draw.text((x, ty),         line, font=title_font, fill=(255, 255, 255, 255))
                ty += line_h

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        await tool_context.save_artifact(
            filename=dst_title,
            artifact=types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=buf.getvalue())
            ),
        )
        run_artifacts = run_artifacts + [dst_title]
        tool_context.state["current_run_artifacts"] = run_artifacts
        results.append({"page_number": 0, "filename": dst_title, "skipped": False})

    # ── 스토리 페이지 (page_01 ~ page_05) ────────────────────────────────
    for page in pages:
        page_number = int(page["page_number"])
        src = f"page_{page_number:02d}.jpeg"
        dst = f"final_page_{page_number:02d}.jpeg"

        if dst in run_artifacts:
            results.append({"page_number": page_number, "filename": dst, "skipped": True})
            continue

        if src not in existing:
            results.append({"page_number": page_number, "error": f"{src} not found"})
            continue

        src_artifact = await tool_context.load_artifact(src)
        img = Image.open(io.BytesIO(src_artifact.inline_data.data)).convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(
            [(0, h - _BANNER_HEIGHT), (w, h)], fill=(0, 0, 0, 185)
        )
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        lines = _wrap(draw, page["text"], text_font, w - _PADDING * 2)
        line_h = _TEXT_SIZE + 6
        text_block_h = len(lines) * line_h
        usable_h = _BANNER_HEIGHT - _PAGE_NUM_AREA - _PADDING
        ty = h - _BANNER_HEIGHT + (usable_h - text_block_h) // 2

        for line in lines:
            lw = draw.textbbox((0, 0), line, font=text_font)[2]
            x = (w - lw) // 2
            draw.text((x + 1, ty + 1), line, font=text_font, fill=(0, 0, 0, 180))
            draw.text((x, ty), line, font=text_font, fill=(255, 255, 255, 255))
            ty += line_h

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
        run_artifacts = run_artifacts + [dst]
        tool_context.state["current_run_artifacts"] = run_artifacts
        results.append({"page_number": page_number, "filename": dst, "skipped": False})

    return {"status": "complete", "results": results}


assemble_book_tool = FunctionTool(func=assemble_book)

book_assembler_agent = Agent(
    name="book_assembler_agent",
    model=MODEL,
    description="Overlays narration text and page numbers onto each illustrated page and saves the final book",
    instruction="""
    You are a book layout artist.
    Call assemble_book once — it overlays the book title onto the cover page and the story text
    and page number onto every story page, saving the results as
    final_page_00.jpeg (cover) through final_page_05.jpeg.

    After the tool returns, report:
    - Total pages assembled
    - Filename of each final page
    - Any pages that were skipped or failed
    """,
    tools=[assemble_book_tool],
    after_agent_callback=on_assembly_done,
)
