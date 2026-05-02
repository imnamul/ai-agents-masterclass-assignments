# Story Book Maker

A multi-agent pipeline that generates an illustrated children's picture book from a user-provided theme.

## Project Structure

```
story_book_maker/
├── .env                          # OPENAI_API_KEY
├── agent.py                      # root_agent (LlmAgent) + pipeline_agent (SequentialAgent)
├── callbacks.py                  # after_agent progress callbacks
├── story_writer/
│   └── agent.py                  # Writes 5-page story → state["story_output"]
├── illustrator/
│   └── agent.py                  # ParallelAgent with 5 per-page illustrator agents
└── book_assembler/
    └── agent.py                  # Overlays text + page number onto each image
```

## Agent Flow

```
[User]
  ↓
[story_book_maker_agent]  ← root_agent (LlmAgent)
  - Greets user, asks for theme
  - Confirms theme before starting
  ↓
[story_book_maker_pipeline]  (SequentialAgent)
  ↓
[story_writer_agent]  (LlmAgent)
  - Writes a 5-page story
  - Saves structured output to state["story_output"]
  ↓
[illustrator_agent]  (ParallelAgent)
  - Runs 5 page_illustrator agents simultaneously
  - Each generates one image via OpenAI gpt-image-1
  - Saves page_01.jpeg ~ page_05.jpeg as artifacts
  ↓
[book_assembler_agent]  (LlmAgent)
  - Reads page images + story text from state
  - Overlays narration text and page number onto each image
  - Saves final_page_01.jpeg ~ final_page_05.jpeg as artifacts
  ↓
[story_book_maker_agent]
  - Presents the completed picture book to the user
```

## Setup

```bash
uv sync
```

Set `OPENAI_API_KEY` in `story_book_maker/.env`.

## Run

```bash
uv run adk web
```

Open http://127.0.0.1:8000 and select `story_book_maker`.
