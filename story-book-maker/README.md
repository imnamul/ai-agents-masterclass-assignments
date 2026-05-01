story_book_maker/
├── .env
├── agent.py                    # root_agent = SequentialAgent
├── callbacks.py                # Progress callbacks
├── story_writer/
│   ├── __init__.py
│   └── agent.py
├── illustrator/
│   ├── __init__.py
│   └── agent.py                # ParallelAgent with 5 page agents
└── book_assembler/
    ├── __init__.py
    └── agent.py                # Overlays text + page number onto each image



[User input]
      ↓
[SequentialAgent]  ← root_agent
      ↓
[Story Writer Agent]
 - Writes 5-page story
 - Saves to state["story_output"]
      ↓
[ParallelAgent]
 - Generates 5 images simultaneously
 - Saves as artifacts (page_01.jpeg ~ page_05.jpeg)
      ↓
[Book Assembler Agent]          ← NEW
 - Reads artifacts + state
 - Overlays narration text on each image
 - Adds page number to bottom center
 - Saves final pages as new artifacts (final_page_01.jpeg ~ final_page_05.jpeg)
      ↓
[Final output: completed picture book]