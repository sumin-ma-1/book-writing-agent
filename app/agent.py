from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

from google.adk.agents import Agent, SequentialAgent
from google.adk.apps import App
from google.adk.models import LiteLlm

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "local")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")

# --- Model Configuration ---

_agent_model = os.environ.get("AGENT_MODEL", "gemma4:31b")
_current_model_name = _agent_model


def _make_ollama_model(name: str) -> LiteLlm:
    timeout = int(os.environ.get("LLM_TIMEOUT", "1800"))
    num_ctx = int(os.environ.get("NUM_CTX", "32768"))
    repeat_penalty = float(os.environ.get("REPEAT_PENALTY", "1.2"))
    no_think = bool(os.environ.get("DISABLE_THINKING"))
    return LiteLlm(
        model=f"ollama_chat/{name}",
        api_base="http://localhost:11434",
        think=not no_think,
        num_ctx=num_ctx,
        repeat_penalty=repeat_penalty,
        temperature=0.7,
        timeout=timeout,
    )


_model = _make_ollama_model(_agent_model)

# --- Sub-Agent Instructions ---

_CONTEXT_BLOCK = """
Book Bible (tone, audience, terminology, consistency — follow strictly):
{book_bible}

Previous chapters (continuity — do not contradict; build on this):
{previous_chapters_summary}
"""

BOOK_BIBLE_INSTRUCTION = """You are a senior book editor creating a Book Bible.

Book: "{book_title}"
Description: {book_description}

Planned chapters:
{chapters_outline}
{writing_guidelines}{language_instruction}

Write a Book Bible in Markdown that will guide every chapter writer. Include:
1. Target audience and reading level
2. Tone and voice (register, person, formality)
3. Terminology glossary (key terms and preferred usage)
4. Consistency rules (always do / never do)
5. How chapters connect as a narrative arc
6. Brief style notes with examples if helpful

Be specific and actionable. Output ONLY the Book Bible in Markdown."""

CHAPTER_SUMMARY_INSTRUCTION = """You summarize a completed book chapter so later chapters stay consistent.

Book: "{book_title}"
Chapter {chapter_number}: {chapter_title}

Book Bible (use the same terminology):
{book_bible}

Full chapter text:
{chapter_text}
{language_instruction}

Write a concise summary (200-400 words) for writers of later chapters. Include:
- Main points, arguments, and conclusions
- Key terms, names, and definitions introduced
- Facts or claims that later chapters must not contradict
- Open threads and how this chapter leads into what follows

Output ONLY the summary. No headings, labels, or meta-commentary."""

OUTLINE_INSTRUCTION = """You are a book chapter outline specialist.

You are writing an outline for a chapter of the book "{book_title}".
Book description: {book_description}
""" + _CONTEXT_BLOCK + """
Current chapter:
- Chapter {current_chapter_number}: {current_chapter_title}
- Description: {current_chapter_description}
{writing_guidelines}{language_instruction}

Create a detailed, hierarchical outline for this chapter. Include:
1. A compelling opening hook or introduction concept
2. Main sections (3-6 sections) with clear headings
3. Key points and sub-topics under each section (2-4 per section)
4. Transition notes between sections
5. A conclusion or chapter summary concept
6. Estimated word count per section (target total: {target_word_count} words)

Write the outline in Markdown with clear hierarchy using headings and bullet points.
Be specific and substantive — this outline guides the writer agent."""

WRITER_INSTRUCTION = """You are an expert book writer.

You are writing a chapter for the book "{book_title}".
Chapter {current_chapter_number}: {current_chapter_title}
""" + _CONTEXT_BLOCK + """
Use the following outline as your guide:

{chapter_outline}
{writing_guidelines}{language_instruction}

Write the FULL chapter as polished, publication-ready prose. Requirements:
- Follow the outline structure exactly
- Write substantive prose paragraphs, NOT bullet points or lists (unless they serve the content)
- Target {target_word_count} words total
- Use a clear, engaging, and authoritative tone
- Start with the chapter title as a level-1 heading: # Chapter {current_chapter_number}: {current_chapter_title}
- Use level-2 headings (##) for main sections
- Use level-3 headings (###) for sub-sections where appropriate
- Include smooth transitions between sections
- End with a strong conclusion that ties back to the chapter's theme

Output ONLY the chapter content in Markdown. No meta-commentary."""

REVIEWER_INSTRUCTION = """You are a professional book editor and reviewer.

You are reviewing a chapter for the book "{book_title}".
Chapter {current_chapter_number}: {current_chapter_title}
""" + _CONTEXT_BLOCK + """
Original outline:
{chapter_outline}

Draft to review:
{chapter_draft}
{writing_guidelines}{language_instruction}

Review the draft and produce an IMPROVED version of the entire chapter. Focus on:
1. Clarity and readability — simplify convoluted sentences
2. Flow — ensure smooth transitions between sections and paragraphs
3. Completeness — fill any gaps where the outline was not fully addressed
4. Consistency — uniform terminology, tone, and style throughout
5. Engagement — strengthen the opening hook and conclusion
6. Accuracy — flag and fix any factual inconsistencies

Output the COMPLETE revised chapter in Markdown. Do NOT output review notes or commentary —
output only the improved chapter text, ready for the finalizer."""

FINALIZER_INSTRUCTION = """You are a book production editor performing the final polish.

You are finalizing a chapter for the book "{book_title}".
Chapter {current_chapter_number}: {current_chapter_title}
""" + _CONTEXT_BLOCK + """
Reviewed draft:
{chapter_review}
{language_instruction}

Produce the FINAL version of this chapter. Ensure:
1. The chapter starts with: # Chapter {current_chapter_number}: {current_chapter_title}
2. Consistent heading hierarchy (## for sections, ### for sub-sections)
3. No orphaned headings (every heading has content below it)
4. Clean Markdown formatting (proper spacing, no double blank lines)
5. Professional tone maintained throughout
6. No meta-commentary, review notes, or TODO markers remain
7. The chapter reads as a cohesive, standalone piece

Output ONLY the final chapter content in clean Markdown."""

# --- Agent Definitions ---

book_bible_agent = Agent(
    name="book_bible_agent",
    model=_model,
    instruction=BOOK_BIBLE_INSTRUCTION,
    output_key="book_bible",
)

chapter_summary_agent = Agent(
    name="chapter_summary_agent",
    model=_model,
    instruction=CHAPTER_SUMMARY_INSTRUCTION,
    output_key="chapter_summary",
)

outline_agent = Agent(
    name="outline_agent",
    model=_model,
    instruction=OUTLINE_INSTRUCTION,
    output_key="chapter_outline",
)

writer_agent = Agent(
    name="writer_agent",
    model=_model,
    instruction=WRITER_INSTRUCTION,
    output_key="chapter_draft",
)

reviewer_agent = Agent(
    name="reviewer_agent",
    model=_model,
    instruction=REVIEWER_INSTRUCTION,
    output_key="chapter_review",
)

finalizer_agent = Agent(
    name="finalizer_agent",
    model=_model,
    instruction=FINALIZER_INSTRUCTION,
    output_key="chapter_final",
)

_agent_registry = {
    "outline": outline_agent,
    "writer": writer_agent,
    "reviewer": reviewer_agent,
    "finalizer": finalizer_agent,
}

_selected = os.environ.get("PIPELINE_AGENTS", "outline,writer,reviewer,finalizer").split(",")
_pipeline_agents = [_agent_registry[n.strip()] for n in _selected if n.strip() in _agent_registry]

chapter_pipeline = SequentialAgent(
    name="chapter_pipeline",
    sub_agents=_pipeline_agents,
)

# --- Root Agent ---

ROOT_INSTRUCTION = """You are a book-writing orchestrator.

You help users write books by processing their table of contents chapter by chapter.
For interactive use, guide the user through providing their book's table of contents
and any preferences about style, tone, or target audience.

For automated overnight runs, the chapter_pipeline sub-agent handles each chapter
through a 4-phase process: outline → write → review → finalize.

The book title is: {book_title}
Total chapters: {total_chapters}
"""

root_agent = Agent(
    name="root_agent",
    model=_model,
    instruction=ROOT_INSTRUCTION,
    sub_agents=[chapter_pipeline],
)

app = App(root_agent=root_agent, name="app")
