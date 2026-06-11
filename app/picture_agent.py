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

_agent_model = os.environ.get("AGENT_MODEL", "gemma4:31b")


def _make_ollama_model(name: str) -> LiteLlm:
    timeout = int(os.environ.get("LLM_TIMEOUT", "1800"))
    num_ctx = int(os.environ.get("NUM_CTX", "8192"))
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

_PAGE_CONTEXT = """
Book: "{book_title}"
Description: {book_description}
Target audience: {target_age}

{age_text_guidance}

Style Bible:
{style_bible}

Storyboard for Page {current_page_number}:
{page_storyboard}
{writing_guidelines}{language_instruction}
"""

STYLE_BIBLE_INSTRUCTION = """You are an art director for an illustrated book (text + artwork per page).

Book: "{book_title}"
Description: {book_description}
Target audience: {target_age}
Visual style: {visual_style}

Characters:
{characters_description}
{writing_guidelines}{language_instruction}

Create a detailed Style Bible. Include:
1. Overall visual style and color palette (appropriate for {target_age})
2. Character appearance sheets (consistent features, clothing, proportions)
3. Background/environment style guidelines
4. Mood and tone for illustrations
5. Things to avoid (e.g. text in images, off-model characters; content inappropriate for {target_age})

Write in clear Markdown. This guide ensures visual consistency across all pages."""

STORYBOARD_INSTRUCTION = """You are a storyboard artist for an illustrated book.

Book: "{book_title}"
Description: {book_description}
Target audience: {target_age}

Style Bible:
{style_bible}

Pages to illustrate:
{pages_description}
{writing_guidelines}{language_instruction}

Create a detailed storyboard as a JSON object with this structure:
{{
  "pages": [
    {{
      "number": 1,
      "scene_description": "Detailed visual description of what appears in the illustration",
      "character_actions": "What characters are doing",
      "composition": "Camera angle, foreground/background layout",
      "mood": "Emotional tone of the scene"
    }}
  ]
}}

Output ONLY valid JSON. Every page from the TOC must have an entry."""

PAGE_OUTLINE_INSTRUCTION = (
    """You are a page outline specialist for an illustrated book.
"""
    + _PAGE_CONTEXT
    + """
Create a concise outline for the TEXT on this ONE page (not the illustration). Include:
1. The narrative beat or idea this page must convey
2. 2-4 bullet points for what the prose should cover
3. Tone and pacing notes for {target_age}
4. What to avoid (describing the artwork; meta commentary)

Output ONLY the outline in Markdown bullets."""
)

PAGE_WRITER_INSTRUCTION = (
    """You are an author writing one page of an illustrated book.
"""
    + _PAGE_CONTEXT
    + """
Page outline:
{page_outline}

Write the FULL page text following the outline (if the outline is empty, use the storyboard). Requirements:
- Follow the age guidance for length and tone
- The illustration accompanies the text; do NOT describe what the art shows
- Write narrative prose that stands on its own
- Do NOT include page numbers or meta-commentary

Output ONLY the page text."""
)

PAGE_REVIEWER_INSTRUCTION = (
    """You are a professional editor reviewing one page of an illustrated book.
"""
    + _PAGE_CONTEXT
    + """
Page outline:
{page_outline}

Draft to review:
{page_text}

Review and output an IMPROVED version of the entire page text. Focus on:
1. Clarity and flow
2. Fit with the outline and storyboard
3. Age-appropriate voice for {target_age}
4. Engagement and emotional impact

Output ONLY the complete revised page text — no review notes."""
)

PAGE_FINALIZER_INSTRUCTION = (
    """You are an editor performing the final polish on one page of an illustrated book.
"""
    + _PAGE_CONTEXT
    + """
Reviewed page text:
{page_text}

Produce the FINAL page text. Ensure:
1. Clean, publication-ready prose
2. Correct length and tone for {target_age}
3. No meta-commentary or placeholders remain
4. The text does not describe the illustration

Output ONLY the final page text."""
)

ILLUSTRATOR_PROMPT_INSTRUCTION = """You are an illustration prompt engineer for an illustrated book.

Book: "{book_title}"
Target audience: {target_age}
Visual style: {visual_style}

{age_illustration_guidance}

Style Bible:
{style_bible}

Page {current_page_number} text: {page_text}

Storyboard for this page:
{page_storyboard}

Write a detailed ENGLISH image generation prompt for this page's illustration. Requirements:
- Describe the scene, characters, actions, and environment vividly
- Reflect the visual style: {visual_style}
- Follow the illustration guidance for {target_age}
- Include character details from the style bible for consistency
- End with: "no text, no words, no letters in the image"
- Keep under 200 words
- Output ONLY the prompt, nothing else"""

style_bible_agent = Agent(
    name="style_bible_agent",
    model=_model,
    instruction=STYLE_BIBLE_INSTRUCTION,
    output_key="style_bible",
)

storyboard_agent = Agent(
    name="storyboard_agent",
    model=_model,
    instruction=STORYBOARD_INSTRUCTION,
    output_key="storyboard",
)

page_outline_agent = Agent(
    name="page_outline_agent",
    model=_model,
    instruction=PAGE_OUTLINE_INSTRUCTION,
    output_key="page_outline",
)

page_writer_agent = Agent(
    name="page_writer_agent",
    model=_model,
    instruction=PAGE_WRITER_INSTRUCTION,
    output_key="page_text",
)

page_reviewer_agent = Agent(
    name="page_reviewer_agent",
    model=_model,
    instruction=PAGE_REVIEWER_INSTRUCTION,
    output_key="page_text",
)

page_finalizer_agent = Agent(
    name="page_finalizer_agent",
    model=_model,
    instruction=PAGE_FINALIZER_INSTRUCTION,
    output_key="page_text",
)

illustrator_prompt_agent = Agent(
    name="illustrator_prompt_agent",
    model=_model,
    instruction=ILLUSTRATOR_PROMPT_INSTRUCTION,
    output_key="image_prompt",
)

setup_pipeline = SequentialAgent(
    name="setup_pipeline",
    sub_agents=[style_bible_agent, storyboard_agent],
)

_agent_registry = {
    "style_bible": style_bible_agent,
    "storyboard": storyboard_agent,
    "outline": page_outline_agent,
    "writer": page_writer_agent,
    "reviewer": page_reviewer_agent,
    "finalizer": page_finalizer_agent,
    "illustrator": illustrator_prompt_agent,
    "page_writer": page_writer_agent,
}

_selected = os.environ.get(
    "PIPELINE_AGENTS", "outline,writer,reviewer,finalizer,illustrator"
).split(",")
_page_agents = [_agent_registry[n.strip()] for n in _selected if n.strip() in _agent_registry]

page_pipeline = SequentialAgent(
    name="page_pipeline",
    sub_agents=_page_agents,
)

app = App(root_agent=page_pipeline, name="picture_book_app")
