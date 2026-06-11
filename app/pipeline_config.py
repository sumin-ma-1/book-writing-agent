"""Pipeline block defaults and TOC/UI merge for book generation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TEXT_AGENT_BLOCKS = [
    {"id": "outline", "label": "개요", "group": "chapter"},
    {"id": "writer", "label": "집필", "group": "chapter"},
    {"id": "reviewer", "label": "퇴고", "group": "chapter"},
    {"id": "finalizer", "label": "마무리", "group": "chapter"},
]

PICTURE_AGENT_BLOCKS = [
    {"id": "outline", "label": "개요", "group": "page"},
    {"id": "writer", "label": "집필", "group": "page"},
    {"id": "reviewer", "label": "퇴고", "group": "page"},
    {"id": "finalizer", "label": "마무리", "group": "page"},
    {"id": "illustrator", "label": "그림 프롬프트", "group": "page"},
]

PICTURE_PAGE_AGENT_IDS = frozenset({
    "outline", "writer", "reviewer", "finalizer", "illustrator", "page_writer",
})

PICTURE_AGENT_ALIASES = {"page_writer": "writer"}

PICTURE_AGENT_ORDER = ["outline", "writer", "reviewer", "finalizer", "illustrator"]


def normalize_picture_agents(agents: list[str] | None) -> list[str]:
    """Keep UI agent selection; fix empty or illustrator-only chains."""
    if not agents:
        return list(PICTURE_AGENT_ORDER)
    canonical: list[str] = []
    for raw in agents:
        agent = PICTURE_AGENT_ALIASES.get(raw, raw)
        if agent in PICTURE_AGENT_ORDER and agent not in canonical:
            canonical.append(agent)
    canonical.sort(key=PICTURE_AGENT_ORDER.index)
    if not canonical:
        return list(PICTURE_AGENT_ORDER)
    if "illustrator" in canonical and "writer" not in canonical:
        return list(PICTURE_AGENT_ORDER)
    return canonical

DEFAULT_TEXT_PIPELINE: dict[str, Any] = {
    "book_bible": True,
    "chapter_summary": True,
    "agents": ["outline", "writer", "reviewer", "finalizer"],
    "publisher": True,
}

DEFAULT_PICTURE_PIPELINE: dict[str, Any] = {
    "style_bible": True,
    "storyboard": True,
    "agents": ["outline", "writer", "reviewer", "finalizer", "illustrator"],
    "image": True,
    "publisher": True,
}

TEXT_SETUP_BLOCKS = [
    {"id": "book_bible", "label": "Book Bible"},
    {"id": "chapter_summary", "label": "장 요약"},
]

PICTURE_SETUP_BLOCKS = [
    {"id": "style_bible", "label": "Style Bible"},
    {"id": "storyboard", "label": "스토리보드"},
]

FINISH_BLOCKS = [
    {"id": "publisher", "label": "PDF 출판"},
]

PICTURE_EXTRA_BLOCKS = [
    {"id": "image", "label": "이미지 생성"},
]


def default_pipeline(book_type: str) -> dict[str, Any]:
    if book_type == "picture":
        return deepcopy(DEFAULT_PICTURE_PIPELINE)
    return deepcopy(DEFAULT_TEXT_PIPELINE)


def normalize_pipeline(book_type: str, raw: dict | None) -> dict[str, Any]:
    base = default_pipeline(book_type)
    if not raw:
        return base

    if book_type == "picture":
        for key in ("style_bible", "storyboard", "image", "publisher"):
            if key in raw:
                base[key] = bool(raw[key])
        base["agents"] = normalize_picture_agents(raw.get("agents"))
        return base

    for key in ("book_bible", "chapter_summary", "publisher"):
        if key in raw:
            base[key] = bool(raw[key])
    if agents := raw.get("agents"):
        valid = {b["id"] for b in TEXT_AGENT_BLOCKS}
        base["agents"] = [a for a in agents if a in valid] or base["agents"]
    return base


def _book_type_from_toc(toc: dict) -> str:
    if toc.get("type") == "picture_book" or toc.get("pages"):
        return "picture"
    return "text"


def pipeline_from_toc(toc: dict) -> dict[str, Any]:
    return normalize_pipeline(_book_type_from_toc(toc), toc.get("pipeline"))


def attach_default_pipeline(toc: dict, book_type: str) -> dict:
    toc = dict(toc)
    if "pipeline" not in toc:
        toc["pipeline"] = default_pipeline(book_type)
    else:
        toc["pipeline"] = normalize_pipeline(book_type, toc["pipeline"])
    return toc


def build_text_cli_args(pipeline: dict[str, Any]) -> list[str]:
    args: list[str] = []
    if not pipeline.get("book_bible", True):
        args.append("--no-bible")
    if not pipeline.get("chapter_summary", True):
        args.append("--no-chapter-summary")

    agents = list(pipeline.get("agents") or [])
    if pipeline.get("publisher", True) and "publisher" not in agents:
        agents.append("publisher")
    elif not pipeline.get("publisher", True):
        agents = [a for a in agents if a != "publisher"]

    if agents:
        args.extend(["--agents", ",".join(agents)])
    return args


def build_picture_cli_args(pipeline: dict[str, Any]) -> list[str]:
    agents: list[str] = []
    if pipeline.get("style_bible", True):
        agents.append("style_bible")
    if pipeline.get("storyboard", True):
        agents.append("storyboard")
    agents.extend(normalize_picture_agents(pipeline.get("agents")))
    if pipeline.get("image", True):
        agents.append("image")
    if pipeline.get("publisher", True):
        agents.append("publisher")

    # preserve order, dedupe
    seen: set[str] = set()
    ordered: list[str] = []
    for a in agents:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ["--agents", ",".join(ordered)] if ordered else []
