"""Import and export TOC (목차) JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.pipeline_config import attach_default_pipeline, normalize_pipeline
from app.toc_paths import ensure_toc_dir

_SAFE_NAME = re.compile(r"^[\w\-. ]+\.json$", re.IGNORECASE)


def _resolve_toc_path(project_root: Path, path: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = (project_root / target).resolve()
    else:
        target = target.resolve()
    root = project_root.resolve()
    if not str(target).startswith(str(root)):
        raise ValueError("Invalid path")
    if not target.is_file():
        raise FileNotFoundError("목차 파일을 찾을 수 없습니다")
    return target


def export_toc_path(project_root: Path, path: str) -> Path:
    return _resolve_toc_path(project_root, path)


def import_toc_file(project_root: Path, content: bytes, filename: str) -> dict:
    name = Path(filename or "imported-toc.json").name
    if not _SAFE_NAME.match(name):
        raise ValueError("파일명은 .json 이어야 합니다")

    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"JSON 형식이 올바르지 않습니다: {e}") from e

    if not isinstance(data, dict) or not data.get("title"):
        raise ValueError("title 필드가 필요합니다")

    book_type = data.get("type", "text_book")
    if book_type == "picture_book" or data.get("pages"):
        if not data.get("pages"):
            raise ValueError("그림책 목차에는 pages 배열이 필요합니다")
        book_type = "picture_book"
        count = len(data["pages"])
        unit = "pages"
    elif data.get("chapters"):
        book_type = "text_book"
        count = len(data["chapters"])
        unit = "chapters"
    else:
        raise ValueError("chapters 또는 pages 배열이 필요합니다")

    dest = project_root / name
    if dest.exists():
        stem = dest.stem
        n = 2
        while (project_root / f"{stem}-{n}.json").exists():
            n += 1
        dest = project_root / f"{stem}-{n}.json"

    bt = "picture" if book_type == "picture_book" else "text"
    data = attach_default_pipeline(data, bt)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        rel_path = str(dest.resolve().relative_to(project_root.resolve()))
    except ValueError:
        rel_path = str(dest)

    return {
        "path": rel_path,
        "filename": dest.name,
        "title": data["title"],
        "type": book_type,
        "count": count,
        "unit": unit,
        "pipeline": data["pipeline"],
    }


def repair_picture_toc_pipeline(data: dict) -> tuple[dict, bool]:
    """Normalize picture-book pipeline; return (data, changed)."""
    if data.get("type") != "picture_book" and not data.get("pages"):
        return data, False
    before = json.dumps(data.get("pipeline"), sort_keys=True)
    fixed = attach_default_pipeline(dict(data), "picture")
    after = json.dumps(fixed.get("pipeline"), sort_keys=True)
    return fixed, before != after


def save_toc_pipeline(project_root: Path, path: str, pipeline: dict) -> dict:
    target = _resolve_toc_path(project_root, path)
    data = json.loads(target.read_text(encoding="utf-8"))
    book_type = _book_type_from_toc(data)
    data["pipeline"] = normalize_pipeline(book_type, pipeline)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(target), "pipeline": data["pipeline"]}


def repair_toc_file_pipeline(project_root: Path, path: Path) -> bool:
    """Rewrite TOC on disk when picture pipeline agents are incomplete."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    fixed, changed = repair_picture_toc_pipeline(data)
    if not changed:
        return False
    path.write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def _book_type_from_toc(toc: dict) -> str:
    if toc.get("type") == "picture_book" or toc.get("pages"):
        return "picture"
    return "text"


def _normalize_text_items(chapters: list) -> list[dict]:
    if not chapters:
        raise ValueError("chapters 배열이 필요합니다")
    normalized = []
    for i, ch in enumerate(chapters, start=1):
        title = str(ch.get("title", "")).strip()
        if not title:
            raise ValueError(f"{i}번 장 제목이 비어 있습니다")
        normalized.append({
            "number": i,
            "title": title,
            "description": str(ch.get("description", "")).strip(),
        })
    return normalized


def _normalize_picture_items(pages: list) -> list[dict]:
    if not pages:
        raise ValueError("pages 배열이 필요합니다")
    normalized = []
    for i, pg in enumerate(pages, start=1):
        scene = str(pg.get("scene", "")).strip()
        if not scene:
            raise ValueError(f"{i}번 페이지 장면 설명이 비어 있습니다")
        normalized.append({
            "number": i,
            "scene": scene,
            "mood": str(pg.get("mood", "")).strip() or "warm",
        })
    return normalized


def load_toc_file(project_root: Path, path: str) -> dict:
    target = _resolve_toc_path(project_root, path)
    data = json.loads(target.read_text(encoding="utf-8"))
    book_type = _book_type_from_toc(data)
    data = attach_default_pipeline(data, book_type)
    unit = "pages" if book_type == "picture" else "chapters"
    count = len(data.get("pages") or data.get("chapters") or [])
    return {
        "path": str(target),
        "filename": target.name,
        "type": "picture_book" if book_type == "picture" else "text_book",
        "count": count,
        "unit": unit,
        "toc": data,
    }


def save_toc_content(project_root: Path, path: str, toc: dict) -> dict:
    target = _resolve_toc_path(project_root, path)
    existing = json.loads(target.read_text(encoding="utf-8"))
    book_type = _book_type_from_toc(toc) if toc.get("pages") or toc.get("type") == "picture_book" else _book_type_from_toc(existing)

    title = str(toc.get("title", "")).strip()
    if not title:
        raise ValueError("title이 필요합니다")

    pipeline = toc.get("pipeline") or existing.get("pipeline")

    if book_type == "picture":
        pages = _normalize_picture_items(toc.get("pages") or [])
        characters = toc.get("characters") or existing.get("characters") or [
            {"name": "주인공", "description": "이야기의 주인공"},
        ]
        data = {
            "type": "picture_book",
            "title": title,
            "description": str(toc.get("description", "")).strip(),
            "language": str(toc.get("language") or existing.get("language") or "ko").strip(),
            "target_age": str(toc.get("target_age") or existing.get("target_age") or "3-5").strip(),
            "style": str(toc.get("style") or existing.get("style") or "").strip()
            or "watercolor children's book illustration, soft pastel colors",
            "writing_guidelines": toc.get("writing_guidelines") or existing.get("writing_guidelines") or [],
            "characters": characters,
            "pages": pages,
        }
        unit = "pages"
    else:
        chapters = _normalize_text_items(toc.get("chapters") or [])
        data = {
            "title": title,
            "description": str(toc.get("description", "")).strip(),
            "chapters": chapters,
        }
        unit = "chapters"

    if pipeline:
        data["pipeline"] = normalize_pipeline(book_type, pipeline)
    else:
        data = attach_default_pipeline(data, book_type)

    # Persist normalized pipeline (e.g. illustrator-only → full page text chain).
    if book_type == "picture" and data.get("pipeline"):
        data["pipeline"] = normalize_pipeline(book_type, data["pipeline"])

    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    count = len(data.get("pages") or data.get("chapters") or [])
    return {
        "path": str(target),
        "filename": target.name,
        "title": data["title"],
        "type": "picture_book" if book_type == "picture" else "text_book",
        "count": count,
        "unit": unit,
        "pipeline": data.get("pipeline"),
        "toc": data,
    }
