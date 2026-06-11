"""Book Bible and previous-chapter context for text book pipeline."""

from __future__ import annotations

from pathlib import Path

BOOK_BIBLE_FILENAME = "book-bible.md"
_MAX_TOTAL_SUMMARY = 6000
_MAX_CHAPTER_TEXT_FOR_AGENT = 28000


def format_chapters_outline(toc: dict) -> str:
    lines = []
    for ch in toc.get("chapters", []):
        desc = ch.get("description", "")
        lines.append(
            f"- Chapter {ch['number']}: {ch['title']}"
            + (f" — {desc}" if desc else "")
        )
    return "\n".join(lines) or "(no chapters)"


def save_book_bible(content: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / BOOK_BIBLE_FILENAME
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def load_book_bible(output_dir: str) -> str:
    path = Path(output_dir) / BOOK_BIBLE_FILENAME
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def strip_yaml_front_matter(content: str) -> str:
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3 :].lstrip()
    return content


def chapter_summary_path(output_dir: str, chapter_number: int) -> Path:
    return Path(output_dir) / f"chapter-{chapter_number:02d}-summary.md"


def save_chapter_summary(chapter_number: int, summary: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = chapter_summary_path(output_dir, chapter_number)
    path.write_text(summary.strip() + "\n", encoding="utf-8")
    return path


def load_chapter_summary(output_dir: str, chapter_number: int) -> str:
    path = chapter_summary_path(output_dir, chapter_number)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def find_chapter_file(output_dir: str, chapter_number: int) -> Path | None:
    out = Path(output_dir)
    matches = sorted(out.glob(f"chapter-{chapter_number:02d}-*.md"))
    for path in matches:
        if path.name.endswith("-summary.md"):
            continue
        return path
    return None


def read_chapter_body(output_dir: str, chapter_number: int) -> str:
    path = find_chapter_file(output_dir, chapter_number)
    if not path:
        return ""
    return strip_yaml_front_matter(path.read_text(encoding="utf-8"))


def truncate_for_summary_agent(text: str, max_chars: int = _MAX_CHAPTER_TEXT_FOR_AGENT) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... middle of chapter omitted for length ...]\n\n"
        + text[-half:]
    )


def build_previous_chapters_summary(
    output_dir: str,
    before_chapter: int,
    toc: dict,
) -> str:
    """Load agent-written summaries for chapters before before_chapter."""
    if before_chapter <= 1:
        return "(First chapter — no prior chapters.)"

    parts: list[str] = []
    missing: list[int] = []

    for ch in sorted(toc.get("chapters", []), key=lambda c: c["number"]):
        num = ch["number"]
        if num >= before_chapter:
            break
        summary = load_chapter_summary(output_dir, num)
        if summary:
            parts.append(f"### Chapter {num}: {ch['title']}\n{summary}")
        elif find_chapter_file(output_dir, num):
            missing.append(num)

    if not parts and not missing:
        return "(No prior chapter files on disk yet.)"

    if missing:
        parts.append(
            f"(Summaries pending for chapter(s): {', '.join(str(n) for n in missing)}.)"
        )

    text = "\n\n".join(parts)
    if len(text) > _MAX_TOTAL_SUMMARY:
        text = text[: _MAX_TOTAL_SUMMARY - 20].rstrip() + "\n\n...(truncated)"
    return text


def list_chapters_needing_summary(
    output_dir: str,
    before_chapter: int,
    toc: dict,
) -> list[dict]:
    """Chapters with a .md file but no -summary.md yet."""
    needed = []
    for ch in sorted(toc.get("chapters", []), key=lambda c: c["number"]):
        num = ch["number"]
        if num >= before_chapter:
            break
        if find_chapter_file(output_dir, num) and not load_chapter_summary(output_dir, num):
            needed.append(ch)
    return needed
