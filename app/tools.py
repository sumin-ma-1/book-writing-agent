from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from slugify import slugify

logger = logging.getLogger("book-writer")


def parse_toc_json(text: str) -> dict:
    return json.loads(text)


def parse_toc_yaml(text: str) -> dict:
    return yaml.safe_load(text)


def parse_toc_text(text: str) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    title = "Untitled Book"
    description = ""
    chapters = []

    if lines and lines[0].startswith("#"):
        title = lines.pop(0).lstrip("# ").strip()

    if lines and not re.match(r"^\d+[\.\)]\s", lines[0]):
        description = lines.pop(0).strip()

    for line in lines:
        m = re.match(r"^(\d+)[\.\)]\s+(.+?)(?:\s*[-–—]\s*(.+))?$", line)
        if m:
            chapters.append({
                "number": int(m.group(1)),
                "title": m.group(2).strip(),
                "description": m.group(3).strip() if m.group(3) else "",
            })

    return {"title": title, "description": description, "chapters": chapters}


def parse_toc(file_path: str) -> dict:
    """Parse a table of contents file (JSON, YAML, or plain text)."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    if path.suffix in (".json",):
        toc = parse_toc_json(content)
    elif path.suffix in (".yaml", ".yml"):
        toc = parse_toc_yaml(content)
    else:
        toc = parse_toc_text(content)

    for i, ch in enumerate(toc.get("chapters", [])):
        if "number" not in ch:
            ch["number"] = i + 1
        if "description" not in ch:
            ch["description"] = ""

    return toc


def save_chapter_to_disk(
    chapter_number: int,
    title: str,
    content: str,
    output_dir: str,
) -> dict:
    """Write a chapter Markdown file to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for old in out.glob(f"chapter-{chapter_number:02d}-*.md"):
        old.unlink()

    slug = slugify(title, max_length=60)
    filename = f"chapter-{chapter_number:02d}-{slug}.md"
    filepath = out / filename

    now = datetime.now(timezone.utc).isoformat()
    front_matter = (
        f"---\n"
        f"chapter: {chapter_number}\n"
        f"title: \"{title}\"\n"
        f"generated_at: \"{now}\"\n"
        f"---\n\n"
    )

    full_content = front_matter + content
    filepath.write_text(full_content, encoding="utf-8")

    word_count = len(content.split())
    return {
        "success": True,
        "file_path": str(filepath),
        "filename": filename,
        "word_count": word_count,
    }


def git_commit_and_push_sync(
    chapter_number: int,
    title: str,
    output_dir: str,
    branch: str = "main",
    message: str | None = None,
) -> dict:
    """Git add, commit, and push for a completed chapter."""
    cwd = Path(output_dir)

    if not (cwd / ".git").exists():
        parent = cwd.parent
        while parent != parent.parent:
            if (parent / ".git").exists():
                cwd = parent
                break
            parent = parent.parent
        else:
            return {"success": False, "message": "Not a git repository"}

    try:
        subprocess.run(
            ["git", "add", "."],
            cwd=str(cwd),
            check=True,
            timeout=60,
            capture_output=True,
        )

        msg = message or f"Add Chapter {chapter_number}: {title}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(cwd),
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )

        commit_hash = ""
        for line in result.stdout.splitlines():
            if line.strip().startswith("["):
                parts = line.split()
                for p in parts:
                    if len(p) >= 7 and p.replace("]", "").isalnum():
                        commit_hash = p.replace("]", "")
                        break

        max_push_attempts = 3
        for attempt in range(1, max_push_attempts + 1):
            pull_result = subprocess.run(
                ["git", "pull", "--rebase", "origin", branch],
                cwd=str(cwd),
                timeout=120,
                capture_output=True,
                text=True,
            )

            if pull_result.returncode != 0:
                return {
                    "success": True,
                    "commit_hash": commit_hash,
                    "pushed": False,
                    "message": f"Committed but pull --rebase failed: {pull_result.stderr.strip()}",
                }

            push_result = subprocess.run(
                ["git", "push", "origin", branch],
                cwd=str(cwd),
                timeout=120,
                capture_output=True,
                text=True,
            )

            if push_result.returncode == 0:
                return {
                    "success": True,
                    "commit_hash": commit_hash,
                    "pushed": True,
                    "message": f"Committed and pushed Chapter {chapter_number}",
                }

            if attempt < max_push_attempts:
                logger.warning(
                    "Push attempt %d/%d failed, retrying: %s",
                    attempt, max_push_attempts, push_result.stderr.strip(),
                )

        return {
            "success": True,
            "commit_hash": commit_hash,
            "pushed": False,
            "message": f"Committed but push failed after {max_push_attempts} attempts: {push_result.stderr.strip()}",
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Git operation timed out"}
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
        return {"success": False, "message": f"Git error: {stderr.strip()}"}


def load_progress(output_dir: str) -> dict:
    """Load progress from .progress.json."""
    path = Path(output_dir) / ".progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "failed": {}, "in_progress": None}


def save_progress(output_dir: str, progress: dict) -> None:
    """Save progress to .progress.json."""
    path = Path(output_dir) / ".progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


_FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n*", re.DOTALL)

_PDF_CSS = """\
@page {
    size: A4;
    margin: 2.5cm 2cm;
    @bottom-center { content: counter(page); }
}
body {
    font-family: "DejaVu Serif", Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
}
h1 { font-size: 22pt; margin-top: 0; }
h2 { font-size: 16pt; margin-top: 1.5em; }
h3 { font-size: 13pt; margin-top: 1.2em; }
.title-page {
    text-align: center;
    padding-top: 30%;
    page-break-after: always;
}
.title-page h1 { font-size: 32pt; margin-bottom: 0.5em; }
.title-page .description {
    font-size: 14pt;
    color: #555;
    font-style: italic;
    max-width: 80%;
    margin: 1em auto;
}
.title-page .author {
    font-size: 13pt;
    color: #444;
    margin-top: 2em;
}
.title-page .orchestrator {
    font-size: 11pt;
    color: #777;
    margin-top: 0.3em;
}
.title-page .version {
    font-size: 12pt;
    color: #777;
    margin-top: 1.5em;
}
.toc-page {
    page-break-after: always;
}
.toc-page h2 {
    text-align: center;
    font-size: 20pt;
    margin-bottom: 1.5em;
}
.toc-page ul {
    list-style: none;
    padding: 0;
    margin: 0;
}
.toc-page li {
    margin-bottom: 0.4em;
    display: flex;
    align-items: baseline;
}
.toc-page .toc-title {
    white-space: nowrap;
}
.toc-page .toc-dots {
    flex: 1;
    border-bottom: 1px dotted #999;
    margin: 0 0.4em;
    min-width: 2em;
}
.toc-page .toc-page-num {
    white-space: nowrap;
}
.toc-page .toc-page-num::after {
    content: target-counter(attr(href), page);
}
.toc-page a {
    text-decoration: none;
    color: #1a1a1a;
}
.chapter { page-break-before: always; }
pre {
    background: #f5f5f5;
    padding: 1em;
    border-radius: 4px;
    font-size: 9pt;
    overflow-x: auto;
}
code {
    font-family: "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 9pt;
}
blockquote {
    border-left: 3px solid #ccc;
    padding-left: 1em;
    color: #555;
    font-style: italic;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #ccc;
    padding: 0.4em 0.8em;
    text-align: left;
}
th {
    background: #f5f5f5;
    font-weight: bold;
}
tr:nth-child(even) { background: #fafafa; }
"""

_CHAPTER_TITLE_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$|\d)([^$\n]+?)(?<!\$)\$(?!\$)")


def _convert_latex_math(html: str) -> str:
    """Replace $...$ and $$...$$ with MathML."""
    from latex2mathml.converter import convert as latex_to_mathml

    def _replace_display(m: re.Match) -> str:
        try:
            return f'<div class="math-block">{latex_to_mathml(m.group(1).strip())}</div>'
        except Exception:
            return m.group(0)

    def _replace_inline(m: re.Match) -> str:
        try:
            return latex_to_mathml(m.group(1).strip())
        except Exception:
            return m.group(0)

    html = _DISPLAY_MATH_RE.sub(_replace_display, html)
    html = _INLINE_MATH_RE.sub(_replace_inline, html)
    return html


def publish_to_pdf(
    output_dir: str,
    title: str,
    description: str = "",
) -> dict:
    """Convert all chapter markdown files into a single PDF book."""
    try:
        import markdown as md
        from weasyprint import HTML
    except ImportError as e:
        return {
            "success": False,
            "message": (
                f"Missing dependency: {e}. "
                "Install with: pip install markdown weasyprint "
                "(weasyprint also needs system libs: pango, gdk-pixbuf)"
            ),
        }

    out = Path(output_dir)
    chapter_files = sorted(out.glob("chapter-*.md"))

    if not chapter_files:
        return {
            "success": False,
            "message": f"No chapter markdown files found in {output_dir}",
        }

    html_chapters = []
    toc_entries = []
    total_words = 0
    for idx, md_file in enumerate(chapter_files):
        raw = md_file.read_text(encoding="utf-8")
        body = _FRONT_MATTER_RE.sub("", raw, count=1).strip()
        total_words += len(body.split())
        chapter_html = md.markdown(
            body,
            extensions=["extra", "toc", "codehilite"],
            extension_configs={"codehilite": {"css_class": "highlight", "guess_lang": True}},
        )
        chapter_html = _convert_latex_math(chapter_html)

        ch_id = f"chapter-{idx + 1}"
        m = _CHAPTER_TITLE_RE.search(chapter_html)
        ch_title = m.group(1) if m else f"Chapter {idx + 1}"

        html_chapters.append(
            f'<section class="chapter" id="{ch_id}">{chapter_html}</section>'
        )
        toc_entries.append((ch_id, ch_title))

    toc_items = "\n".join(
        f'<li>'
        f'<a class="toc-title" href="#{cid}">{ctitle}</a>'
        f'<span class="toc-dots"></span>'
        f'<a class="toc-page-num" href="#{cid}"></a>'
        f'</li>'
        for cid, ctitle in toc_entries
    )
    toc_html = (
        '<div class="toc-page">\n'
        "<h2>Table of Contents</h2>\n"
        f"<ul>{toc_items}</ul>\n"
        "</div>"
    )

    pdf_slug = slugify(title, max_length=60)
    existing_versions = [
        int(m.group(1))
        for f in out.glob(f"{pdf_slug}-v*.pdf")
        if (m := re.search(r"-v(\d+)\.pdf$", f.name))
    ]
    version = max(existing_versions, default=0) + 1

    desc_html = (
        f'<p class="description">{description}</p>' if description else ""
    )
    author_html = (
        '<p class="author">기획 MSM</p>'
        '<p class="orchestrator">book-writing-agent</p>'
    )
    version_html = f'<p class="version">Version {version}</p>'
    chapters_html = "\n".join(html_chapters)

    from pygments.formatters import HtmlFormatter

    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".highlight")

    html_doc = (
        "<!DOCTYPE html>\n"
        "<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>{_PDF_CSS}\n{pygments_css}</style>\n"
        "</head>\n<body>\n"
        f'<div class="title-page"><h1>{title}</h1>{desc_html}{author_html}{version_html}</div>\n'
        f"{toc_html}\n"
        f"{chapters_html}\n"
        "</body>\n</html>"
    )

    pdf_filename = f"{pdf_slug}-v{version}.pdf"
    pdf_path = out / pdf_filename

    try:
        HTML(string=html_doc).write_pdf(str(pdf_path))
    except Exception as e:
        return {"success": False, "message": f"PDF rendering failed: {e}"}

    logger.info("PDF written to %s (version %d)", pdf_path, version)

    return {
        "success": True,
        "file_path": str(pdf_path),
        "filename": pdf_filename,
        "version": version,
        "total_chapters": len(chapter_files),
        "total_words": total_words,
    }
