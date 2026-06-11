from __future__ import annotations

import base64
import json
import logging
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from slugify import slugify

logger = logging.getLogger("picture-book")

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_IMAGE_MODEL = "x/flux2-klein:4b"
DEFAULT_PDF_PLANNER = "MSM"
BOOK_META_FILENAME = "book-meta.json"


def parse_picture_toc(file_path: str) -> dict:
    """Parse a picture book TOC file (JSON or YAML)."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        toc = yaml.safe_load(content)
    else:
        toc = json.loads(content)

    if toc.get("type") and toc["type"] != "picture_book":
        raise ValueError(f"Expected type 'picture_book', got '{toc['type']}'")

    pages = toc.get("pages", [])
    for i, page in enumerate(pages):
        if "number" not in page:
            page["number"] = i + 1
        for key in ("scene", "mood"):
            if key not in page:
                page[key] = ""

    toc.setdefault("style", "illustrated book art, cohesive visual style for the target audience")
    toc.setdefault("characters", [])
    toc.setdefault("target_age", "general")
    toc.setdefault("image_width", 512)
    toc.setdefault("image_height", 512)

    return toc


def format_age_text_guidance(target_age: str) -> str:
    """Prompt block: page text length/tone scales with audience, not fixed toddler rules."""
    age = (target_age or "general").strip()
    return f"""Page text guidance for audience ({age}):
- This is an illustrated book: each page has art, but the text is real book prose — not image captions.
- Adapt length to the audience (do NOT default to toddler brevity unless the audience is young):
  · Ages 3-6 / preschool: 1-3 short sentences, simple words, read-aloud rhythm
  · Ages 7-12: about one paragraph (roughly 3-6 sentences), clear and engaging
  · Ages 13-17: one or more paragraphs; richer voice and emotional depth
  · Adults (18+, e.g. "18세 이상", "adult"): full prose as the page needs — multiple paragraphs OK; literary tone matching the book
  · "전 연령" / general: accessible but not artificially shortened
- Interpret custom age labels sensibly from the book description and writing guidelines."""


def format_age_illustration_guidance(target_age: str, visual_style: str) -> str:
    """Prompt block: illustration style follows audience and visual_style, not always toddler art."""
    age = (target_age or "general").strip()
    style = (visual_style or "illustrated book").strip()
    return f"""Illustration guidance for audience ({age}), visual style: {style}
- Match art style and mood to the target audience — not every illustrated book is for toddlers.
- Young children: gentle, warm, age-appropriate picture-book illustration when the audience is young
- Older children / teens: can be more dynamic, detailed, or stylized per the style bible
- Adults: cinematic, painterly, graphic-novel, or other mature styles as {style} implies
- Keep character consistency from the style bible; lighting and tone should fit the scene mood"""


def check_ollama_image_model(model: str) -> bool:
    """Verify Ollama is running and the image model is available."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        available = [m["name"] for m in data.get("models", [])]
        matched = any(
            model in name or name.startswith(model) or name.split(":")[0] == model.split(":")[0]
            for name in available
        )
        if not matched:
            logger.error(
                "Image model '%s' not found in Ollama. Available: %s",
                model,
                ", ".join(available) or "(none)",
            )
            logger.error("Pull it with: ollama pull %s", model)
            return False
        logger.info("Ollama OK — image model '%s' available", model)
        return True
    except Exception as e:
        logger.error("Cannot reach Ollama at %s: %s", OLLAMA_BASE_URL, e)
        logger.error("Start Ollama with: ollama serve")
        return False


def generate_ollama_image(
    prompt: str,
    output_path: str,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 1024,
    height: int = 1024,
    reference_image: str | None = None,
    timeout: int = 600,
) -> dict:
    """Generate an illustration via Ollama's image generation API (macOS mainly)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "width": width,
        "height": height,
    }

    if reference_image and Path(reference_image).exists():
        image_b64 = base64.b64encode(Path(reference_image).read_bytes()).decode("utf-8")
        payload["images"] = [image_b64]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except urllib.error.URLError as e:
        return {"success": False, "message": f"Ollama image request failed: {e}"}

    result = json.loads(body)
    image_b64 = result.get("image", "")
    if not image_b64:
        return {
            "success": False,
            "message": "Ollama returned no image. Ensure you use an image model "
            f"(e.g. {DEFAULT_IMAGE_MODEL}) and a recent Ollama version.",
        }

    out.write_bytes(base64.b64decode(image_b64))
    logger.info("Image saved to %s", out)
    return {"success": True, "image_path": str(out)}


def save_style_bible(content: str, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "style-bible.md"
    path.write_text(content, encoding="utf-8")
    return {"success": True, "file_path": str(path)}


def save_storyboard(content: str, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "storyboard.json"

    try:
        parsed = json.loads(content)
        path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    except json.JSONDecodeError:
        path = out / "storyboard.md"
        path.write_text(content, encoding="utf-8")

    return {"success": True, "file_path": str(path)}


def load_storyboard(output_dir: str) -> dict | None:
    path = Path(output_dir) / "storyboard.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_style_bible(output_dir: str) -> str:
    path = Path(output_dir) / "style-bible.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def save_page_to_disk(
    page_number: int,
    text: str,
    image_prompt: str,
    output_dir: str,
    image_path: str | None = None,
    scene: str = "",
    mood: str = "",
) -> dict:
    """Save page metadata as JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"page-{page_number:02d}.json"
    filepath = out / filename

    page_data = {
        "number": page_number,
        "scene": scene,
        "mood": mood,
        "text": text.strip(),
        "image_prompt": image_prompt.strip(),
        "image_path": image_path or "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    filepath.write_text(
        json.dumps(page_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "success": True,
        "file_path": str(filepath),
        "filename": filename,
        "word_count": len(text.split()),
    }


def load_page(output_dir: str, page_number: int) -> dict | None:
    path = Path(output_dir) / f"page-{page_number:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_progress(output_dir: str) -> dict:
    path = Path(output_dir) / ".progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "completed": [],
        "failed": {},
        "in_progress": None,
        "setup_completed": [],
    }


def save_progress(output_dir: str, progress: dict) -> None:
    path = Path(output_dir) / ".progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _compute_picture_book_layout(pages: list[dict]) -> dict:
    """Pick one image/text split and font size for the whole book (longest page wins)."""
    texts = [p.get("text", "").strip() for p in pages if p.get("text", "").strip()]
    if not texts:
        return {
            "image_pct": 62,
            "text_pct": 38,
            "font_size_pt": 13,
            "line_height": 1.5,
            "text_align": "center",
            "max_chars": 0,
        }

    max_chars = max(len(t) for t in texts)
    tiers = (
        (80, 65, 14),
        (180, 58, 13),
        (320, 52, 12),
        (480, 46, 11),
        (650, 40, 10),
        (10**9, 34, 9),
    )
    image_pct, font_size = 62, 13
    for limit, img_pct, fs in tiers:
        if max_chars <= limit:
            image_pct, font_size = img_pct, fs
            break

    return {
        "image_pct": image_pct,
        "text_pct": 100 - image_pct,
        "font_size_pt": font_size,
        "line_height": 1.45 if font_size <= 11 else 1.5,
        "text_align": "left" if max_chars > 150 else "center",
        "max_chars": max_chars,
    }


def _picture_pdf_css(layout: dict) -> str:
    img_pct = layout["image_pct"]
    text_pct = layout["text_pct"]
    font_pt = layout["font_size_pt"]
    line_h = layout["line_height"]
    align = layout["text_align"]
    return f"""\
@page {{
    size: 210mm 210mm;
    margin: 0;
}}
body {{
    font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
    margin: 0;
    padding: 0;
    color: #2d2d2d;
}}
.page {{
    page-break-after: always;
    width: 210mm;
    height: 210mm;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}}
.page .illustration {{
    width: 100%;
    height: {img_pct}%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0ebe4;
    overflow: hidden;
    flex-shrink: 0;
}}
.page .illustration img {{
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
}}
.page .text {{
    height: {text_pct}%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    text-align: {align};
    padding: 8mm 14mm 10mm;
    font-size: {font_pt}pt;
    line-height: {line_h};
    background: #fffaf5;
    box-sizing: border-box;
    overflow: hidden;
}}
.page .text p {{
    margin: 0 0 0.35em;
}}
.page .text p:last-child {{
    margin-bottom: 0;
}}
.page.text-only .text {{
    height: 100%;
    font-size: {min(font_pt + 2, 16)}pt;
    justify-content: center;
}}
.title-page {{
    page-break-after: always;
    width: 210mm;
    height: 210mm;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: linear-gradient(135deg, #fff5e6 0%, #ffe4cc 100%);
    padding: 20mm;
    box-sizing: border-box;
}}
.title-page h1 {{
    font-size: 32pt;
    margin: 0 0 0.5em;
    color: #5c3d2e;
}}
.title-page .description {{
    font-size: 14pt;
    color: #7a5c4f;
    font-style: italic;
    max-width: 80%;
}}
.title-page .colophon {{
    margin-top: 2em;
    font-size: 11pt;
    color: #9a7b6c;
    line-height: 1.7;
}}
"""


def save_book_meta(output_dir: str, meta: dict) -> None:
    path = Path(output_dir) / BOOK_META_FILENAME
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_book_meta(output_dir: str) -> dict:
    path = Path(output_dir) / BOOK_META_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _format_image_model_label(meta: dict) -> str:
    model = (meta.get("image_model") or "").strip()
    backend = (meta.get("image_backend") or "").strip()
    if model:
        return model
    if backend == "automatic1111":
        return "Forge / automatic1111"
    return backend or "—"


def _build_colophon_lines(meta: dict) -> list[str]:
    planner = (
        meta.get("planner")
        or meta.get("producer")
        or meta.get("author")
        or DEFAULT_PDF_PLANNER
    )
    lines = [f"기획 {planner}"]
    date = (meta.get("generated_at") or "")[:10]
    if date:
        lines.append(date)
    text_model = (meta.get("text_model") or "").strip()
    if text_model:
        lines.append(f"텍스트 {text_model}")
    image_label = _format_image_model_label(meta)
    if image_label != "—":
        lines.append(f"이미지 {image_label}")
    return lines


def _colophon_html(meta: dict) -> str:
    lines = _build_colophon_lines(meta)
    body = "<br>\n".join(_escape_html(line) for line in lines)
    return f'<div class="colophon">{body}</div>'


def _next_pdf_path(out: Path, title: str) -> tuple[Path, str, int]:
    pdf_slug = slugify(title, max_length=60)
    existing_versions = [
        int(m.group(1))
        for f in out.glob(f"{pdf_slug}-v*.pdf")
        if (m := re.search(r"-v(\d+)\.pdf$", f.name))
    ]
    version = max(existing_versions, default=0) + 1
    pdf_filename = f"{pdf_slug}-v{version}.pdf"
    return out / pdf_filename, pdf_filename, version


def _load_picture_pages(out: Path) -> tuple[list[dict], int]:
    page_files = sorted(out.glob("page-*.json"))
    pages = []
    total_words = 0
    for page_file in page_files:
        page = json.loads(page_file.read_text(encoding="utf-8"))
        text = page.get("text", "").strip()
        total_words += len(text.split())
        pages.append(page)
    return pages, total_words


def _image_pixel_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def _fit_image_in_box(
    img_w: int, img_h: int, box_w: float, box_h: float
) -> tuple[float, float, float, float]:
    """Return draw size (w, h) and top-left offset (x, y) to contain image in box."""
    if img_w <= 0 or img_h <= 0:
        return box_w, box_h, 0.0, 0.0
    scale = min(box_w / img_w, box_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    return draw_w, draw_h, (box_w - draw_w) / 2, (box_h - draw_h) / 2


def _format_page_text_html(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ""
    return "".join(f"<p>{_escape_html(p)}</p>" for p in paragraphs)


def _windows_korean_font() -> str | None:
    if sys.platform != "win32":
        return None
    for path in (
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunsl.ttf"),
    ):
        if path.exists():
            return str(path)
    return None


def _publish_picture_book_fpdf(
    output_dir: str,
    title: str,
    description: str = "",
    meta: dict | None = None,
) -> dict:
    """Fallback PDF builder for Windows (no GTK/Pango required)."""
    try:
        from fpdf import FPDF
    except ImportError as e:
        return {
            "success": False,
            "message": f"fpdf2 not installed: {e}. Run: pip install fpdf2",
        }

    out = Path(output_dir)
    pages, total_words = _load_picture_pages(out)
    if not pages:
        return {"success": False, "message": f"No page JSON files found in {output_dir}"}

    pdf_path, pdf_filename, version = _next_pdf_path(out, title)
    font_path = _windows_korean_font()

    pdf = FPDF(orientation="P", unit="mm", format=(210, 210))
    pdf.set_auto_page_break(auto=False)
    if font_path:
        pdf.add_font("BookFont", "", font_path)
        body_font = "BookFont"
    else:
        body_font = "Helvetica"

    meta = {**load_book_meta(output_dir), **(meta or {})}
    meta.setdefault("planner", DEFAULT_PDF_PLANNER)
    layout = _compute_picture_book_layout(pages)
    logger.info(
        "Picture PDF layout: image=%d%% text=%d%% font=%dpt (max_chars=%d)",
        layout["image_pct"],
        layout["text_pct"],
        layout["font_size_pt"],
        layout["max_chars"],
    )

    page_w = 210.0
    page_h = 210.0
    img_h = page_h * layout["image_pct"] / 100
    text_top = img_h
    text_h = page_h - img_h
    body_font_size = layout["font_size_pt"]
    line_h_mm = body_font_size * 0.38 * layout["line_height"]

    # Title page
    pdf.add_page()
    pdf.set_font(body_font, size=22)
    pdf.set_xy(15, 58)
    pdf.multi_cell(180, 12, title, align="C")
    y = 88
    if description:
        pdf.set_font(body_font, size=11)
        pdf.set_xy(18, y)
        pdf.multi_cell(174, 7, description, align="C")
        y += 28
    pdf.set_font(body_font, size=10)
    for line in _build_colophon_lines(meta):
        pdf.set_xy(18, y)
        pdf.multi_cell(174, 6, line, align="C")
        y += 8

    for page in pages:
        pdf.add_page()
        text = page.get("text", "").strip()
        image_path = page.get("image_path", "")

        if image_path and Path(image_path).exists():
            try:
                px_size = _image_pixel_size(Path(image_path))
                if px_size:
                    draw_w, draw_h, off_x, off_y = _fit_image_in_box(
                        px_size[0], px_size[1], page_w, img_h
                    )
                    pdf.image(image_path, x=off_x, y=off_y, w=draw_w, h=draw_h)
                else:
                    pdf.image(image_path, x=0, y=0, w=page_w, h=img_h)
            except Exception as e:
                logger.warning("fpdf image embed failed for page %s: %s", page.get("number"), e)

        if text:
            pdf.set_font(body_font, size=body_font_size)
            x_margin = 14.0
            text_w = page_w - x_margin * 2
            align = "L" if layout["text_align"] == "left" else "C"
            y = text_top + 6.0
            for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
                pdf.set_xy(x_margin, y)
                pdf.multi_cell(text_w, line_h_mm, para, align=align)
                y = pdf.get_y() + line_h_mm * 0.25
                if y > page_h - 8:
                    break

    try:
        pdf.output(str(pdf_path))
    except Exception as e:
        return {"success": False, "message": f"fpdf PDF write failed: {e}"}

    logger.info("Picture book PDF written to %s (backend: fpdf2, version %d)", pdf_path, version)
    return {
        "success": True,
        "file_path": str(pdf_path),
        "filename": pdf_filename,
        "version": version,
        "total_pages": len(pages),
        "total_words": total_words,
        "backend": "fpdf2",
    }


def publish_picture_book_to_pdf(
    output_dir: str,
    title: str,
    description: str = "",
    meta: dict | None = None,
) -> dict:
    """Combine page images and text into a square picture-book PDF."""
    merged_meta = {**load_book_meta(output_dir), **(meta or {})}
    merged_meta.setdefault("planner", DEFAULT_PDF_PLANNER)

    weasyprint_error = ""
    try:
        from weasyprint import HTML  # noqa: F401
    except (ImportError, OSError) as e:
        weasyprint_error = str(e)
        logger.warning(
            "WeasyPrint unavailable (%s). Trying fpdf2 fallback.",
            e,
        )
        result = _publish_picture_book_fpdf(output_dir, title, description, merged_meta)
        if result["success"]:
            return result
        if weasyprint_error:
            result["message"] = (
                f"WeasyPrint: {weasyprint_error}. Fallback also failed: {result['message']}"
            )
        return result

    out = Path(output_dir)
    page_files = sorted(out.glob("page-*.json"))

    if not page_files:
        return {
            "success": False,
            "message": f"No page JSON files found in {output_dir}",
        }

    pages = [json.loads(p.read_text(encoding="utf-8")) for p in page_files]
    layout = _compute_picture_book_layout(pages)
    logger.info(
        "Picture PDF layout: image=%d%% text=%d%% font=%dpt (max_chars=%d)",
        layout["image_pct"],
        layout["text_pct"],
        layout["font_size_pt"],
        layout["max_chars"],
    )

    pages_html = []
    total_words = 0

    for page in pages:
        text = page.get("text", "").strip()
        total_words += len(text.split())
        image_path = page.get("image_path", "")

        text_html = f'<div class="text">{_format_page_text_html(text)}</div>'

        if image_path and Path(image_path).exists():
            rel = Path(image_path).resolve().as_uri()
            pages_html.append(
                f'<div class="page">'
                f'<div class="illustration"><img src="{rel}" alt="Page {page["number"]}" /></div>'
                f"{text_html}</div>"
            )
        else:
            pages_html.append(f'<div class="page text-only">{text_html}</div>')

    desc_html = (
        f'<p class="description">{_escape_html(description)}</p>' if description else ""
    )
    colophon_html = _colophon_html(merged_meta)

    html_doc = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{_escape_html(title)}</title>\n"
        f"<style>{_picture_pdf_css(layout)}</style>\n"
        "</head>\n<body>\n"
        f'<div class="title-page">'
        f"<h1>{_escape_html(title)}</h1>"
        f"{desc_html}{colophon_html}</div>\n"
        + "\n".join(pages_html)
        + "\n</body>\n</html>"
    )

    pdf_path, pdf_filename, version = _next_pdf_path(out, title)

    try:
        HTML(string=html_doc, base_url=str(out.resolve())).write_pdf(str(pdf_path))
    except Exception as e:
        logger.warning("WeasyPrint render failed (%s). Trying fpdf2 fallback.", e)
        fallback = _publish_picture_book_fpdf(output_dir, title, description, merged_meta)
        if fallback["success"]:
            return fallback
        return {
            "success": False,
            "message": f"WeasyPrint render failed: {e}. Fallback: {fallback['message']}",
        }

    logger.info("Picture book PDF written to %s (backend: weasyprint, version %d)", pdf_path, version)
    return {
        "success": True,
        "file_path": str(pdf_path),
        "filename": pdf_filename,
        "version": version,
        "total_pages": len(page_files),
        "total_words": total_words,
        "backend": "weasyprint",
    }


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def format_characters_for_prompt(characters: list[dict]) -> str:
    if not characters:
        return "No specific characters defined."
    lines = []
    for ch in characters:
        name = ch.get("name", "Unknown")
        desc = ch.get("description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def format_pages_for_prompt(pages: list[dict]) -> str:
    lines = []
    for page in pages:
        lines.append(
            f"Page {page['number']}: scene={page.get('scene', '')}, "
            f"mood={page.get('mood', '')}"
        )
    return "\n".join(lines)


def get_storyboard_page(storyboard: dict | str | None, page_number: int) -> dict:
    if not storyboard:
        return {}
    if isinstance(storyboard, str):
        try:
            storyboard = json.loads(storyboard)
        except json.JSONDecodeError:
            return {"description": storyboard}

    pages = storyboard.get("pages", storyboard if isinstance(storyboard, list) else [])
    for page in pages:
        if page.get("number") == page_number:
            return page
    return {}
