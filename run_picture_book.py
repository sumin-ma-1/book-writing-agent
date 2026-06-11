#!/usr/bin/env python3
"""Picture book runner using Ollama text + image models.

Usage:
    python run_picture_book.py --toc toc/sample-picture-toc.json --model llama3:8b --no-push
    python run_picture_book.py --toc toc/sample-picture-toc.json --model llama3:8b --resume --no-push
    python run_picture_book.py --toc toc/sample-picture-toc.json --agents image,publisher --no-push
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from slugify import slugify
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.image_backends import (
    DEFAULT_A1111_URL,
    DEFAULT_DIFFUSERS_MODEL,
    DEFAULT_IMAGE_MODEL,
    check_image_backend,
    generate_page_image,
)
from app.picture_tools import (
    format_age_illustration_guidance,
    format_age_text_guidance,
    format_characters_for_prompt,
    format_pages_for_prompt,
    get_storyboard_page,
    load_page,
    load_progress,
    load_storyboard,
    load_style_bible,
    parse_picture_toc,
    load_book_meta,
    publish_picture_book_to_pdf,
    save_book_meta,
    save_page_to_disk,
    save_progress,
    save_storyboard,
    save_style_bible,
)

logger = logging.getLogger("picture-book")

LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "de": "German", "en": "English",
    "es": "Spanish", "fr": "French", "hi": "Hindi", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "my": "Burmese (Myanmar)", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
}


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def setup_logging(output_dir: str) -> None:
    from app.log_setup import configure_runner_logging

    configure_runner_logging(logger, output_dir, "picture-book.log")


def check_ollama_text_model(model: str) -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        available = [m["name"] for m in data.get("models", [])]
        matched = any(model in name or name.startswith(model) for name in available)
        if not matched:
            logger.error(
                "Text model '%s' not found. Available: %s",
                model, ", ".join(available) or "(none)",
            )
            logger.error("Pull it with: ollama pull %s", model)
            return False
        logger.info("Ollama OK - text model '%s' available", model)
        return True
    except Exception as e:
        logger.error("Cannot reach Ollama at localhost:11434: %s", e)
        return False


def _language_instruction(toc: dict, lang_override: str = "") -> str:
    lang = lang_override or toc.get("language", "")
    if lang and lang.lower() != "en":
        lang_name = _language_name(lang)
        return (
            f"\n\nIMPORTANT: Write ALL page text in {lang_name}. "
            f"Do NOT write in English."
        )
    return ""


def _writing_guidelines(toc: dict) -> str:
    guidelines = toc.get("writing_guidelines", [])
    if not guidelines:
        return ""
    lines = "\n".join(f"- {g}" for g in guidelines)
    return f"\n\nWriting guidelines:\n{lines}"


async def run_setup(
    runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    lang: str = "",
    stream: bool = False,
) -> tuple[str | None, str | None]:
    """Run style_bible + storyboard setup pipeline."""
    target_age = toc.get("target_age", "general")
    visual_style = toc.get("style", "")
    state = {
        "book_title": toc["title"],
        "book_description": toc.get("description", ""),
        "target_age": target_age,
        "visual_style": visual_style,
        "characters_description": format_characters_for_prompt(toc.get("characters", [])),
        "pages_description": format_pages_for_prompt(toc["pages"]),
        "language_instruction": _language_instruction(toc, lang),
        "writing_guidelines": _writing_guidelines(toc),
        "age_text_guidance": format_age_text_guidance(target_age),
        "age_illustration_guidance": format_age_illustration_guidance(target_age, visual_style),
    }

    session = await session_service.create_session(
        app_name="picture-book",
        user_id="picture-book",
        state=state,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Create the style bible and storyboard for '{toc['title']}'.")],
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if stream else None

    logger.info(
        "LLM setup started: style_bible -> storyboard (%d pages). Large models may take several minutes.",
        len(toc["pages"]),
    )

    style_bible = ""
    storyboard = ""
    async for event in runner.run_async(
        user_id="picture-book",
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
        if event.content and event.content.parts and not getattr(event, "partial", False):
            if event.author == "style_bible_agent":
                for part in event.content.parts:
                    if part.text:
                        style_bible = part.text
                if style_bible:
                    logger.info("Style bible done (%d chars)", len(style_bible))
            elif event.author == "storyboard_agent":
                for part in event.content.parts:
                    if part.text:
                        storyboard = part.text
                if storyboard:
                    logger.info("Storyboard done (%d chars)", len(storyboard))

    if not style_bible or not storyboard:
        session = await session_service.get_session(
            app_name="picture-book",
            user_id="picture-book",
            session_id=session.id,
        )
        style_bible = style_bible or session.state.get("style_bible", "")
        storyboard = storyboard or session.state.get("storyboard", "")

    return style_bible or None, storyboard or None


async def run_page(
    runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    page: dict,
    style_bible: str,
    storyboard_data: dict | None,
    lang: str = "",
    stream: bool = False,
) -> tuple[str | None, str | None]:
    """Run page_writer + illustrator pipeline for one page."""
    page_storyboard = get_storyboard_page(storyboard_data, page["number"])
    page_storyboard_text = json.dumps(page_storyboard, ensure_ascii=False, indent=2)

    target_age = toc.get("target_age", "general")
    visual_style = toc.get("style", "")
    state = {
        "book_title": toc["title"],
        "book_description": toc.get("description", ""),
        "target_age": target_age,
        "visual_style": visual_style,
        "style_bible": style_bible,
        "current_page_number": str(page["number"]),
        "page_storyboard": page_storyboard_text,
        "page_outline": "",
        "page_text": "",
        "language_instruction": _language_instruction(toc, lang),
        "writing_guidelines": _writing_guidelines(toc),
        "age_text_guidance": format_age_text_guidance(target_age),
        "age_illustration_guidance": format_age_illustration_guidance(target_age, visual_style),
    }

    session = await session_service.create_session(
        app_name="picture-book",
        user_id="picture-book",
        state=state,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=f"Write page {page['number']}: {page.get('scene', '')}"
        )],
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if stream else None

    page_text = ""
    image_prompt = ""
    async for event in runner.run_async(
        user_id="picture-book",
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
        if event.content and event.content.parts and not getattr(event, "partial", False):
            if event.author in (
                "page_writer_agent",
                "page_reviewer_agent",
                "page_finalizer_agent",
            ):
                for part in event.content.parts:
                    if part.text:
                        page_text = part.text
            elif event.author == "illustrator_prompt_agent":
                for part in event.content.parts:
                    if part.text:
                        image_prompt = part.text

    if not page_text or not image_prompt:
        session = await session_service.get_session(
            app_name="picture-book",
            user_id="picture-book",
            session_id=session.id,
        )
        page_text = page_text or session.state.get("page_text", "")
        image_prompt = image_prompt or session.state.get("image_prompt", "")

    return page_text or None, image_prompt or None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Picture book writer (text + illustrations via Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_picture_book.py --toc toc/sample-picture-toc.json --model llama3:8b --no-push\n"
            "  python run_picture_book.py --toc toc/sample-picture-toc.json --model llama3:8b --resume --no-push\n"
            "  python run_picture_book.py --toc toc/sample-picture-toc.json --agents image,publisher --no-push\n"
        ),
    )
    parser.add_argument("--toc", required=True, help="Path to picture book TOC JSON/YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: ./picture-book/<slug>)")
    parser.add_argument("--model", default=None, help="Ollama text model (e.g. llama3:8b, gemma3:27b)")
    parser.add_argument(
        "--image-backend",
        default="automatic1111",
        choices=["automatic1111", "diffusers", "ollama"],
        help="Image backend (default: automatic1111 for Linux/Windows)",
    )
    parser.add_argument(
        "--image-api-url", default=DEFAULT_A1111_URL,
        help=f"SD WebUI/Forge API URL (default: {DEFAULT_A1111_URL})",
    )
    parser.add_argument(
        "--image-model", default="",
        help="Model/checkpoint: A1111 checkpoint name, diffusers HF id, or Ollama model",
    )
    parser.add_argument("--retry", type=int, default=3, help="Retries per page")
    parser.add_argument("--resume", action="store_true", help="Resume from progress")
    parser.add_argument(
        "--timeout",
        type=int,
        default=2400,
        help="Min. timeout per page LLM chain (seconds); scaled up by agent count",
    )
    parser.add_argument("--image-timeout", type=int, default=600, help="Timeout per image (seconds)")
    parser.add_argument("--stream", action="store_true", help="Stream LLM output")
    parser.add_argument("--no-think", action="store_true", help="Disable model thinking")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window for text model")
    parser.add_argument(
        "--agents",
        default="style_bible,storyboard,outline,writer,reviewer,finalizer,illustrator,image,publisher",
        help="Pipeline stages (default: full pipeline)",
    )
    parser.add_argument("--lang", default=None, help="Language override (e.g. ko)")
    parser.add_argument("--rewrite", type=int, nargs="+", metavar="N", help="Rewrite page(s)")
    parser.add_argument("--rewrite-all", action="store_true", help="Rewrite all pages")
    parser.add_argument("--skip", type=int, nargs="+", metavar="N", help="Skip page(s)")
    parser.add_argument("--no-push", action="store_true", help="Skip git operations")
    args = parser.parse_args()

    toc = parse_picture_toc(args.toc)
    output_dir = args.output_dir or f"./picture-book/{slugify(toc['title'], max_length=60)}"
    setup_logging(output_dir)

    requested = [a.strip() for a in args.agents.split(",") if a.strip()]
    run_setup_stages = "style_bible" in requested or "storyboard" in requested
    from app.pipeline_config import PICTURE_PAGE_AGENT_IDS

    run_page_llm = any(a in PICTURE_PAGE_AGENT_IDS for a in requested)
    run_image = "image" in requested
    run_publisher = "publisher" in requested
    page_llm_names = [a for a in requested if a in PICTURE_PAGE_AGENT_IDS]
    n_page_llm = len(page_llm_names) or 5
    per_agent_budget = 480
    page_llm_timeout = max(args.timeout, n_page_llm * per_agent_budget)

    needs_text_model = run_setup_stages or run_page_llm
    if needs_text_model and not args.model:
        parser.error("--model is required when running text agents")

    if needs_text_model:
        os.environ["AGENT_MODEL"] = args.model
        os.environ["LLM_TIMEOUT"] = str(max(600, per_agent_budget + 120))
        os.environ["NUM_CTX"] = str(args.num_ctx)
        if args.no_think:
            os.environ["DISABLE_THINKING"] = "1"
        if not check_ollama_text_model(args.model):
            sys.exit(1)

    image_model = args.image_model or {
        "ollama": DEFAULT_IMAGE_MODEL,
        "diffusers": DEFAULT_DIFFUSERS_MODEL,
        "automatic1111": "",
    }[args.image_backend]

    if run_image and not check_image_backend(
        args.image_backend,
        api_url=args.image_api_url,
        model=image_model,
    ):
        sys.exit(1)

    resolved_image_label = image_model
    if run_image and args.image_backend == "automatic1111" and not image_model:
        from app.image_backends import _resolve_a1111_checkpoint

        resolved_image_label = _resolve_a1111_checkpoint(args.image_api_url, "") or "Forge"

    from app.pipeline_config import PICTURE_AGENT_ALIASES

    page_llm_agents = [
        PICTURE_AGENT_ALIASES.get(a, a)
        for a in requested
        if a in PICTURE_PAGE_AGENT_IDS
    ]
    seen_agents: set[str] = set()
    ordered_page_agents: list[str] = []
    for a in page_llm_agents:
        if a not in seen_agents:
            seen_agents.add(a)
            ordered_page_agents.append(a)
    os.environ["PIPELINE_AGENTS"] = ",".join(ordered_page_agents) if ordered_page_agents else ""

    logger.info("Picture Book starting - title: %s", toc["title"])
    logger.info("Pages: %d | Output: %s", len(toc["pages"]), output_dir)
    logger.info("Agents: %s", ", ".join(requested))
    if run_image:
        logger.info("Image backend: %s", args.image_backend)

    if args.rewrite_all:
        rewrite_set = {p["number"] for p in toc["pages"]}
    else:
        rewrite_set = set(args.rewrite) if args.rewrite else None

    if rewrite_set:
        progress = load_progress(output_dir)
        progress["completed"] = [c for c in progress.get("completed", []) if c not in rewrite_set]
    elif args.resume:
        progress = load_progress(output_dir)
    else:
        progress = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed": [],
            "failed": {},
            "in_progress": None,
            "setup_completed": [],
        }

    completed = set(progress.get("completed", []))
    setup_completed = set(progress.get("setup_completed", []))
    skip_set = set(args.skip) if args.skip else set()
    start_time = time.time()
    reference_image: str | None = None

    style_bible = load_style_bible(output_dir)
    storyboard_data = load_storyboard(output_dir)

    # --- Setup: style bible + storyboard ---
    if run_setup_stages and not {"style_bible", "storyboard"}.issubset(setup_completed):
        from app.picture_agent import setup_pipeline

        logger.info("=" * 60)
        logger.info("SETUP: Style bible + storyboard")
        logger.info("=" * 60)
        progress["phase"] = "setup"
        progress["in_progress"] = "setup"
        save_progress(output_dir, progress)

        session_service = InMemorySessionService()
        setup_runner = Runner(
            agent=setup_pipeline,
            app_name="picture-book",
            session_service=session_service,
        )

        try:
            sb, st = await asyncio.wait_for(
                run_setup(setup_runner, session_service, toc, lang=args.lang or "", stream=args.stream),
                timeout=args.timeout * 2,
            )
            if sb:
                save_style_bible(sb, output_dir)
                style_bible = sb
                setup_completed.add("style_bible")
                logger.info("Style bible saved")
            if st:
                save_storyboard(st, output_dir)
                storyboard_data = load_storyboard(output_dir)
                setup_completed.add("storyboard")
                logger.info("Storyboard saved")
        except Exception:
            logger.exception("Setup pipeline failed")
            sys.exit(1)

        progress["setup_completed"] = list(setup_completed)
        progress["in_progress"] = None
        progress["phase"] = "pages"
        save_progress(output_dir, progress)

    if not style_bible and (run_page_llm or run_image):
        logger.error("Style bible missing. Run with --agents style_bible,storyboard first.")
        sys.exit(1)

    # --- Per-page pipeline ---
    if run_page_llm or run_image:
        page_runner = None
        page_session_service = None

        if run_page_llm:
            from app.picture_agent import page_pipeline

            page_session_service = InMemorySessionService()
            page_runner = Runner(
                agent=page_pipeline,
                app_name="picture-book",
                session_service=page_session_service,
            )
            logger.info(
                "Page LLM timeout: %ds (%d text agents per page, model=%s)",
                page_llm_timeout,
                n_page_llm,
                args.model,
            )

        out_path = Path(output_dir)
        for page in toc["pages"]:
            page_num = page["number"]

            if page_num in skip_set:
                logger.info("Skipping page %d (--skip)", page_num)
                continue

            existing = load_page(output_dir, page_num)
            if page_num in completed and existing:
                if run_image and existing.get("image_path") and Path(existing["image_path"]).exists():
                    logger.info("Skipping page %d (already complete)", page_num)
                    if page_num == 1 or reference_image is None:
                        reference_image = existing.get("image_path")
                    continue
                if not run_image:
                    logger.info("Skipping page %d (already complete)", page_num)
                    continue

            logger.info("--- Page %d/%d ---", page_num, len(toc["pages"]))
            progress["in_progress"] = page_num
            save_progress(output_dir, progress)

            page_text = existing.get("text", "") if existing else ""
            image_prompt = existing.get("image_prompt", "") if existing else ""

            if run_page_llm and (not page_text or not image_prompt or page_num in (rewrite_set or set())):
                for attempt in range(1, args.retry + 1):
                    try:
                        logger.info("LLM attempt %d/%d for page %d", attempt, args.retry, page_num)
                        pt, ip = await asyncio.wait_for(
                            run_page(
                                page_runner, page_session_service, toc, page,
                                style_bible, storyboard_data,
                                lang=args.lang or "", stream=args.stream,
                            ),
                            timeout=page_llm_timeout,
                        )
                        if pt and ip:
                            page_text, image_prompt = pt, ip
                            break
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Page %d LLM timed out after %ds (attempt %d, %d agents)",
                            page_num,
                            page_llm_timeout,
                            attempt,
                            n_page_llm,
                        )
                    except Exception:
                        logger.exception("Page %d LLM failed (attempt %d)", page_num, attempt)

            image_path = existing.get("image_path", "") if existing else ""
            if run_image and image_prompt:
                images_dir = out_path / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                target_image = str(images_dir / f"page-{page_num:02d}.png")

                if not image_path or not Path(image_path).exists() or page_num in (rewrite_set or set()):
                    full_prompt = f"{toc.get('style', '')}. {image_prompt}"
                    for attempt in range(1, args.retry + 1):
                        try:
                            logger.info("Image attempt %d/%d for page %d", attempt, args.retry, page_num)
                            result = generate_page_image(
                                prompt=full_prompt,
                                output_path=target_image,
                                backend=args.image_backend,
                                model=image_model,
                                api_url=args.image_api_url,
                                width=toc.get("image_width", 512),
                                height=toc.get("image_height", 512),
                                reference_image=reference_image,
                                timeout=args.image_timeout,
                            )
                            if result["success"]:
                                image_path = result["image_path"]
                                if reference_image is None:
                                    reference_image = image_path
                                break
                            logger.warning("Image gen failed: %s", result["message"])
                        except Exception:
                            logger.exception("Page %d image failed (attempt %d)", page_num, attempt)

            if page_text and image_prompt:
                save_page_to_disk(
                    page_num, page_text, image_prompt, output_dir,
                    image_path=image_path or None,
                    scene=page.get("scene", ""),
                    mood=page.get("mood", ""),
                )
                has_image = bool(image_path and Path(image_path).exists())
                logger.info("Saved page %d (text: %d words, image: %s)",
                            page_num, len(page_text.split()),
                            "yes" if has_image else "no")
                page_done = (not run_image) or has_image
                if page_done:
                    progress["completed"].append(page_num)
                    completed.add(page_num)
                elif run_image:
                    logger.warning(
                        "Page %d text saved but image missing - will retry on --resume",
                        page_num,
                    )
            else:
                logger.error("SKIPPING page %d after failures", page_num)
                progress["failed"][str(page_num)] = f"Failed after {args.retry} attempts"

            progress["in_progress"] = None
            save_progress(output_dir, progress)

    # --- Publisher ---
    pub_result = None
    if run_publisher:
        logger.info("=" * 60)
        logger.info("PUBLISHING: Picture book PDF")
        logger.info("=" * 60)
        progress["phase"] = "publish"
        progress["in_progress"] = "publish"
        save_progress(output_dir, progress)

        existing_meta = load_book_meta(output_dir)
        pdf_meta = {
            "planner": "MSM",
            "title": toc["title"],
            "description": toc.get("description", ""),
            "text_model": args.model or existing_meta.get("text_model", ""),
            "image_model": (
                resolved_image_label if run_image else existing_meta.get("image_model", "")
            ),
            "image_backend": (
                args.image_backend if run_image else existing_meta.get("image_backend", "")
            ),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        save_book_meta(output_dir, pdf_meta)

        pub_result = publish_picture_book_to_pdf(
            output_dir=output_dir,
            title=toc["title"],
            description=toc.get("description", ""),
            meta=pdf_meta,
        )

        if pub_result["success"]:
            logger.info(
                "PDF published: %s v%d (%d pages)",
                pub_result["filename"], pub_result["version"], pub_result["total_pages"],
            )
        else:
            logger.error("PDF publishing failed: %s", pub_result["message"])

        progress["in_progress"] = None
        progress["phase"] = "done" if pub_result and pub_result.get("success") else "publish_failed"
        save_progress(output_dir, progress)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PICTURE BOOK COMPLETE")
    logger.info("Title: %s", toc["title"])
    logger.info("Pages completed: %d/%d", len(completed), len(toc["pages"]))
    logger.info("Failed: %d", len(progress.get("failed", {})))
    if pub_result and pub_result.get("success"):
        logger.info("PDF: %s", pub_result["filename"])
    logger.info("Total time: %.1f minutes", elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
