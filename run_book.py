#!/usr/bin/env python3
"""Overnight book-writing runner using local Ollama models.

Usage:
    # GitHub URL — auto-clones repo, reads TOC, writes chapters to the same directory:
    python run_book.py --toc https://github.com/user/repo/blob/main/my-book/toc.json --model gemma4:31b

    # Local file:
    python run_book.py --toc toc.json --output-dir ./my-book --model llama3:8b

    # Resume after interruption:
    python run_book.py --toc https://github.com/user/repo/blob/main/my-book/toc.json --model gemma4:31b --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

from slugify import slugify
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logger = logging.getLogger("book-writer")

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

GITHUB_BLOB_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)"
)
GITHUB_TREE_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<branch>[^/]+)/(?P<path>.+)"
)


def resolve_github_toc(url: str, clone_base: str = "./repos") -> dict:
    """Parse a GitHub blob URL, clone/pull the repo, return local paths.

    Returns dict with keys: toc_path, output_dir, repo_dir, branch, repo_url
    """
    m = GITHUB_BLOB_RE.match(url)
    if not m:
        raise ValueError(
            f"Not a valid GitHub blob URL: {url}\n"
            "Expected: https://github.com/owner/repo/blob/branch/path/to/toc.json"
        )

    owner = m.group("owner")
    repo = m.group("repo")
    branch = m.group("branch")
    file_path = m.group("path")

    repo_url = f"git@github.com:{owner}/{repo}.git"
    repo_dir = Path(clone_base) / repo
    toc_path = repo_dir / file_path
    output_dir = toc_path.parent

    if (repo_dir / ".git").exists():
        logger.info("Repo already cloned at %s — pulling latest", repo_dir)
        subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=str(repo_dir),
            check=True,
            timeout=120,
            capture_output=True,
        )
    else:
        logger.info("Cloning %s into %s", repo_url, repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-b", branch, repo_url, str(repo_dir)],
            check=True,
            timeout=120,
        )

    if not toc_path.exists():
        raise FileNotFoundError(f"TOC file not found at {toc_path}")

    return {
        "toc_path": str(toc_path),
        "output_dir": str(output_dir),
        "repo_dir": str(repo_dir),
        "branch": branch,
        "repo_url": repo_url,
    }


def is_github_url(s: str) -> bool:
    return s.startswith("https://github.com/") or s.startswith("http://github.com/")


def setup_logging(output_dir: str) -> None:
    from app.log_setup import configure_runner_logging

    configure_runner_logging(logger, output_dir, "book-writer.log")


def check_ollama(model: str) -> bool:
    """Verify Ollama is running and the model is available."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        available = [m["name"] for m in data.get("models", [])]
        matched = any(model in name or name.startswith(model) for name in available)
        if not matched:
            logger.error(
                "Model '%s' not found in Ollama. Available: %s",
                model,
                ", ".join(available),
            )
            logger.error("Pull it with: ollama pull %s", model)
            return False
        logger.info("Ollama OK — model '%s' available", model)
        return True
    except Exception as e:
        logger.error("Cannot reach Ollama at localhost:11434: %s", e)
        logger.error("Start Ollama with: ollama serve")
        return False


def setup_git_repo(output_dir: str, repo_url: str | None, branch: str) -> None:
    """Initialize or clone a git repo in the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if (out / ".git").exists():
        logger.info("Git repo already exists at %s", out)
        return

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=str(out),
        capture_output=True,
        timeout=10,
    )
    if result.returncode == 0:
        logger.info("Output dir %s is inside an existing git repo — skipping clone", out)
        return

    if repo_url:
        logger.info("Cloning %s into %s", repo_url, out)
        subprocess.run(
            ["git", "clone", repo_url, str(out)],
            check=True,
            timeout=120,
        )
    else:
        logger.info("Initializing new git repo at %s", out)
        subprocess.run(["git", "init"], cwd=str(out), check=True, timeout=30)
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=str(out),
            capture_output=True,
            timeout=30,
        )


async def run_book_bible(
    runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    stream: bool = False,
    lang: str = "",
) -> str | None:
    """Generate Book Bible once before chapter writing."""
    from app.book_context import format_chapters_outline

    lang = lang or toc.get("language", "")
    if lang and lang.lower() != "en":
        lang_name = _language_name(lang)
        language_instruction = (
            f"\n\nIMPORTANT: Write the Book Bible in {lang_name}."
        )
    else:
        language_instruction = ""

    guidelines = toc.get("writing_guidelines", [])
    if guidelines:
        lines = "\n".join(f"- {g}" for g in guidelines)
        writing_guidelines = f"\n\nWriting guidelines:\n{lines}"
    else:
        writing_guidelines = ""

    state = {
        "book_title": toc["title"],
        "book_description": toc.get("description", ""),
        "chapters_outline": format_chapters_outline(toc),
        "language_instruction": language_instruction,
        "writing_guidelines": writing_guidelines,
    }

    session = await session_service.create_session(
        app_name="book-writer",
        user_id="book-writer",
        state=state,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Create the Book Bible for '{toc['title']}'.")],
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if stream else None
    final_text = ""
    async for event in runner.run_async(
        user_id="book-writer",
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
        if (
            event.content
            and event.content.parts
            and event.author == "book_bible_agent"
            and not getattr(event, "partial", False)
        ):
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    if not final_text:
        session = await session_service.get_session(
            app_name="book-writer",
            user_id="book-writer",
            session_id=session.id,
        )
        final_text = session.state.get("book_bible", "")

    return final_text or None


async def run_chapter_summary(
    runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    chapter: dict,
    chapter_text: str,
    book_bible: str,
    stream: bool = False,
    lang: str = "",
) -> str | None:
    """Summarize one completed chapter for later continuity."""
    from app.book_context import truncate_for_summary_agent

    lang = lang or toc.get("language", "")
    if lang and lang.lower() != "en":
        lang_name = _language_name(lang)
        language_instruction = f"\n\nWrite the summary in {lang_name}."
    else:
        language_instruction = ""

    state = {
        "book_title": toc["title"],
        "chapter_number": str(chapter["number"]),
        "chapter_title": chapter["title"],
        "book_bible": book_bible or "(No Book Bible.)",
        "chapter_text": truncate_for_summary_agent(chapter_text),
        "language_instruction": language_instruction,
    }

    session = await session_service.create_session(
        app_name="book-writer",
        user_id="book-writer",
        state=state,
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=f"Summarize Chapter {chapter['number']}: {chapter['title']}."
        )],
    )

    run_config = RunConfig(streaming_mode=StreamingMode.SSE) if stream else None
    final_text = ""
    async for event in runner.run_async(
        user_id="book-writer",
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
        if (
            event.content
            and event.content.parts
            and event.author == "chapter_summary_agent"
            and not getattr(event, "partial", False)
        ):
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    if not final_text:
        session = await session_service.get_session(
            app_name="book-writer",
            user_id="book-writer",
            session_id=session.id,
        )
        final_text = session.state.get("chapter_summary", "")

    return final_text or None


async def ensure_prior_chapter_summaries(
    summary_runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    output_dir: str,
    before_chapter: int,
    book_bible: str,
    stream: bool,
    lang: str,
    timeout: int,
) -> None:
    """Generate missing summaries for chapters before before_chapter."""
    from app.book_context import (
        list_chapters_needing_summary,
        read_chapter_body,
        save_chapter_summary,
    )

    for ch in list_chapters_needing_summary(output_dir, before_chapter, toc):
        body = read_chapter_body(output_dir, ch["number"])
        if not body:
            continue
        logger.info("Summarizing Chapter %d for continuity...", ch["number"])
        try:
            summary = await asyncio.wait_for(
                run_chapter_summary(
                    summary_runner,
                    session_service,
                    toc,
                    ch,
                    body,
                    book_bible,
                    stream=stream,
                    lang=lang,
                ),
                timeout=timeout,
            )
            if summary:
                save_chapter_summary(ch["number"], summary, output_dir)
                logger.info("Chapter %d summary saved", ch["number"])
            else:
                logger.warning("Chapter %d summary was empty", ch["number"])
        except asyncio.TimeoutError:
            logger.warning("Chapter %d summary timed out", ch["number"])
        except Exception:
            logger.exception("Chapter %d summary failed", ch["number"])


async def run_chapter(
    runner: Runner,
    session_service: InMemorySessionService,
    toc: dict,
    chapter: dict,
    output_dir: str,
    book_bible: str,
    stream: bool = False,
    lang: str = "",
) -> str | None:
    """Run the 4-phase pipeline for a single chapter. Returns the final content."""
    from app.book_context import build_previous_chapters_summary

    lang = lang or toc.get("language", "")
    if lang and lang.lower() != "en":
        lang_name = _language_name(lang)
        language_instruction = (
            f"\n\nIMPORTANT: You MUST write ALL content in {lang_name}. "
            f"Every sentence and paragraph must be in {lang_name}. "
            f"Do NOT write in English. Technical terms and code may remain in English, "
            f"but all explanatory text must be in {lang_name}."
        )
    else:
        language_instruction = ""

    guidelines = toc.get("writing_guidelines", [])
    if guidelines:
        lines = "\n".join(f"- {g}" for g in guidelines)
        writing_guidelines = f"\n\nWriting guidelines:\n{lines}"
    else:
        writing_guidelines = ""

    previous_summary = build_previous_chapters_summary(
        output_dir, chapter["number"], toc
    )

    state = {
        "book_title": toc["title"],
        "book_description": toc.get("description", ""),
        "book_bible": book_bible or "(No Book Bible available.)",
        "previous_chapters_summary": previous_summary,
        "current_chapter_number": str(chapter["number"]),
        "current_chapter_title": chapter["title"],
        "current_chapter_description": chapter.get("description", ""),
        "total_chapters": str(len(toc["chapters"])),
        "target_word_count": os.environ.get("CHAPTER_WORD_COUNT", "3000-5000"),
        "language_instruction": language_instruction,
        "writing_guidelines": writing_guidelines,
    }

    session = await session_service.create_session(
        app_name="book-writer",
        user_id="book-writer",
        state=state,
    )

    prompt = (
        f"Write Chapter {chapter['number']}: {chapter['title']}. "
        f"Description: {chapter.get('description', 'No additional description.')}"
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )

    agent_name_map = {
        "outline": "outline_agent",
        "writer": "writer_agent",
        "reviewer": "reviewer_agent",
        "finalizer": "finalizer_agent",
    }
    selected = os.environ.get("PIPELINE_AGENTS", "outline,writer,reviewer,finalizer").split(",")
    agent_order = [agent_name_map[s.strip()] for s in selected if s.strip() in agent_name_map]
    last_agent = agent_order[-1] if agent_order else "finalizer_agent"

    run_config = None
    if stream:
        run_config = RunConfig(streaming_mode=StreamingMode.SSE)

    final_text = ""
    current_author = None
    async for event in runner.run_async(
        user_id="book-writer",
        session_id=session.id,
        new_message=message,
        run_config=run_config,
    ):
        if stream:
            if getattr(event, "turn_complete", False) and current_author:
                idx = agent_order.index(current_author) if current_author in agent_order else -1
                if idx >= 0 and idx + 1 < len(agent_order):
                    next_agent = agent_order[idx + 1]
                    print(f"\n[{current_author} done → {next_agent} starting...]", flush=True)
                else:
                    print(f"\n[{current_author} done]", flush=True)

            if event.content and event.content.parts:
                if event.author != current_author:
                    if current_author is not None:
                        print(flush=True)
                    current_author = event.author
                    print(f"\n[{event.author}]", flush=True)

                if getattr(event, "partial", False):
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)

        if (
            event.content
            and event.content.parts
            and event.author == last_agent
            and not getattr(event, "partial", False)
        ):
            for part in event.content.parts:
                if part.text:
                    final_text = part.text

    if stream and current_author is not None:
        print(flush=True)

    if not final_text:
        session = await session_service.get_session(
            app_name="book-writer",
            user_id="book-writer",
            session_id=session.id,
        )
        output_keys = ["chapter_final", "chapter_review", "chapter_draft", "chapter_outline"]
        for key in output_keys:
            final_text = session.state.get(key, "")
            if final_text:
                break

    return final_text if final_text else None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overnight book writer",
        epilog=(
            "Examples:\n"
            "  python run_book.py --toc https://github.com/user/repo/blob/main/my-book/toc.json --model gemma4:31b\n"
            "  python run_book.py --toc toc.json --output-dir ./my-book --model llama3:8b\n"
            "  python run_book.py --toc https://github.com/user/repo/blob/main/my-book/toc.json --model gemma4:31b --resume\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--toc", required=True,
        help="Path or GitHub URL to table of contents file (e.g. https://github.com/user/repo/blob/main/book/toc.json)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-detected from GitHub URL)")
    parser.add_argument("--model", default=None, help="Ollama model name (e.g. gemma4:31b, llama3:8b)")
    parser.add_argument("--branch", default=None, help="Git branch (auto-detected from GitHub URL)")
    parser.add_argument("--repo", default=None, help="Git remote repo URL (auto-detected from GitHub URL)")
    parser.add_argument("--clone-dir", default="./repos", help="Base directory for cloned repos (default: ./repos)")
    parser.add_argument("--retry", type=int, default=3, help="Retries per chapter")
    parser.add_argument("--resume", action="store_true", help="Resume from progress")
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Timeout per chapter (seconds)"
    )
    parser.add_argument(
        "--words", default="3000-5000",
        help="Target word count range per chapter (default: 3000-5000)",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream LLM output to console in real-time",
    )
    parser.add_argument(
        "--no-think", action="store_true",
        help="Disable model thinking (recommended for qwen3 models)",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=32768,
        help="Context window size (default: 32768, use 4096-8192 for small models)",
    )
    parser.add_argument(
        "--repeat-penalty", type=float, default=1.2,
        help="Repetition penalty (default: 1.2, use 1.5+ for small models)",
    )
    parser.add_argument(
        "--agents", default="outline,writer,reviewer,finalizer,publisher",
        help="Comma-separated pipeline stages (default: outline,writer,reviewer,finalizer,publisher)",
    )
    parser.add_argument(
        "--lang", default=None,
        help="Language for the book content (e.g. my, es, fr). Overrides TOC language field.",
    )
    parser.add_argument(
        "--rewrite", type=int, nargs="+", metavar="N",
        help="Rewrite specific chapter(s) then stop (e.g. --rewrite 1 or --rewrite 1 3 5)",
    )
    parser.add_argument(
        "--rewrite-all", action="store_true",
        help="Rewrite all chapters from scratch, ignoring existing progress",
    )
    parser.add_argument(
        "--skip", type=int, nargs="+", metavar="N",
        help="Skip specific chapter(s) even during --rewrite-all (e.g. --skip 1 2)",
    )
    parser.add_argument(
        "--no-push", action="store_true", help="Skip git push (commit only)"
    )
    parser.add_argument(
        "--no-bible", action="store_true",
        help="Skip Book Bible generation (no book-wide consistency guide)",
    )
    parser.add_argument(
        "--regenerate-bible", action="store_true",
        help="Regenerate Book Bible even if book-bible.md exists",
    )
    parser.add_argument(
        "--no-chapter-summary", action="store_true",
        help="Skip per-chapter summary agent (no chapter-XX-summary.md)",
    )
    args = parser.parse_args()

    # Resolve GitHub URL or use local paths
    if is_github_url(args.toc):
        gh = resolve_github_toc(args.toc, clone_base=args.clone_dir)
        toc_path = gh["toc_path"]
        output_dir = args.output_dir or gh["output_dir"]
        branch = args.branch or gh["branch"]
        repo_url = args.repo or gh["repo_url"]
    else:
        toc_path = args.toc
        output_dir = args.output_dir or "./book"
        branch = args.branch or "main"
        repo_url = args.repo

    # When rewriting, rename the output folder if the book title changed
    if args.rewrite_all and Path(output_dir).exists():
        from app.tools import parse_toc as _parse_toc
        toc_for_rename = _parse_toc(toc_path)
        new_slug = slugify(toc_for_rename["title"], max_length=80)
        current_dir = Path(output_dir)
        if current_dir.name != new_slug:
            new_dir = current_dir.parent / new_slug
            if new_dir.exists():
                logger.warning(
                    "Cannot rename to '%s' — directory already exists", new_dir,
                )
            else:
                current_dir.rename(new_dir)
                output_dir = str(new_dir)
                # Update toc_path if it lived inside the old directory
                old_toc = Path(toc_path)
                if old_toc.parts[: len(current_dir.parts)] == current_dir.parts:
                    toc_path = str(new_dir / old_toc.relative_to(current_dir))
                logger.info(
                    "Renamed output folder: %s → %s", current_dir.name, new_slug,
                )

    setup_logging(output_dir)

    requested_agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    run_publisher = "publisher" in requested_agents
    llm_agents = [a for a in requested_agents if a != "publisher"]
    has_llm_agents = len(llm_agents) > 0

    if has_llm_agents and not args.model:
        parser.error("--model is required when running LLM agents")

    os.environ["PIPELINE_AGENTS"] = ",".join(llm_agents) if llm_agents else ""

    if has_llm_agents:
        os.environ["AGENT_MODEL"] = args.model
        os.environ["CHAPTER_WORD_COUNT"] = args.words
        os.environ["LLM_TIMEOUT"] = str(args.timeout)
        os.environ["NUM_CTX"] = str(args.num_ctx)
        os.environ["REPEAT_PENALTY"] = str(args.repeat_penalty)
        if args.no_think:
            os.environ["DISABLE_THINKING"] = "1"
        logger.info("Book Writer starting — model: %s", args.model)

        if not check_ollama(args.model):
            sys.exit(1)
    else:
        logger.info("Book Writer starting — publisher only (no LLM)")

    from app.tools import (
        git_commit_and_push_sync,
        load_progress,
        parse_toc,
        save_chapter_to_disk,
        save_progress,
    )

    toc = parse_toc(toc_path)
    lang = args.lang or toc.get("language", "")
    if lang:
        logger.info(
            "Loaded TOC: '%s' with %d chapters (language: %s)",
            toc["title"], len(toc["chapters"]), _language_name(lang),
        )
    else:
        logger.info(
            "Loaded TOC: '%s' with %d chapters", toc["title"], len(toc["chapters"])
        )
    logger.info("Output directory: %s", output_dir)

    if repo_url or not args.no_push:
        setup_git_repo(output_dir, repo_url, branch)

    completed: set[int] = set()
    total_words = 0
    start_time = time.time()

    if has_llm_agents:
        from app.agent import book_bible_agent, chapter_pipeline, chapter_summary_agent
        from app.book_context import (
            BOOK_BIBLE_FILENAME,
            load_book_bible,
            save_book_bible,
            save_chapter_summary,
        )

        session_service = InMemorySessionService()
        runner = Runner(
            agent=chapter_pipeline,
            app_name="book-writer",
            session_service=session_service,
        )
        summary_runner = None
        if not args.no_chapter_summary:
            summary_runner = Runner(
                agent=chapter_summary_agent,
                app_name="book-writer",
                session_service=session_service,
            )

        book_bible = ""
        if not args.no_bible:
            book_bible = load_book_bible(output_dir)
            if book_bible and args.regenerate_bible:
                book_bible = ""
            if not book_bible:
                logger.info("=" * 60)
                logger.info("SETUP: Book Bible")
                logger.info("=" * 60)
                bible_runner = Runner(
                    agent=book_bible_agent,
                    app_name="book-writer",
                    session_service=session_service,
                )
                try:
                    generated = await asyncio.wait_for(
                        run_book_bible(
                            bible_runner,
                            session_service,
                            toc,
                            stream=args.stream,
                            lang=args.lang or "",
                        ),
                        timeout=args.timeout,
                    )
                    if generated:
                        save_book_bible(generated, output_dir)
                        book_bible = generated
                        logger.info("Book Bible saved to %s", BOOK_BIBLE_FILENAME)
                    else:
                        logger.warning("Book Bible generation returned empty content")
                except asyncio.TimeoutError:
                    logger.warning("Book Bible generation timed out — continuing without it")
                except Exception:
                    logger.exception("Book Bible generation failed — continuing without it")
            else:
                logger.info("Using existing Book Bible (%s)", BOOK_BIBLE_FILENAME)
        else:
            logger.info("Book Bible skipped (--no-bible)")

        if args.rewrite_all:
            rewrite_set = {ch["number"] for ch in toc["chapters"]}
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
            }

        completed = set(progress.get("completed", []))

        out_path = Path(output_dir)
        for chapter in toc["chapters"]:
            ch_num = chapter["number"]
            if ch_num not in completed and list(out_path.glob(f"chapter-{ch_num:02d}-*.md")):
                completed.add(ch_num)
                if ch_num not in progress["completed"]:
                    progress["completed"].append(ch_num)
                    save_progress(output_dir, progress)

        logger.info("=" * 60)
        logger.info("Starting book generation: %s", toc["title"])
        logger.info("Chapters: %d | Already done: %d", len(toc["chapters"]), len(completed))
        if rewrite_set:
            logger.info("Rewriting chapter(s): %s", ", ".join(str(c) for c in sorted(rewrite_set)))
        logger.info("=" * 60)

        skip_set = set(args.skip) if args.skip else set()

        for chapter in toc["chapters"]:
            ch_num = chapter["number"]

            if ch_num in skip_set:
                logger.info("Skipping Chapter %d (--skip)", ch_num)
                continue

            if ch_num in completed:
                logger.info("Skipping Chapter %d (already complete)", ch_num)
                continue

            logger.info(
                "--- Chapter %d/%d: %s ---",
                ch_num,
                len(toc["chapters"]),
                chapter["title"],
            )

            progress["in_progress"] = ch_num
            save_progress(output_dir, progress)

            if summary_runner:
                await ensure_prior_chapter_summaries(
                    summary_runner,
                    session_service,
                    toc,
                    output_dir,
                    ch_num,
                    book_bible,
                    args.stream,
                    args.lang or "",
                    args.timeout,
                )

            content = None
            for attempt in range(1, args.retry + 1):
                try:
                    ch_start = time.time()
                    logger.info("Attempt %d/%d", attempt, args.retry)

                    content = await asyncio.wait_for(
                        run_chapter(
                            runner,
                            session_service,
                            toc,
                            chapter,
                            output_dir=output_dir,
                            book_bible=book_bible,
                            stream=args.stream,
                            lang=args.lang or "",
                        ),
                        timeout=args.timeout,
                    )

                    if content:
                        elapsed = time.time() - ch_start
                        logger.info(
                            "Chapter %d written in %.1f minutes", ch_num, elapsed / 60
                        )
                        break
                    else:
                        logger.warning("Chapter %d returned empty content", ch_num)

                except asyncio.TimeoutError:
                    logger.warning(
                        "Chapter %d timed out after %ds (attempt %d)",
                        ch_num,
                        args.timeout,
                        attempt,
                    )
                except Exception:
                    logger.exception("Chapter %d failed (attempt %d)", ch_num, attempt)

            if content:
                result = save_chapter_to_disk(
                    ch_num, chapter["title"], content, output_dir
                )
                logger.info(
                    "Saved: %s (%d words)", result["filename"], result["word_count"]
                )
                total_words += result["word_count"]

                if summary_runner:
                    try:
                        summary = await asyncio.wait_for(
                            run_chapter_summary(
                                summary_runner,
                                session_service,
                                toc,
                                chapter,
                                content,
                                book_bible,
                                stream=args.stream,
                                lang=args.lang or "",
                            ),
                            timeout=args.timeout,
                        )
                        if summary:
                            save_chapter_summary(ch_num, summary, output_dir)
                            logger.info("Chapter %d summary saved", ch_num)
                    except Exception:
                        logger.exception("Chapter %d summary failed (chapter still saved)", ch_num)

                if not args.no_push:
                    git_result = git_commit_and_push_sync(
                        ch_num, chapter["title"], output_dir, branch
                    )
                    if git_result["success"]:
                        pushed = "pushed" if git_result.get("pushed") else "committed only"
                        logger.info("Git: %s (%s)", git_result["message"], pushed)
                    else:
                        logger.warning("Git: %s", git_result["message"])

                progress["completed"].append(ch_num)
                completed.add(ch_num)
            else:
                logger.error(
                    "SKIPPING Chapter %d after %d failures", ch_num, args.retry
                )
                progress["failed"][str(ch_num)] = f"Failed after {args.retry} attempts"

            progress["in_progress"] = None
            save_progress(output_dir, progress)

    # --- Publisher ---
    pub_result = None
    if run_publisher:
        from app.tools import publish_to_pdf

        logger.info("=" * 60)
        logger.info("PUBLISHING: Converting chapters to PDF")
        logger.info("=" * 60)

        pub_result = publish_to_pdf(
            output_dir=output_dir,
            title=toc["title"],
            description=toc.get("description", ""),
        )

        if pub_result["success"]:
            logger.info(
                "PDF published: %s v%d (%d chapters, %d words)",
                pub_result["filename"],
                pub_result["version"],
                pub_result["total_chapters"],
                pub_result["total_words"],
            )
            git_result = git_commit_and_push_sync(
                0, "", output_dir, branch,
                message=f"Publish book PDF v{pub_result['version']}: {toc['title']}",
            )
            if git_result["success"]:
                pushed = "pushed" if git_result.get("pushed") else "committed only"
                logger.info("Git: %s (%s)", git_result["message"], pushed)
            else:
                logger.warning("Git: %s", git_result["message"])
        else:
            logger.error("PDF publishing failed: %s", pub_result["message"])

    elapsed_total = time.time() - start_time
    logger.info("=" * 60)
    logger.info("BOOK COMPLETE")
    logger.info("Title: %s", toc["title"])
    if has_llm_agents:
        logger.info("Chapters completed: %d/%d", len(completed), len(toc["chapters"]))
        logger.info("Failed: %d", len(progress.get("failed", {})))
        logger.info("Total words: %d", total_words)
    if pub_result and pub_result.get("success"):
        logger.info("PDF: %s", pub_result["filename"])
    logger.info("Total time: %.1f minutes", elapsed_total / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
