"""Generate TOC JSON files via Ollama from user inputs."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("book-agent")

from slugify import slugify

from app.pipeline_config import attach_default_pipeline
from app.system_info import OLLAMA_BASE_URL

TEXT_TOC_SCHEMA = """{
  "title": "책 제목",
  "description": "책 전체 설명 (2-3문장)",
  "chapters": [
    {"number": 1, "title": "장 제목", "description": "이 장에서 다룰 내용"}
  ]
}"""

PICTURE_TOC_SCHEMA = """{
  "type": "picture_book",
  "title": "그림책 제목",
  "description": "줄거리 요약",
  "language": "ko",
  "target_age": "대상 연령 (예: 3-5, 10-14, 18세 이상)",
  "style": "일러스트 화풍 (대상 연령·장르에 맞게)",
  "writing_guidelines": ["대상 연령에 맞는 문체·분량"],
  "characters": [
    {"name": "캐릭터명", "description": "외모·특징"}
  ],
  "pages": [
    {"number": 1, "scene": "장면 설명", "mood": "분위기"}
  ]
}"""


def _ollama_chat(
    model: str,
    prompt: str,
    system: str,
    base_url: str = OLLAMA_BASE_URL,
    *,
    num_predict: int = 4096,
    timeout: float = 180,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": num_predict,
            "temperature": 0.3,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode() if e.fp else str(e)
        raise RuntimeError(f"Ollama 오류: {detail}") from e
    except TimeoutError as e:
        raise RuntimeError(
            f"Ollama 응답 시간 초과 ({int(timeout)}초). "
            f"큰 모델({model})은 SSH 터널에서 수 분 걸릴 수 있습니다."
        ) from e
    except Exception as e:
        err = str(e).lower()
        if "timed out" in err or "timeout" in err:
            raise RuntimeError(
                f"Ollama 응답 시간 초과 ({int(timeout)}초). "
                f"큰 모델({model})은 SSH 터널에서 수 분 걸릴 수 있습니다."
            ) from e
        raise RuntimeError(f"Ollama 연결 실패: {e}") from e

    message = body.get("message") or {}
    content = message.get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama가 빈 응답을 반환했습니다")
    return content


def _close_truncated_json(text: str) -> str:
    """Append closing quotes/brackets when the model output was cut off."""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()

    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return text + suffix


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start > 0:
        text = text[start:]

    candidates = [text, _close_truncated_json(text)]
    last_err: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e

    raise ValueError(f"JSON 파싱 실패: {last_err}") from last_err


def _validate_text_toc(toc: dict, count: int) -> dict:
    if not toc.get("title"):
        raise ValueError("title이 없습니다")
    chapters = toc.get("chapters") or []
    if not chapters:
        raise ValueError("chapters가 비어 있습니다")
    normalized = []
    for i, ch in enumerate(chapters[:count], start=1):
        normalized.append({
            "number": ch.get("number", i),
            "title": str(ch.get("title", f"Chapter {i}")).strip(),
            "description": str(ch.get("description", "")).strip(),
        })
    while len(normalized) < count:
        n = len(normalized) + 1
        normalized.append({
            "number": n,
            "title": f"Chapter {n}",
            "description": "",
        })
    result = {
        "title": str(toc["title"]).strip(),
        "description": str(toc.get("description", "")).strip(),
        "chapters": normalized[:count],
    }
    return attach_default_pipeline(result, "text")


def _coalesce_picture_pages(toc: dict) -> list:
    """Recover page list when the model uses another key or the text-book schema."""
    pages = toc.get("pages")
    if isinstance(pages, list) and pages:
        return pages
    for key in ("page_list", "scenes", "storyboard"):
        alt = toc.get(key)
        if isinstance(alt, list) and alt:
            return alt
    chapters = toc.get("chapters")
    if isinstance(chapters, list) and chapters:
        return [
            {
                "number": ch.get("number", i),
                "scene": str(
                    ch.get("scene") or ch.get("title") or ch.get("description") or f"Page {i}"
                ).strip(),
                "mood": str(ch.get("mood", "warm")).strip(),
            }
            for i, ch in enumerate(chapters, start=1)
        ]
    return []


def _validate_picture_toc(toc: dict, count: int, language: str, target_age: str) -> dict:
    if not toc.get("title"):
        raise ValueError("title이 없습니다")
    pages = _coalesce_picture_pages(toc)
    if not pages:
        logger.warning(
            "Picture TOC from LLM had no pages; using %d placeholder page(s)",
            count,
        )
        desc = str(toc.get("description", "")).strip()
        title = str(toc.get("title", "Page")).strip()
        pages = [
            {
                "number": i,
                "scene": (desc[:80] if i == 1 and desc else f"{title} — 장면 {i}"),
                "mood": "warm",
            }
            for i in range(1, count + 1)
        ]
    normalized_pages = []
    for i, pg in enumerate(pages[:count], start=1):
        normalized_pages.append({
            "number": pg.get("number", i),
            "scene": str(pg.get("scene", "")).strip() or f"Page {i} scene",
            "mood": str(pg.get("mood", "")).strip() or "warm",
        })
    while len(normalized_pages) < count:
        n = len(normalized_pages) + 1
        normalized_pages.append({
            "number": n,
            "scene": f"Page {n}",
            "mood": "warm",
        })

    characters = toc.get("characters") or []
    if not characters:
        characters = [{"name": "주인공", "description": "이야기의 주인공"}]

    result = {
        "type": "picture_book",
        "title": str(toc["title"]).strip(),
        "description": str(toc.get("description", "")).strip(),
        "language": toc.get("language") or language,
        "target_age": toc.get("target_age") or target_age,
        "style": toc.get("style") or "illustrated book art suited to the target audience",
        "writing_guidelines": toc.get("writing_guidelines") or [
            f"Match tone, vocabulary, and page text length to target audience ({target_age})",
        ],
        "characters": characters,
        "pages": normalized_pages[:count],
    }
    return attach_default_pipeline(result, "picture")


def _build_prompt(
    book_type: str,
    title: str,
    topic: str,
    count: int,
    language: str,
    target_age: str,
    extra_notes: str,
) -> tuple[str, str]:
    if book_type == "picture":
        system = (
            "You are an illustrated-book planner (text + art per page; not always for toddlers). "
            "Respond with ONLY valid JSON matching the schema. No markdown."
        )
        prompt = f"""그림책 TOC(JSON)를 작성하세요. (페이지마다 글 + 일러스트가 붙는 책)

사용자 입력:
- 제목: {title}
- 주제/줄거리: {topic}
- 페이지 수: {count}
- 언어: {language}
- 대상 연령: {target_age}
{f'- 추가 요청: {extra_notes}' if extra_notes else ''}

스키마:
{PICTURE_TOC_SCHEMA}

규칙:
- type 필드는 반드시 "picture_book"
- pages 배열은 반드시 포함, 정확히 {count}개 (비우지 말 것)
- 각 page에 number, scene(장면), mood(분위기) 포함
- scene/mood는 각 40자 이내로 짧게
- characters 1~3명
- language 필드는 "{language}"
- target_age는 "{target_age}"
- style·writing_guidelines는 대상 연령({target_age})과 장르에 맞게 (유아용으로 고정하지 말 것)
- JSON만 출력, 마크다운 금지, 응답을 끝까지 완성할 것
"""
        return system, prompt

    system = (
        "You are a non-fiction book planner. "
        "Respond with ONLY valid JSON matching the schema. No markdown."
    )
    prompt = f"""텍스트 책 TOC(JSON)를 작성하세요.

사용자 입력:
- 제목: {title}
- 주제: {topic}
- 장(챕터) 수: {count}
{f'- 추가 요청: {extra_notes}' if extra_notes else ''}

스키마:
{TEXT_TOC_SCHEMA}

규칙:
- chapters는 정확히 {count}개
- 각 chapter에 number, title, description 포함
- description은 1문장, 80자 이내 (짧게)
- JSON만 출력, 따옴표는 반드시 이스케이프, 마크다운 금지
- 응답을 끝까지 완성할 것 (잘리면 안 됨)
"""
    return system, prompt


def generate_and_save_toc(
    project_root: Path,
    book_type: str,
    title: str,
    topic: str,
    count: int,
    text_model: str,
    language: str = "ko",
    target_age: str = "3-5",
    extra_notes: str = "",
    pipeline: dict | None = None,
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    title = title.strip()
    topic = topic.strip()
    if not title:
        raise ValueError("책 제목을 입력하세요")
    if not topic:
        raise ValueError("주제/줄거리를 입력하세요")
    count = max(1, min(count, 30))

    system, prompt = _build_prompt(
        book_type, title, topic, count, language, target_age, extra_notes
    )
    per_item = 700 if book_type == "picture" else 500
    num_predict = max(3072, min(12288, count * per_item + 1536))
    ollama_timeout = max(300, min(900, 180 + count * 25))
    model_l = text_model.lower()
    if any(tag in model_l for tag in ("31b", "32b", "30b", "27b", "70b")):
        ollama_timeout = max(ollama_timeout, 600)
    last_error: Exception | None = None
    toc: dict | None = None

    logger.info(
        "TOC generate: model=%s type=%s count=%d timeout=%ds",
        text_model,
        book_type,
        count,
        ollama_timeout,
    )

    for attempt in range(1, 4):
        try:
            retry_prompt = prompt
            if attempt > 1:
                if book_type == "picture":
                    retry_prompt += (
                        f"\n\n[재시도 {attempt}] pages 배열이 비었거나 JSON이 잘렸습니다. "
                        f"type은 picture_book, pages는 정확히 {count}개(scene, mood)로 "
                        "완전한 JSON만 출력하세요. scene/mood는 각 40자 이내로 짧게."
                    )
                else:
                    retry_prompt += (
                        f"\n\n[재시도 {attempt}] chapters 배열이 비었거나 JSON이 잘렸습니다. "
                        f"chapters는 정확히 {count}개, description은 40자 이내로 "
                        "완전한 JSON만 출력하세요."
                    )
            raw = _ollama_chat(
                text_model,
                retry_prompt,
                system,
                base_url,
                num_predict=num_predict,
                timeout=ollama_timeout,
            )
            parsed = _extract_json(raw)
            if book_type == "picture":
                toc = _validate_picture_toc(parsed, count, language, target_age)
            else:
                toc = _validate_text_toc(parsed, count)
            break
        except (ValueError, RuntimeError) as e:
            last_error = e
            logger.warning("TOC generate attempt %d failed: %s", attempt, e)
            if attempt >= 3:
                raise ValueError(str(e)) from e

    if toc is None:
        raise ValueError(str(last_error or "목차 생성 실패"))

    if pipeline:
        from app.pipeline_config import normalize_pipeline

        toc["pipeline"] = normalize_pipeline(book_type, pipeline)

    from app.toc_paths import ensure_toc_dir

    slug = slugify(title, max_length=50) or "book"
    filename = f"{slug}-toc.json"
    dest_dir = ensure_toc_dir(project_root)
    dest = dest_dir / filename
    if dest.exists():
        n = 2
        while (dest_dir / f"{slug}-toc-{n}.json").exists():
            n += 1
        filename = f"{slug}-toc-{n}.json"
        dest = dest_dir / filename

    dest.write_text(json.dumps(toc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    unit = "pages" if book_type == "picture" else "chapters"
    item_count = len(toc.get("pages") or toc.get("chapters") or [])
    try:
        rel_path = str(dest.resolve().relative_to(project_root.resolve()))
    except ValueError:
        rel_path = dest.name
    return {
        "path": rel_path,
        "filename": filename,
        "title": toc["title"],
        "type": "picture_book" if book_type == "picture" else "text_book",
        "count": item_count,
        "unit": unit,
        "toc": toc,
    }
