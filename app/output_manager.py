"""List and clean up generated book outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAFE_ROOTS = [PROJECT_ROOT / "book", PROJECT_ROOT / "picture-book"]


def _is_safe(path: Path) -> bool:
    resolved = path.resolve()
    for root in SAFE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def list_outputs() -> list[dict]:
    outputs = []
    for root in SAFE_ROOTS:
        if not root.exists():
            continue
        for item in sorted(root.iterdir()):
            if not item.is_dir():
                continue
            progress_path = item / ".progress.json"
            progress = {}
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            chapters = list(item.glob("chapter-*.md"))
            pages = list(item.glob("page-*.json"))
            images = list((item / "images").glob("*.png")) if (item / "images").exists() else []
            pdfs = sorted(
                item.glob("*.pdf"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)

            outputs.append({
                "id": f"{root.name}/{item.name}",
                "path": str(item),
                "name": item.name,
                "type": "picture" if root.name == "picture-book" else "text",
                "completed": len(progress.get("completed", [])),
                "failed": len(progress.get("failed", {})),
                "in_progress": progress.get("in_progress"),
                "chapters": len(chapters),
                "pages": len(pages),
                "images": len(images),
                "pdfs": len(pdfs),
                "latest_pdf": str(pdfs[0]) if pdfs else "",
                "latest_pdf_name": pdfs[0].name if pdfs else "",
                "size_mb": round(size_mb, 1),
            })
    return outputs


def _rmtree_windows(path: Path) -> list[str]:
    locked: list[str] = []

    def onerror(func, p, exc_info):
        import os
        import stat

        exc = exc_info[1]
        if isinstance(exc, PermissionError):
            locked.append(p)
            return
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            locked.append(p)

    shutil.rmtree(path, onerror=onerror)
    return locked


def delete_output(path: str) -> dict:
    target = Path(path).resolve()
    if not _is_safe(target):
        return {"success": False, "message": "Path not allowed"}
    if not target.exists():
        return {"success": False, "message": "Not found"}
    try:
        locked = _rmtree_windows(target)
    except Exception as e:
        return {
            "success": False,
            "message": (
                f"삭제 실패: {e}. "
                "책 생성 작업이 실행 중이거나 로그 파일이 에디터에서 열려 있으면 "
                "작업 중지 후 파일을 닫고 다시 시도하세요."
            ),
        }
    if locked:
        remaining = target.exists()
        names = ", ".join(Path(p).name for p in locked[:3])
        if remaining:
            return {
                "success": False,
                "message": (
                    f"일부 파일이 사용 중이라 삭제하지 못했습니다 ({names}). "
                    "책 생성을 중지하고 picture-book.log 탭을 닫은 뒤 다시 시도하세요."
                ),
            }
    return {"success": True, "message": f"Deleted {target.name}"}


def list_output_files(path: str) -> list[dict]:
    target = Path(path).resolve()
    if not _is_safe(target):
        return []
    files = []
    for f in sorted(target.rglob("*")):
        if f.is_file() and f.name != ".progress.json":
            rel = f.relative_to(target)
            files.append({
                "path": str(f),
                "name": str(rel),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "type": f.suffix.lstrip("."),
            })
    return files
