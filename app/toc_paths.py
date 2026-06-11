"""Conventions for TOC (목차) file locations."""

from __future__ import annotations

from pathlib import Path

TOC_DIR = "toc"

# Legacy: sample files and older TOCs may live at project root.
ROOT_TOC_PATTERNS = (
    "sample-toc.json",
    "sample-picture-toc.json",
    "*-toc.json",
    "*toc*.json",
)


def ensure_toc_dir(project_root: Path) -> Path:
    path = project_root / TOC_DIR
    path.mkdir(exist_ok=True)
    return path


def discover_toc_files(project_root: Path) -> list[Path]:
    """Find TOC JSON files at project root (legacy) and under toc/ (recursive)."""
    root = project_root.resolve()
    by_rel: dict[str, Path] = {}

    for pattern in ROOT_TOC_PATTERNS:
        for p in root.glob(pattern):
            if p.is_file():
                rel = str(p.resolve().relative_to(root))
                by_rel[rel] = p

    toc_root = root / TOC_DIR
    if toc_root.is_dir():
        for p in toc_root.rglob("*.json"):
            if p.is_file():
                rel = str(p.resolve().relative_to(root))
                by_rel[rel] = p

    return sorted(by_rel.values(), key=lambda p: str(p).lower())
