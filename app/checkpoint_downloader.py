"""Civitai checkpoint search and download into Forge models folder."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.model_manager import CHECKPOINT_EXTS, get_forge_models_dir

CIVITAI_API = "https://civitai.com/api/v1"
USER_AGENT = "book-writing-agent/1.0"
CIVITAI_TOKEN = os.environ.get("CIVITAI_API_TOKEN", "")

# SD 1.5 checkpoints suited for ~6GB VRAM (free on Civitai)
RECOMMENDED_CHECKPOINTS = [
    {
        "name": "DreamShaper 8",
        "version_id": 128713,
        "filename": "dreamshaper_8.safetensors",
        "size_gb": 2.0,
        "base_model": "SD 1.5",
        "note": "그림책·일러스트 추천",
    },
    {
        "name": "Realistic Vision V5.1 Hyper",
        "version_id": 501240,
        "filename": "realisticVisionV60B1_v51HyperVAE.safetensors",
        "size_gb": 2.0,
        "base_model": "SD 1.5",
        "note": "사실적 이미지",
    },
    {
        "name": "MeinaMix V12",
        "version_id": 948574,
        "filename": "meinamix_v12Final.safetensors",
        "size_gb": 2.0,
        "base_model": "SD 1.5",
        "note": "애니·일러스트",
    },
    {
        "name": "Anything V3",
        "version_id": 75,
        "filename": "anythingV3_fp16.safetensors",
        "size_gb": 2.0,
        "base_model": "SD 1.5",
        "note": "범용",
    },
]


def _api_get(path: str, params: dict | None = None) -> dict | list | None:
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    headers = {"User-Agent": USER_AGENT}
    if CIVITAI_TOKEN:
        headers["Authorization"] = f"Bearer {CIVITAI_TOKEN}"
    req = urllib.request.Request(f"{CIVITAI_API}{path}{query}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _pick_model_file(files: list[dict]) -> dict | None:
    for f in files:
        if f.get("type") != "Model":
            continue
        name = (f.get("name") or "").lower()
        if name.endswith(".safetensors"):
            return f
    for f in files:
        if f.get("type") == "Model":
            return f
    return files[0] if files else None


def _normalize_version(model: dict, version: dict) -> dict | None:
    files = version.get("files") or []
    file_info = _pick_model_file(files)
    if not file_info:
        return None
    filename = file_info.get("name") or f"model_{version['id']}.safetensors"
    if Path(filename).suffix.lower() not in {".safetensors", ".ckpt"}:
        filename = f"{filename}.safetensors"
    size_kb = file_info.get("sizeKB") or 0
    return {
        "model_id": model.get("id"),
        "version_id": version.get("id"),
        "name": model.get("name", "?"),
        "version_name": version.get("name", ""),
        "filename": filename,
        "size_gb": round(size_kb / (1024 * 1024), 2) if size_kb else None,
        "base_model": version.get("baseModel", ""),
        "nsfw": model.get("nsfw", False),
    }


def get_recommended_checkpoints() -> list[dict]:
    return RECOMMENDED_CHECKPOINTS.copy()


def search_checkpoints(query: str, limit: int = 15) -> dict:
    query = query.strip()
    if not query:
        return {"items": [], "message": "검색어를 입력하세요"}
    data = _api_get(
        "/models",
        {
            "query": query,
            "types": "Checkpoint",
            "limit": min(limit, 30),
            "sort": "Highest Rated",
        },
    )
    if not data or "items" not in data:
        return {"items": [], "message": "Civitai 검색 실패 (네트워크 또는 API 제한)"}

    items = []
    for model in data["items"]:
        versions = model.get("modelVersions") or []
        if not versions:
            continue
        entry = _normalize_version(model, versions[0])
        if entry:
            items.append(entry)
    return {"items": items, "message": ""}


def resolve_checkpoint(version_id: int) -> dict | None:
    for item in RECOMMENDED_CHECKPOINTS:
        if item["version_id"] == version_id:
            return item.copy()
    data = _api_get(f"/model-versions/{version_id}")
    if not data:
        return None
    model = data.get("model") or {}
    entry = _normalize_version(
        {
            "id": data.get("modelId") or model.get("id"),
            "name": model.get("name", "?"),
            "nsfw": model.get("nsfw", False),
        },
        data,
    )
    return entry


@dataclass
class DownloadTask:
    id: str
    version_id: int
    filename: str
    status: Literal["downloading", "completed", "failed"] = "downloading"
    progress_pct: float = 0.0
    downloaded_mb: float = 0.0
    total_mb: float = 0.0
    message: str = ""
    dest_path: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version_id": self.version_id,
            "filename": self.filename,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "downloaded_mb": self.downloaded_mb,
            "total_mb": self.total_mb,
            "message": self.message,
            "dest_path": self.dest_path,
        }


class CheckpointDownloadManager:
    def __init__(self) -> None:
        self._tasks: dict[str, DownloadTask] = {}
        self._lock = threading.Lock()

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def start_download(self, version_id: int, filename: str | None = None) -> dict:
        info = resolve_checkpoint(version_id)
        if not info:
            return {"success": False, "message": "모델 정보를 찾을 수 없습니다"}
        dest_name = filename or info["filename"]
        if Path(dest_name).suffix.lower() not in CHECKPOINT_EXTS:
            dest_name = f"{dest_name}.safetensors"

        models_dir = get_forge_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / dest_name
        if dest.exists():
            return {
                "success": False,
                "message": f"이미 존재합니다: {dest_name}",
            }

        task_id = uuid.uuid4().hex[:12]
        task = DownloadTask(
            id=task_id,
            version_id=version_id,
            filename=dest_name,
            dest_path=str(dest),
        )
        with self._lock:
            self._tasks[task_id] = task

        thread = threading.Thread(
            target=self._run_download,
            args=(task_id, version_id, dest),
            daemon=True,
        )
        thread.start()
        return {"success": True, "task_id": task_id, "filename": dest_name}

    def _run_download(self, task_id: str, version_id: int, dest: Path) -> None:
        params = {"type": "Model", "format": "SafeTensor"}
        if CIVITAI_TOKEN:
            params["token"] = CIVITAI_TOKEN
        url = f"https://civitai.com/api/download/models/{version_id}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": USER_AGENT}
        if CIVITAI_TOKEN:
            headers["Authorization"] = f"Bearer {CIVITAI_TOKEN}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                disp = resp.headers.get("Content-Disposition") or ""
                match = re.search(r'filename="?([^";]+)"?', disp)
                if match and not dest.exists():
                    suggested = match.group(1).strip()
                    if Path(suggested).suffix.lower() in CHECKPOINT_EXTS:
                        dest = dest.parent / suggested
                        with self._lock:
                            t = self._tasks.get(task_id)
                            if t:
                                t.filename = suggested
                                t.dest_path = str(dest)

                downloaded = 0
                chunk_size = 1024 * 1024
                with open(dest, "wb") as out:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        total_mb = total / (1024 * 1024) if total else 0
                        downloaded_mb = downloaded / (1024 * 1024)
                        pct = (downloaded / total * 100) if total else 0
                        with self._lock:
                            t = self._tasks.get(task_id)
                            if t:
                                t.downloaded_mb = round(downloaded_mb, 1)
                                t.total_mb = round(total_mb, 1)
                                t.progress_pct = round(pct, 1)

            with self._lock:
                t = self._tasks.get(task_id)
                if t:
                    t.status = "completed"
                    t.progress_pct = 100.0
                    t.message = "다운로드 완료"
        except Exception as e:
            if dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            with self._lock:
                t = self._tasks.get(task_id)
                if t:
                    t.status = "failed"
                    msg = str(e)
                    if "401" in msg or "403" in msg:
                        msg += " — Civitai API 토큰이 필요할 수 있습니다 (CIVITAI_API_TOKEN)"
                    t.message = msg


download_manager = CheckpointDownloadManager()
