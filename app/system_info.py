"""System specs and remote model discovery (Ollama / Forge)."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
FORGE_API_URL = os.environ.get("FORGE_API_URL", "http://localhost:7860")


def _http_get_json(url: str, timeout: int = 10) -> dict | list | None:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _http_post_json(url: str, payload: dict | None = None, timeout: int = 15) -> bool:
    try:
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _refresh_forge_checkpoints(api_url: str) -> None:
    _http_post_json(f"{api_url.rstrip('/')}/sdapi/v1/refresh-checkpoints")


def _disk_forge_checkpoints() -> list[dict]:
    from app.model_manager import list_forge_checkpoints

    return list_forge_checkpoints().get("checkpoints", [])


def get_ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                return round(mem.ullTotalPhys / (1024 ** 3), 1)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    return None


def get_gpu_info() -> list[dict]:
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "vram_total_mb": int(float(parts[1])),
                        "vram_free_mb": int(float(parts[2])),
                    })
    except Exception:
        pass
    return gpus


def get_disk_free_gb(path: str = ".") -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024 ** 3), 1)


def get_system_info() -> dict:
    gpus = get_gpu_info()
    ollama_ok = _http_get_json(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) is not None
    forge_data = _http_get_json(f"{FORGE_API_URL}/sdapi/v1/sd-models", timeout=5)
    forge_ok = forge_data is not None
    forge_count = len(forge_data) if isinstance(forge_data, list) else 0

    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": get_ram_gb(),
        "gpus": gpus,
        "disk_free_gb": get_disk_free_gb(),
        "ollama": {
            "url": OLLAMA_BASE_URL,
            "online": ollama_ok,
            "note": "SSH 터널(11434)로 서버 Ollama 연결" if OLLAMA_BASE_URL.endswith("11434") else "",
        },
        "forge": {
            "url": FORGE_API_URL,
            "online": forge_ok,
            "checkpoint_count": forge_count,
        },
    }


def _model_name_matches(requested: str, running_name: str) -> bool:
    if not requested or not running_name:
        return False
    if running_name == requested:
        return True
    # Ollama may report "gemma4:31b" while request uses same, or with digest suffix
    return running_name.startswith(requested + ":") or requested.startswith(running_name + ":")


def list_ollama_running(base_url: str = OLLAMA_BASE_URL, model: str = "") -> dict:
    data = _http_get_json(f"{base_url.rstrip('/')}/api/ps", timeout=3)
    if data is None:
        return {"online": False, "models": [], "running": False, "requested_model": model}
    models = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model") or ""
        models.append({
            "name": name,
            "size_vram": m.get("size_vram"),
        })
    running = any(_model_name_matches(model, m["name"]) for m in models) if model else bool(models)
    return {
        "online": True,
        "models": models,
        "requested_model": model,
        "running": running,
        "url": base_url,
    }


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> dict:
    data = _http_get_json(f"{base_url.rstrip('/')}/api/tags")
    if not data:
        return {"online": False, "models": [], "url": base_url}
    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        size_gb = round(m.get("size", 0) / (1024 ** 3), 1)
        models.append({"name": name, "size_gb": size_gb})
    return {"online": True, "models": models, "url": base_url}


def list_image_models(
    backend: str = "automatic1111",
    api_url: str = FORGE_API_URL,
) -> dict:
    if backend == "automatic1111":
        base = api_url.rstrip("/")
        data = _http_get_json(f"{base}/sdapi/v1/sd-models")
        if data is None:
            disk = _disk_forge_checkpoints()
            if disk:
                models = [
                    {"name": f["name"], "title": f["name"]}
                    for f in disk
                ]
                return {
                    "online": False,
                    "backend": backend,
                    "models": models,
                    "url": api_url,
                    "source": "disk",
                    "message": "Forge 연결 안 됨 — 폴더의 체크포인트만 표시",
                }
            return {"online": False, "backend": backend, "models": [], "url": api_url}

        if isinstance(data, list) and not data:
            _refresh_forge_checkpoints(api_url)
            data = _http_get_json(f"{base}/sdapi/v1/sd-models") or []

        models = [
            {"name": m.get("model_name", m.get("title", "?")), "title": m.get("title", "")}
            for m in (data if isinstance(data, list) else [])
        ]
        source = "forge_api"
        message = ""

        if not models:
            disk = _disk_forge_checkpoints()
            if disk:
                models = [{"name": f["name"], "title": f["name"]} for f in disk]
                source = "disk"
                message = "Forge 목록 갱신 필요 — 폴더의 체크포인트 표시 중"

        return {
            "online": True,
            "backend": backend,
            "models": models,
            "url": api_url,
            "source": source,
            "message": message,
        }

    if backend == "diffusers":
        return {
            "online": True,
            "backend": backend,
            "models": [
                {"name": "runwayml/stable-diffusion-v1-5", "title": "SD 1.5 (default)"},
            ],
            "url": "local",
        }

    if backend == "ollama":
        result = list_ollama_models(api_url.replace("/sdapi", "") if "/sdapi" in api_url else OLLAMA_BASE_URL)
        image_models = [m for m in result["models"] if "flux" in m["name"].lower() or "image" in m["name"].lower()]
        return {"online": result["online"], "backend": backend, "models": image_models, "url": OLLAMA_BASE_URL}

    return {"online": False, "backend": backend, "models": [], "url": api_url}


def list_toc_files(project_root: Path) -> list[dict]:
    from app.toc_paths import discover_toc_files

    root = project_root.resolve()
    files = []
    for p in discover_toc_files(project_root):
        if not p.is_file():
            continue
        rel_path = str(p.resolve().relative_to(root)).replace("\\", "/")
        try:
            from app.toc_io import repair_toc_file_pipeline

            repair_toc_file_pipeline(project_root, p)
            data = json.loads(p.read_text(encoding="utf-8"))
            book_type = data.get("type")
            if book_type not in ("picture_book", "text_book"):
                book_type = "picture_book" if data.get("pages") else "text_book"
            if book_type == "picture_book":
                count = len(data.get("pages", []))
                unit = "pages"
            else:
                count = len(data.get("chapters", []))
                unit = "chapters"
            from app.pipeline_config import pipeline_from_toc

            files.append({
                "path": rel_path,
                "name": p.name,
                "title": data.get("title", p.stem),
                "type": book_type,
                "count": count,
                "unit": unit,
                "pipeline": pipeline_from_toc(data),
            })
        except Exception:
            files.append({
                "path": rel_path,
                "name": p.name,
                "title": p.stem,
                "type": "unknown",
                "count": 0,
                "unit": "",
            })
    seen = set()
    unique = []
    for f in files:
        if f["path"] not in seen:
            seen.add(f["path"])
            unique.append(f)
    return unique
