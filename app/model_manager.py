"""Manage text (Ollama) and image (Forge checkpoint) models."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from app.system_info import FORGE_API_URL, OLLAMA_BASE_URL, list_ollama_models

CHECKPOINT_EXTS = {".safetensors", ".ckpt", ".pt", ".pth"}


def get_forge_models_dir() -> Path:
    if env := os.environ.get("FORGE_MODELS_DIR"):
        return Path(env)
    home = Path.home()
    for name in ("stable-diffusion-webui-forge", "stable-diffusion-webui"):
        p = home / name / "models" / "Stable-diffusion"
        if p.exists():
            return p
    return home / "stable-diffusion-webui-forge" / "models" / "Stable-diffusion"


def list_forge_checkpoints() -> dict:
    models_dir = get_forge_models_dir()
    checkpoints = []
    if models_dir.exists():
        for f in sorted(models_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in CHECKPOINT_EXTS:
                checkpoints.append({
                    "name": f.name,
                    "path": str(f),
                    "size_gb": round(f.stat().st_size / (1024 ** 3), 2),
                })

    return {
        "models_dir": str(models_dir),
        "exists": models_dir.exists(),
        "checkpoints": checkpoints,
        "forge_url": FORGE_API_URL,
        "download_hint": "Civitai에서 .safetensors 받아 models/Stable-diffusion/ 에 넣거나 Forge UI에서 다운로드",
    }


def delete_forge_checkpoint(filename: str) -> dict:
    models_dir = get_forge_models_dir().resolve()
    target = (models_dir / filename).resolve()

    if not str(target).startswith(str(models_dir)):
        return {"success": False, "message": "Invalid path"}
    if not target.exists():
        return {"success": False, "message": "File not found"}
    if target.suffix.lower() not in CHECKPOINT_EXTS:
        return {"success": False, "message": "Not a checkpoint file"}

    target.unlink()
    return {"success": True, "message": f"Deleted {filename}"}


def delete_ollama_model(name: str, base_url: str = OLLAMA_BASE_URL) -> dict:
    payload = json.dumps({"name": name}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/delete",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return {"success": True, "message": f"Deleted {name}"}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else str(e)
        return {"success": False, "message": body}
    except Exception as e:
        return {"success": False, "message": str(e)}


def get_models_overview() -> dict:
    text = list_ollama_models()
    image = list_forge_checkpoints()
    return {
        "text": text,
        "image": image,
        "links": {
            "forge_ui": FORGE_API_URL,
            "civitai": "https://civitai.com",
            "ollama_library": "https://ollama.com/library",
        },
    }
