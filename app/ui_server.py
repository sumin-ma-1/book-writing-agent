"""Book Writing Agent — Web UI server."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.job_manager import job_manager
from app.checkpoint_downloader import (
    download_manager,
    get_recommended_checkpoints,
    search_checkpoints,
)
from app.toc_generator import generate_and_save_toc
from app.pipeline_config import (
    DEFAULT_PICTURE_PIPELINE,
    DEFAULT_TEXT_PIPELINE,
    FINISH_BLOCKS,
    PICTURE_AGENT_BLOCKS,
    PICTURE_EXTRA_BLOCKS,
    PICTURE_SETUP_BLOCKS,
    TEXT_AGENT_BLOCKS,
    TEXT_SETUP_BLOCKS,
    default_pipeline,
)
from app.toc_io import (
    export_toc_path,
    import_toc_file,
    load_toc_file,
    save_toc_content,
    save_toc_pipeline,
)
from app.model_manager import (
    delete_forge_checkpoint,
    delete_ollama_model,
    get_models_overview,
    list_forge_checkpoints,
)
from app.output_manager import delete_output, list_output_files, list_outputs
from app.system_info import (
    FORGE_API_URL,
    OLLAMA_BASE_URL,
    get_system_info,
    list_image_models,
    list_ollama_models,
    list_ollama_running,
    list_toc_files,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Book Writing Agent UI", version="0.1.0")


class StartJobRequest(BaseModel):
    book_type: str = "text"
    toc_path: str
    text_model: str
    image_backend: str = "automatic1111"
    image_model: str = ""
    output_dir: str = ""
    resume: bool = False
    rewrite_all: bool = False
    rewrite: list[int] | None = None
    pipeline: dict | None = None


class DeleteRequest(BaseModel):
    path: str


class DeleteModelRequest(BaseModel):
    name: str
    type: str = "image"


class DownloadCheckpointRequest(BaseModel):
    version_id: int
    filename: str = ""


class GenerateTocRequest(BaseModel):
    book_type: str = "text"
    title: str
    topic: str
    count: int = 5
    language: str = "ko"
    target_age: str = "3-5"
    extra_notes: str = ""
    text_model: str
    pipeline: dict | None = None


class SavePipelineRequest(BaseModel):
    path: str
    pipeline: dict


class SaveTocRequest(BaseModel):
    path: str
    toc: dict


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon = STATIC_DIR / "favicon-32.png"
    if not icon.is_file():
        icon = PROJECT_ROOT / "bookagent_nobg.png"
    if not icon.is_file():
        raise HTTPException(404, "Favicon not found")
    return FileResponse(icon, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "UI not found")
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/system")
def api_system():
    return get_system_info()


@app.get("/api/models/text")
def api_text_models(url: str = OLLAMA_BASE_URL):
    return list_ollama_models(url)


@app.get("/api/models/running")
def api_models_running(model: str = "", url: str = OLLAMA_BASE_URL):
    return list_ollama_running(url, model)


@app.get("/api/models/image")
def api_image_models(
    backend: str = "automatic1111",
    api_url: str = FORGE_API_URL,
):
    return list_image_models(backend, api_url)


@app.get("/api/models/manage")
def api_models_manage():
    return get_models_overview()


@app.get("/api/models/checkpoints")
def api_checkpoints():
    return list_forge_checkpoints()


@app.get("/api/models/checkpoints/recommended")
async def api_checkpoints_recommended():
    return {"items": get_recommended_checkpoints()}


@app.get("/api/models/checkpoints/search")
async def api_checkpoints_search(q: str = "", limit: int = 15):
    return search_checkpoints(q, limit=limit)


@app.post("/api/models/checkpoints/download")
async def api_checkpoint_download(req: DownloadCheckpointRequest):
    result = download_manager.start_download(req.version_id, req.filename or None)
    if not result.get("success"):
        raise HTTPException(400, result.get("message", "Download failed"))
    return result


@app.get("/api/models/checkpoints/download/{task_id}")
async def api_checkpoint_download_status(task_id: str):
    task = download_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Download task not found")
    return task


@app.delete("/api/models")
async def api_delete_model(req: DeleteModelRequest):
    if req.type == "text":
        result = delete_ollama_model(req.name)
    elif req.type == "image":
        result = delete_forge_checkpoint(req.name)
    else:
        raise HTTPException(400, "type must be text or image")
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.get("/api/toc-files")
def api_toc_files():
    return list_toc_files(PROJECT_ROOT)


@app.get("/api/pipeline/schema")
async def api_pipeline_schema():
    return {
        "default_text": DEFAULT_TEXT_PIPELINE,
        "default_picture": DEFAULT_PICTURE_PIPELINE,
        "setup_text": TEXT_SETUP_BLOCKS,
        "setup_picture": PICTURE_SETUP_BLOCKS,
        "agents_text": TEXT_AGENT_BLOCKS,
        "agents_picture": PICTURE_AGENT_BLOCKS,
        "finish": FINISH_BLOCKS,
        "extra_picture": PICTURE_EXTRA_BLOCKS,
    }


@app.get("/api/toc/content")
def api_toc_content(path: str):
    try:
        return load_toc_file(PROJECT_ROOT, path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/toc/save")
def api_save_toc(req: SaveTocRequest):
    try:
        return save_toc_content(PROJECT_ROOT, req.path, req.toc)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/toc/export")
async def api_export_toc(path: str):
    try:
        target = export_toc_path(PROJECT_ROOT, path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return FileResponse(target, filename=target.name, media_type="application/json")


@app.post("/api/toc/pipeline")
def api_save_pipeline(req: SavePipelineRequest):
    try:
        return save_toc_pipeline(PROJECT_ROOT, req.path, req.pipeline)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/toc/import")
async def api_import_toc(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "파일이 없습니다")
    content = await file.read()
    try:
        return import_toc_file(PROJECT_ROOT, content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/toc/generate")
async def api_generate_toc(req: GenerateTocRequest):
    if req.book_type not in ("text", "picture"):
        raise HTTPException(400, "book_type must be text or picture")
    if not req.text_model:
        raise HTTPException(400, "text_model is required")
    try:
        pipeline = req.pipeline or default_pipeline(req.book_type)
        return await run_in_threadpool(
            generate_and_save_toc,
            PROJECT_ROOT,
            book_type=req.book_type,
            title=req.title,
            topic=req.topic,
            count=req.count,
            text_model=req.text_model,
            language=req.language,
            target_age=req.target_age,
            extra_notes=req.extra_notes,
            pipeline=pipeline,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.get("/api/outputs")
def api_outputs():
    return list_outputs()


@app.get("/api/outputs/files")
def api_output_files(path: str):
    return list_output_files(path)


@app.delete("/api/outputs")
async def api_delete_output(req: DeleteRequest):
    job_manager.stop_jobs_for_output(req.path)
    result = delete_output(req.path)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


@app.get("/api/jobs")
def api_jobs():
    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/jobs")
def api_start_job(req: StartJobRequest):
    if not req.text_model:
        raise HTTPException(400, "text_model is required")
    try:
        return job_manager.start_job(
            book_type=req.book_type,
            toc_path=req.toc_path,
            text_model=req.text_model,
            image_backend=req.image_backend,
            image_model=req.image_model,
            output_dir=req.output_dir,
            resume=req.resume,
            rewrite_all=req.rewrite_all,
            rewrite=req.rewrite,
            pipeline=req.pipeline,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/jobs/{job_id}/stop")
async def api_stop_job(job_id: str):
    if not job_manager.stop_job(job_id):
        raise HTTPException(404, "Job not found or not running")
    return {"success": True}


@app.get("/api/files/serve")
async def api_serve_file(path: str, download: bool = False):
    target = Path(path).resolve()
    allowed_roots = [
        (PROJECT_ROOT / "book").resolve(),
        (PROJECT_ROOT / "picture-book").resolve(),
    ]
    ok = any(
        str(target).startswith(str(root))
        for root in allowed_roots
    )
    if not ok or not target.exists():
        raise HTTPException(404, "File not found")

    suffix = target.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".md": "text/markdown; charset=utf-8",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    if download:
        return FileResponse(
            target,
            media_type=media_type,
            filename=target.name,
            headers={"Content-Disposition": f'attachment; filename="{target.name}"'},
        )
    return FileResponse(target, media_type=media_type)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
