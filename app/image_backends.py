"""Image generation backends for picture books (free, local).

Backends:
  - automatic1111: Stable Diffusion WebUI / Forge API (Linux server + SSH tunnel)
  - diffusers:       In-process Stable Diffusion via Hugging Face (local GPU)
  - ollama:          Ollama image models (macOS mainly)
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("picture-book")

DEFAULT_IMAGE_MODEL = "x/flux2-klein:4b"
DEFAULT_A1111_URL = "http://localhost:7860"
DEFAULT_DIFFUSERS_MODEL = "runwayml/stable-diffusion-v1-5"

CHILDREN_NEGATIVE_PROMPT = (
    "ugly, blurry, low quality, watermark, text, letters, words, caption, "
    "signature, scary, horror, realistic photo, photograph, deformed, "
    "bad anatomy, extra limbs"
)


def check_image_backend(backend: str, *, api_url: str = "", model: str = "") -> bool:
    if backend == "ollama":
        from app.picture_tools import check_ollama_image_model

        return check_ollama_image_model(model or DEFAULT_IMAGE_MODEL)
    if backend == "automatic1111":
        return check_automatic1111(api_url or DEFAULT_A1111_URL)
    if backend == "diffusers":
        return check_diffusers(model or DEFAULT_DIFFUSERS_MODEL)
    logger.error("Unknown image backend: %s", backend)
    return False


def generate_page_image(
    prompt: str,
    output_path: str,
    *,
    backend: str = "automatic1111",
    model: str = "",
    api_url: str = DEFAULT_A1111_URL,
    width: int = 512,
    height: int = 512,
    reference_image: str | None = None,
    timeout: int = 600,
) -> dict:
    if backend == "ollama":
        from app.picture_tools import generate_ollama_image

        return generate_ollama_image(
            prompt=prompt,
            output_path=output_path,
            model=model or DEFAULT_IMAGE_MODEL,
            width=width,
            height=height,
            reference_image=reference_image,
            timeout=timeout,
        )
    if backend == "automatic1111":
        return generate_automatic1111(
            prompt=prompt,
            output_path=output_path,
            api_url=api_url,
            model=model,
            width=width,
            height=height,
            reference_image=reference_image,
            timeout=timeout,
        )
    if backend == "diffusers":
        return generate_diffusers(
            prompt=prompt,
            output_path=output_path,
            model=model or DEFAULT_DIFFUSERS_MODEL,
            width=width,
            height=height,
            reference_image=reference_image,
        )
    return {"success": False, "message": f"Unknown image backend: {backend}"}


def _a1111_get_json(api_url: str, path: str, *, timeout: int = 15) -> object:
    url = f"{api_url.rstrip('/')}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _resolve_a1111_checkpoint(api_url: str, model: str = "") -> str:
    """Pick a Forge/A1111 checkpoint (loaded, requested, or first on disk)."""
    if model:
        return model

    base = api_url.rstrip("/")
    try:
        options = _a1111_get_json(base, "/sdapi/v1/options")
        loaded = options.get("sd_model_checkpoint")
        if loaded:
            return loaded
    except Exception:
        pass

    models = _a1111_get_json(base, "/sdapi/v1/sd-models")
    if not models:
        return ""

    first = models[0].get("model_name") or models[0].get("title", "")
    if first.endswith(".safetensors"):
        first = first[: -len(".safetensors")]
    return first


def check_automatic1111(api_url: str) -> bool:
    base = api_url.rstrip("/")
    try:
        models = _a1111_get_json(base, "/sdapi/v1/sd-models")
        names = [m.get("model_name", m.get("title", "?")) for m in models]
        logger.info(
            "Stable Diffusion WebUI/Forge OK at %s - %d model(s)",
            api_url, len(names),
        )
        if names:
            logger.info("Available checkpoints: %s", ", ".join(names[:5]))

        loaded = ""
        try:
            options = _a1111_get_json(base, "/sdapi/v1/options")
            loaded = options.get("sd_model_checkpoint") or ""
        except Exception:
            pass

        if not loaded and names:
            logger.warning(
                "Forge has checkpoints on disk but none loaded in UI. "
                "Will auto-select '%s' for API requests.",
                names[0],
            )
        elif loaded:
            logger.info("Forge loaded checkpoint: %s", loaded)
        return bool(names)
    except Exception as e:
        logger.error("Cannot reach SD WebUI at %s: %s", api_url, e)
        logger.error(
            "On the Linux server, start Forge/A1111 with API enabled, then tunnel:\n"
            "  ssh -N -L 7860:localhost:7860 user@server\n"
            "Example server command:\n"
            "  ./webui.sh --api --listen"
        )
        return False


def generate_automatic1111(
    prompt: str,
    output_path: str,
    api_url: str = DEFAULT_A1111_URL,
    model: str = "",
    width: int = 512,
    height: int = 512,
    reference_image: str | None = None,
    timeout: int = 600,
) -> dict:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = api_url.rstrip("/")

    checkpoint = _resolve_a1111_checkpoint(base, model)
    if not checkpoint:
        return {
            "success": False,
            "message": "No SD checkpoint found. Add a model under models/Stable-diffusion in Forge.",
        }

    common = {
        "prompt": prompt,
        "negative_prompt": CHILDREN_NEGATIVE_PROMPT,
        "width": width,
        "height": height,
        "steps": 28,
        "cfg_scale": 7.5,
        "sampler_name": "Euler a",
        "seed": -1,
        "override_settings": {"sd_model_checkpoint": checkpoint},
    }
    logger.info("Forge image gen using checkpoint: %s", checkpoint)

    if reference_image and Path(reference_image).exists():
        init_b64 = base64.b64encode(Path(reference_image).read_bytes()).decode("utf-8")
        payload = {
            **common,
            "init_images": [init_b64],
            "denoising_strength": 0.55,
        }
        endpoint = f"{base}/sdapi/v1/img2img"
    else:
        payload = common
        endpoint = f"{base}/sdapi/v1/txt2img"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        detail = body
        try:
            parsed = json.loads(body)
            detail = parsed.get("message") or parsed.get("error") or body
        except json.JSONDecodeError:
            pass
        return {"success": False, "message": f"SD WebUI HTTP {e.code}: {detail}"}
    except urllib.error.URLError as e:
        return {"success": False, "message": f"SD WebUI request failed: {e}"}

    images = result.get("images", [])
    if not images:
        return {"success": False, "message": "SD WebUI returned no images"}

    out.write_bytes(base64.b64decode(images[0]))
    logger.info("Image saved to %s (backend: automatic1111)", out)
    return {"success": True, "image_path": str(out)}


_diffusers_pipelines: dict = {}


def check_diffusers(model_id: str) -> bool:
    try:
        import torch
        from diffusers import StableDiffusionPipeline  # noqa: F401
    except ImportError:
        logger.error(
            "diffusers backend requires extra packages.\n"
            "Install with: pip install book-writing-agent[images]"
        )
        return False

    if torch.cuda.is_available():
        logger.info("diffusers OK — CUDA GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning(
            "diffusers OK but no CUDA GPU detected — generation will be very slow on CPU"
        )
    logger.info("Model: %s", model_id)
    return True


def _get_diffusers_txt2img(model_id: str):
    if model_id not in _diffusers_pipelines:
        import torch
        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        _diffusers_pipelines[model_id] = pipe
    return _diffusers_pipelines[model_id]


def _get_diffusers_img2img(model_id: str):
    key = f"{model_id}:img2img"
    if key not in _diffusers_pipelines:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        _diffusers_pipelines[key] = pipe
    return _diffusers_pipelines[key]


def generate_diffusers(
    prompt: str,
    output_path: str,
    model: str = DEFAULT_DIFFUSERS_MODEL,
    width: int = 512,
    height: int = 512,
    reference_image: str | None = None,
) -> dict:
    try:
        from PIL import Image
    except ImportError:
        return {
            "success": False,
            "message": "Pillow required. Install with: pip install book-writing-agent[images]",
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if reference_image and Path(reference_image).exists():
            pipe = _get_diffusers_img2img(model)
            init_image = Image.open(reference_image).convert("RGB")
            init_image = init_image.resize((width, height))
            result = pipe(
                prompt=prompt,
                negative_prompt=CHILDREN_NEGATIVE_PROMPT,
                image=init_image,
                strength=0.55,
                num_inference_steps=28,
                guidance_scale=7.5,
            )
        else:
            pipe = _get_diffusers_txt2img(model)
            result = pipe(
                prompt=prompt,
                negative_prompt=CHILDREN_NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_inference_steps=28,
                guidance_scale=7.5,
            )
        result.images[0].save(out)
    except Exception as e:
        return {"success": False, "message": f"diffusers generation failed: {e}"}

    logger.info("Image saved to %s (backend: diffusers)", out)
    return {"success": True, "image_path": str(out)}
