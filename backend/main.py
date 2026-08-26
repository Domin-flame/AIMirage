"""MirageAI backend.

This service creates a PNG from a text prompt. If the diffusion model is not
available yet, it falls back to a styled placeholder image so the app still works.
"""

import gc
import io

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw

try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception:  # Optional dependencies may not be installed in a lightweight env.
    torch = None
    StableDiffusionPipeline = None

app = FastAPI(
    title="MirageAI",
    version="0.1.0",
    description="Generate marketing-ready images from text prompts.",
)

if torch is not None:
    torch.set_grad_enabled(False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipe = None


def get_pipe():
    """Create the SD pipeline once and reuse it for later requests."""
    global _pipe
    if _pipe is None and StableDiffusionPipeline is not None and torch is not None:
        _pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float32,
        )
        _pipe.enable_attention_slicing()
        _pipe.to("cpu")
    return _pipe


def generate_fallback_image(prompt: str, width: int = 768, height: int = 512) -> Image.Image:
    """Build a placeholder image when the real model is unavailable."""
    image = Image.new("RGB", (width, height), color=(24, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(84, 120, 180), width=4)
    draw.text((24, 24), "MirageAI", fill=(255, 255, 255))
    draw.text((24, 70), "Fallback preview", fill=(120, 220, 255))
    draw.text((24, 120), prompt[:160], fill=(230, 230, 230))
    return image


def generate_avatar_fallback_image(prompt: str, width: int = 768, height: int = 1024) -> Image.Image:
    """Build a portrait/avatar placeholder optimized for a bust shot."""
    image = Image.new("RGB", (width, height), color=(14, 20, 32))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=28, outline=(146, 165, 255), width=6)
    draw.rectangle((0, int(height * 0.62), width, height), fill=(21, 36, 56))

    center_x = width // 2
    center_y = int(height * 0.34)
    draw.ellipse((center_x - 150, center_y - 120, center_x + 150, center_y + 120), fill=(249, 214, 182))
    draw.ellipse((center_x - 175, center_y - 160, center_x + 175, center_y + 20), fill=(49, 51, 67))
    draw.rectangle((center_x - 170, center_y + 120, center_x + 170, height - 120), fill=(78, 93, 120))
    draw.rounded_rectangle((center_x - 220, height - 220, center_x + 220, height - 50), radius=36, fill=(104, 130, 232))

    draw.text((40, 40), "MirageAI Avatar", fill=(255, 255, 255), font=None)
    draw.text((40, 90), "Portrait preview", fill=(160, 214, 255), font=None)
    draw.text((40, height - 120), prompt[:120], fill=(235, 240, 255), font=None)
    return image


def build_generated_image(prompt: str, quality: str, is_avatar: bool = False) -> Image.Image:
    """Generate either a concept image or a portrait/avatar using the app pipeline."""
    pipe = get_pipe()

    if quality == "quality":
        steps = 8
        height = 1024 if is_avatar else 640
        width = 768 if is_avatar else 640
    else:
        steps = 4
        height = 768 if is_avatar else 512
        width = 576 if is_avatar else 512

    if pipe is None:
        if is_avatar:
            return generate_avatar_fallback_image(prompt, width=width, height=height)
        return generate_fallback_image(prompt, width=width, height=height)

    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,
        height=height,
        width=width,
    ).images[0]
    return image


@app.get("/health")
def health_check():
    """Basic health endpoint for checking system readiness."""
    return {
        "status": "ok",
        "model_available": StableDiffusionPipeline is not None and torch is not None,
    }


@app.post("/generate")
def generate_image(
    prompt: str = Query(..., min_length=1, max_length=500),
    quality: str = Query(default="fast", pattern=r"^(fast|quality)$"),
):
    """Generate a PNG response from the prompt."""
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        image = build_generated_image(clean_prompt, quality=quality, is_avatar=False)
    except Exception:
        image = generate_fallback_image(clean_prompt)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    gc.collect()

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/generate-avatar")
def generate_avatar(
    prompt: str = Query(..., min_length=1, max_length=500),
    quality: str = Query(default="fast", pattern=r"^(fast|quality)$"),
):
    """Generate a portrait-oriented avatar using the same API contract as /generate."""
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        image = build_generated_image(clean_prompt, quality=quality, is_avatar=True)
    except Exception:
        image = generate_avatar_fallback_image(clean_prompt)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    gc.collect()

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def root():
    return {"message": "MirageAI backend is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)