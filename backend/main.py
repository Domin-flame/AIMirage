"""MirageAI backend.

This service creates a PNG from a text prompt. If the diffusion model is not
available yet, it falls back to a styled placeholder image so the app still works.
"""

import gc
import io
import math
import os
import wave
from pathlib import Path

import httpx
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


def load_local_env_file():
    """Load environment variables from a .env file in the project root when present."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


load_local_env_file()


def get_cloud_model_config():
    """Return the configured cloud model settings if the app is set up for external inference."""
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL")
    if token and model:
        return {
            "token": token,
            "model": model,
            "url": f"https://api-inference.huggingface.co/models/{model}",
        }
    return None


def call_cloud_generation(prompt: str, quality: str, is_avatar: bool = False) -> Image.Image | None:
    """Generate an image using a hosted Hugging Face inference endpoint when configured."""
    config = get_cloud_model_config()
    if config is None:
        return None

    payload = {"inputs": prompt}
    if quality == "quality":
        payload["parameters"] = {"guidance_scale": 7.5, "num_inference_steps": 25}
    else:
        payload["parameters"] = {"guidance_scale": 6.0, "num_inference_steps": 12}

    if is_avatar:
        payload["parameters"]["negative_prompt"] = "blurry, low quality, duplicate face, deformed hands, text, watermark"

    response = httpx.post(
        config["url"],
        headers={"Authorization": f"Bearer {config['token']}"},
        json=payload,
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Cloud model request failed: {response.status_code}")

    image_bytes = response.content
    if not image_bytes:
        raise RuntimeError("Cloud model returned an empty response.")

    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


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
    cloud_image = None
    if get_cloud_model_config() is not None:
        cloud_image = call_cloud_generation(prompt, quality=quality, is_avatar=is_avatar)

    if cloud_image is not None:
        return cloud_image

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


def generate_speech_wav(text: str, sample_rate: int = 22050, duration_per_char: float = 0.09) -> bytes:
    """Generate a lightweight synthetic WAV file from text for local demo use."""
    cleaned = text.strip()[:200]
    if not cleaned:
        raise ValueError("Text cannot be empty.")

    sample_count = int(sample_rate * max(0.7, len(cleaned) * duration_per_char))
    frames = bytearray()
    for idx, char in enumerate(cleaned):
        base_freq = 180 + (ord(char.lower()) % 26) * 12
        mod = 1.0 + (idx % 5) * 0.18
        phase = idx * 0.35
        for sample_index in range(int(sample_rate * duration_per_char)):
            t = sample_index / sample_rate
            value = math.sin(2 * math.pi * base_freq * mod * (t + phase)) * 0.35
            if char.isspace():
                value *= 0.2
            if char in ",.!?;:":
                value *= 0.8
            pcm = int(max(-1.0, min(1.0, value)) * 32767)
            frames.extend(int(pcm).to_bytes(2, byteorder="little", signed=True))

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames[:sample_count * 2]))
    return wav_buffer.getvalue()


def generate_avatar_gif(prompt: str, size: tuple[int, int] = (256, 256), frames: int = 6) -> bytes:
    """Generate a lightweight animated GIF by varying a portrait-like fallback over several frames."""
    palette = [
        (18, 24, 36),
        (28, 38, 52),
        (66, 88, 108),
        (104, 138, 170),
        (147, 176, 195),
        (120, 144, 178),
    ]

    frame_images = []
    for frame_index in range(frames):
        image = Image.new("RGB", size, palette[frame_index % len(palette)])
        draw = ImageDraw.Draw(image)
        cx = size[0] // 2
        cy = size[1] // 2
        head_y = 10 + frame_index % 3
        draw.ellipse((cx - 52, cy - 64 + head_y, cx + 52, cy + 20 + head_y), fill=(246, 210, 180))
        draw.ellipse((cx - 68, cy - 80 + head_y, cx + 68, cy + 10 + head_y), fill=(40, 52, 68))
        draw.rectangle((cx - 80, cy + 20 + head_y, cx + 80, cy + 92 + head_y), fill=(77, 96, 122))
        draw.text((12, 12), "MirageAI", fill=(255, 255, 255))
        draw.text((12, 34), prompt[:20], fill=(200, 220, 255))
        frame_images.append(image)

    gif_buffer = io.BytesIO()
    frame_images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=frame_images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )
    return gif_buffer.getvalue()


@app.get("/health")
def health_check():
    """Basic health endpoint for checking system readiness."""
    return {
        "status": "ok",
        "local_model_available": StableDiffusionPipeline is not None and torch is not None,
        "cloud_model_configured": get_cloud_model_config() is not None,
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


@app.post("/speak")
def speak_text(
    text: str = Query(..., min_length=1, max_length=500),
):
    """Generate a lightweight WAV response from text for local voice previewing."""
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_bytes = generate_speech_wav(clean_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/generate-video")
def generate_video(
    prompt: str = Query(..., min_length=1, max_length=500),
):
    """Generate a small animated GIF row for avatar/video demo flows."""
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        gif_bytes = generate_avatar_gif(clean_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(gif_bytes),
        media_type="image/gif",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def root():
    return {"message": "MirageAI backend is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)