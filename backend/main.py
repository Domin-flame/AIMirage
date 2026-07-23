from pathlib import Path
import uuid
import gc
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw

try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception:
    torch = None
    StableDiffusionPipeline = None

app = FastAPI()
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
    global _pipe
    if _pipe is None and StableDiffusionPipeline is not None:
        _pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype = torch.float32
        )
        
        _pipe.enable_attention_slicing()
        _pipe = _pipe.to("cpu")
    return _pipe


def generate_fallback_image(prompt: str, path: Path):
    image = Image.new("RGB", (768, 512), color=(24, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Fallback mode (diffusers/torch not installed)", fill=(230, 230, 230))
    draw.text((20, 70), f"Prompt: {prompt[:140]}", fill=(180, 220, 255))
    



@app.post("/generate")
def generate_image(prompt: str, quality: str = "fast"):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    buffer = io.BytesIO()
    

    pipe = get_pipe()
    if quality == "quality":
        steps = 8
        height = 640
        width = 640
    else:
        steps = 4
        image = pipe(prompt,
                     num_inference_steps=4,
                     guidance_scale=0.0,
                     height=512,
                     width=512).images[0]
    image.save(buffer, format="PNG")
    buffer.seek(0)

    gc.collect()
   

    return StreamingResponse(buffer, media_type="image/png")

import wopt