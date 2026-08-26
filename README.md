# MirageAI

A lightweight app for generating marketing-style images from text prompts. The project combines a FastAPI backend with a simple HTML/JavaScript frontend so it is easy to run locally and extend.

## Highlights

- Text-to-image generation endpoint with a safe fallback mode
- FastAPI backend with CORS enabled for local front-end testing
- Lightweight browser interface with prompt input and quality selector
- Graceful behavior when the diffusion model is unavailable
- Regression tests to protect the API contract

## Current status

This project is now in a runnable baseline state for local development. The backend can generate a placeholder image even when the heavy AI dependencies are not installed yet, and it automatically switches to the real model when available.

## Tech stack

- Python
- FastAPI
- Pillow
- Diffusers
- PyTorch
- HTML/CSS/JavaScript

## Local setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Open the frontend

Open the file [frontend/IAGen.html](frontend/IAGen.html) in a browser.

## API

### Generate an image

```http
POST /generate?prompt=your+prompt&quality=fast
```

Returns a PNG image response.

### Health check

```http
GET /health
```

## Project roadmap

- [x] Basic FastAPI app structure
- [x] Prompt validation and error handling
- [x] Simple browser-based frontend
- [x] Fallback image generation
- [ ] Real portrait/avatar generation
- [ ] Text-to-speech integration
- [ ] Video export and avatar pipeline
- [ ] Auth and persistence

## Contributing

Open a pull request with a clear description of the change and the expected behavior. The project is intentionally small and easy to extend, so keep changes focused and well-tested.

## License

This project is currently under active development and does not yet have a final release license.
