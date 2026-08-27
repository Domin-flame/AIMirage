import io

from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_generate_requires_prompt():
    response = client.post("/generate")
    assert response.status_code == 422


def test_generate_returns_png_for_valid_prompt():
    response = client.post("/generate?prompt=hello%20world")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_generate_avatar_returns_png_for_valid_prompt():
    response = client.post("/generate-avatar?prompt=portrait%20of%20a%20futuristic%20creator")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_generate_uses_cloud_when_configured(monkeypatch):
    import backend.main as main

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    buffer.seek(0)

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "image/png"}
        content = buffer.getvalue()

    def fake_post(url, headers=None, json=None):
        assert "api-inference.huggingface.co" in url
        assert headers["Authorization"] == "Bearer demo-token"
        return FakeResponse()

    monkeypatch.setenv("HF_TOKEN", "demo-token")
    monkeypatch.setenv("HF_MODEL", "stabilityai/sd-turbo")
    monkeypatch.setattr(main, "httpx", type("FakeHTTPX", (), {"post": staticmethod(fake_post)})())

    response = client.post("/generate?prompt=cloud%20portrait")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")


def test_speak_returns_wav_for_valid_text():
    response = client.post("/speak?text=hello%20friend")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")


def test_generate_video_returns_gif_for_valid_prompt():
    response = client.post("/generate-video?prompt=animated%20portrait%20of%20a%20creator")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/gif")
