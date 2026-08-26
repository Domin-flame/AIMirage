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
