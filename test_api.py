from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API ML opérationnelle"}


def test_predict():
    payload = {
        "age": 30,
        "salary": 50000,
        "score": 0.8,
        "history": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_validation_error():
    payload = {
        "age": 30,
        "salary": 50000,
        "score": 0.8
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422