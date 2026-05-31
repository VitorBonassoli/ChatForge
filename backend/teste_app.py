import pytest
import json
from unittest.mock import patch, MagicMock
import app as flask_app


@pytest.fixture
def client():
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        yield client


# --- Testes da rota home ---

def test_home_status_200(client):
    response = client.get("/")
    assert response.status_code == 200


def test_home_content(client):
    response = client.get("/")
    assert b"API rodando" in response.data


# --- Testes da rota /chat ---

def test_chat_sucesso(client):
    mock_response = MagicMock()
    mock_response.text = "Olá! Como posso ajudar?"

    with patch.object(flask_app.client.models, "generate_content", return_value=mock_response):
        response = client.post(
            "/chat",
            data=json.dumps({"message": "Oi"}),
            content_type="application/json",
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["response"] == "Olá! Como posso ajudar?"


def test_chat_retorna_json(client):
    mock_response = MagicMock()
    mock_response.text = "resposta qualquer"

    with patch.object(flask_app.client.models, "generate_content", return_value=mock_response):
        response = client.post(
            "/chat",
            data=json.dumps({"message": "teste"}),
            content_type="application/json",
        )

    assert response.content_type == "application/json"


def test_chat_erro_gemini(client):
    with patch.object(
        flask_app.client.models,
        "generate_content",
        side_effect=Exception("Falha na API"),
    ):
        response = client.post(
            "/chat",
            data=json.dumps({"message": "teste"}),
            content_type="application/json",
        )

    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
    assert "Falha na API" in data["error"]


def test_chat_sem_mensagem(client):
    mock_response = MagicMock()
    mock_response.text = "resposta"

    with patch.object(flask_app.client.models, "generate_content", return_value=mock_response):
        response = client.post(
            "/chat",
            data=json.dumps({}),
            content_type="application/json",
        )

    assert response.status_code == 200


def test_chat_mensagem_longa(client):
    mock_response = MagicMock()
    mock_response.text = "resumo"

    long_message = "palavra " * 500

    with patch.object(flask_app.client.models, "generate_content", return_value=mock_response):
        response = client.post(
            "/chat",
            data=json.dumps({"message": long_message}),
            content_type="application/json",
        )

    assert response.status_code == 200
    assert response.get_json()["response"] == "resumo"