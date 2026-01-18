import pytest
import pandas as pd
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app


@pytest.fixture
def client():
    with patch("app.main.ModelService") as mock_service_class:
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance
        
        with TestClient(app) as c:
            c.app.model_service = mock_instance
            yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success(client):
    client.app.model_service.predict.return_value = pd.Series([10.5, 20.0])

    csv_content = "Store,Dept,Date\n1,1,2023-01-01\n1,1,2023-01-08"
    files = {
        "file": ("test.csv", csv_content, "text/csv")
    }

    response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert response.json()["predictions"] == [10.5, 20.0]

def test_predict_invalid_extension(client):
    files = {
        "file": ("test.txt", "conteudo qualquer", "text/plain")
    }
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]

def test_predict_empty_csv(client):
    csv_content = "Store,Dept,Date"
    files = {
        "file": ("test.csv", csv_content, "text/csv")
    }
    response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Uploaded CSV is empty" in response.json()["detail"]

def test_predict_internal_error(client):
    client.app.model_service.predict.side_effect = Exception("Erro genérico de ML")

    csv_content = "Store,Dept,Date\n1,1,2023-01-01"
    files = {
        "file": ("test.csv", csv_content, "text/csv")
    }
    
    response = client.post("/predict", files=files)
    
    assert response.status_code == 500
    assert "Erro genérico de ML" in response.json()["detail"]