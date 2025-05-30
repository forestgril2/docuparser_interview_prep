from fastapi.testclient import TestClient
from docparser.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Docparser API"}

def test_create_document():
    response = client.post(
        "/documents/",
        json={
            "title": "Test Document",
            "content": "This is a test document",
            "metadata": {"author": "Test User"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Document"
    assert data["status"] == "processing"

def test_get_document():
    response = client.get("/documents/123")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 123
    assert data["status"] == "completed"

def test_get_nonexistent_document():
    response = client.get("/documents/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found" 