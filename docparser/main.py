from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="Docparser API",
    description="Document parsing and processing service",
    version="0.1.0"
)

class DocumentRequest(BaseModel):
    title: str
    content: str
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    id: int
    title: str
    status: str
    processed_at: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Welcome to Docparser API"}

@app.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentRequest,
    background_tasks: BackgroundTasks
):
    # TODO: Implement actual document processing
    doc_id = 123
    
    # Add background processing
    background_tasks.add_task(process_document_background, doc_id)
    
    return DocumentResponse(
        id=doc_id,
        title=document.title,
        status="processing"
    )

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int):
    # TODO: Implement actual document retrieval
    if document_id not in [123, 456]:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=document_id,
        title="Sample Document",
        status="completed"
    )

def process_document_background(document_id: int):
    # TODO: Implement actual background processing
    print(f"Processing document {document_id}")

if __name__ == "__main__":
    uvicorn.run("docparser.main:app", host="0.0.0.0", port=8000, reload=True) 