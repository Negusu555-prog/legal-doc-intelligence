# main.py
from fastapi import FastAPI, UploadFile, File
from document_processor import extract_text_from_pdf
from rag_engine import RAGEngine
import tempfile
import os

app = FastAPI()
rag = RAGEngine()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    chunks = extract_text_from_pdf(tmp_path)
    rag.vector_store.add_chunks(chunks)
    os.unlink(tmp_path)
    
    return {"message": f"עובדו {len(chunks)} חתיכות בהצלחה"}

@app.post("/ask")
async def ask_question(question: str):
    result = rag.answer(question)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}