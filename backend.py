"""
Document-Based Question Answering System - Backend
FastAPI server with RAG implementation
"""

import os
import re
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Document processing
import PyPDF2
from docx import Document as DocxDocument
import pdfplumber

# NLP and embeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# Data models
class Question(BaseModel):
    question: str
    top_k: int = 5

class Answer(BaseModel):
    answer: str
    confidence: float
    source_text: str
    source_document: str
    chunk_id: str

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


# Global storage (in-memory)
class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        if self.model is None:
            print("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded successfully!")


# Initialize global store
doc_store = DocumentStore()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    doc_store.initialize_model()
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Document QA System",
    description="RAG-based Question Answering System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Document Processing Functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e2:
            raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {e2}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract DOCX text: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Split text into overlapping chunks"""
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "start_idx": len(chunks)})
            
            overlap_words = []
            overlap_length = 0
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len <= overlap:
                    overlap_words.insert(0, s)
                    overlap_length += s_len
                else:
                    break
            
            current_chunk = overlap_words
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({"text": chunk_text, "start_idx": len(chunks)})
    
    return chunks


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Document QA System API"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        if file_ext == '.pdf':
            text = extract_text_from_pdf(tmp_path)
        elif file_ext == '.docx':
            text = extract_text_from_docx(tmp_path)
        else:
            text = extract_text_from_txt(tmp_path)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        doc_id = hashlib.md5(f"{file.filename}_{len(text)}".encode()).hexdigest()[:12]
        chunks = chunk_text(text)
        
        doc_store.initialize_model()
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = doc_store.model.encode(chunk_texts, convert_to_numpy=True)
        
        for i, chunk in enumerate(chunks):
            chunk["doc_id"] = doc_id
            chunk["filename"] = file.filename
            chunk["chunk_id"] = f"{doc_id}_chunk_{i}"
            chunk["embedding"] = chunk_embeddings[i]
            doc_store.chunks.append(chunk)
        
        if doc_store.embeddings is None:
            doc_store.embeddings = chunk_embeddings
        else:
            doc_store.embeddings = np.vstack([doc_store.embeddings, chunk_embeddings])
        
        doc_store.documents[doc_id] = {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "full_text": text
        }
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "message": f"Document processed successfully! Created {len(chunks)} chunks."
        }
    
    finally:
        os.unlink(tmp_path)


@app.post("/upload-text")
async def upload_text(text: str = Form(...), title: str = Form("Direct Input")):
    """Process directly entered text"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    doc_id = hashlib.md5(f"{title}_{len(text)}".encode()).hexdigest()[:12]
    chunks = chunk_text(text)
    
    doc_store.initialize_model()
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = doc_store.model.encode(chunk_texts, convert_to_numpy=True)
    
    for i, chunk in enumerate(chunks):
        chunk["doc_id"] = doc_id
        chunk["filename"] = title
        chunk["chunk_id"] = f"{doc_id}_chunk_{i}"
        chunk["embedding"] = chunk_embeddings[i]
        doc_store.chunks.append(chunk)
    
    if doc_store.embeddings is None:
        doc_store.embeddings = chunk_embeddings
    else:
        doc_store.embeddings = np.vstack([doc_store.embeddings, chunk_embeddings])
    
    doc_store.documents[doc_id] = {
        "filename": title,
        "num_chunks": len(chunks),
        "full_text": text
    }
    
    return {
        "success": True,
        "doc_id": doc_id,
        "filename": title,
        "num_chunks": len(chunks)
    }


@app.post("/ask", response_model=List[Answer])
async def ask_question(question: Question):
    """Answer a question based on uploaded documents"""
    if not doc_store.chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded. Please upload documents first."
        )
    
    doc_store.initialize_model()
    question_embedding = doc_store.model.encode(question.question, convert_to_numpy=True)
    
    similarities = util.cos_sim(question_embedding, doc_store.embeddings)[0].numpy()
    top_indices = np.argsort(similarities)[::-1][:question.top_k]
    
    answers = []
    for idx in top_indices:
        chunk = doc_store.chunks[idx]
        confidence = float(similarities[idx])
        
        if confidence > 0.1:
            sentences = sent_tokenize(chunk["text"])
            sentence_embeddings = doc_store.model.encode(sentences, convert_to_numpy=True)
            sentence_sims = util.cos_sim(question_embedding, sentence_embeddings)[0].numpy()
            
            best_sentence_idx = np.argmax(sentence_sims)
            best_sentence = sentences[best_sentence_idx]
            
            answers.append(Answer(
                answer=best_sentence,
                confidence=round(confidence * 100, 2),
                source_text=chunk["text"],
                source_document=chunk["filename"],
                chunk_id=chunk["chunk_id"]
            ))
    
    if not answers:
        answers.append(Answer(
            answer="I couldn't find a relevant answer in the uploaded documents.",
            confidence=0.0,
            source_text="N/A",
            source_document="N/A",
            chunk_id="N/A"
        ))
    
    return answers


@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    docs = []
    for doc_id, info in doc_store.documents.items():
        docs.append({
            "doc_id": doc_id,
            "filename": info["filename"],
            "num_chunks": info["num_chunks"]
        })
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks"""
    if doc_id not in doc_store.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_store.chunks = [c for c in doc_store.chunks if c["doc_id"] != doc_id]
    
    if doc_store.chunks:
        doc_store.embeddings = np.vstack([c["embedding"] for c in doc_store.chunks])
    else:
        doc_store.embeddings = None
    
    del doc_store.documents[doc_id]
    return {"success": True, "message": f"Document {doc_id} deleted successfully"}


@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents"""
    doc_store.documents.clear()
    doc_store.chunks.clear()
    doc_store.embeddings = None
    return {"success": True, "message": "All documents cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
