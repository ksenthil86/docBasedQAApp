# 📚 Document-Based Question Answering System
 
A RAG (Retrieval Augmented Generation) based Question Answering application.

## 👥 Team Members

This project was a collaborative effort by **Group 72**. Below are the details of the team members and their contributions:

| Student Name | Student ID | Contribution |
| :--- | :--- | :--- |
| SENTHILKUMAR K. | 2024aa05227 | 100% |
| SENTHIL KUMAR K. | 2024aa05233 | 100% |
| SAVITA SHENDE. | 2024ac05946 | 100% |
| ANAND K NAIR. | 2024aa05086 | 100% |
| DIVYA CHAUDHARY | 2024aa05426 | 100% |
 
## 🛠️ Technology Stack
 
| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Document Processing | PyPDF2, python-docx, pdfplumber |
 
## 🚀 Quick Start
 
### Step 1: Install Dependencies
 
```bash
# Create virtual environment
python3 -m venv venv
 
# Activate it
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows
 
# Upgrade pip
pip install --upgrade pip
 
# Install PyTorch CPU-only version (to avoid large CUDA dependencies)
pip install torch --index-url https://download.pytorch.org/whl/cpu
 
# Install other dependencies
python -m pip install --no-cache-dir --only-binary=:all  -r requirements.txt
 
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
 
### Step 2: Start the Backend
 
```bash
python backend.py
```
 
The API will be available at `http://localhost:8000`
 
### Step 3: Start the Frontend (New Terminal)
 
```bash
source venv/bin/activate  # Activate venv first
streamlit run frontend.py
```
 
The web interface will open at `http://localhost:8501`
 
## 📖 Usage
 
1. **Upload Documents**: Go to "Upload Documents" tab and upload PDF, DOCX, or TXT files
2. **Ask Questions**: Go to "Ask Questions" tab, type your question, and click Search
3. **View History**: Go to "History" tab to see past questions and answers
 
## 📁 Project Structure
 
```
doc-qa-system/
├── backend.py          # FastAPI server with RAG implementation
├── frontend.py         # Streamlit UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
 
## 🔌 API Endpoints
 
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload a document |
| POST | `/upload-text` | Upload text directly |
| POST | `/ask` | Ask a question |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| DELETE | `/documents` | Clear all documents |
 
## 🐛 Troubleshooting
 
### Port already in use
```bash
# Kill process on port 8000 (Mac/Linux)
kill -9 $(lsof -t -i:8000)
 
# On Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```
 
### Module not found
```bash
# Make sure venv is activated
source venv/bin/activate
pip install -r requirements.txt
```
 
### NLTK data not found
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
