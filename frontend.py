"""
Document-Based Question Answering System - Frontend
Streamlit application for document Q&A
"""

import streamlit as st
import requests
import time
from typing import Optional, Dict, List

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="📚 Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { color: white; margin-bottom: 0.5rem; }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; }
    .source-text {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        border: 1px solid #e9ecef;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []


def check_api_health() -> bool:
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_document(file) -> Optional[Dict]:
    """Upload a document to the backend"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Please ensure the server is running.")
        return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def upload_text(text: str, title: str) -> Optional[Dict]:
    """Upload text directly to the backend"""
    try:
        data = {"text": text, "title": title}
        response = requests.post(f"{API_URL}/upload-text", data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def ask_question(question: str, top_k: int = 5) -> Optional[List[Dict]]:
    """Send a question to the backend"""
    try:
        payload = {"question": question, "top_k": top_k}
        response = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Query error: {str(e)}")
        return None


def get_documents() -> List[Dict]:
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []


def delete_document(doc_id: str) -> bool:
    """Delete a specific document"""
    try:
        response = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
        return response.status_code == 200
    except:
        return False


def clear_all_documents() -> bool:
    """Clear all documents"""
    try:
        response = requests.delete(f"{API_URL}/documents", timeout=10)
        return response.status_code == 200
    except:
        return False


def highlight_answer_in_source(source_text: str, answer: str) -> str:
    """Highlight the answer within the source text"""
    if answer and answer in source_text:
        highlighted = source_text.replace(
            answer, 
            f'<mark style="background-color: #fff3cd; padding: 2px 4px; border-radius: 3px;">{answer}</mark>'
        )
        return highlighted
    return source_text


# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Document-Based Question Answering System</h1>
        <p>Upload documents and ask questions using RAG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API status
    api_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        if api_status:
            st.success("✅ Backend API Connected")
        else:
            st.error("❌ Backend API Offline")
            st.info("Start the backend with:\n```\npython backend.py\n```")
        
        st.markdown("---")
        st.markdown("### 📁 Uploaded Documents")
        
        if st.button("🔄 Refresh", key="refresh_docs"):
            st.session_state.documents = get_documents()
        
        docs = get_documents()
        
        if docs:
            for doc in docs:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**📄 {doc['filename']}**")
                        st.caption(f"Chunks: {doc['num_chunks']}")
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete"):
                            if delete_document(doc['doc_id']):
                                st.rerun()
            
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if clear_all_documents():
                    st.success("All documents cleared!")
                    st.rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses:
        - **RAG** (Retrieval Augmented Generation)
        - **Sentence Transformers** for embeddings
        - **FastAPI** backend
        - **Streamlit** frontend
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload Documents", 
        "❓ Ask Questions", 
        "📜 History"
    ])
    
    # Tab 1: Upload Documents
    with tab1:
        st.markdown("### Upload Documents")
        st.markdown("Upload PDF, DOCX, or TXT files to build your knowledge base.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📁 File Upload")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload one or more documents"
            )
            
            if uploaded_files:
                if st.button("📤 Process Documents", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        result = upload_document(file)
                        
                        if result and result.get("success"):
                            st.success(f"✅ {file.name}: {result.get('message', 'Processed successfully')}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("Processing complete!")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            st.markdown("#### 📝 Direct Text Input")
            text_title = st.text_input("Document Title", value="My Document")
            text_content = st.text_area(
                "Enter text directly",
                height=200,
                placeholder="Paste or type your text here..."
            )
            
            if st.button("📤 Process Text", type="primary"):
                if text_content.strip():
                    with st.spinner("Processing text..."):
                        result = upload_text(text_content, text_title)
                        if result and result.get("success"):
                            st.success(f"✅ Text processed successfully!")
                            st.info(f"Created {result.get('num_chunks', 0)} chunks")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.warning("Please enter some text")
    
    # Tab 2: Ask Questions
    with tab2:
        st.markdown("### Ask Questions")
        st.markdown("Enter your question below to search through the uploaded documents.")
        
        question = st.text_input(
            "Your Question",
            placeholder="What is the main topic of the document?",
            key="question_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_answers = st.slider("Number of answers to retrieve", 1, 10, 5)
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        
        if search_button and question:
            with st.spinner("Searching through documents..."):
                answers = ask_question(question, num_answers)
                
                if answers:
                    st.session_state.qa_history.append({
                        "question": question,
                        "answers": answers,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.markdown("---")
                    st.markdown("### 📋 Answers")
                    
                    for i, answer in enumerate(answers):
                        confidence = answer.get("confidence", 0)
                        
                        with st.expander(f"**Answer {i+1}** - Confidence: {confidence}%", expanded=(i==0)):
                            st.markdown("#### 💡 Answer")
                            st.info(answer.get("answer", "No answer found"))
                            
                            st.markdown("#### 📊 Confidence Score")
                            st.progress(min(confidence / 100, 1.0))
                            
                            if confidence >= 70:
                                st.success(f"High confidence: {confidence}%")
                            elif confidence >= 40:
                                st.warning(f"Medium confidence: {confidence}%")
                            else:
                                st.error(f"Low confidence: {confidence}%")
                            
                            st.markdown("#### 📄 Source Document")
                            st.text(answer.get("source_document", "Unknown"))
                            
                            st.markdown("#### 📝 Source Context")
                            source_text = answer.get("source_text", "")
                            answer_text = answer.get("answer", "")
                            highlighted_source = highlight_answer_in_source(source_text, answer_text)
                            st.markdown(
                                f'<div class="source-text">{highlighted_source}</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.warning("No answers found. Try uploading more documents or rephrasing your question.")
        
        elif search_button and not question:
            st.warning("Please enter a question")
    
    # Tab 3: History
    with tab3:
        st.markdown("### Question & Answer History")
        
        if st.session_state.qa_history:
            for i, item in enumerate(reversed(st.session_state.qa_history)):
                with st.expander(f"**Q: {item['question']}** - {item['timestamp']}", expanded=(i==0)):
                    st.markdown(f"**Question:** {item['question']}")
                    st.markdown("**Answers:**")
                    
                    for j, answer in enumerate(item['answers'][:3]):
                        st.markdown(f"""
                        **{j+1}.** {answer.get('answer', 'N/A')}
                        - *Confidence:* {answer.get('confidence', 0)}%
                        - *Source:* {answer.get('source_document', 'Unknown')}
                        """)
            
            if st.button("🗑️ Clear History"):
                st.session_state.qa_history = []
                st.rerun()
        else:
            st.info("No questions asked yet. Go to 'Ask Questions' tab to start!")


if __name__ == "__main__":
    main()
