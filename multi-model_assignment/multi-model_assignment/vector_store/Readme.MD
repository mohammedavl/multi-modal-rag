Multi-Modal Document Intelligence (RAG-Based QA System)

A complete end-to-end Retrieval-Augmented Generation pipeline designed to process real-world IMF Article IV reports containing text, tables, charts, and scanned images — and answer questions with LLM-generated, citation-backed responses.

📌 Project Overview

Modern financial and policy documents are multi-modal in nature.
This project builds a multi-modal RAG system capable of:

Extracting text, tables, and images (with OCR)

Converting them into structured chunks

Generating embeddings for all modalities

Storing them in a vector DB (Chroma)

Performing similarity-based retrieval

Answering questions using an LLM with proper attribution

This is a fully functional prototype aligned with real enterprise document-intelligence systems.

🏗 System Architecture
                    ┌───────────────┐
                    │    PDF File    │
                    └───────┬───────┘
                            │
                  Multi-Modal Ingestion
                            │
       ┌───────────┬──────────┬───────────┐
       │           │          │           │
   Text Chunks   Table Rows   OCR Images   Metadata
       │           │          │           │
       └───────────▼──────────▼───────────┘
                 Chunking & Cleaning
                            │
                  Embeddings (MiniLM)
                            │
                 Chroma Vector Database
                            │
                Retrieval-Augmented QA
                            │
              Streamlit Chatbot + Citations

✨ Features
🔍 Multi-Modal Document Ingestion

Text extraction (PyMuPDF)

Table extraction (Camelot)

Image extraction + OCR (Tesseract)

Per-page chunking and metadata tagging

🧠 Vector Indexing

Sentence-Transformers MiniLM

Unified multi-modal embedding space

Stored in ChromaDB (chroma.sqlite3 + chunks.pkl)

🤖 Question Answering

Similarity search

LLM answer generation (Flan-T5 or fallback SimpleQA)

Citation-backed responses

Page-level attribution

💬 Interactive Streamlit App

Chat interface

Retrieval visualization

Chunk statistics

Clear document insights

📦 Project Structure

multi-model-assignment/
│
├── app.py                     # Streamlit UI
├── process_document.py        # Multi-modal ingestion
├── create_embeddings.py       # Embedding generation
├── vector_store.py            # Chroma store wrapper
├── llm_qa.py                  # RAG + citations
├── config.py                  # Directories & model configs
│
├── data/
│   ├── raw/                   # Input PDFs
│   ├── images/                # Extracted images
│   ├── processed/             # Extracted chunks (JSON)
│   └── vector_store/          # Chroma files + chunks.pkl
│
└── requirements.txt

⚙️ Installation
1. Clone the Repository
git clone <your-repo-url>
cd multi-model_assignment

2. Install Dependencies
pip install -r requirements.txt

3. Install Tesseract (For OCR)

Download from:
https://github.com/tesseract-ocr/tesseract

▶️ How to Run the Project
Step 1 — Place PDF in data/raw/
data/raw/qatar_test_doc.pdf

Step 2 — Run Document Processing
python process_document.py

Step 3 — Create Embeddings
python create_embeddings.py

Step 4 — Run Streamlit App
python -m streamlit run app.py


You should see:

✔ Ready to use
✔ Total chunks loaded
✔ Text, table, image counts
✔ Chat interface active

🧪 Example Queries
Type	Sample Question
Text	“Summarize Qatar’s economic outlook.”
Tables	“What does the table on page 43 show?”
OCR Images	“What text was extracted from scanned images?”
Cross-Modal	“Compare inflation values from text and tables.”
🧠 Future Improvements (Excellence Track)

Cross-modal reranking using CLIP

Hybrid search: RRF fusion (BM25 + dense vectors)

Structured KPI extraction

Analytics dashboard (latency, recall charts)

Long-context LLM integration (Llama-3-Long, GPT-4.1-1106)

🎥 Video Demonstration

A 3–5 minute video script is included covering:

Pipeline explanation

Code walkthrough

Application demo

Insights

This satisfies the assignment’s mandatory video requirement.

👤 Author

Mohammed
Multi-Modal RAG Engineer
Big AIR Lab Candidate Submission