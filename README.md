# Product Manual Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers technical questions about product documentation. Built with LangChain v1, HuggingFace, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-v1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Demo

<!-- Add a screenshot or GIF of the Gradio interface here -->
<!-- ![Demo Screenshot](docs/demo.png) -->

## Overview

This application allows users to ask natural language questions about product documentation and receive accurate, context-grounded answers. Instead of searching through hundreds of pages manually, the chatbot retrieves the most relevant sections and generates a focused response.

Drop any product manual PDF into the `data/manuals/` directory, run the ingestion pipeline, and start asking questions.

### How It Works

1. **Document Ingestion** — PDF manuals are loaded, split into chunks, embedded using sentence-transformers, and stored in a ChromaDB vector database.
2. **Query Processing** — When a user asks a question, the query is embedded and used to find the most similar document chunks via vector similarity search.
3. **Response Generation** — The retrieved chunks are injected as context into a system prompt, and an LLM generates an answer grounded in the documentation.

### Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Embed Query │────▶│  ChromaDB    │────▶│ Top-K Chunks│
│  (MiniLM)   │     │  Similarity  │     │  Retrieved  │
└─────────────┘     │  Search      │     └──────┬──────┘
                    └──────────────┘            │
                                               ▼
                                    ┌──────────────────┐
                                    │  Dynamic Prompt   │
                                    │  (System + Context│
                                    │   + User Query)   │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   HuggingFace    │
                                    │   Inference API  │
                                    │   (GPT-OSS-20B)  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                       Answer to User
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain v1 |
| LLM | openai/gpt-oss-20b via HuggingFace Inference API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | ChromaDB (persistent, local) |
| Frontend | Gradio |
| Agent Runtime | LangGraph (via create_agent) |
| Document Loader | LangChain PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |

## Project Structure

```
product-manual-assistant/
├── app.py                  # Gradio UI + RAG agent with dynamic prompt middleware
├── ingest.py               # Document loading, chunking, and embedding pipeline
├── config.py               # Centralized configuration (models, chunk sizes, etc.)
├── requirements.txt        # Python dependencies
├── .env                    # HuggingFace API token (not committed)
├── .gitignore
├── README.md
├── data/
│   └── manuals/            # Product manual PDFs (not committed)
├── vectorstore/            # ChromaDB persistent storage (not committed)
└── docs/
    └── architecture.md     # Detailed architecture notes (optional)
```

## Getting Started

### Prerequisites

- Python 3.11
- Anaconda or Miniconda
- A [HuggingFace account](https://huggingface.co/) with an API token

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/product-manual-assistant.git
   cd product-manual-assistant
   ```

2. Create and activate the conda environment:
   ```bash
   conda create -n product-manual-rag python=3.11 -y
   conda activate product-manual-rag
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```
   HF_API_TOKEN=your_huggingface_token_here
   ```

5. Place one or more product manual PDFs in `data/manuals/`.

### Running

**Step 1: Ingest documents** (only needed once, or when adding new PDFs)
```bash
python ingest.py
```

**Step 2: Launch the chatbot**
```bash
python app.py
```

Open the URL shown in the terminal (default: `http://127.0.0.1:7860`) to start chatting.

## Configuration

All key parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `openai/gpt-oss-20b` | HuggingFace model for response generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for text embeddings |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

## Design Decisions

- **LangChain v1 with create_agent**: Uses the current recommended approach with dynamic prompt middleware for 2-step RAG, rather than deprecated chain patterns.
- **2-step RAG over agentic RAG**: For a documentation Q&A bot, always retrieving context is more predictable and requires only one LLM call per query.
- **HuggingFace Inference API**: Zero cost, no GPU required, and demonstrates ability to work with open-source models.
- **ChromaDB**: Lightweight, persistent, no external server needed — appropriate for a portfolio-scale project.
- **RecursiveCharacterTextSplitter**: Respects natural text boundaries (paragraphs, sentences) while maintaining target chunk size.

## Potential Improvements

- [ ] Structure-aware chunking to better handle tables and section headers in PDFs
- [ ] Contextual retrieval (adding section summaries to each chunk before embedding)
- [ ] Conversational memory for multi-turn follow-up questions
- [ ] Evaluation pipeline to measure retrieval accuracy and response quality
- [ ] Support for multiple document formats (DOCX, HTML, Markdown)
- [ ] Deployment to HuggingFace Spaces for a live demo

## License

MIT