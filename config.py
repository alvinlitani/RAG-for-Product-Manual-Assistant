import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector store
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "product_manuals"

# Retrieval
TOP_K = 4