import os
from dotenv import load_dotenv

# import variables from .env file
load_dotenv()

# HuggingFace API token
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# HuggingFace models used
LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # popular model trained on English text, change model if multi-lingual manual 

# document chunking
CHUNK_SIZE = 1000       # size target of 1000 characters
CHUNK_OVERLAP = 200     # each chunk shares 200 characters with the chunk before it

# vector store
VECTORSTORE_DIR = "vectorstore"         # filesystem path
COLLECTION_NAME = "product_manuals"     # table name

# Number of retrieved chunks sent to LLM
TOP_K = 4