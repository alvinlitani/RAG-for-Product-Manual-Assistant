from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSTORE_DIR,
    COLLECTION_NAME,
)
import os
import glob


def load_pdfs(data_dir="data/manuals"):
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return []

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)")
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store created at '{VECTORSTORE_DIR}' with collection '{COLLECTION_NAME}'")
    return vectorstore


def main():
    docs = load_pdfs()
    if not docs:
        return

    chunks = chunk_documents(docs)
    create_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()