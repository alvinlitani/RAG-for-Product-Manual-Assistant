# import libraries and variables
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings     # runs the model locally model (alt HuggingFaceEndpointEmbeddings)
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

# function to load PDF files from the directory ./data/manuals
def load_pdfs(data_dir = "data/manuals"):

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))                  # get all .pdf files filepath
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return []                                                           # return empty list if no pdf files found

    docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")                                       
        pdf_loader = PyPDFLoader(pdf_path, mode="page")                     # load the pdf per page into Document objects                               
        docs.extend(pdf_loader.load())                                      # insert the Document objects into docs

    print(f"Loaded {len(docs)} pages from {len(pdf_files)} PDF(s)")
    return docs

# function to split the document into chunks 
def chunk_documents(docs):
    
    # RecursiveCharacterTextSplitter will split at character delimiters and not strict number of characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    chunks = splitter.split_documents(docs)                                 # split each Document into chunks
    print(f"Split into {len(chunks)} chunks")
    return chunks

# function to create vector table from the chunks
def create_vectorstore(chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)          # which embedding model is used

    # Transform the chunks into Chroma vectors and store it in persist_directory with table name of collection_name 
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
        return      # end the program if docs is empty

    chunks = chunk_documents(docs)
    create_vectorstore(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()