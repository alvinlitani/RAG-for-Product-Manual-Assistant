import os
import gradio as gr
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from config import (
    HF_API_TOKEN,
    LLM_MODEL,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    TOP_K,
)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN


# Load vector store
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id=LLM_MODEL,
    task="text-generation",
    max_new_tokens=2048,
    do_sample=False,
    repetition_penalty=1.03,
)
model = ChatHuggingFace(llm=llm)


# 2-step RAG: always retrieve context before LLM call
@dynamic_prompt
def rag_prompt(request: ModelRequest) -> str:
    """Retrieve relevant docs and inject as system prompt context."""
    last_message = request.state["messages"][-1].text
    retrieved_docs = vectorstore.similarity_search(last_message, k=TOP_K)

    docs_content = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return (
        "You are a helpful technical assistant that answers questions "
        "about product documentation.\n"
        "Use only the provided context to answer. If the context doesn't "
        "contain enough information, say so honestly.\n\n"
        f"Context:\n{docs_content}"
    )


# Create agent with RAG middleware
agent = create_agent(
    model=model,
    tools=[],
    middleware=[rag_prompt],
)


def respond(message, history):
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].text


demo = gr.ChatInterface(
    fn=respond,
    title="Product Manual Assistant",
    description="Ask questions about product documentation. Powered by RAG.",
    examples=[
        "What safety precautions should I follow?",
        "How do I configure the network settings?",
        "What are the technical specifications?",
    ],
)

if __name__ == "__main__":
    demo.launch()