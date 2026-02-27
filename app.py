# import libraries and variables
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings     # runs the model locally (alt HuggingFaceEndpointEmbeddings)
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
import os
import gradio

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_TOKEN


# load vector store from persist_directory with table name of collection_name 
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)

# initialize the LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id=LLM_MODEL,
    task="text-generation",
    max_new_tokens=2048,                # maximum number of output tokens
    do_sample=False,                    # less variable responses making more consistent/factual answers      
    repetition_penalty=1.03,            # small penalty if it repeats
)
# wrap the LLM in a chat model to handle sent/received message formatting for any particular LLM
model = ChatHuggingFace(llm=llm)


# 2-step RAG: always retrieve context before LLM call
@dynamic_prompt         # dynamically generate system prompt before every LLM call
def rag_prompt(request: ModelRequest) -> str:       # function to retrieve relevant docs and use them to build system prompt 
    
    # embeds the user's question and searches ChromaDB for the chunks with most similar embedding
    user_message = request.state["messages"][-1].text
    retrieved_docs = vectorstore.similarity_search(user_message, k=TOP_K)

    # loops thru the chunks and adds metadata alongside actual content
    docs_content = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}]\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )

    # the actual system prompt
    return (
        "You are a helpful technical assistant that answers questions about product documentation.\n"
        "Use only the provided context to answer. If the context doesn't contain enough information, say so honestly."
        f"Context:\n{docs_content}"
    )


# create agent 
agent = create_agent(
    model=model,
    tools=[],                       # no additional tools used, change to [search_manual] for agentic RAG    
    middleware=[rag_prompt],        # run rag_prompt before every LLM call, remove for agentic RAG
)

# function to handle the actual messages to and from the LLM
def respond(message, history):
    try:
        messages = []

        # when calling LLM, append last 2 call/response history beforehand
        for user_msg, assistant_msg in history[-2:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
       
        messages.append({"role": "user", "content": message})

        result = agent.invoke({"messages": messages})
        return result["messages"][-1].text
    
    except Exception as e:
        return f"Error: {str(e)}"       # print error message if occurs

# chat frontend using gradio
demo = gradio.ChatInterface(
    fn=respond,
    title="Product Manual Assistant",
    description="Ask questions about Raspberry Pi Pico product documentation.",
    examples=[
        "What safety precautions should I follow?",
        "How do I program the flash memory?",
        "What are the technical specifications?",
    ],
    retry_btn="Retry",
    undo_btn="Undo",
    theme=gradio.themes.Glass(),
)

if __name__ == "__main__":
    demo.launch()