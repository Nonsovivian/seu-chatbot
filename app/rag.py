import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="vector_store")
    index = load_index_from_storage(storage_context)
    return index

def get_rag_answer(question, index):
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )
    response = query_engine.query(question)
    
    answer = str(response)
    
    sources = []
    if response.source_nodes:
        for node in response.source_nodes:
            filename = node.metadata.get("file_name", "Unknown source")
            sources.append(filename)
    sources = list(dict.fromkeys(sources))
    
    if not response.source_nodes or len(answer.strip()) < 10:
        return {
            "answer": "I don't have enough information in the knowledge base to answer that confidently. Could you rephrase your question or ask about a specific CSC302 topic such as deadlines, submission rules, or marking criteria?",
            "sources": [],
            "mode": "rag"
        }
    
    return {
        "answer": answer,
        "sources": sources,
        "mode": "rag"
    }

def get_baseline_answer(question):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.complete(
        f"You are a university student support assistant. Answer this question: {question}"
    )
    return {
        "answer": str(response),
        "sources": [],
        "mode": "baseline"
    }

if __name__ == "__main__":
    print("Loading index...")
    index = load_index()
    print("Index loaded. Type your question or 'quit' to exit.")
    print("Commands: 'rag: your question' or 'base: your question'\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        
        if user_input.startswith("base:"):
            question = user_input[5:].strip()
            result = get_baseline_answer(question)
            print(f"\n[BASELINE] {result['answer']}\n")
        else:
            question = user_input.replace("rag:", "").strip()
            result = get_rag_answer(question, index)
            print(f"\n[RAG] {result['answer']}")
            if result["sources"]:
                print(f"Sources: {', '.join(result['sources'])}\n")
            else:
                print()
