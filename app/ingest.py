import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

print("Loading documents...")

documents = SimpleDirectoryReader(
    input_dir="kb",
    recursive=True,
    required_exts=[".docx", ".pdf", ".txt", ".md"]
).load_data()

print(f"{len(documents)} chunks loaded")
print("Building index...")

index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

index.storage_context.persist(persist_dir="vector_store")

print("Vector store saved to vector_store/")
