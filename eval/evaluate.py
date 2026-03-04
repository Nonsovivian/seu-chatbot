import json
import time
import datetime
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def load_index():
    sc = StorageContext.from_defaults(persist_dir="vector_store")
    return load_index_from_storage(sc)

def get_rag_answer(q, idx):
    qe = idx.as_query_engine(similarity_top_k=5, response_mode="compact")
    r = qe.query(q)
    ans = str(r)
    srcs = list(dict.fromkeys([n.metadata.get("file_name", "Unknown") for n in (r.source_nodes or [])]))
    has_sources = bool(r.source_nodes) and len(ans.strip()) >= 10
    return {"answer": ans, "sources": srcs, "has_sources": has_sources}

def get_baseline_answer(q):
    r = OpenAI(model="gpt-3.5-turbo", temperature=0).complete(
        f"You are a university student support assistant. Answer this question: {q}"
    )
    return {"answer": str(r), "sources": [], "has_sources": False}

def run_evaluation():
    os.makedirs("eval", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    with open("eval/questions.jsonl", "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    print(f"Loaded {len(questions)} questions")
    print("Loading index...")
    idx = load_index()
    print("Index loaded. Starting evaluation...\n")

    results = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        category = q["category"]

        print(f"[{qid}] {question}")

        start = time.time()
        rag = get_rag_answer(question, idx)
        rag_time = round(time.time() - start, 2)

        start = time.time()
        baseline = get_baseline_answer(question)
        base_time = round(time.time() - start, 2)

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "rag_answer": rag["answer"],
            "rag_sources": rag["sources"],
            "rag_has_sources": rag["has_sources"],
            "rag_response_time": rag_time,
            "baseline_answer": baseline["answer"],
            "baseline_response_time": base_time
        })

        print(f"  RAG ({rag_time}s) — sources: {', '.join(rag['sources']) if rag['sources'] else 'none'}")
        print(f"  BASELINE ({base_time}s)\n")

        time.sleep(0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"eval/results_{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. Results saved to {output_path}")
    print(f"Total questions: {len(results)}")
    print(f"RAG answered with sources: {sum(1 for r in results if r['rag_has_sources'])}/{len(results)}")

if __name__ == "__main__":
    run_evaluation()
