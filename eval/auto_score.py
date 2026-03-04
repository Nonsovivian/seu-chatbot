import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def score_answer(question, answer, sources, mode):
    source_text = ", ".join(sources) if sources else "none"

    prompt = f"""You are evaluating a university chatbot answer. Score strictly and fairly.

Question: {question}
Mode: {mode}
Sources cited: {source_text}
Answer: {answer}

Return ONLY a JSON object with these exact keys:
- citation_correct: 0, 1, or 2 (0=no citation or wrong, 1=partial, 2=fully supported). For BASELINE always 0.
- hallucination: 0 or 1 (0=nothing made up, 1=contains unsupported claim)
- completeness: 0, 1, or 2 (0=not answered, 1=partly answered, 2=fully answered)
- refusal_quality: 0, 1, or 2 (2=correctly refused with follow-up, 1=refused unclearly, 0=wrongly refused a valid question). Score 2 if answer is good and NOT a refusal.
- notes: one short sentence explaining your scores

JSON only, no other text."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())


def run_auto_score():
    eval_dir = "eval"
    results_files = sorted([f for f in os.listdir(eval_dir) if f.startswith("results_") and f.endswith(".jsonl")])

    if not results_files:
        print("No results file found in eval/")
        return

    results_file = os.path.join(eval_dir, results_files[-1])
    print(f"Using: {results_file}\n")

    with open(results_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    scored = []
    for i, r in enumerate(results):
        print(f"[{r['id']}] Scoring RAG + Baseline...")

        rag_scores = score_answer(r["question"], r["rag_answer"], r["rag_sources"], "RAG")
        time.sleep(0.5)
        base_scores = score_answer(r["question"], r["baseline_answer"], [], "BASELINE")
        time.sleep(0.5)

        scored.append({
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "rag_answer": r["rag_answer"],
            "rag_sources": r["rag_sources"],
            "rag_response_time": r["rag_response_time"],
            "rag_citation_correct": rag_scores["citation_correct"],
            "rag_hallucination": rag_scores["hallucination"],
            "rag_completeness": rag_scores["completeness"],
            "rag_refusal_quality": rag_scores["refusal_quality"],
            "rag_notes": rag_scores["notes"],
            "baseline_answer": r["baseline_answer"],
            "baseline_response_time": r["baseline_response_time"],
            "baseline_hallucination": base_scores["hallucination"],
            "baseline_completeness": base_scores["completeness"],
            "baseline_notes": base_scores["notes"],
        })

        print(f"  RAG — citation:{rag_scores['citation_correct']} hallucination:{rag_scores['hallucination']} completeness:{rag_scores['completeness']}")
        print(f"  BASE — hallucination:{base_scores['hallucination']} completeness:{base_scores['completeness']}\n")

    output_path = os.path.join(eval_dir, "scored_results.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    print(f"\nScoring complete. Saved to {output_path}")
    print(f"Total questions scored: {len(scored)}")
    print(f"\nRAG Summary:")
    print(f"  Avg Citation Score:  {sum(r['rag_citation_correct'] for r in scored)/len(scored):.2f} / 2")
    print(f"  Hallucinations:      {sum(r['rag_hallucination'] for r in scored)} / {len(scored)}")
    print(f"  Avg Completeness:    {sum(r['rag_completeness'] for r in scored)/len(scored):.2f} / 2")
    print(f"\nBaseline Summary:")
    print(f"  Hallucinations:      {sum(r['baseline_hallucination'] for r in scored)} / {len(scored)}")
    print(f"  Avg Completeness:    {sum(r['baseline_completeness'] for r in scored)/len(scored):.2f} / 2")

if __name__ == "__main__":
    run_auto_score()
