# CSC302 Student Support Chatbot

**Chinonso Ojinnaka — Student No: 2317501**  
**Supervisor: Dr Daniyal Haider — 7CS077 Dissertation**  
**South-East University (fictitious case study)**

---

## What is this?

This is the chatbot I built for my Master's dissertation. The idea came from a real problem, students in Nigerian universities often struggle to get quick, reliable answers about their modules. Lecturers are busy, office hours are limited, and asking on WhatsApp groups usually gets you five different answers.

So I built a chatbot that reads the actual module documents and answers questions directly from them. Ask it about a deadline, a submission rule, or a marking criterion, it tells you exactly what the handbook says and shows you which document it got the answer from.

The project uses a technique called Retrieval-Augmented Generation (RAG), which means the chatbot doesn't just guess, it looks things up first, then answers. This makes it far more reliable than a standard AI chatbot which can confidently make things up.

---

## How it works

When a student asks a question, the system searches through the module documents to find the most relevant sections, then passes those sections to the language model to generate a clear answer. The source documents are always shown so the student can verify the answer themselves.

There is also a Baseline mode (RAG turned off) which lets the chatbot answer without looking anything up. This was included purely for dissertation comparison, to show how much better grounded answers are compared to ungrounded ones.

---

## Project Structure

```
student_chatbot/
├── app/
│   ├── app.py          — the web interface
│   ├── ingest.py       — reads the KB documents and builds the vector database
│   └── rag.py          — handles retrieval, generation, and citations
├── assets/
│   ├── logo.png
│   ├── student.jpg
│   └── building.jpg
├── eval/
│   ├── questions.jsonl         — 40 test questions used in evaluation
│   ├── evaluate.py             — runs questions through both RAG and Baseline
│   ├── auto_score.py           — scores answers automatically using AI
│   ├── results_*.jsonl         — raw results
│   └── scored_results.jsonl    — final scored results
├── kb/                  — all module documents the chatbot reads from
├── logs/                — chat logs and flagged responses
├── vector_store/        — the saved vector database
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting it running

### 1. Clone the repo

```bash
git clone https://github.com/Nonsovivian/seu-chatbot.git
cd seu-chatbot
```

### 2. Set up the virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Copy `.env.example` to a new file called `.env` and paste your key in:

```
OPENAI_API_KEY=your-key-here
```

### 5. Build the vector database

This only needs to be done once. It reads all the documents in the `kb/` folder and converts them into searchable embeddings.

```bash
python app/ingest.py
```

### 6. Start the chatbot

```bash
streamlit run app/app.py
```

Then open `http://localhost:8501` in your browser.

---

## Running the evaluation

To run all 40 test questions through both RAG and Baseline:

```bash
python eval/evaluate.py
```

To automatically score the results:

```bash
python eval/auto_score.py
```

---

## Some decisions I made along the way

I used LlamaIndex's built-in SimpleVectorStore instead of ChromaDB because ChromaDB requires C++ build tools to install on Windows and I wanted this to run on any machine without extra setup. It works just as well for a project this size.

I set the LLM temperature to 0 so answers are consistent and deterministic, important when comparing RAG vs Baseline across 40 questions.

I chose GPT-3.5-turbo because it is fast, cheap, and more than capable enough for question answering over short document chunks. The evaluation results confirmed this.

---

## What the evaluation found

After running 40 questions through both modes:

| Metric | RAG | Baseline |
|---|---|---|
| Citation accuracy | 95% | 0% |
| Hallucinations | 0 out of 40 | 0 out of 40 |
| Completeness | 1.98 out of 2 | 1.98 out of 2 |
| Answers with sources | 39 out of 40 | 0 out of 40 |

Both modes answered questions well, but only RAG showed where each answer came from. For a student making decisions about deadlines or submissions, that traceability matters a lot.

---

## Ethics

Every participant was told this is a research prototype and not an official university service. No names or personal details were collected. The chatbot itself displays a disclaimer on every page and directs students to their lecturer for anything high-stakes.

---

## Acknowledgements

Thanks to Dr Daniyal Haider for supervising this project, and to all the students who took the time to test the chatbot and fill in the questionnaire.

Built at the University of Wolverhampton, 2026.
