import os
import json
import datetime
import base64
from pathlib import Path
from dotenv import load_dotenv
import nltk
os.makedirs("/tmp/nltk_data", exist_ok=True)
nltk.data.path.append("/tmp/nltk_data")
nltk.download("punkt_tab", download_dir="/tmp/nltk_data", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", download_dir="/tmp/nltk_data", quiet=True)
import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

st.set_page_config(
    page_title="SEU Student Support — CSC302",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_html = '<span style="font-size:20px;font-weight:800;color:white;">SOUTH-EAST UNIVERSITY</span>'
logo_path = Path("assets/logo.png")
if logo_path.exists():
    logo_html = f'<img src="data:image/png;base64,{img_to_b64(logo_path)}" style="height:50px;object-fit:contain;">'

student_tag = ""
student_path = Path("assets/student.jpg")
if student_path.exists():
    sb64 = img_to_b64(student_path)
    student_tag = f'<img src="data:image/jpeg;base64,{sb64}" style="height:280px;object-fit:contain;object-position:bottom;align-self:flex-end;filter:drop-shadow(0 4px 24px rgba(0,0,0,0.4));margin-right:40px;flex-shrink:0;">'

building_bg = "background:linear-gradient(135deg,#081f4a 0%,#0d3272 60%,#1a4a9e 100%);"
building_path = Path("assets/building.jpg")
if building_path.exists():
    bb64 = img_to_b64(building_path)
    building_bg = f"background-image:linear-gradient(to right,rgba(8,31,74,0.92) 0%,rgba(13,50,114,0.82) 50%,rgba(13,50,114,0.55) 100%),url('data:image/jpeg;base64,{bb64}');background-size:cover;background-position:center;"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');
:root {{--navy:#0d3272;--navy-dark:#081f4a;--navy-light:#1a4a9e;--gold:#c9a84c;--cream:#f9f7f2;--border:#dde3ed;--muted:#6b7280;}}
html,body,[class*="css"]{{font-family:'Source Sans 3',sans-serif;}}
.stApp{{background-color:var(--cream);}}
header[data-testid="stHeader"]{{background:transparent;height:0;}}
.seu-topbar{{background:linear-gradient(135deg,var(--navy-dark) 0%,var(--navy) 60%,var(--navy-light) 100%);padding:0 40px;display:flex;align-items:center;justify-content:space-between;height:72px;position:fixed;top:0;left:0;right:0;z-index:999;box-shadow:0 2px 20px rgba(8,31,74,0.35);}}
.seu-topbar-right{{font-size:12px;color:rgba(255,255,255,0.65);text-align:right;line-height:1.6;}}
.seu-topbar-right span{{color:var(--gold);font-weight:600;}}
.seu-hero{{{building_bg}min-height:340px;margin-top:72px;padding:56px 56px 0 56px;display:flex;align-items:flex-end;justify-content:space-between;overflow:hidden;}}
.hero-text{{padding-bottom:40px;max-width:580px;}}
.hero-eyebrow{{font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--gold);margin-bottom:12px;}}
.hero-title{{font-family:'Playfair Display',serif;font-size:44px;font-weight:700;color:white;line-height:1.15;margin:0 0 14px 0;}}
.hero-title span{{color:var(--gold);}}
.hero-sub{{font-size:15px;color:rgba(255,255,255,0.78);line-height:1.65;font-weight:300;}}
.main-content{{margin-top:72px;padding:24px 40px 80px 40px;max-width:900px;}}
.disclaimer{{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid var(--gold);border-radius:8px;padding:10px 16px;font-size:13px;color:#78350f;margin-bottom:20px;line-height:1.6;}}
.source-tag{{display:inline-block;background:#eef2fb;border:1px solid #c7d4f0;border-radius:20px;padding:4px 12px;font-size:12px;color:var(--navy);margin-top:8px;font-weight:500;}}
.badge-rag{{display:inline-block;background:#dcfce7;color:#166534;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;margin-bottom:6px;}}
.badge-base{{display:inline-block;background:#fef3c7;color:#92400e;border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;margin-bottom:6px;}}
[data-testid="stChatMessage"]{{background:white !important;border-radius:12px !important;border:1px solid var(--border) !important;padding:16px !important;margin-bottom:10px !important;box-shadow:0 1px 4px rgba(0,0,0,0.04) !important;max-width:820px;}}
[data-testid="stSidebar"]{{background:var(--navy-dark) !important;}}
[data-testid="stSidebar"] *{{color:white !important;}}
[data-testid="stSidebar"] hr{{border-color:rgba(255,255,255,0.15) !important;}}
.sidebar-section{{font-size:10px;font-weight:700;letter-spacing:1.5px;color:rgba(255,255,255,0.4) !important;text-transform:uppercase;margin:16px 0 8px 0;}}
.status-rag{{background:rgba(22,163,74,0.2);border:1px solid rgba(22,163,74,0.4);border-radius:8px;padding:10px 14px;font-size:13px;color:#86efac !important;font-weight:500;}}
.status-base{{background:rgba(234,179,8,0.15);border:1px solid rgba(234,179,8,0.3);border-radius:8px;padding:10px 14px;font-size:13px;color:#fde047 !important;font-weight:500;}}
.stButton button{{background:transparent !important;border:1px solid rgba(255,255,255,0.25) !important;color:rgba(255,255,255,0.75) !important;border-radius:8px !important;font-size:13px !important;}}
</style>
<div class="seu-topbar">
    {logo_html}
    <div class="seu-topbar-right">
        <span>CSC302</span> Artificial Intelligence &amp; Applications<br>
        Student Support Portal &nbsp;·&nbsp; Research Prototype
    </div>
</div>
<div class="seu-hero">
    <div class="hero-text">
        <div class="hero-eyebrow">CSC302 · South-East University, Nigeria</div>
        <h1 class="hero-title">Relax! <span>How can we help</span><br>you today?</h1>
        <p class="hero-sub">Your AI-powered student support assistant — ask anything about CSC302 deadlines, submissions, marking criteria, or module policies.</p>
    </div>
    {student_tag}
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_index():
    sc = StorageContext.from_defaults(persist_dir="vector_store")
    return load_index_from_storage(sc)

def get_rag_answer(q, idx):
    qe = idx.as_query_engine(similarity_top_k=5, response_mode="compact")
    r = qe.query(q)
    ans = str(r)
    srcs = list(dict.fromkeys([n.metadata.get("file_name", "Unknown") for n in (r.source_nodes or [])]))
    if not r.source_nodes or len(ans.strip()) < 10:
        return {"answer": "I don't have enough information in the knowledge base to answer that confidently. Please rephrase or ask about CSC302 deadlines, submission rules, or marking criteria.", "sources": [], "mode": "rag"}
    return {"answer": ans, "sources": srcs, "mode": "rag"}

def get_baseline_answer(q):
    r = OpenAI(model="gpt-3.5-turbo", temperature=0).complete(f"You are a university student support assistant. Answer: {q}")
    return {"answer": str(r), "sources": [], "mode": "baseline"}

def save_log(q, a, s, m):
    os.makedirs("logs", exist_ok=True)
    with open("logs/chat_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "mode": m, "question": q, "answer": a, "sources": s}) + "\n")

def save_issue(q):
    os.makedirs("logs", exist_ok=True)
    with open("logs/issues.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(), "reported_question": q}) + "\n")


with st.sidebar:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Answer Mode</div>', unsafe_allow_html=True)
    rag_mode = st.toggle("RAG — Knowledge Base", value=True)
    show_sources = st.toggle("Show Sources", value=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    if rag_mode:
        st.markdown('<div class="status-rag">✓ Grounded in CSC302 documents</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-base">⚡ Baseline — no document grounding</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Project Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:rgba(255,255,255,0.45);line-height:1.9;">
    Dissertation research prototype evaluating AI-powered student support in Nigerian HEIs.<br><br>
    <strong style="color:rgba(255,255,255,0.75)">Chinonso Ojinnaka</strong><br>
    Supervisor: Dr Daniyal Haider<br>
    Course: 7CS077 Dissertation
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sidebar-section">Theme</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px;color:rgba(255,255,255,0.6);line-height:1.8;">
    Want light or dark mode? Click the <strong style="color:white;">☰ menu</strong> top right, then <strong style="color:white;">Settings</strong>.
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="main-content">', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>⚠ Prototype notice:</strong> This is a research prototype, not an official university service.
    Answers are based on CSC302 module documents only. For extensions, appeals or grade queries always
    contact <strong>Dr Amaka Obi</strong> (a.obi@seu.edu.ng) or the Student Office directly.
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("mode"):
            bc = "badge-rag" if msg["mode"] == "rag" else "badge-base"
            bt = "KB Grounded" if msg["mode"] == "rag" else "Baseline"
            st.markdown(f'<span class="{bc}">{bt}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            st.markdown(f'<div class="source-tag">📄 {" · ".join(msg["sources"])}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about CSC302..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..." if rag_mode else "Generating response..."):
            try:
                idx = load_index()
                res = get_rag_answer(prompt, idx) if rag_mode else get_baseline_answer(prompt)
                bc = "badge-rag" if res["mode"] == "rag" else "badge-base"
                bt = "KB Grounded" if res["mode"] == "rag" else "Baseline"
                st.markdown(f'<span class="{bc}">{bt}</span>', unsafe_allow_html=True)
                st.markdown(res["answer"])
                if show_sources and res.get("sources"):
                    st.markdown(f'<div class="source-tag">📄 {" · ".join(res["sources"])}</div>', unsafe_allow_html=True)
                c1, c2 = st.columns([10, 1])
                with c2:
                    if st.button("🚩", key=f"r{len(st.session_state.messages)}", help="Report issue"):
                        save_issue(prompt)
                        st.toast("Issue reported.")
                save_log(prompt, res["answer"], res.get("sources", []), res["mode"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"], "sources": res.get("sources", []), "mode": res["mode"]})
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)
