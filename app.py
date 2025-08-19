import os
import json
import time
import csv
from pathlib import Path
from typing import List, Dict

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Client as GroqClient

# ========================
# CONFIG
# ========================
DATA_DIR = Path("DATA")
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.index"
METADATA_PATH = DATA_DIR / "passage_metadata.json"
LOG_CSV = DATA_DIR / "interaction_log.csv"

EMB_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-8b-8192"
K_RETRIEVE = 5

# ========================
# LOAD MODELS + INDEX
# ========================
@st.cache_resource
def load_models_and_index():
    embed_model = SentenceTransformer(EMB_MODEL_NAME)
    idx = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)
    return embed_model, idx, passages

embed_model, index, passages = load_models_and_index()

# ========================
# RETRIEVAL
# ========================
def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, I = index.search(np.array(q_emb, dtype="float32"), top_k)
    hits = []
    for ii in I[0]:
        if 0 <= ii < len(passages):
            hits.append(passages[ii])
    return hits

# ========================
# GROQ GENERATION
# ========================
#groq_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
groq_client = GroqClient(api_key="gsk_g2Gu0krzPZOCvThl43WGdyb3FY9CQPJKsxqFiNiuCA8i947EyX")

def groq_generate(prompt: str, model: str = GROQ_MODEL, max_tokens: int = 512) -> str:
    resp = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# ========================
# PROMPT BUILDER + QA
# ========================
def build_prompt(question: str, retrieved: List[Dict]) -> str:
    snippets = []
    for i, s in enumerate(retrieved):
        snippets.append(f"[{i+1}] (Chapter: {s.get('chapter')})\n{s.get('text')}\n")
    context_block = "\n\n".join(snippets)
    prompt = f"""
You are an AI tutor for NCERT Class 8 Science. 
Answer the student's question using ONLY the facts from the provided textbook snippets below.

If the question cannot be answered from the textbook, reply exactly:
"I'm focused on Class 8 Science content. I don't have a textbook answer for that."

Provide:
1) A short, grade-appropriate answer (1–6 sentences).
2) A 'Sources' line listing the snippet indices you used, e.g. [1],[3].

Context snippets:
{context_block}

Question:
{question}

Answer:
"""
    return prompt

def answer_question(question: str, top_k: int = K_RETRIEVE) -> Dict:
    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        return {
            "answer": "I'm focused on Class 8 Science; I don't have content to answer that question.",
            "retrieved": []
        }
    prompt = build_prompt(question, retrieved)
    gen = groq_generate(prompt, model=GROQ_MODEL, max_tokens=512)

    # log interaction
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), question, gen, "|".join([r["id"] for r in retrieved])])

    return {"answer": gen, "retrieved": retrieved}

# ========================
# STREAMLIT UI
# ========================
st.set_page_config(page_title="NCERT Class 8 Science Tutor", layout="wide")

st.title("📘 NCERT Class 8 Science Tutor")
st.write("Ask questions from the Class 8 NCERT Science textbook. The AI will answer only from textbook content.")

# input box
user_q = st.text_input("❓ Enter your question:")

if st.button("Get Answer") and user_q:
    with st.spinner("Retrieving and generating answer..."):
        out = answer_question(user_q, top_k=K_RETRIEVE)

    st.subheader("📌 Answer")
    st.write(out["answer"])

    if out["retrieved"]:
        st.subheader("📖 Sources")
        for i, r in enumerate(out["retrieved"], start=1):
            with st.expander(f"Snippet {i} (Chapter: {r['chapter']})"):
                st.write(r["text"])
