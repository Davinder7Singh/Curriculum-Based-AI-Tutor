<img width="1028" height="551" alt="image" src="https://github.com/user-attachments/assets/bf499609-a508-4b7e-9a1e-452a9727f7e2" />


# Curriculum-Based-AI-Tutor
An AI-powered tutor that answers student questions from NCERT Class 8 Science using a Retrieval-Augmented Generation (RAG) pipeline.
It combines semantic search (FAISS) with Groq’s LLaMA3 LLM to deliver curriculum-focused, transparent, and interactive responses.

# Project Overview

- Answers student queries from NCERT Class 8 Science textbook
- Uses RAG pipeline for accurate and grounded answers
- Connects FAISS-based semantic search with Groq LLaMA3 LLM
- Provides source references for transparency

# Key Components

- Data Prep: PDF → JSONL → Passages
- Embeddings: all-MiniLM-L6-v2 (Sentence-Transformers)
- Retriever: FAISS similarity search (L2 distance)
- LLM: Groq LLaMA3 (8B parameters)
- Interface: Streamlit single-page chatbot
- Evaluation: BLEU, ROUGE-L, BERTScore


# Code & Implementation

- Notebook (ai tutor.ipynb)
Data cleaning, chunking, embeddings, FAISS index creation
RAG pipeline assembly & evaluation scripts
- App (app.py)
Loads embeddings + index
Runs retriever + Groq LLM
Provides Streamlit-based interactive UI
