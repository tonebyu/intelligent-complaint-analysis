# 🧠 Intelligent Complaint Analysis for Financial Services

**10 Academy – AI Mastery Challenge (Week 2)**  
**Date:** July 2 – July 8, 2025  
**Team:** CrediTrust AI Solutions

---

## 🚀 Project Overview

CrediTrust Financial, operating across East Africa, receives thousands of customer complaints monthly through its mobile platform. These complaints contain valuable insights but are largely unstructured and difficult to analyze.

This project presents a **Retrieval-Augmented Generation (RAG) AI chatbot** that enables internal teams to:

- Ask **natural language questions**.
- Receive **contextual answers** grounded in real complaint narratives.
- Identify **emerging product issues** and customer pain points rapidly.

---

## 🎯 Objectives

- 📌 Detect major complaint trends **within minutes**.
- 📌 Reduce analyst workload by automating complaint insights.
- 📌 Support **proactive product and compliance decisions**.

---

## ⚙️ Technical Stack

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Data Source**      | CFPB public complaint dataset (filtered for key products)                   |
| **Preprocessing**    | Narrative cleaning, product filtering, EDA visualizations                   |
| **Embedding**        | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace                    |
| **Vector Store**     | FAISS for fast similarity search and metadata indexing                      |
| **LLM Generator**    | `distilgpt2` from HuggingFace (`text-generation` pipeline)                  |
| **RAG Pipeline**     | Custom Retriever + Prompt Template + LLM response                           |
| **Frontend**         | Streamlit UI for live question answering with source traceability           |

---

## 🧪 Sample Evaluation

| ❓ Question                         | ✅ Answer Summary                                      | ⭐ Rating (1–5) |
|------------------------------------|--------------------------------------------------------|----------------|
| Why are users unhappy with BNPL?   | Auto-renewals, hidden fees, unclear repayment terms    | 4              |
| Issues with money transfers?       | Delayed transactions and unreachable support teams     | 5              |

 Over 80% of responses accurately reflected source complaints.  
🔁 Future iterations can include larger LLMs and keyword clustering for deeper insights.

---

## 📂 Project Structure

├── data/ # Raw and filtered datasets
├── notebooks/ # EDA and development notebooks
├── src/ # Core logic: RAG pipeline, preprocessing, utilities
├── vector_store/ # FAISS index and metadata
├── reports/ # Visualizations, evaluation, and screenshots
├── app.py # Streamlit UI
├── README.md # Project overview
