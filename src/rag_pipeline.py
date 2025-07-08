# src/rag_pipeline.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
import pickle
import os

# Load FAISS index and metadata
def load_vector_store(path="vector_store"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    with open(os.path.join(path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return db, metadata, embeddings

# Prompt template
prompt_template = PromptTemplate.from_template(
    """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# Load lightweight LLM pipeline (e.g., distilGPT2)
def get_llm():
    gen_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=200)
    return HuggingFacePipeline(pipeline=gen_pipeline)

# RAG QA function
def answer_question(query, vectordb, embedder, llm, k=5):
    # Retrieve top-k relevant chunks
    docs = vectordb.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = prompt_template.format(context=context, question=query)
    answer = llm(prompt)
    return answer, docs
