# app.py

import streamlit as st
from src.rag_pipeline import load_vector_store, get_llm, answer_question

st.set_page_config(page_title="CrediTrust Complaint Assistant", layout="centered")
st.title("CrediTrust Complaint Analysis Chatbot")
st.write("Ask any question about customer complaints. Example: 'Why are users unhappy with BNPL?'")

# Load model and vector store
@st.cache_resource(show_spinner=False)
def load_resources():
    return load_vector_store("vector_store"), get_llm()

(vectordb, metadata, embedder), llm = load_resources()

# User input
query = st.text_input("Enter your question")
ask = st.button("Ask")
clear = st.button("Clear")

if ask and query:
    with st.spinner("Generating response..."):
        answer, sources = answer_question(query, vectordb, embedder, llm)
        st.markdown("### Answer")
        st.success(answer)

        st.markdown("---")
        st.markdown("### Source Excerpts")
        for i, doc in enumerate(sources[:2]):
            with st.expander(f"Source {i+1}"):
                st.write(doc.page_content)

if clear:
    st.experimental_rerun()
