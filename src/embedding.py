# src/embedding.py

import os
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle

class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=True)

class ComplaintChunker:
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, texts: List[str]):
        return self.splitter.create_documents(texts)

def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

def save_index(index, metadata, save_path="vector_store/"):
    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, os.path.join(save_path, "faiss_index.index"))
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
