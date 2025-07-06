# src/preprocessing.py

import os
import pandas as pd
import re

ALLOWED_PRODUCTS = [
    'Credit card',
    'Personal loan',
    'Buy Now, Pay Later (BNPL)',
    'Savings account',
    'Money transfers'
]

class ComplaintDataLoader:
    def __init__(self, filename='data/raw/complaints.csv'):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        self.path = filename

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

def filter_products(df: pd.DataFrame, allowed: list = ALLOWED_PRODUCTS) -> pd.DataFrame:
    return df[df["Product"].isin(allowed)].copy()

def remove_missing_narratives(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Consumer complaint narrative'].notnull()].copy()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text
