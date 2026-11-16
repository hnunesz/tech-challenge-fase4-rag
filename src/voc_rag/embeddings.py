"""Embeddings e índice vetorial (ChromaDB persistente, distância de cosseno)."""

from functools import lru_cache

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

METADATA_COLS = [
    "review_id", "order_id", "review_score", "sentiment", "review_date", "review_year",
    "review_ym", "order_status", "delivered_late", "delivery_days", "days_late",
    "product_category", "product_category_en", "n_items", "total_price", "text_norm",
]


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_passages(texts, batch_size: int = 128) -> np.ndarray:
    return get_embedder().encode(
        [f"passage: {t}" for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(text: str) -> np.ndarray:
    return get_embedder().encode([f"query: {text}"], normalize_embeddings=True)[0]


def get_collection(create: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create:
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION_NAME)
        return client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
        )
    return client.get_collection(COLLECTION_NAME)


def build_index(df: pd.DataFrame, batch_size: int = 5000) -> None:
    """Gera os embeddings e grava texto + vetor + metadados no ChromaDB.

    Textos idênticos são embutidos uma única vez e o vetor é reaproveitado
    (há ~5 mil repetições como "Muito bom"), economizando tempo de CPU.
    """
    unique_texts = df["text"].drop_duplicates().tolist()
    vectors = embed_passages(unique_texts)
    lookup = dict(zip(unique_texts, vectors))
    all_vectors = np.stack([lookup[t] for t in df["text"]])

    collection = get_collection(create=True)
    meta = df[METADATA_COLS].to_dict("records")
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        collection.add(
            ids=df["review_id"].iloc[start:end].tolist(),
            documents=df["text"].iloc[start:end].tolist(),
            embeddings=all_vectors[start:end].tolist(),
            metadatas=meta[start:end],
        )
    print(f"Índice criado em {CHROMA_DIR} com {collection.count():,} documentos.")
