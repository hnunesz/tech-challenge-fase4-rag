"""Construção da base de conhecimento: seleção, limpeza e enriquecimento das avaliações.

Cada avaliação com comentário vira um documento. Os comentários da Olist são curtos
(máx. ~200 caracteres), então não há chunking: 1 avaliação = 1 documento = 1 vetor,
o que preserva a rastreabilidade (review_id) de cada evidência.
"""

import re
import unicodedata

import pandas as pd

from .config import PROCESSED_PATH, RAW_DIR

_WS = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([!?.,])\1{2,}")
_LETTER = re.compile(r"[a-zA-ZÀ-ÿ]")


def clean_text(text) -> str:
    """Normaliza unicode, remove quebras de linha e pontuação repetida.

    Acentos e caixa são mantidos: o modelo de embeddings é multilíngue e foi
    treinado com texto natural, então "não" e "nao" já ficam próximos.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _REPEATED_PUNCT.sub(r"\1\1", text)
    return _WS.sub(" ", text).strip()


def normalize_for_dedup(text: str) -> str:
    """Forma canônica usada para agrupar comentários idênticos ("Ótimo" == "otimo!")."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]", " ", text)).strip()


def is_informative(text: str) -> bool:
    """Descarta comentários sem conteúdo (".", "ok", "...", "10").

    Mantemos textos a partir de 2 palavras com letras: "Não recebi" é curto,
    mas é exatamente o tipo de evidência que interessa.
    """
    words = [w for w in text.split() if _LETTER.search(w)]
    return len(words) >= 2


def _order_features(raw_dir) -> pd.DataFrame:
    """Uma linha por pedido com informações de entrega e categoria do produto."""
    orders = pd.read_csv(
        raw_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    items = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(raw_dir / "product_category_name_translation.csv")

    products = products.merge(translation, on="product_category_name", how="left")
    items = items.merge(
        products[["product_id", "product_category_name", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    # Pedidos com vários itens: usamos a categoria mais frequente do pedido.
    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        product_category=(
            "product_category_name",
            lambda s: s.mode().iat[0] if s.notna().any() else None,
        ),
        product_category_en=(
            "product_category_name_english",
            lambda s: s.mode().iat[0] if s.notna().any() else None,
        ),
        total_price=("price", "sum"),
    )

    o = orders.set_index("order_id")
    delivered = o["order_delivered_customer_date"]
    estimated = o["order_estimated_delivery_date"]
    feats = pd.DataFrame(index=o.index)
    feats["order_status"] = o["order_status"]
    feats["purchase_date"] = o["order_purchase_timestamp"]
    feats["delivery_days"] = (delivered - o["order_purchase_timestamp"]).dt.days
    feats["days_late"] = (delivered - estimated).dt.days
    feats["delivered_late"] = (delivered > estimated).where(delivered.notna())
    return feats.join(per_order, how="left")


def build_knowledge_base(raw_dir=RAW_DIR) -> pd.DataFrame:
    reviews = pd.read_csv(
        raw_dir / "olist_order_reviews_dataset.csv",
        parse_dates=["review_creation_date"],
    )
    n_total = len(reviews)

    # 1) Seleção: só avaliações com algum texto (título ou comentário).
    reviews["title"] = reviews["review_comment_title"].map(clean_text)
    reviews["message"] = reviews["review_comment_message"].map(clean_text)
    reviews = reviews[(reviews["title"] != "") | (reviews["message"] != "")].copy()

    # Título e comentário são unidos: o título costuma ser um resumo ("Produto com defeito").
    def _join(row):
        if row.title and row.message and normalize_for_dedup(row.title) not in normalize_for_dedup(row.message):
            return f"{row.title}. {row.message}"
        return row.message or row.title

    reviews["text"] = reviews.apply(_join, axis=1)

    # 2) Limpeza: remove textos sem informação e review_id duplicado
    #    (a mesma avaliação aparece ligada a mais de um pedido).
    reviews = reviews[reviews["text"].map(is_informative)]
    reviews = reviews.drop_duplicates("review_id")

    # 3) Enriquecimento com metadados do pedido.
    df = reviews.merge(_order_features(raw_dir), left_on="order_id", right_index=True, how="left")

    # 4) Valores ausentes. O ChromaDB não aceita None em metadados, então usamos
    #    sentinelas explícitas (-1 / "desconhecida") em vez de descartar a avaliação:
    #    o texto continua útil mesmo sem a informação de entrega ou categoria.
    df["product_category"] = (
        df["product_category"].fillna("desconhecida").str.replace("_", " ")
    )
    df["product_category_en"] = df["product_category_en"].fillna("unknown").str.replace("_", " ")
    df["order_status"] = df["order_status"].fillna("desconhecido")
    df["delivered_late"] = df["delivered_late"].map({True: 1, False: 0}).fillna(-1).astype(int)
    for col in ("delivery_days", "days_late", "n_items"):
        df[col] = df[col].fillna(-1).astype(int)
    df["total_price"] = df["total_price"].fillna(-1.0).round(2)

    df["sentiment"] = pd.cut(
        df["review_score"], bins=[0, 2, 3, 5], labels=["negativa", "neutra", "positiva"]
    ).astype(str)
    df["review_date"] = df["review_creation_date"].dt.strftime("%Y-%m-%d")
    df["review_year"] = df["review_creation_date"].dt.year.astype(int)
    df["review_ym"] = (df["review_creation_date"].dt.year * 100 + df["review_creation_date"].dt.month).astype(int)
    df["text_norm"] = df["text"].map(normalize_for_dedup)

    cols = [
        "review_id", "order_id", "text", "text_norm", "review_score", "sentiment",
        "review_date", "review_year", "review_ym", "order_status", "delivered_late",
        "delivery_days", "days_late", "product_category", "product_category_en",
        "n_items", "total_price",
    ]
    df = df[cols].reset_index(drop=True)
    print(f"Avaliações na base original: {n_total:,}")
    print(f"Documentos na base de conhecimento: {len(df):,} ({df['text_norm'].nunique():,} textos distintos)")
    return df


def load_or_build(force: bool = False) -> pd.DataFrame:
    if PROCESSED_PATH.exists() and not force:
        return pd.read_parquet(PROCESSED_PATH)
    df = build_knowledge_base()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df
