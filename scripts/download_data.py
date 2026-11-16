"""Baixa o dataset da Olist do Kaggle (via kagglehub) e copia os CSVs usados para data/raw/.

Uso: uv run --with kagglehub python scripts/download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

FILES = [
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    src = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(src / f, RAW / f)
        print(f"ok  {f}")
