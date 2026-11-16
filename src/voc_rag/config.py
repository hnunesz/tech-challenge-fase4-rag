"""Configurações centrais do projeto (caminhos, modelos e hiperparâmetros do RAG)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- Caminhos ---------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "reviews.parquet"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION_NAME = "olist_reviews"

# --- Modelos ----------------------------------------------------------------
# multilingual-e5-small: 118M parâmetros, bom em português, rápido em CPU.
# Exige os prefixos "query: " (pergunta) e "passage: " (documento).
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "transformers")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Recuperação -------------------------------------------------------------
TOP_K_CANDIDATES = 200  # candidatos trazidos do índice vetorial (pool amplo para o MMR achar temas distintos)
TOP_K_CONTEXT = 12      # evidências que vão para o contexto do LLM (após MMR)
MMR_LAMBDA = 0.5        # 1.0 = só relevância, 0.0 = só diversidade
MIN_SIMILARITY = 0.86   # similaridade de cosseno mínima para uma avaliação contar como evidência (calibrada em scripts/evaluate.py)
MIN_EVIDENCES = 3       # abaixo disso o sistema declara evidência insuficiente
MAX_NEW_TOKENS = 450
