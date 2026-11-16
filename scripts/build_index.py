"""Etapa offline: prepara a base de conhecimento e cria o índice vetorial.

Uso: uv run python scripts/build_index.py [--force]
"""

import argparse
import time

from voc_rag.data_prep import load_or_build
from voc_rag.embeddings import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="refaz a preparação dos dados")
    args = parser.parse_args()

    t0 = time.time()
    df = load_or_build(force=args.force)
    print(f"Base pronta: {len(df):,} documentos")
    build_index(df)
    print(f"Concluído em {time.time() - t0:.0f}s")
