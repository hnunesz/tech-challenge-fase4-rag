"""Faz uma pergunta ao RAG pelo terminal.

Uso: uv run python scripts/ask.py "O que os clientes relatam sobre problemas na entrega?"
"""

import sys

from voc_rag.rag import ask


def print_answer(res) -> None:
    r = res.retrieval
    print(f"\nPergunta: {res.question}")
    print(f"Busca por: {r.plan.search_text!r} | filtros: {', '.join(r.plan.notes) or 'nenhum'}")
    print(
        f"Candidatos: {r.n_candidates} | acima do limite: {r.n_relevant} | "
        f"similaridade máx.: {r.max_similarity} | status: {res.status} | {res.seconds:.1f}s\n"
    )
    print(res.answer)
    if res.reason:
        print(f"(motivo: {res.reason})")
    for w in res.warnings:
        print(f"\n[aviso] {w}")
    print("\nEvidências:")
    for e in r.evidences:
        mark = "*" if e.ref in res.cited_refs else " "
        m = e.metadata
        print(
            f" {mark}[{e.ref}] sim={e.similarity:.3f} nota={m['review_score']} "
            f"cat={m['product_category']} review_id={e.review_id}\n      {e.text}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print_answer(ask(" ".join(sys.argv[1:])))
