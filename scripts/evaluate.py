"""Roda o conjunto de perguntas de exemplo e gera docs/exemplos.md.

Inclui perguntas fora do escopo para verificar se o sistema reconhece
quando não há evidência suficiente.

Uso: uv run python scripts/evaluate.py [--calibrate]
"""

import argparse
import time
from pathlib import Path

from voc_rag import config
from voc_rag.rag import ask
from voc_rag.retriever import analyze_question, retrieve

IN_SCOPE = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?",
    "O que os clientes mais elogiam na categoria cama mesa banho?",
    "Quais temas aparecem nas avaliações com entrega atrasada?",
]
OUT_OF_SCOPE = [
    "Qual é a capital da França?",
    "Como faço uma receita de bolo de cenoura?",
    "Quem ganhou a Copa do Mundo de 2002?",
    "Qual o preço da ação da Petrobras hoje?",
]

ROOT = Path(__file__).resolve().parents[1]


def calibration_report() -> None:
    """Mede as duas salvaguardas pré-geração em 40 perguntas (20 dentro e 20 fora do escopo).

    GATE = barrada pelo limite de similaridade; SIM/NAO = veredito do verificador (LLM).
    """
    from calibration_questions import IN_SCOPE as CAL_IN, OUT_OF_SCOPE as CAL_OUT

    from voc_rag.generator import judge_evidence

    accepted = {"DENTRO": 0, "FORA": 0}
    for label, qs in (("DENTRO", CAL_IN), ("FORA", CAL_OUT)):
        for q in qs:
            r = retrieve(analyze_question(q))
            if len(r.evidences) < config.MIN_EVIDENCES:
                verdict = "GATE"
            else:
                verdict = "SIM" if judge_evidence(q, r) else "NAO"
                accepted[label] += verdict == "SIM"
            print(f"{label:6} max={r.max_similarity:.3f} evid={len(r.evidences):2d} {verdict:4} | {q}")
    print(f"\nAceitas: {accepted['DENTRO']}/{len(CAL_IN)} dentro do escopo, {accepted['FORA']}/{len(CAL_OUT)} fora do escopo")


def full_report() -> None:
    lines = [
        "# Exemplos de perguntas e respostas",
        "",
        f"Gerado por `scripts/evaluate.py` · embeddings `{config.EMBED_MODEL}` · "
        f"LLM `{config.LLM_PROVIDER}`/`{config.LLM_MODEL}` · similaridade mínima {config.MIN_SIMILARITY}",
        "",
    ]
    for q in IN_SCOPE + OUT_OF_SCOPE:
        t0 = time.time()
        res = ask(q)
        r = res.retrieval
        print(f"[{res.status}] {time.time() - t0:5.1f}s | {q}")
        lines += [
            f"## {q}",
            "",
            f"- **Status:** `{res.status}` · **tempo:** {res.seconds:.1f}s",
            f"- **Busca semântica por:** _{r.plan.search_text}_",
            f"- **Filtros aplicados:** {', '.join(r.plan.notes) or 'nenhum'}",
            f"- **Recuperação:** {r.n_relevant} de {r.n_candidates} candidatos acima do limite "
            f"(similaridade máx. {r.max_similarity:.3f})",
            "",
            "**Resposta:**",
            "",
            res.answer,
            "",
        ]
        if res.reason:
            lines += [f"> Motivo da recusa: {res.reason}.", ""]
        for w in res.warnings:
            lines.append(f"> ⚠️ {w}\n")
        if res.status == "ok":
            lines += ["**Evidências citadas:**", "", "| ref | nota | categoria | comentário | review_id |", "|---|---|---|---|---|"]
            for e in res.cited_evidences:
                text = e.text.replace("|", "/")
                lines.append(
                    f"| [{e.ref}] | {e.metadata['review_score']} | {e.metadata['product_category']} | {text} | `{e.review_id}` |"
                )
            lines.append("")
    out = ROOT / "docs" / "exemplos.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório salvo em {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="avalia só as salvaguardas (limite + verificador)")
    args = parser.parse_args()
    calibration_report() if args.calibrate else full_report()
