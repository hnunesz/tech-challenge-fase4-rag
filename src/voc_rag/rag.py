"""Orquestração do pipeline RAG com as salvaguardas contra respostas sem evidência."""

import re
import time
from dataclasses import dataclass, field

from . import config
from .generator import generate_answer, judge_evidence, llm_enabled
from .retriever import RetrievalResult, analyze_question, retrieve

INSUFFICIENT_MSG = (
    "Não encontrei avaliações suficientes na base para responder a essa pergunta com segurança. "
    "A base contém apenas comentários de clientes da Olist (2016–2018) sobre compras, produtos, "
    "entregas e atendimento. Tente reformular a pergunta ou relaxar os filtros."
)
_CITE = re.compile(r"\[(\d+)\]")


@dataclass
class RAGAnswer:
    question: str
    status: str                  # ok | insuficiente | sem_llm
    answer: str
    retrieval: RetrievalResult
    cited_refs: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    reason: str = ""             # por que a resposta foi recusada, quando for o caso
    seconds: float = 0.0

    @property
    def cited_evidences(self):
        return [e for e in self.retrieval.evidences if e.ref in self.cited_refs]


def _check_citations(raw: str, valid: set[int]) -> tuple[str, list[int], list[str]]:
    """Remove citações inexistentes e descarta tópicos que não citam nenhuma avaliação.

    Um tópico sem citação é uma afirmação sem evidência rastreável, então não é exibido.
    """
    warnings, kept, dropped, invalid = [], [], 0, set()
    for line in raw.splitlines():
        # Modelos pequenos às vezes repetem o bloco de tópicos ("Formato padrão: ...").
        if line.strip() and (line in kept or line.strip().lower().startswith("formato")):
            continue
        refs = [int(n) for n in _CITE.findall(line)]
        invalid |= {n for n in refs if n not in valid}
        line = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", line).rstrip()
        is_topic = line.lstrip().startswith(("-", "*", "•")) or re.match(r"\s*\d+[.)]\s", line)
        if is_topic and not _CITE.search(line):
            dropped += 1
            continue
        kept.append(line)
    if invalid:
        warnings.append(f"Citações a avaliações inexistentes foram removidas: {sorted(invalid)}")
    if dropped:
        warnings.append(f"{dropped} tópico(s) sem citação de evidência foram omitidos.")
    answer = "\n".join(kept).strip()
    cited = sorted({int(n) for n in _CITE.findall(answer)})
    return answer, cited, warnings


def ask(
    question: str,
    auto_filters: bool = True,
    manual_filters: dict | None = None,
    min_similarity: float = config.MIN_SIMILARITY,
    k_context: int = config.TOP_K_CONTEXT,
) -> RAGAnswer:
    t0 = time.time()

    def refuse(retrieval, reason):
        return RAGAnswer(question, "insuficiente", INSUFFICIENT_MSG, retrieval,
                         reason=reason, seconds=time.time() - t0)

    # 1-4) Transformação da pergunta, embedding, busca e seleção das evidências.
    plan = analyze_question(question, auto_filters=auto_filters, manual_filters=manual_filters)
    retrieval = retrieve(plan, k_context=k_context, min_similarity=min_similarity)

    # Salvaguarda 1 (sem LLM): poucas avaliações acima do limite de similaridade.
    # Barra perguntas claramente fora do escopo e filtros sem resultado, sem dar
    # ao modelo a chance de "completar" a resposta com conhecimento próprio.
    if len(retrieval.evidences) < config.MIN_EVIDENCES:
        return refuse(retrieval, f"apenas {len(retrieval.evidences)} avaliação(ões) acima da similaridade mínima {min_similarity}")

    if not llm_enabled():
        return RAGAnswer(
            question, "sem_llm",
            "LLM desativado (LLM_PROVIDER=none). Seguem as avaliações mais relevantes encontradas.",
            retrieval, seconds=time.time() - t0,
        )

    # Salvaguarda 2 (LLM como juiz): casos de fronteira em que a busca vetorial sempre
    # devolve "algo parecido" (ex.: "cotação do dólar" recupera avaliações sobre preço).
    if not judge_evidence(question, retrieval):
        return refuse(retrieval, "o verificador de relevância julgou que as avaliações recuperadas não tratam do tema da pergunta")

    # 5) Construção do contexto e geração.
    raw = generate_answer(question, retrieval)

    # Salvaguarda 3: só sobrevivem afirmações com citação válida.
    answer, cited, warnings = _check_citations(raw, {e.ref for e in retrieval.evidences})
    if not cited:
        return refuse(retrieval, "a resposta gerada não citou nenhuma avaliação")

    return RAGAnswer(question, "ok", answer, retrieval, cited, warnings, seconds=time.time() - t0)
