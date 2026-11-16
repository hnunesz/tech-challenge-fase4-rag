"""Etapas de recuperação do RAG.

pergunta -> análise (filtros de metadados + reescrita) -> embedding da pergunta
-> busca vetorial filtrada (top-N) -> corte por similaridade -> deduplicação
-> MMR (relevância x diversidade) -> evidências para o contexto.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import config
from .data_prep import normalize_for_dedup
from .embeddings import embed_query, get_collection


@dataclass
class QueryPlan:
    original: str
    search_text: str
    filters: dict = field(default_factory=dict)  # campo -> condição no formato do Chroma
    notes: list = field(default_factory=list)    # explicação legível dos filtros aplicados

    def where(self):
        clauses = [{k: v} for k, v in self.filters.items()]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@dataclass
class Evidence:
    ref: int
    review_id: str
    text: str
    similarity: float
    metadata: dict
    duplicates: int = 1  # quantas avaliações com o mesmo texto entre os candidatos


# --- Transformação da pergunta ---------------------------------------------------
_WS = re.compile(r"\s+")

# Palavras de "moldura" da pergunta não descrevem o conteúdo de uma avaliação
# e só adicionam ruído ao embedding ("Quais são os principais ... relatados pelos clientes?").
_BOILERPLATE = re.compile(
    r"\b(quais|qual|que|o que|existem|existe|há|sao|são|os|as|principais|mais|"
    r"frequentes?|frequencia|frequência|aparecem|com maior|relatad[oa]s?|relatam|"
    r"pelos|clientes?|consumidores?|nas|nos|em suas|suas|seus|avaliações|avaliacoes|"
    r"comentários|comentarios|podem ser|identificad[oa]s|padrões|padroes|recorrentes|"
    r"sustentam|conclusão|conclusao|temas?|exemplos?|"
    r"sobre|de|da|do|dos|das|na|no|em|com|um|uma|o|a|é|à|ao|categoria|relacionad[oa]s?)\b",
    re.IGNORECASE,
)

_RULES = [
    # (regex na pergunta normalizada, campo, condição, explicação)
    (r"\bnota (1|um)\b", "review_score", {"$eq": 1}, "nota = 1"),
    (r"\bnota (5|cinco)\b", "review_score", {"$eq": 5}, "nota = 5"),
    (r"insatisf|nota(s)? baixa|negativ|detrator|irritad|decepcion", "review_score", {"$lte": 2}, "nota ≤ 2 (insatisfeitos)"),
    (r"reclama|problema|queixa|defeito|falha|ruim|pessim", "review_score", {"$lte": 3}, "nota ≤ 3 (reclamações)"),
    (r"(?<!in)satisfeit|elogi|positiv|nota(s)? alta|valoriz|gostam|gostaram|promotor", "review_score", {"$gte": 4}, "nota ≥ 4 (positivas)"),
    (r"atras|fora do prazo|depois do prazo", "delivered_late", {"$eq": 1}, "entrega atrasada"),
    (r"\b(no prazo|antes do prazo)\b", "delivered_late", {"$eq": 0}, "entrega no prazo"),
    (r"\b2016\b", "review_year", {"$eq": 2016}, "ano 2016"),
    (r"\b2017\b", "review_year", {"$eq": 2017}, "ano 2017"),
    (r"\b2018\b", "review_year", {"$eq": 2018}, "ano 2018"),
]


def _known_categories() -> list[str]:
    from .data_prep import load_or_build

    cats = load_or_build()["product_category"].value_counts()
    return [c for c in cats.index if c != "desconhecida"]


_CATEGORIES: list[str] | None = None


def analyze_question(question: str, auto_filters: bool = True, manual_filters: dict | None = None) -> QueryPlan:
    """Reescreve a pergunta e extrai filtros de metadados (nota, entrega, ano, categoria).

    A primeira regra que define um campo vence, por isso as regras mais específicas
    vêm antes ("nota 1" antes de "insatisfeito"). Filtros escolhidos na interface
    (manual_filters) têm prioridade sobre os automáticos.
    """
    global _CATEGORIES
    norm = normalize_for_dedup(question)
    plan = QueryPlan(original=question, search_text=question)

    if auto_filters:
        for pattern, fld, cond, note in _RULES:
            if fld not in plan.filters and re.search(pattern, norm):
                plan.filters[fld] = cond
                plan.notes.append(note)

        if _CATEGORIES is None:
            _CATEGORIES = _known_categories()
        # Categorias mais longas primeiro para "moveis escritorio" vencer "moveis".
        for cat in sorted(_CATEGORIES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(cat)}\b", norm):
                plan.filters["product_category"] = {"$eq": cat}
                plan.notes.append(f"categoria = {cat}")
                break

    for fld, cond in (manual_filters or {}).items():
        plan.filters[fld] = cond
        plan.notes.append(f"{fld} {cond} (manual)")

    stripped = _WS.sub(" ", _BOILERPLATE.sub(" ", question)).strip(" ?.,")
    # Se a reescrita esvaziar a pergunta, buscamos pela pergunta original.
    plan.search_text = stripped if stripped else question
    return plan



# --- Recuperação ------------------------------------------------------------------
def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: evita que o contexto seja 12 variações de "muito bom"."""
    sim_q = cand_vecs @ query_vec
    sim_c = cand_vecs @ cand_vecs.T
    selected = [int(np.argmax(sim_q))]
    while len(selected) < min(k, len(cand_vecs)):
        rest = [i for i in range(len(cand_vecs)) if i not in selected]
        scores = [lam * sim_q[i] - (1 - lam) * sim_c[i, selected].max() for i in rest]
        selected.append(rest[int(np.argmax(scores))])
    return selected


@dataclass
class RetrievalResult:
    plan: QueryPlan
    evidences: list
    n_candidates: int          # retornados pelo índice
    n_relevant: int            # acima do limite de similaridade
    max_similarity: float
    stats: dict


def retrieve(
    plan: QueryPlan,
    k_candidates: int = config.TOP_K_CANDIDATES,
    k_context: int = config.TOP_K_CONTEXT,
    min_similarity: float = config.MIN_SIMILARITY,
    mmr_lambda: float = config.MMR_LAMBDA,
) -> RetrievalResult:
    q_vec = embed_query(plan.search_text)
    res = get_collection().query(
        query_embeddings=[q_vec.tolist()],
        n_results=k_candidates,
        where=plan.where(),
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    ids, docs = res["ids"][0], res["documents"][0]
    metas, dists = res["metadatas"][0], res["distances"][0]
    vecs = np.asarray(res["embeddings"][0]) if ids else np.empty((0, len(q_vec)))
    sims = [1 - d for d in dists]  # distância de cosseno -> similaridade
    max_sim = max(sims, default=0.0)

    # Corte por similaridade: o que está abaixo não é evidência, só "o menos distante".
    keep = [i for i, s in enumerate(sims) if s >= min_similarity]
    n_relevant = len(keep)

    # Deduplicação: textos idênticos viram uma evidência com contador.
    groups: dict[str, int] = {}
    counts: Counter = Counter()
    for i in keep:
        key = metas[i]["text_norm"]
        counts[key] += 1
        if key not in groups:
            groups[key] = i
    unique_idx = list(groups.values())

    chosen = []
    if unique_idx:
        order = _mmr(q_vec, vecs[unique_idx], k_context, mmr_lambda)
        chosen = [unique_idx[j] for j in order]

    evidences = [
        Evidence(
            ref=n,
            review_id=ids[i],
            text=docs[i],
            similarity=round(sims[i], 4),
            metadata=metas[i],
            duplicates=counts[metas[i]["text_norm"]],
        )
        for n, i in enumerate(chosen, start=1)
    ]
    return RetrievalResult(
        plan=plan,
        evidences=evidences,
        n_candidates=len(ids),
        n_relevant=n_relevant,
        max_similarity=round(max_sim, 4),
        stats=_stats([metas[i] for i in keep]),
    )


def _stats(metas: list[dict]) -> dict:
    """Agregados das avaliações relevantes: dão base quantitativa a afirmações de frequência."""
    if not metas:
        return {}
    n = len(metas)
    late = [m["delivered_late"] for m in metas if m["delivered_late"] >= 0]
    return {
        "n": n,
        "nota_media": round(sum(m["review_score"] for m in metas) / n, 2),
        "dist_notas": dict(sorted(Counter(m["review_score"] for m in metas).items())),
        "pct_entrega_atrasada": round(100 * sum(late) / len(late), 1) if late else None,
        "top_categorias": Counter(m["product_category"] for m in metas).most_common(3),
    }
