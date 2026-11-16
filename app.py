"""Interface Streamlit do Voice of Customer Intelligence.

Uso: uv run streamlit run app.py
"""

import pandas as pd
import streamlit as st

from voc_rag import config
from voc_rag.data_prep import load_or_build
from voc_rag.rag import ask

st.set_page_config(page_title="VoC Intelligence – Olist", page_icon="💬", layout="wide")

EXAMPLES = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "O que os clientes mais elogiam na categoria cama mesa banho?",
    "Qual é a capital da França?",
]


@st.cache_data
def categories() -> list[str]:
    return sorted(load_or_build()["product_category"].unique())


with st.sidebar:
    st.header("Filtros de metadados")
    auto = st.toggle("Detectar filtros na pergunta", value=True,
                     help="Ex.: 'insatisfeitos' → nota ≤ 2; 'atraso' → entrega atrasada; nome de categoria → categoria.")
    score = st.slider("Nota da avaliação", 1, 5, (1, 5))
    cats = st.multiselect("Categoria do produto", categories(), placeholder="Todas")
    late = st.radio("Entrega", ["Todas", "Atrasada", "No prazo"], horizontal=True)
    years = st.multiselect("Ano da avaliação", [2016, 2017, 2018], placeholder="Todos")
    st.header("Recuperação")
    k_context = st.slider("Evidências no contexto", 4, 20, config.TOP_K_CONTEXT)
    min_sim = st.slider("Similaridade mínima", 0.70, 0.95, config.MIN_SIMILARITY, 0.01)
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`  \nLLM: `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`")

manual = {}
if score != (1, 5):
    manual["review_score"] = {"$in": list(range(score[0], score[1] + 1))}
if cats:
    manual["product_category"] = {"$in": cats}
if late != "Todas":
    manual["delivered_late"] = {"$eq": 1 if late == "Atrasada" else 0}
if years:
    manual["review_year"] = {"$in": years}

st.title("💬 Voice of Customer Intelligence")
st.caption("Perguntas em linguagem natural sobre ~39 mil avaliações de clientes da Olist (2016–2018), "
           "respondidas com base nas avaliações recuperadas.")

example = st.selectbox("Exemplos", EXAMPLES, index=None, placeholder="Escolha uma pergunta de exemplo") or ""
question = st.text_input("Sua pergunta", value=example)

if st.button("Perguntar", type="primary", disabled=not question.strip()):
    with st.spinner("Recuperando avaliações e gerando a resposta..."):
        res = ask(question, auto_filters=auto, manual_filters=manual,
                  min_similarity=min_sim, k_context=k_context)
    r = res.retrieval

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", res.status)
    c2.metric("Avaliações relevantes", f"{r.n_relevant}/{r.n_candidates}")
    c3.metric("Similaridade máx.", f"{r.max_similarity:.3f}")
    c4.metric("Tempo", f"{res.seconds:.1f}s")
    st.caption(f"Busca semântica por: *{r.plan.search_text}* · Filtros: {', '.join(r.plan.notes) or 'nenhum'}")

    if res.status == "insuficiente":
        st.warning(res.answer)
        st.caption(f"Motivo: {res.reason}.")
    else:
        st.markdown(res.answer)
    for w in res.warnings:
        st.info(w)

    if r.stats:
        st.subheader("Perfil das avaliações relevantes")
        s1, s2, s3 = st.columns(3)
        s1.metric("Nota média", r.stats["nota_media"])
        s2.metric("% entrega atrasada", r.stats["pct_entrega_atrasada"])
        s3.write("**Categorias mais comuns**  \n" + "  \n".join(f"{c} ({n})" for c, n in r.stats["top_categorias"]))

    if r.evidences:
        if res.status == "insuficiente":
            st.subheader("Avaliações mais próximas")
            st.caption("Não foram usadas como evidência: estão listadas só para transparência da recusa.")
        else:
            st.subheader("Evidências recuperadas")
            st.caption("✅ = citada na resposta. Cada linha é uma avaliação real, identificada pelo review_id.")
        rows = [{
            "ref": f"[{e.ref}]",
            "citada": "✅" if e.ref in res.cited_refs else "",
            "similaridade": e.similarity,
            "nota": e.metadata["review_score"],
            "comentário": e.text,
            "repetições": e.duplicates,
            "categoria": e.metadata["product_category"],
            "entrega": {1: "atrasada", 0: "no prazo"}.get(e.metadata["delivered_late"], "—"),
            "data": e.metadata["review_date"],
            "review_id": e.review_id,
        } for e in r.evidences]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
