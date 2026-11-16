"""Construção do contexto e geração da resposta pelo LLM.

Duas chamadas ao modelo, cada uma com uma tarefa simples (importante para LLMs pequenos):
1. judge_evidence: "as avaliações recuperadas respondem à pergunta?" -> SIM/NAO (poucos tokens)
2. generate_answer: síntese em formato fixo, com cada tópico citando as avaliações [n]

Provedores suportados (LLM_PROVIDER no .env):
- transformers : modelo do Hugging Face rodando localmente em CPU (padrão, sem API key)
- ollama       : servidor Ollama local
- openai_compat: qualquer API compatível com OpenAI (Groq, OpenRouter, LM Studio...)
- none         : sem LLM; devolve só as evidências (modo extrativo)
"""

from functools import lru_cache

import requests

from . import config

# --- Prompts ------------------------------------------------------------------------
JUDGE_PROMPT = """Você verifica se avaliações de clientes de uma loja online servem para responder a uma pergunta.
Responda apenas SIM ou NAO.
SIM: pelo menos parte das avaliações fala do assunto da pergunta (ex.: entrega, prazo, produto, atendimento, qualidade, elogios, reclamações).
NAO: a pergunta é sobre outro assunto (ex.: geografia, esportes, cotações, receitas, conhecimentos gerais) ou as avaliações não tratam do tema perguntado."""

ANSWER_PROMPT = """Você é um analista de Voice of Customer de um e-commerce brasileiro.
Responda à pergunta usando SOMENTE as avaliações numeradas fornecidas. Não use conhecimento próprio.

Formato obrigatório:
Resumo: <uma ou duas frases respondendo à pergunta>
- <padrão encontrado nas avaliações> [n][n]
- <padrão encontrado nas avaliações> [n]
(3 a 6 tópicos)

Regras:
- Agrupe as avaliações em temas DISTINTOS (um tema por tópico) e tente cobrir todas as avaliações relevantes.
- Todo tópico TERMINA com os números das avaliações que o comprovam, entre colchetes.
- Só cite um número se aquela avaliação realmente disser o que o tópico afirma.
- Não invente números, porcentagens, produtos ou causas.

Exemplo:
Avaliações:
[1] (nota 1) "Produto veio quebrado"
[2] (nota 2) "chegou com a tampa rachada, péssima embalagem"
[3] (nota 1) "não recebi até hoje"
Pergunta: Quais problemas os clientes relatam?
Resposta:
Resumo: Os clientes relatam principalmente produtos danificados e pedidos não entregues.
- Produtos chegam quebrados ou danificados, às vezes por embalagem ruim [1][2]
- Pedidos que nunca chegaram ao cliente [3]"""


def format_evidence(ev) -> str:
    m = ev.metadata
    entrega = {1: "atrasada", 0: "no prazo"}.get(m["delivered_late"], "sem informação")
    repetida = f", {ev.duplicates} clientes escreveram o mesmo" if ev.duplicates > 1 else ""
    return (
        f"[{ev.ref}] (nota {m['review_score']}, {m['product_category']}, "
        f"entrega {entrega}{repetida}) \"{ev.text}\""
    )


def build_context(retrieval) -> str:
    return "\n".join(format_evidence(ev) for ev in retrieval.evidences)


def judge_messages(question: str, retrieval, n: int = 6) -> list[dict]:
    # As mais similares, não a ordem do MMR: o juiz avalia se *existe* evidência sobre o tema,
    # e a diversidade do MMR traria avaliações periféricas para a amostra.
    top = sorted(retrieval.evidences, key=lambda e: e.similarity, reverse=True)[:n]
    sample = "\n".join(f"- \"{ev.text}\"" for ev in top)
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Pergunta: {question}\n\nAvaliações:\n{sample}\n\nSIM ou NAO?"},
    ]


def answer_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": f"Avaliações:\n{context}\n\nPergunta: {question}\nResposta:"},
    ]


# --- Provedores ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _hf_pipeline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, dtype=torch.float32)
    model.eval()
    return tok, model


def _generate_transformers(messages, max_tokens) -> str:
    import torch

    tok, model = _hf_pipeline()
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,          # determinístico: mesma pergunta, mesma resposta
            repetition_penalty=1.05,
        )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _generate_ollama(messages, max_tokens) -> str:
    r = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _generate_openai_compat(messages, max_tokens) -> str:
    r = requests.post(
        f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_PROVIDERS = {
    "transformers": _generate_transformers,
    "ollama": _generate_ollama,
    "openai_compat": _generate_openai_compat,
}


def llm_enabled() -> bool:
    return config.LLM_PROVIDER != "none"


def generate(messages, max_tokens: int = config.MAX_NEW_TOKENS) -> str:
    try:
        fn = _PROVIDERS[config.LLM_PROVIDER]
    except KeyError:
        raise ValueError(f"LLM_PROVIDER inválido: {config.LLM_PROVIDER}") from None
    return fn(messages, max_tokens).strip()


def judge_evidence(question: str, retrieval) -> bool:
    """True se o LLM considera que as avaliações recuperadas tratam do tema da pergunta."""
    verdict = generate(judge_messages(question, retrieval), max_tokens=3).upper()
    return not verdict.startswith("N")


def generate_answer(question: str, retrieval) -> str:
    return generate(answer_messages(question, build_context(retrieval)))
