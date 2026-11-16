# Voice of Customer Intelligence — Sistema RAG para avaliações da Olist

Projeto desenvolvido no **Tech Challenge — Fase 4** com o objetivo de facilitar a consulta e a análise de avaliações de clientes de um e-commerce. A solução utiliza **Retrieval-Augmented Generation (RAG)** para permitir que uma pessoa faça perguntas em linguagem natural e receba respostas fundamentadas em avaliações reais da base, com referências às evidências utilizadas.

Em vez de depender apenas do conhecimento geral de um modelo de linguagem, o sistema primeiro recupera comentários relevantes da base da Olist e, em seguida, utiliza esse conteúdo como contexto para elaborar a resposta. Quando não encontra evidências suficientes ou a pergunta não pertence ao escopo dos dados, o sistema informa que não consegue responder com segurança.

## Objetivo e problema de negócio

Empresas recebem grandes volumes de comentários e avaliações sobre produtos, entregas e atendimento. Embora esses registros contenham informações importantes sobre a experiência dos clientes, localizar manualmente os relatos relacionados a um assunto específico pode consumir tempo e dificultar a identificação de padrões.

Este projeto busca tornar essa exploração mais simples. Por meio de perguntas como:

- Quais são os principais problemas relatados pelos clientes?
- O que os clientes dizem sobre atrasos na entrega?
- Quais aspectos da experiência de compra são mais elogiados?
- O que aparece nas avaliações negativas de uma categoria específica?

o usuário consegue consultar os relatos e visualizar as avaliações que sustentam a resposta.

**Benefício esperado:** apoiar análises de Voz do Cliente (Voice of Customer — VoC), agilizar investigações sobre reclamações e elogios e facilitar a identificação de temas recorrentes. A solução é um protótipo analítico e não substitui a validação humana nem representa, por si só, uma ferramenta de decisão automatizada.

## Base de dados

O projeto utiliza o conjunto público **Brazilian E-Commerce Public Dataset by Olist**, disponibilizado no Kaggle. São utilizados arquivos relacionados a avaliações, pedidos, itens dos pedidos, produtos e tradução das categorias.

A base de conhecimento é construída a partir das avaliações que possuem comentários textuais. Cada avaliação é mantida como um documento individual, preservando seu identificador (`review_id`) para permitir a rastreabilidade das evidências.

Além do texto, são associados metadados que ajudam a contextualizar e filtrar os resultados, como:

- Nota da avaliação e classificação de sentimento (negativa, neutra ou positiva);
- Data e ano da avaliação;
- Categoria do produto;
- Situação do pedido;
- Indicador de entrega atrasada, dias para entrega e dias de atraso;
- Quantidade de itens e valor total dos itens do pedido.

O tratamento inclui normalização dos textos, remoção de comentários sem conteúdo informativo, deduplicação e tratamento de valores ausentes nos metadados.

## Como a solução funciona

O fluxo foi organizado em etapas:

1. **Ingestão e preparação:** os arquivos CSV da Olist são obtidos e tratados. Os comentários são limpos, avaliados quanto ao conteúdo informativo e enriquecidos com dados dos pedidos e produtos.
2. **Geração de embeddings:** cada comentário é convertido em uma representação vetorial usando o modelo multilíngue `intfloat/multilingual-e5-small`. Isso permite comparar semanticamente a pergunta com os textos, mesmo quando as palavras utilizadas são diferentes.
3. **Indexação:** os vetores, textos e metadados são armazenados em uma coleção persistente do **ChromaDB**, com busca por similaridade de cosseno.
4. **Análise da pergunta e recuperação:** o sistema prepara a consulta, identifica filtros aplicáveis e recupera os comentários mais próximos semanticamente. Os resultados passam por um limite mínimo de similaridade, deduplicação e seleção por **Maximal Marginal Relevance (MMR)**, que busca equilibrar relevância e diversidade das evidências.
5. **Verificação e geração:** antes de gerar a resposta, o sistema verifica se há evidências suficientes e utiliza o modelo de linguagem como um verificador adicional de pertinência. Se a consulta for adequada, o modelo sintetiza os comentários recuperados e deve associar os tópicos às referências das avaliações.
6. **Validação das citações:** referências inexistentes são removidas e tópicos sem citação válida são descartados. Assim, a resposta apresentada mantém vínculo com as avaliações recuperadas.

### Salvaguardas

O sistema foi projetado para reduzir respostas sem sustentação na base:

- Exige uma quantidade mínima de evidências relevantes antes de responder;
- Utiliza um verificador para avaliar se os comentários recuperados tratam do assunto perguntado;
- Orienta o modelo a responder somente com base nas avaliações fornecidas;
- Verifica as referências citadas e remove citações inválidas;
- Quando não há evidência suficiente ou a pergunta está fora do escopo, informa a limitação em vez de inventar uma resposta.

Essas medidas ajudam a controlar alucinações, mas não garantem que toda resposta gerada esteja correta. As evidências devem ser conferidas no contexto da análise.

## Interface e funcionalidades

A aplicação foi desenvolvida com **Streamlit** e oferece:

- Campo para perguntas em linguagem natural e sugestões de perguntas;
- Filtros por nota, categoria do produto, situação da entrega e ano da avaliação;
- Opção para detectar automaticamente filtros presentes na pergunta;
- Ajuste da quantidade de evidências utilizadas e da similaridade mínima;
- Resposta gerada acompanhada das avaliações recuperadas;
- Exibição de métricas da recuperação, como quantidade de avaliações relevantes, similaridade máxima e tempo de execução;
- Indicadores resumidos das avaliações relevantes, como nota média, percentual de entregas atrasadas e categorias mais frequentes.

Também é possível consultar o sistema pelo terminal, sem utilizar a interface gráfica.

## Tecnologias utilizadas

| Tecnologia | Utilização |
|---|---|
| Python 3.12+ | Linguagem principal |
| Pandas | Leitura, tratamento e preparação dos dados |
| Sentence Transformers | Geração dos embeddings |
| multilingual-e5-small | Modelo de embeddings multilíngue |
| ChromaDB | Armazenamento persistente e busca vetorial |
| NumPy | Operações com vetores e cálculo de similaridade |
| Transformers / Hugging Face | Execução local do modelo de linguagem |
| Qwen2.5-3B-Instruct | Modelo de linguagem padrão |
| Streamlit | Interface interativa |
| python-dotenv | Configurações por variáveis de ambiente |
| uv | Gerenciamento do ambiente e das dependências |

O provedor do modelo de linguagem é configurável: execução local com Transformers (padrão), Ollama, API compatível com OpenAI ou modo sem LLM, que retorna as evidências recuperadas.

## Estrutura do repositório

```text
.
├── app.py                       # Interface Streamlit
├── docs/
│   └── exemplos.md              # Relatório de perguntas, respostas e evidências
├── scripts/
│   ├── ask.py                   # Consulta pelo terminal
│   ├── build_index.py           # Preparação da base e criação do índice
│   ├── calibration_questions.py # Perguntas de calibração
│   ├── download_data.py         # Download dos dados da Olist
│   └── evaluate.py              # Avaliação e geração do relatório
├── src/
│   └── voc_rag/
│       ├── config.py            # Configurações e parâmetros
│       ├── data_prep.py         # Limpeza e construção da base de conhecimento
│       ├── embeddings.py        # Embeddings e integração com ChromaDB
│       ├── generator.py         # Verificação e geração com o LLM
│       ├── rag.py               # Orquestração do pipeline e salvaguardas
│       └── retriever.py         # Análise da pergunta e recuperação de evidências
├── .env.example                 # Exemplo de configuração
├── pyproject.toml               # Metadados e dependências do projeto
└── README.md
```

Os dados brutos, arquivos processados, índice vetorial e credenciais não são versionados no Git. Eles são criados localmente durante a execução e estão cobertos pelo `.gitignore`.

## Como executar o projeto

### 1. Pré-requisitos

- Python 3.12 ou superior;
- [uv](https://docs.astral.sh/uv/) instalado;
- Acesso à internet na primeira execução para baixar os dados e os modelos;
- Memória e espaço em disco suficientes para carregar os modelos localmente.

### 2. Clonar o repositório e instalar as dependências

```bash
git clone https://github.com/hnunesz/tech-challenge-fase4-rag.git
cd tech-challenge-fase4-rag

uv sync
```

### 3. Configurar as variáveis de ambiente

Crie seu arquivo local `.env` a partir do exemplo:

**Linux/macOS**
```bash
cp .env.example .env
```

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

O padrão já utiliza o modelo de linguagem local via Transformers, sem necessidade de chave de API. Para utilizar outro provedor, consulte as opções na seção de configuração.

### 4. Baixar os dados

O script utiliza o `kagglehub` para obter o dataset público da Olist e copiar os CSVs necessários para `data/raw/`:

```bash
uv run --with kagglehub python scripts/download_data.py
```

### 5. Preparar os dados e construir o índice vetorial

```bash
uv run python scripts/build_index.py
```

Essa etapa cria a base processada em `data/processed/`, gera os embeddings e persiste o índice no ChromaDB em `data/chroma/`. O processamento inicial pode levar alguns minutos, dependendo do equipamento.

Para reconstruir a base processada e o índice:

```bash
uv run python scripts/build_index.py --force
```

### 6. Executar a aplicação

```bash
uv run streamlit run app.py
```

O Streamlit exibirá no terminal o endereço local para abrir a interface no navegador.

### Consulta pelo terminal

Depois de construir o índice, também é possível fazer perguntas diretamente pelo terminal:

```bash
uv run python scripts/ask.py "O que os clientes relatam sobre problemas na entrega?"
```

## Avaliação e calibração

O projeto inclui scripts para verificar o comportamento da recuperação e das respostas:

**Executar o conjunto de perguntas e gerar o relatório**

```bash
uv run python scripts/evaluate.py
```

O script executa perguntas dentro e fora do escopo da base, registra o status, o tempo, os filtros aplicados, os resultados recuperados, as respostas e as evidências citadas. O relatório é salvo em `docs/exemplos.md`.

**Executar a calibração das salvaguardas**

```bash
uv run python scripts/evaluate.py --calibrate
```

A calibração utiliza perguntas de referência dentro e fora do escopo para observar o comportamento do limite de similaridade e do verificador de relevância. Ela auxilia o ajuste dos parâmetros, mas não deve ser interpretada isoladamente como uma avaliação definitiva da qualidade do sistema.

## Configurações principais

As configurações ficam centralizadas em `src/voc_rag/config.py` e podem ser sobrescritas por variáveis no arquivo `.env`.

| Variável | Padrão | Descrição |
|---|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Modelo responsável pelos embeddings |
| `LLM_PROVIDER` | `transformers` | Provedor do modelo de linguagem |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Identificador do modelo de linguagem |
| `OLLAMA_URL` | `http://localhost:11434` | Endereço do servidor Ollama |
| `OPENAI_BASE_URL` | vazio | URL-base da API compatível com OpenAI |
| `OPENAI_API_KEY` | vazio | Chave da API, quando necessária |

Valores possíveis para `LLM_PROVIDER`:

- `transformers`: executa o modelo do Hugging Face localmente (padrão);
- `ollama`: utiliza um servidor Ollama local;
- `openai_compat`: utiliza uma API compatível com o formato da OpenAI;
- `none`: desativa a geração e retorna as evidências recuperadas.

Ao utilizar `openai_compat`, preencha `OPENAI_BASE_URL`, `OPENAI_API_KEY` e o identificador apropriado em `LLM_MODEL`. Nunca compartilhe ou versione o arquivo `.env`.

Outros parâmetros, como quantidade de candidatos, quantidade de evidências no contexto, limite mínimo de similaridade e peso do MMR, podem ser ajustados em `config.py`.

## Limitações e próximos passos

- A base representa avaliações da Olist referentes ao período de 2016 a 2018; portanto, não reflete necessariamente a experiência atual dos consumidores.
- A qualidade das respostas depende da cobertura, qualidade e representatividade dos comentários recuperados.
- O modelo pode interpretar incorretamente uma avaliação ou sintetizar um padrão de forma incompleta; por isso, as evidências são exibidas para conferência.
- O modelo local pode exigir recursos computacionais consideráveis e ter maior tempo de resposta em máquinas com pouca memória.
- O sistema é um protótipo acadêmico. Antes de uso operacional, seriam necessários testes adicionais, monitoramento, avaliação com usuários e revisão dos controles de qualidade.

## Contexto acadêmico

Desenvolvido como entrega do **Tech Challenge — Fase 4**, com foco na aplicação de técnicas de recuperação semântica e modelos de linguagem para apoiar a análise de Voz do Cliente.

**Repositório:** https://github.com/hnunesz/tech-challenge-fase4-rag
