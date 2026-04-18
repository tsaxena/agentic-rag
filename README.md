# RAG Agent with LLaMA 3.1 + LangGraph

A Retrieval-Augmented Generation (RAG) agent that uses LLaMA 3.1 (via Ollama) locally, with LangGraph for agentic orchestration and LangSmith for tracing.

## How it works

1. **Retrieve** - fetches relevant documents from a vector store (pre-loaded from web URLs)
2. **Grade** - scores each retrieved document for relevance using LLM-as-judge
3. **Web Search** - falls back to Tavily web search if retrieved docs are not relevant
4. **Generate** - produces a concise answer using the relevant documents

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- A [Tavily](https://tavily.com) API key
- A [LangSmith](https://smith.langchain.com) API key

## Setup

### 1. Pull the required Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export TAVILY_API_KEY="your-tavily-api-key"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
```

> **Note:** The LangSmith API key in `rag_agent.py` is hardcoded and should be replaced with your own or set via the environment variable above.

## Run

```bash
python rag_agent.py
```

This will run the agent with the default question: *"What are the types of agent memory?"*

To ask a different question, edit the `example` dict at the bottom of `rag_agent.py`:

```python
example = {"input": "Your question here"}
response = predict_custom_agent_answer(example)
print(response)
```

## Output

The agent returns a dict with:
- `response` - the generated answer
- `steps` - list of pipeline steps executed (e.g. `retriever_documents`, `grade_document_retrival`, `web_search`, `generate_answers`)
