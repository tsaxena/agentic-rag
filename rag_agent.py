"""
RAG Agent using LLaMA 3.1 (via Ollama) + LangSmith tracing + LangGraph
"""

import os
from typing import List
from typing_extensions import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import END, StateGraph


# --- LangSmith config ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-agent-llama3"
os.environ["LANGCHAIN_API_KEY"] = ""

local_llm = "llama3.1"

# --- Document loading ---
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
retriever = vectorstore.as_retriever(k=4)

# --- Web search tool ---
web_search_tool = TavilySearchResults()


# --- RAG Chain ---
llm = ChatOllama(model=local_llm, temperature=0)

rag_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.

Use the following documents to answer the question.

If you don't know the answer, just say that you don't know.

Use three sentences maximum and keep the answer concise:
Question: {question}
Documents: {documents}
Answer:""",
    input_variables=["question", "documents"],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = rag_prompt | llm | StrOutputParser()


# --- Retrieval Grader ---
grader_llm = ChatOllama(model=local_llm, format="json", temperature=0)

grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question.

Here is the retrieved document:

{document}

Here is the user question: {question}

If the document contains keywords related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = grader_prompt | grader_llm | JsonOutputParser()


# ============================================================
# LangGraph State + Nodes
# ============================================================

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retriever_documents")
    # Retrieval
    return {"documents": documents, "question": question, "steps": steps}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrival")
    # Score each doc
    filtered_docs = []
    search = "No" 
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "search": search, "steps": steps, }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    
    # Web search
    web_results = web_search_tool.invoke({"query": question})
    
    # Check if web_results is a list of dictionaries
    if isinstance(web_results, list) and all(isinstance(item, dict) for item in web_results):
        new_documents = [
            Document(page_content=d.get("content", ""), metadata={"url": d.get("url", "")})
            for d in web_results
        ]
    else:
        # If web_results is not in the expected format, create a single document
        new_documents = [Document(page_content=str(web_results), metadata={"url": ""})]
    
    documents.extend(new_documents)
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answers")
    return {"documents": documents, "question": question, "generation": generation, "steps": steps}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    search = state["search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# --- Build Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"web_search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


def query(question: str) -> str:
    result = app.invoke({"question": question, "generation": "", "web_search": False, "documents": []})
    return result["generation"]

def predict_custom_agent_answer(example: dict):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = app.invoke(
        {"question": example["input"],
         "steps": []},
        config
    )
    return {
        "response": state_dict["generation"],
        "steps": state_dict["steps"]
    }


if __name__ == "__main__":
    
    example = {"input": "What are the types of agent memory?"}
    response = predict_custom_agent_answer(example)
    print(response)
    # questions = [
    #     "What is RAG?",
    #     "What is LangSmith used for?",
    #     "How can I run LLaMA locally?",
    # ]

    # for q in questions:
    #     print(f"\nQ: {q}")
    #     print(f"A: {query(q)}")
