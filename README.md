# Internal Engineering Copilot (Agentic RAG)

## The Business Problem
Software engineers and IT staff spend countless hours searching through disparate, fragmented internal documentation (Jira tickets, Confluence pages, Slack discussions, etc.) to find historical context on system architecture and resolved bugs. 

This project implements an **Agentic RAG (Retrieval-Augmented Generation) Workflow** to act as an Internal Engineering Copilot. It ingests simulated corporate documentation and uses LLMs accessed throuh Gorq API to accurately answer engineering queries while strictly maintaining data privacy.

## Key Technologies
* **LLM:** OpenAI’s GPT OSS 120B. Accessed through Gorq API.
* **Framework:** LangGraph / LangChain (Agentic Workflow).
* **Vector Store:** FAISS (Local in-memory execution).
* **Dataset:** Hugging Face `aeriesec/orgforge` (Simulated internal corporate Wiki & Ticketing system).

## Project layout

The project consits of the RAG Pipeline and a small testing suite. The source code for these can be found in `src/pipeline.py` and `src/testing.py` respectively.

### `src/pipeline.py`

This file contains the LangGraph Agentic workflow. It consits of a state class, base and wrapper classes for LLMs and retrievers, and the Pipeline class itself.

* The `RAGState` class stores all relevant information to a single workflow. These are the user question, alternative search queries, retrieved documents, and the generated answer. The Agent reads and updates these variables in each LangGraph node, representing workflow steps.
*  The `BaseLLM` and `BaseRetriever` classes are abstract classes for all LLM and retriever types that the RAG pipeline uses. This Adapter design pattern was chosen so all LLM providers, and Retrievers with different complexity have the same interface toward the RAG pipeline. `ChatModelWrapper` and `FAISSRetrieverWrapper` are wrapper classes inherited from the abstract classes that we use later on.
*  The `AgenticRAGPipeline` class contains the core RAG logic. It consists of nodes, that represent a set of subtasks the Agent must complete each time it is queried. Each node!s role and significance is explained in the `ADR_documentation.pdf` pdf file (page 2, "Cosen approach: Agentic Workflow" section).

### `src/testing.py`

This file contains the automated evaluation suite for the Agentic RAG pipeline. It uses the `ragas` framework to evaluate the system.

* The `RAGTester` class acts as the core of the testing suite. It is initialized with the active RAG pipeline, alongside a separate evaluator LLM (since ragas uses LLM-as-a-judge metrics) and an embedding model used for semantic similarity checks. Using a different LLM and embedding model to evaluate the system is the key to ensure fair results.
* The `generate_evaluation_dataset` method orchestrates the test runs. It feeds the benchmark questions into the RAG pipeline and records the generated `answer` and retrieved `contexts`. Crucially, it includes a `generate_gt_answers` flag; when enabled, it uses the evaluator LLM to dynamically translate the benchmark's raw JSON metadata into a natural language `ground_truth`. This ensures `ragas` can perform accurate text-to-text semantic comparisons.
* The `run_evaluation` method executes the actual `ragas` grading process against the generated dataset. By default, it measures three critical metrics: `faithfulness` (to ensure the model doesn't hallucinate beyond the retrieved documents), `answer_relevancy` (to ensure it directly answers the prompt), and `context_precision` (to verify the vector store retrieved the correct corporate knowledge).

## Quick Start Guide

### 1. Prerequisites
You must have Python 3.10+ installed. This application is designed to run entirely locally on a standard laptop, however it relies on external API keys (Gorq).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/boiboozled/delaware_rag_assignement.git
cd delaware_rag_assignement

# Create a virtual environment
python -m venv venv

# Activate it on Mac/Linux:
source venv/bin/activate
# OR activate it on Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Setting up the Groq API Key

* Go to (https://groq.com/)[https://groq.com/], and select 'Free API key' under the Developers menu point,
* After registering/logging in, generate an API key.
* Save it to the project root folder, under ```groq_api_key.txt```.

The system will use the above txt file to set the ```GROQ_API_KEY``` environment feature so the pipeline can run smoothly.

## 4. Demo usage

The delaware_rag_proto.ipynb notebook provides a complete walk-through of the system.

### Step 1: Data Ingestion and chunking

After imports and setting the Groq api key, the notebook downloads the OrgForge Dataset and chunks it using LangCHain's `RecursiveCharacterSplitter`:

```bash
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the dataset
events_dataset = load_dataset("aeriesec/orgforge", data_dir="corpus", split="train")

raw_docs = []
for row in events_dataset:
    content = (
        f"Title: {row.get('title', 'Unknown Title')}\n"
        f"Document Type: {row.get('doc_type', 'Unknown')}\n"
        f"Content: {row.get('body', '')}"
    )
    raw_docs.append(Document(page_content=content))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = text_splitter.split_documents(raw_docs)
```

### Step 2: Vector Store initialization

The chunked documents are then embedded and loaded into a local, in-memory FAISS vector database:

```bash
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = FAISSRetrieverWrapper(lc_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}))
```

### Step 3: Running the Agentic Workflow

The pipeline uses the Groq API for rapid inference with the massive 120B parameter MoE model. Here is how to query the agent:

```bash
from langchain_groq import ChatGroq

print("Loading Groq LLM (gpt-oss-120b)...")
groq_chat = ChatGroq(
    model="gpt-oss-120b",
    temperature=0.1,
    max_tokens=300
)
llm = ChatModelWrapper(groq_chat)

# Example Usage
query = "The lactate_metric client is getting timed out. Why is this?"
print(f"\nRunning query: '{query}'")

final_answer = agentic_rag.run(query)

print("\n=== FINAL ANSWER ===")
print(final_answer)
```

## 5. Evaluation

To test the reliabbility of the framework, run the following code:

```bash
# Initialize the Ragas Evaluator Models
print("Loading Ragas Evaluator Models...")

# Ragas Embeddings
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Ragas Judge LLM
ragas_judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1 
)

# Initialize the Tester
agentic_rag.verbose = False  # Turn off verbose for cleaner evaluation outputs
tester = RAGTester(
    rag_pipeline=agentic_rag, 
    evaluator_llm=ragas_judge_llm, 
    evaluator_embeddings=ragas_embeddings
)

prepared_dataset = tester.generate_evaluation_dataset(test_questions)

evaluation_results = tester.run_evaluation(prepared_dataset)
print("\nEvaluation Complete!")
```
