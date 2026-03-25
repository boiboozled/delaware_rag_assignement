from typing import List, TypedDict
from urllib import response
from langchain_core.documents import Document
from abc import ABC, abstractmethod
from typing import List
from langgraph.graph import StateGraph, END
import re

class RAGState(TypedDict):
    """
    State of the RAG graph. Stores the user question, retrieved documents, and the generated answer.
    """
    question: str
    documents: List[Document]
    search_queries: List[str]
    answer: str

class BaseLLM(ABC):
    """Abstract base class for all LLMs used in the pipeline."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Takes a raw string prompt and returns the generated text."""
        pass

class BaseRetriever(ABC):
    """Abstract base class for all Document Retrievers used in the pipeline."""

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """Takes a search query and returns a list of LangChain Document objects."""
        pass

class HuggingFaceLLMWrapper(BaseLLM):
    def __init__(self, lc_llm):
        """
        Wraps a LangChain HuggingFacePipeline or HuggingFaceEndpoint object.
        """
        self.llm = lc_llm

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)

        if "Answer:" in prompt and "Answer:" in response:
             return response.split("Answer:")[-1].strip()
        if "Alternatives:" in prompt and "Alternatives:" in response:
             return response.split("Alternatives:")[-1].strip()

        return response.strip()

class ChatModelWrapper(BaseLLM):
    def __init__(self, chat_model):
        """
        Wraps a LangChain ChatModel (like ChatGroq or ChatMistralAI).
        """
        self.llm = chat_model

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt).content

        if "Answer:" in prompt and "Answer:" in response:
             return response.split("Answer:")[-1].strip()
        if "Alternatives:" in prompt and "Alternatives:" in response:
             return response.split("Alternatives:")[-1].strip()

        return response.strip()

class FAISSRetrieverWrapper(BaseRetriever):
    def __init__(self, lc_retriever):
        """
        Wraps a LangChain VectorStoreRetriever (like FAISS or Qdrant).
        """
        self.retriever = lc_retriever

    def retrieve(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)
    
class AgenticRAGPipeline:
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever, verbose: bool = False):
        """
        Initializes the AgenticRAGPipeline with injected LLM and retriever dependencies.

        Args:
            - llm (BaseLLM): The language model wrapper used for generation and decision making.
            - retriever (BaseRetriever): The document retriever wrapper used for fetching context.
            - verbose (bool): If True, prints intermediate steps and retrieved documents to the console.

        """
        self.llm = llm
        self.retriever = retriever
        self.verbose = verbose
        self.app = self._build_graph()

    def generate_queries(self, state: RAGState):
        """
        Expands the original user query into multiple alternative perspectives to improve retrieval.

        Args:
            - state (RAGState): The current state of the LangGraph, containing the original question.

        Returns:
            - dict (dict): A dictionary updating the state with the original question and the generated 'search_queries' list.
        """
        if self.verbose: print("\n---GENERATE ALTERNATIVE QUERIES---")
        question = state["question"]

        prompt = (
            "You are an expert search query optimization assistant for an internal software engineering database.\n"
            "Your task is to generate exactly 2 alternative ways to phrase the user's question to improve keyword matching in a Jira and Confluence vector store.\n"
            "Rules:\n"
            "1. Do not answer the question.\n"
            "2. Only provide the queries.\n"
            "3. You MUST wrap each alternative query exactly inside <query> and </query> tags.\n\n"
            f"Original question: {question}\n\n"
            "Alternatives:\n"
        )

        response = self.llm.generate(prompt)
        alternatives = re.findall(r'<query>(.*?)</query>', response, re.IGNORECASE | re.DOTALL)        
        alternatives = [q.strip() for q in alternatives if q.strip()][:2]

        if not alternatives:
            if self.verbose: print("  -> LLM failed formatting. Falling back to original query only.")
            search_queries = [question]
        else:
            search_queries = [question] + alternatives

        if self.verbose:
            print(f"  -> Original: '{question}'")
            for i, q in enumerate(alternatives):
                print(f"  -> Alt {i+1}: '{q}'")

        return {"search_queries": search_queries, "question": question}

    def retrieve(self, state: RAGState):
        """
        Retrieves documents from the vector store for all search queries and deduplicates them.

        Args:
            - state (RAGState): The current state of the LangGraph, containing the search queries.

        Returns:
            - dict (dict): A dictionary updating the state with the deduplicated list of 'documents' and the original 'question'.
        """
        if self.verbose: print("\n---RETRIEVE DOCUMENTS FOR ALL QUERIES---")
        search_queries = state.get("search_queries", [state["question"]])

        all_docs = []
        seen_contents = set()

        for q in search_queries:
            if self.verbose: print(f"  -> Fetching for: '{q}'")
            docs = self.retriever.retrieve(q)

            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)

        if self.verbose:
            print(f"  -> Total unique documents retrieved: {len(all_docs)}")

        return {"documents": all_docs, "question": state["question"]}

    def grade_documents(self, state: RAGState):
        """
        Evaluates the retrieved documents to filter out irrelevant information based on the user's question.

        Args:
            - state (RAGState): The current state of the LangGraph, containing the question and retrieved documents.

        Returns:
            - dict (dict): A dictionary updating the state with only the relevant 'documents' and the original 'question'.
        """
        if self.verbose: print("\n---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        relevant_docs = []

        for d in documents:
            prompt = (
                f"You are a grader assessing relevance of a retrieved document to a user question.\n"
                f"Document: \n\n {d.page_content} \n\n"
                f"Question: {question} \n"
                f"Respond ONLY with the word 'yes' if the document is relevant to the question, or 'no' if it is not."
            )

            score = self.llm.generate(prompt).lower()

            if "yes" in score:
                if self.verbose: print(f"  -> Document graded: RELEVANT")
                relevant_docs.append(d)
            else:
                if self.verbose: print(f"  -> Document graded: IRRELEVANT")

        return {"documents": relevant_docs, "question": question}

    def generate(self, state: RAGState):
        """
        Generates the final answer to the user's question using only the relevant retrieved context.

        Args:
            - state (RAGState): The current state of the LangGraph, containing the question and relevant documents.

        Returns:
            - dict (dict): A dictionary updating the state with the final generated 'answer', the 'documents', and the 'question'.
        """
        if self.verbose: print("\n---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]

        docs_content = "\n\n".join(doc.page_content for doc in documents)

        prompt = (
            f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            f"If you don't know the answer or the context is empty, just say that you don't know.\n"
            f"Question: {question}\n"
            f"Context: {docs_content}\n"
            f"Answer:"
        )

        generation = self.llm.generate(prompt)

        return {"documents": documents, "question": question, "answer": generation}

    def _build_graph(self):
        """
        Constructs and compiles the LangGraph state machine workflow.

        Args:
            - None: This method does not take any arguments.

        Returns:
            - app (CompiledGraph): The compiled LangGraph application ready for execution.
        """
        workflow = StateGraph(RAGState)

        workflow.add_node("generate_queries", self.generate_queries)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, query: str, return_docs:bool=False):
        """
        Executes the compiled LangGraph agentic RAG pipeline for a given user query.

        Args:
            - query (str): The initial user question to be answered.
            - return_docs (bool): If True, also returns the retrieved documents along with the answer.

        Returns:
            - answer (str): The final generated answer from the pipeline.
            - documents (List[Document], optional): The list of retrieved documents if return_docs is True.
        """
        if self.verbose: print(f"\nInitializing run for query: '{query}'")
        inputs = {"question": query}

        for output in self.app.stream(inputs):
            for key, value in output.items():
                if self.verbose: print(f"Finished node: '{key}'")

        if return_docs:
            return value["answer"], value["documents"]
        else:
            return value["answer"]