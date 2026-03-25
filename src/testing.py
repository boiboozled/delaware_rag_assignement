import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# LangChain wrappers needed for Ragas
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings, ChatHuggingFace
from transformers import pipeline

class RAGTester:
    def __init__(self, rag_pipeline, evaluator_llm, evaluator_embeddings):
        """
        Initializes the testing suite.
        
        Args:
            rag_pipeline: Your instantiated AgenticRAGPipeline.
            evaluator_llm: A LangChain BaseChatModel wrapper (used by Ragas as the judge).
            evaluator_embeddings: A LangChain Embeddings wrapper (used by Ragas).
        """
        self.rag_pipeline = rag_pipeline
        self.evaluator_llm = evaluator_llm
        self.evaluator_embeddings = evaluator_embeddings

    def generate_evaluation_dataset(self, test_dataset, generate_gt_answers=False) -> Dataset:
        """
        Runs the RAG pipeline against the test questions to generate answers and contexts.
        Ragas requires a dataset with specific columns: question, answer, contexts, ground_truth.
        """
        print("\n--- GENERATING ANSWERS FOR EVALUATION ---")
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        for item in test_dataset:
            question = item['question_text']
            ground_truth = item['ground_truth']
            
            print(f"Processing query: {question[:50]}...")
            
            # Run your RAG pipeline (ensure return_docs=True)
            answer, documents = self.rag_pipeline.run(question, return_docs=True)
            
            # Ragas expects contexts as a list of strings
            contexts = [doc.page_content for doc in documents]

            if generate_gt_answers:
                gt_prompt = (
                    f"You are a helpful data assistant generating ground-truth answers for an AI evaluation.\n"
                    f"Convert the following JSON metadata into a concise, natural language answer to the user's question.\n\n"
                    f"Question: {question}\n"
                    f"True Metadata (Use this to answer the question): {ground_truth}\n\n"
                    f"Rules:\n"
                    f"1. Provide ONLY the natural language answer.\n"
                    f"2. Do not include introductory filler like 'The answer is' or 'Based on the metadata'.\n"
                    f"Answer:"
                )
                generated_ground_truth = self.evaluator_llm.invoke(gt_prompt).content.strip()
            
                if self.rag_pipeline.verbose:
                    print(f"  -> Generated Ground Truth: {generated_ground_truth}")

                ground_truth = generated_ground_truth
            
            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(ground_truth)

        return Dataset.from_dict(eval_data)

    def run_evaluation(self, eval_dataset: Dataset, metrics: list = None):
        """
        Runs the Ragas evaluation suite against the generated dataset.
        """
        print("\n--- RUNNING RAGAS EVALUATION ---")
        if metrics is None:
            # Default metrics
            metrics = [faithfulness, answer_relevancy, context_precision]

        # Ragas evaluate function takes the dataset, metrics, and your local models
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
            raise_exceptions=False # Prevents the whole suite from crashing if one eval fails
        )
        
        return results