from src.components.database.vector_store import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.llm import LLMChain
from typing import Optional, Dict
from src.components.experiment_tracker import MetricsCollector
import time

class RAG:
    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str,
        prompt_template: Optional[str] = None
    ):
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided")
        self.vector_store = vector_store
        self.llm = OpenAI(openai_api_key=openai_api_key)
        self.metrics_collector = MetricsCollector()

        self.prompt_template = PromptTemplate(
            input_variables=['context', 'question'],
            template=prompt_template or self._default_prompt_template()
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _default_prompt_template(self) -> str:
        return """
        You are an AI assistant knowledgeable about computer science research.

        Context:
        {context}

        Question:
        {question}

        Provide a detailed and accurate answer based on the context provided.
        """

    def answer_question(self, question: str, k: int = 3) -> Dict[str, any]:
        start_time = time.time()
        try:
            # Get context with metrics
            context_result = self.get_context(question, k)
            context = context_result["context"]
            metrics = context_result["metrics"]

            # Generate response
            response = self.llm_chain.run(context=context, question=question)

            # Add response metrics
            response_stats = self.metrics_collector.get_text_stats(response)
            metrics.update({
                "response_length": len(response),
                "response_tokens": response_stats["token_count"],
                "total_processing_time": time.time() - start_time,
                "success": True
            })

            return {
                "response": response,
                "metrics": metrics
            }
        except Exception as e:
            return {
                "response": f"Error generating answer: {str(e)}",
                "metrics": {
                    "error": str(e),
                    "success": False,
                    "total_processing_time": time.time() - start_time
                }
            }

    def get_context(self, question: str, k: int = 3) -> Dict[str, any]:
        start_time = time.time()
        try:
            relevant_docs = self.vector_store.similarity_search(question, k=k)
            context = "\n\n".join([doc for doc, _ in relevant_docs])

            # Collect metrics
            context_stats = self.metrics_collector.get_text_stats(context)
            question_stats = self.metrics_collector.get_text_stats(question)

            metrics = {
                "context_length": len(context),
                "context_tokens": context_stats["token_count"],
                "context_chunks": len(relevant_docs),
                "question_length": len(question),
                "question_tokens": question_stats["token_count"],
                "retrieval_time": time.time() - start_time,
                "success": True
            }

            return {
                "context": context,
                "metrics": metrics
            }
        except Exception as e:
            return {
                "context": "",
                "metrics": {
                    "error": str(e),
                    "success": False,
                    "retrieval_time": time.time() - start_time
                }
            }

