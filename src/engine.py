import os
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import our custom prompt versions
from src.prompts import PROMPT_V2  

class RAGEngine:
    def __init__(self, vectorstore):
        """
        Initializes the RAG Engine with Groq as primary and OpenRouter as fallbacks.
        """
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3} 
        )

        # 1. PRIMARY MODEL (Groq - High Speed)
        # Using Llama 3.3 70B for high-quality reasoning
        self.primary_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 2. FALLBACK MODELS (OpenRouter - Free/High Availability)
        # These models are tried in order if Groq fails or is rate-limited.
        fallback_model_ids = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "x-ai/grok-4.1-fast:free",
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-chat" # Highly reliable backup
        ]

        self.fallbacks = [
            ChatOpenAI(
                model=m_id,
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
                default_headers={
                    "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
                    "X-Title": "Intern Policy RAG"
                }
            ) for m_id in fallback_model_ids
        ]

        # Combine into a resilient LLM unit
        self.llm_chain = self.primary_llm.with_fallbacks(self.fallbacks)

    def _format_docs(self, docs):
        """Formats retrieved documents into a clean string for the prompt."""
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Policy')
            formatted.append(f"--- Document: {source} (Chunk {i+1}) ---\n{doc.page_content}")
        return "\n\n".join(formatted)

    def get_response(self, query: str) -> Dict:
        """
        Executes the RAG pipeline with automatic error handling and fallback logic.
        """
        # 1. Retrieve
        retrieved_docs = self.retriever.invoke(query)
        context_str = self._format_docs(retrieved_docs)

        # 2. Edge Case: If no documents are found
        if not retrieved_docs:
            return {
                "answer": "I'm sorry, I couldn't find any relevant company policies to answer your question.",
                "sources": [],
                "model_used": "None"
            }

        # 3. Define Chain
        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | PROMPT_V2 
            | self.llm_chain 
            | StrOutputParser()
        )

        # 4. Generate Answer with a general catch-all for API issues
        try:
            answer = chain.invoke(query)
            return {
                "answer": answer,
                "sources": list(set([doc.metadata.get('source') for doc in retrieved_docs])),
            }
        except Exception as e:
            return {
                "answer": f"All providers (Groq & OpenRouter) failed. Error: {str(e)}",
                "sources": []
            }