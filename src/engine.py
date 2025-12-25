import os
import random
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

# Import LangChain components
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Ensure environment variables are loaded
load_dotenv()

# THE FIX: Satisfy the OpenAI validator so it doesn't crash on start
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-placeholder-for-validator"

from src.prompts import PROMPT_V2

class RAGEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        # Search for top 3 relevant chunks
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Load multiple keys from .env (comma-separated)
        self.groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
        self.or_keys = [k.strip() for k in os.getenv("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
        
        # Build the resilient fallback chain
        self.llm_chain = self._build_fallback_chain()

    def _build_fallback_chain(self):
        """Constructs a list of LLMs using rotated keys for maximum uptime."""
        all_models = []

        # TIER 1: Groq Models (Fastest)
        # We cycle through all provided Groq keys for each model type
        groq_model_names = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        for m_name in groq_model_names:
            for key in self.groq_keys:
                all_models.append(ChatGroq(
                    model_name=m_name,
                    groq_api_key=key,
                    temperature=0.1
                ))

        # TIER 2: OpenRouter Models (Highest Reliability)
        # 15 Priority Models selected from your assessment data
        or_model_ids = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "x-ai/grok-4.1-fast:free",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini",
            "qwen/qwen3-coder:free",
            "kwaipilot/kat-coder-pro:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "google/gemini-2.5-flash-lite",
            "mistralai/mistral-small-24b:free",
            "z-ai/glm-4.5-air",
            "deepseek/deepseek-v3.2-exp",
            "qwen/qwen3-30b-a3b",
            "mistralai/pixtral-12b",
            "openai/gpt-4.1-nano"
        ]

        for m_id in or_model_ids:
            # Randomly assign one of your OpenRouter keys to this model
            key = random.choice(self.or_keys) if self.or_keys else os.getenv("OPENAI_API_KEY")
            all_models.append(ChatOpenAI(
                model=m_id,
                api_key=key, # Use the actual OpenRouter key here
                base_url="https://openrouter.ai/api/v1",
                temperature=0.1,
                streaming=True,
                default_headers={"X-Title": "PolicyPilot RAG Assistant"}
            ))

        # The 'with_fallbacks' method tells LangChain: 
        # "Try model 0. If it fails (rate limit/error), try model 1, then model 2..."
        return all_models[0].with_fallbacks(all_models[1:])

    def get_response(self, query: str, chat_history: List[Dict] = None):
        """
        Processes the query, retrieves context, and streams the answer.
        """
        # 1. Retrieval
        docs = self.retriever.invoke(query)
        context = "\n\n".join([f"Source: {d.metadata.get('source')}\n{d.page_content}" for d in docs])

        # 2. History Processing (Keep only last 5 exchanges to save tokens)
        formatted_history = []
        if chat_history:
            for msg in chat_history[-5:]:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_history.append(AIMessage(content=msg["content"]))

        # 3. Create LCEL Chain
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | PROMPT_V2 
            | self.llm_chain 
            | StrOutputParser()
        )

        # 4. Return Generator for Streaming and the Docs for citations
        return chain.stream({"question": query, "history": formatted_history}), docs