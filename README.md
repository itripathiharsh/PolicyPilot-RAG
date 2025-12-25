
# AI Policy Assistant: Prompt Engineering & RAG System

A specialized Retrieval-Augmented Generation (RAG) assistant designed to provide accurate, grounded, and concise answers based on company policy documents. Built to demonstrate high-performance retrieval, effective prompt engineering, and robust hallucination control.

## 🚀 Quick Start

### 1. Prerequisites

* Python 3.9+
* [Groq API Key](https://console.groq.com/) (Primary LLM)
* [OpenRouter API Key](https://openrouter.ai/) (Fallback LLMs)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rag-intern-assignment.git
cd rag-intern-assignment

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Environment Setup

Create a `.env` file in the root directory:

```text
GROQ_API_KEY=your_gsk_key_here
OPENROUTER_API_KEY=your_sk_or_key_here

```

### 4. Run the Assistant

```bash
python main.py

```

---

## 🏗️ System Architecture

The system implements a **Modular RAG** architecture with the following components:

| Component | Technology | Reasoning |
| --- | --- | --- |
| **Data Ingestion** | `PyPDFLoader` & `RecursiveCharacterTextSplitter` | Handles complex formatting and ensures policy rules aren't split mid-sentence. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, local CPU-based execution. Zero API cost for the embedding layer. |
| **Vector Store** | `FAISS` | Provides high-speed semantic search for top-k retrieval. |
| **Primary LLM** | **Llama-3.3-70B** (via Groq) | Sub-second inference speed with reasoning capabilities comparable to GPT-4. |
| **Fallbacks** | Gemini 2.0 Flash / Llama 3.3 (via OpenRouter) | Ensures system reliability even if primary providers hit rate limits. |

---

## 🧠 Prompt Engineering & Iteration

The core of this project was moving from a naive "answer this" prompt to a structured, production-grade instruction set.

### Version 1 (Naive)

> *Prompt:* "Answer the question based on this context: {context}. Question: {question}"

* **Result:** Often hallucinated when information was missing; lacked professional tone.

### Version 2 (Production - Final)

> *Prompt:* Includes Persona setting (Policy Expert), strict grounding rules ("Use ONLY provided context"), and refusal instructions ("If not found, strictly say...").

* **Why it's better:** * **Hallucination Control:** Forces the model to admit ignorance instead of guessing.
* **Citations:** Automatically appends source document names for auditability.
* **Logic Enforcement:** Explicitly tells the model to follow multi-step reasoning.



---

## 📊 Evaluation Results

I evaluated the system across three categories to verify factual accuracy and guardrail strength.

| Test Case Category | Query Example | Result | Score |
| --- | --- | --- | --- |
| **Factual Retrieval** | "What is the refund window?" | Correctly identified 30 days and packaging requirements. | ✅ 1.0 |
| **Complex Logic** | "Cancel custom sofa after 6hrs?" | Correctly calculated  (production start) and denied request. | ✅ 1.0 |
| **Hallucination Check** | "Who is the company CEO?" | Correctly refused to answer as info was not in docs. | ✅ 1.0 |
| **Out-of-Scope** | "Policy on drone delivery?" | Gracefully handled missing info without guessing. | ✅ 1.0 |

---

## 🛠️ Design Trade-offs & Future Improvements

1. **Local vs. Cloud Embeddings:** I chose local `HuggingFace` embeddings to reduce latency and API costs, though a cloud model like OpenAI `text-embedding-3-small` might offer slightly higher retrieval precision for much larger datasets.
2. **Chunk Size:** Chose 600 characters with 10% overlap. Larger chunks would provide more context but risk "noise" during retrieval; smaller chunks might split a single policy rule into two pieces.
3. **Future Step - Reranking:** With more time, I would implement a **Cross-Encoder Reranker** (e.g., Cohere or BGE-Reranker) to refine the top-k results before sending them to the LLM.
4. **Future Step - Hybrid Search:** Combine BM25 (keyword search) with semantic search to better handle specific terms like SKU numbers or exact fee amounts.

---

### **Next Step Recommendation**

Would you like me to help you create a **GitHub Action** to automatically run your "eval" command and generate a test report on every push?