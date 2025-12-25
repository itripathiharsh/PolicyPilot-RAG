
# PolicyPilot-RAG

🔗 **Live Demo (Hugging Face Space):**  
https://huggingface.co/spaces/The-Arthur-Morgan/PolicyPilot-RAG


### High-Availability Retrieval-Augmented AI Policy Assistant

PolicyPilot-RAG is a **production-grade RAG system** built to deliver reliable, low-latency answers over policy documents. It combines **hybrid document ingestion**, **local vector search**, and a **15-tier LLM fallback engine** to maintain availability even under API failures or rate limits.

This project was developed as a **technical assessment**, with a strong focus on system resilience, grounding, and real-world deployment concerns.

---

## ✨ Key Highlights

* **High Availability by Design**
  15-model priority-based fallback across Groq and OpenRouter ensures uninterrupted responses.

* **Ultra-Low Latency Inference**
  Primary routing through Groq’s LPU enables near-instant responses.

* **Hybrid RAG Ingestion**

  * Pre-indexed system policies (`/data`)
  * Runtime PDF uploads via UI (no restart required)

* **Hallucination-Safe Generation**
  Strict grounding prompts, refusal logic for out-of-scope queries, and source-based answering.

* **Streaming Chat Experience**
  GPT-style real-time token streaming with multi-turn session memory.

---

## 🧠 System Architecture

| Layer         | Technology                     | Purpose                         |
| ------------- | ------------------------------ | ------------------------------- |
| Frontend      | Streamlit                      | Chat UI, uploads, session state |
| Orchestration | LangChain (LCEL)               | RAG pipeline control            |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2` | Local, zero-cost embeddings     |
| Vector Store  | FAISS                          | Fast semantic retrieval         |
| LLM Routing   | Groq + OpenRouter              | Multi-tier failover execution   |

---

## 🔁 LLM Fallback Strategy (15 Tiers)

To prevent downtime from rate limits or provider outages, the engine routes requests through a prioritized queue:

**Tier 1 – Performance (Groq)**

* `llama-3.3-70b-versatile`

**Tier 2 – Speed (Groq)**

* `llama-3.1-8b-instant`

**Tier 3 – Free / Stable (OpenRouter)**

* `gemini-2.0-flash-exp:free`
* `llama-3.3-70b-instruct:free`
* `grok-4.1-fast:free`

**Tier 4 – Deep Backup**

* `deepseek-chat`
* `gpt-4o-mini`
* `qwen3-coder`
* `nemotron-nano`
* * additional compatible models

**API Key Rotation**
Multiple keys per provider are supported and rotated automatically to maximize throughput and avoid hard limits.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/itripathiharsh/PolicyPilot-RAG.git
cd PolicyPilot-RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEYS=gsk_key1,gsk_key2
OPENROUTER_API_KEYS=sk-or-key1,sk-or-key2
OPENAI_API_KEY=placeholder_for_validator
```

> `OPENAI_API_KEY` is included only for compatibility with validation utilities.

---

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Evaluation Methodology

The system was evaluated using the **RAG Triad**:

* **Context Relevance**
* **Groundedness**
* **Answer Relevance**

### Sample Results

| Test Type    | Query                             | Outcome                            |
| ------------ | --------------------------------- | ---------------------------------- |
| Factual      | “What is the refund window?”      | Correct (30 days)                  |
| Policy Logic | “Cancel custom sofa after 6 hrs?” | Correct (Denied – 4 hr rule)       |
| Adversarial  | “Who is the CEO?”                 | Correct refusal (no hallucination) |

---

## 📁 Project Structure

```text
├── app.py                # Streamlit UI and session logic
├── src/
│   ├── engine.py         # 15-model routing & failover logic
│   ├── loader.py         # Document parsing & chunking
│   ├── vectorstore.py    # FAISS index management
│   └── prompts.py        # Grounding & system prompts
├── data/                 # Base policy documents
├── requirements.txt
└── README.md
```

---

## 🎯 Design Principles

* **Fail safely, never hallucinate**
* **Prefer local computation where possible**
* **Optimize for latency and uptime**
* **Production-first, demo-second**

---

## 📌 Notes

This project emphasizes **system reliability, prompt discipline, and infrastructure thinking**, rather than just model performance.

---

**Developed as a Technical Assessment Project**
Author: **Harsh Vardhan Tripathi**

---

