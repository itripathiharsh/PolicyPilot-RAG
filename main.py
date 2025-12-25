import os
import sys
from dotenv import load_dotenv

# Import our custom modules
from src.loader import ingest_documents
from src.vectorstore import VectorStoreManager
from src.engine import RAGEngine

# Load environment variables from .env
load_dotenv()

def initialize_system():
    """Sets up the data folders and vector store."""
    DATA_PATH = "data/"
    STORE_PATH = "vectorstore/db_faiss"
    
    # 1. Check for API Keys
    if not os.getenv("GROQ_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        print(" Error: GROQ_API_KEY or OPENROUTER_API_KEY missing in .env")
        sys.exit(1)

    # 2. Ensure Data Directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f" Created '{DATA_PATH}' folder. Please add your policy PDFs there and restart.")
        sys.exit(0)

    # 3. Initialize Vector Manager
    vm = VectorStoreManager(store_path=STORE_PATH)

    # 4. Ingest if DB doesn't exist
    if not vm.exists():
        print(" Initializing System: Building vector database for the first time...")
        if not os.listdir(DATA_PATH):
            print(f" No files found in '{DATA_PATH}'. Add PDFs to enable retrieval.")
            return None
        
        # This uses loader.py to chunk and embed using local HuggingFace models
        ingest_documents(DATA_PATH, STORE_PATH)
        print(" Database built and saved locally.")
    
    return vm.load()

def main():
    # Setup and Load Vector Store
    vectorstore = initialize_system()
    
    if not vectorstore:
        return

    # Initialize the RAG Engine (with Groq + OpenRouter fallbacks)
    engine = RAGEngine(vectorstore=vectorstore)

    print("\n" + "="*60)
    print(" AI POLICY ASSISTANT (Groq/OpenRouter Hybrid)")
    print("Commands: 'exit' to quit | 'eval' to run test set")
    print("="*60)

    # Pre-defined Evaluation Set (As required by the assignment)
    eval_questions = [
        "What is the standard refund window?",              # Grounded
        "Do you offer free shipping for international?",   # Grounded
        "Who is the current CEO of the company?",           # Out-of-bounds
        "How do I cancel my order if it already shipped?"  # Complex
    ]

    while True:
        user_input = input("\n Ask a policy question: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'eval':
            print("\n--- Running Automated Evaluation Set ---")
            for q in eval_questions:
                print(f"\n[QUERY]: {q}")
                res = engine.get_response(q)
                print(f"[ANSWER]: {res['answer']}")
                print(f"[SOURCES]: {', '.join(res['sources'])}")
            continue

        if not user_input:
            continue

        # Process Query
        print(" Thinking...")
        result = engine.get_response(user_input)
        
        print("\n" + "-"*30)
        print(f" RESPONSE:\n{result['answer']}")
        print(f"\n SOURCES: {', '.join(result['sources'])}")
        print("-"*30)

if __name__ == "__main__":
    main()