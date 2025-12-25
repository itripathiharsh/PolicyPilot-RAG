import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, store_path="vectorstore/db_faiss"):
        """
        Manages the FAISS vector database: Loading, Saving, and Searching.
        """
        self.store_path = store_path
        
        # Local Embedding Model (Must be identical to the one in loader.py)
        # We use CPU for the internship environment to ensure it runs anywhere.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def exists(self) -> bool:
        """Checks if the FAISS index already exists on the local disk."""
        return os.path.exists(os.path.join(self.store_path, "index.faiss"))

    def load(self):
        """
        Loads the vector store from disk. 
        Uses allow_dangerous_deserialization=True for local pickle loading.
        """
        if self.exists():
            print(f" Loading existing vector store from {self.store_path}...")
            return FAISS.load_local(
                self.store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print(" No existing vector store found.")
            return None

    def create_or_update(self, chunks):
        """
        Creates a new index or adds documents to an existing one.
        """
        if self.exists():
            # Load existing and add new chunks
            vector_db = self.load()
            vector_db.add_documents(chunks)
        else:
            # Create fresh from chunks
            vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the updated index back to disk
        os.makedirs(self.store_path, exist_ok=True)
        vector_db.save_local(self.store_path)
        return vector_db

    def get_retriever(self, k=3):
        """
        Returns the vector store as a retriever for the RAG engine.
        """
        vector_db = self.load()
        if vector_db:
            return vector_db.as_retriever(search_kwargs={"k": k})
        return None