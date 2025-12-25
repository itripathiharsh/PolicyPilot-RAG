import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, store_path="vectorstore/db_faiss"):
        """
        Manages the loading and saving of the FAISS vector database.
        
        :param store_path: The directory where the FAISS index files will be stored.
        """
        self.store_path = store_path
        
        # We use a standard local model that maps text to a 384-dimensional vector space.
        # This must be the SAME model used during ingestion in loader.py.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def exists(self) -> bool:
        """
        Checks if the FAISS index files already exist in the specified path.
        """
        # FAISS creates two files: index.faiss and index.pkl
        index_file = os.path.join(self.store_path, "index.faiss")
        return os.path.exists(index_file)

    def load(self):
        """
        Loads the FAISS vector store from the local disk.
        
        :return: A FAISS vectorstore instance or None if it doesn't exist.
        """
        if self.exists():
            print(f"📊 Loading existing vector store from {self.store_path}...")
            # allow_dangerous_deserialization is required for loading FAISS .pkl files locally.
            return FAISS.load_local(
                self.store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("⚠️ No local vector store found.")
            return None

    def create_and_save(self, chunks):
        """
        Creates a new FAISS index from document chunks and saves it locally.
        
        :param chunks: List of LangChain Document objects.
        :return: The created FAISS vectorstore instance.
        """
        if not chunks:
            raise ValueError("Cannot create a vector store from an empty list of chunks.")
            
        print(f"🛠️ Creating new vector store with {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Ensure the directory exists
        os.makedirs(self.store_path, exist_ok=True)
        vectorstore.save_local(self.store_path)
        
        print(f"💾 Vector store saved locally to {self.store_path}")
        return vectorstore