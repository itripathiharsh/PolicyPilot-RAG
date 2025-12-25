import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentProcessor:
    def __init__(self, data_path: str = "data/", store_path: str = "vectorstore/db_faiss"):
        self.data_path = data_path
        self.store_path = store_path
        
        # Consistent Embedding Model across the system
        # Running on CPU for local cost-efficiency
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Policy-optimized splitting strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_and_save(self, existing_vectorstore=None):
        """
        Loads documents from the data directory, chunks them, and 
        updates/creates the FAISS index.
        """
        # 1. Load all PDFs and Markdown files in the data directory
        pdf_loader = DirectoryLoader(self.data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(self.data_path, glob="./*.md", loader_cls=TextLoader)
        
        documents = pdf_loader.load() + txt_loader.load()
        
        if not documents:
            print(" No documents found to ingest.")
            return existing_vectorstore

        # 2. Split into semantic chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # 3. Create or Update Vector Store
        if existing_vectorstore:
            # If we already have a DB, just add the new documents
            existing_vectorstore.add_documents(chunks)
            vectorstore = existing_vectorstore
        else:
            # Create a fresh DB
            vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # 4. Save locally
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        vectorstore.save_local(self.store_path)
        
        print(f" Successfully indexed {len(chunks)} chunks.")
        return vectorstore

# Functional wrapper for main.py/app.py use
def ingest_documents(data_path: str, store_path: str, existing_vs=None):
    processor = DocumentProcessor(data_path, store_path)
    return processor.process_and_save(existing_vs)