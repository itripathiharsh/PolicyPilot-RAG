import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents(data_path: str, store_path: str):
    """
    Reads documents, chunks them, generates local embeddings, 
    and saves a FAISS vector database.
    """
    
    # 1. Load Data
    # Support for both PDFs and Markdown/Text policy files
    pdf_loader = DirectoryLoader(data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(data_path, glob="./*.md", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + txt_loader.load()
    
    if not docs:
        raise ValueError(f"No documents found in {data_path}. Please add some policy files.")

    # 2. Chunking Strategy
    # We use 600 characters with a 60-character overlap.
    # This ensures that policy rules aren't cut in half and context is preserved.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)

    # 3. Local Embeddings (Free & Fast)
    # This model runs on your CPU. No API key required.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Create and Save Vector Store
    # FAISS is used for efficient similarity search
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    vectorstore.save_local(store_path)
    
    print(f" Ingestion complete. {len(chunks)} chunks saved to {store_path}")
    return vectorstore

if __name__ == "__main__":
    # Test block for local debugging
    DATA_DIR = "data"
    STORE_DIR = "vectorstore/db_faiss"
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR} folder. Please add your PDFs there.")
    else:
        ingest_documents(DATA_DIR, STORE_DIR)