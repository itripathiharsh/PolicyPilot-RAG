import streamlit as st
import os
import tempfile
from src.loader import ingest_documents
from src.vectorstore import VectorStoreManager
from src.engine import RAGEngine

# --- PAGE CONFIG ---
st.set_page_config(page_title="PolicyPilot AI", page_icon="🤖", layout="centered")

# --- INITIALIZATION ---
# We store the engine in session state so it doesn't reload on every click
if "engine" not in st.session_state:
    with st.spinner("Initializing System & Loading Policies..."):
        vm = VectorStoreManager()
        # Initialize with default data
        if not vm.exists():
            ingest_documents("data/", "vectorstore/db_faiss")
        st.session_state.vectorstore = vm.load()
        st.session_state.engine = RAGEngine(st.session_state.vectorstore)
        st.session_state.messages = []

# --- SIDEBAR (Uploads & Settings) ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Add more PDFs to the system:", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Update Knowledge Base"):
        if uploaded_files:
            with st.spinner("Processing new documents..."):
                # Save to temp and ingest
                for uploaded_file in uploaded_files:
                    with open(os.path.join("data", uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Update the vectorstore
                st.session_state.vectorstore = ingest_documents(
                    "data/", "vectorstore/db_faiss", st.session_state.vectorstore
                )
                # Re-initialize engine with updated data
                st.session_state.engine = RAGEngine(st.session_state.vectorstore)
                st.success("Policies updated successfully!")
        else:
            st.warning("Please upload files first.")

# --- CHAT INTERFACE ---
st.title(" PolicyPilot Chat")
st.caption("I answer based on official policies and your uploaded documents.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a policy question..."):
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Call the engine (Streaming)
        stream_gen, docs = st.session_state.engine.get_response(
            prompt, st.session_state.messages
        )
        
        for chunk in stream_gen:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Display Sources
        if docs:
            sources = list(set([d.metadata.get('source', 'Unknown') for d in docs]))
            st.caption(f"Sources: {', '.join(sources)}")

    # 3. Add Assistant Message to History
    st.session_state.messages.append({"role": "assistant", "content": full_response})