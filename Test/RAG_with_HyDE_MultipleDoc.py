import os
import shutil
import qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
import nest_asyncio

nest_asyncio.apply()

UPLOAD_DIR = "uploaded_files"

# Function to reset pipeline status
def reset_pipeline_generated():
    st.session_state['pipeline_generated'] = False
    st.session_state.pop('messages', None)

# Function to handle file upload
def upload_file():
    files = st.sidebar.file_uploader(
        'Upload your document', 
        accept_multiple_files=True, 
        on_change=remove_unselected_files, 
        key="file_uploader"
    )
    if files:
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = []
        
        for file in files:
            if file.name not in st.session_state['uploaded_files']:
                file_path = save_uploaded_file(file)
                if file_path:
                    st.session_state['uploaded_files'].append(file.name)
        loaded_file = SimpleDirectoryReader(input_files=[os.path.join(UPLOAD_DIR, file_name) for file_name in st.session_state['uploaded_files']]).load_data()
        return loaded_file
    return None

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to list existing files in the upload directory
def list_existing_files():
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

# Function to initialize the vector store
def initialize_vector_store():
    client = qdrant_client.QdrantClient(location=':memory:')
    vector_store = QdrantVectorStore(client=client, collection_name="sampledata")
    return vector_store

# Function to delete a file
def delete_file(file_name):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        st.success(f"File {file_name} deleted successfully.")
        # Remove the file from the session state
        if 'uploaded_files' in st.session_state and file_name in st.session_state['uploaded_files']:
            st.session_state['uploaded_files'].remove(file_name)
        # Remove the file from the sidebar immediately
       


# Function to remove unselected files
def remove_unselected_files():
    if 'file_uploader' in st.session_state:
        current_files = st.session_state.get("file_uploader", [])
        current_file_names = [file.name for file in current_files] if current_files else []
        existing_files = list_existing_files()

        for file in existing_files:
            if file not in current_file_names:
                delete_file(file)

# Main function
def main():
    # Initialize models and settings
    Settings.llm = Ollama(model="llama3", request_timeout=400.0)
    Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed")
    Settings.text_splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model)
    
    st.set_page_config(page_title="QuickChat")
    st.title("💬 QuickChat")
    st.caption('📝 Chat with your document')

    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    existing_files = list_existing_files()

    if existing_files:
        st.sidebar.markdown("### Existing Files")
        for file_name in existing_files:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(file_name)
            if col2.button("Delete", key=f"delete_{file_name}"):
                delete_file(file_name)
                st.experimental_rerun()

    file = upload_file()

    # Check if there are still existing files after handling file upload and deletion
    existing_files = list_existing_files()

    if existing_files:
        if st.sidebar.button("Generate RAG Pipeline"):
            with st.spinner("Generating RAG Pipeline..."):
                vector_store = initialize_vector_store()
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                index = VectorStoreIndex.from_documents(
                    documents=file,
                    storage_context=storage_context,
                    show_progress=True,
                    transformations=[Settings.text_splitter]
                )

                index.storage_context.persist(persist_dir="dir")

                query_engine = index.as_query_engine(
                    response_mode="tree_summarize",
                    verbose=True,
                    similarity_top_k=10,
                )

                hyde_transform = HyDEQueryTransform(include_original=True)
                hyde_query_engine = TransformQueryEngine(query_engine, hyde_transform)

                st.session_state['hyde_query_engine'] = hyde_query_engine
                st.session_state['pipeline_generated'] = True

    if st.session_state.get('pipeline_generated', False):
        if 'messages' not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Enter your query", key='query'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.chat_message('user').write(prompt)

            if 'hyde_query_engine' in st.session_state:
                system_prompt = (
                    "You are an AI assistant specialized in providing information from the uploaded document. "
                    "Please ensure that your responses are strictly derived from the content of the document. "
                    "If the information is not found in the document, please indicate that explicitly."
                )
                query_with_prompt = f"{system_prompt}\nUser query: {prompt}"

                query_engine = st.session_state['hyde_query_engine']
                response = query_engine.query(query_with_prompt)
                msg = response.response

                st.session_state.messages.append({'role': 'assistant', 'content': msg})
                st.chat_message("assistant").write(msg)
            else:
                st.error("Query engine is not initialized")
    elif not existing_files and file is None:
        st.error("Please upload a file")

if __name__ == '__main__':
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    main()
