import os
import tempfile
import qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import nest_asyncio
nest_asyncio.apply()

# Function to reset pipeline status
def reset_pipeline_generated():
    st.session_state['pipeline_generated'] = False
    st.session_state.pop('messages', None)

# Function to handle file upload
def upload_file():
    file = st.sidebar.file_uploader('Upload your document here', on_change=reset_pipeline_generated)
    if file:
        file_path = save_uploaded_file(file)
        if file_path:
            loaded_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
            return loaded_file
    return None

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Function to initialize the vector store
def initialize_vector_store():
    client = qdrant_client.QdrantClient(location=':memory:')
    vector_store = QdrantVectorStore(client=client, collection_name="sampledata")
    return vector_store


# Main function
def main():
    # Initialize models and settings
    Settings.llm = Ollama(model="llama3", request_timeout=400.0,temperature=0)
    Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed")
    Settings.text_splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model)

    st.set_page_config(page_title="QuickChat")
    st.title("💬 QuickGPT")
    st.caption('📝 Chat with your document')

    file = upload_file()
    
    if file and st.sidebar.button("Generate RAG Pipeline"):
        with st.spinner("Generating RAG Pipeline..."):
            vector_store = initialize_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_documents(
                documents=file,
                storage_context=storage_context,
                show_progress=True,
                transformations=[Settings.text_splitter]
            )

            index.storage_context.persist(persist_dir="SingleDoc_dir")

            chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                response_mode="no_text",
                verbose=True,
                similarity_top_k=10,
            )
            
            st.session_state['chat_engine'] = chat_engine
            st.session_state['pipeline_generated'] = True

    if st.session_state.get('pipeline_generated', False):
        if 'messages' not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Enter your query", key='query'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.chat_message('user').write(prompt)
            
            if 'chat_engine' in st.session_state:
                system_prompt = (
                                """
                                You are a helpful AI assistant named QuickGPT, created by Quickfox Consulting. Your primary function is to provide comprehensive answers based solely on the information contained in the given context documents. Please adhere to the following guidelines:

                                Using the information contained in the context,
                                give a comprehensive answer to the question.
                                Respond only to the question asked, response should be concise and relevant to the question.
                                If the answer cannot be deduced from the given context, do not give an answer.
                                Context documents:
                                {context_str}
                                Your task is to provide detailed answers to user questions based exclusively on the above documents. 
                                Remember, if the information isn't in the context, simply state that you don't know.
                                </s>
                                """
                                )

                query_with_prompt = f"{system_prompt}\nUser query: {prompt}"

                chat_engine = st.session_state['chat_engine']
                response = chat_engine.chat(query_with_prompt)
                msg = response.response

                st.session_state.messages.append({'role': 'assistant', 'content': msg})
                st.chat_message("assistant").write(msg)
            else:
                st.error("Query engine is not initialized")
    elif file is None:
        st.error("Please upload a file to start the conversation")

if __name__ == '__main__':
    main()
