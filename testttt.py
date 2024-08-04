import os
import tempfile
import qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
import pymupdf4llm

import nest_asyncio
nest_asyncio.apply()

# Function to reset pipeline status
def reset_pipeline_generated():
    st.session_state['pipeline_generated'] = False
    st.session_state.pop('messages', None)

# Function to handle file upload
def upload_file():
    file = st.sidebar.file_uploader('Upload your document', on_change=reset_pipeline_generated)
    if file:
        file_paths = save_uploaded_file(file)
        if file_paths:
            loaded_file = pymupdf4llm.LlamaMarkdownReader().load_data(file_path=file_paths)
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

            # Text QA Prompt
            chat_text_qa_msgs = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are an AI assistant specialized in providing information from the uploaded {document} only and not capable of using your extra knowledge."
                        "Please consider all CONTEXT of the {document} to find the answer of the user query"
                        "Please ensure that your responses are derived only from the CONTEXT of the {document}."
                        "If the information is not found in the document, please indicate that you don't know and never answer the question by your own extra knowledge."
                        "Striclty stop using other external sources and your training data."
                        "Never provide any general information by refering to any other sources which is not available in the document."
                        "You cannot answer the question by your own knowledge, you instead must report your lack of information and state- I don't know and don't give extra general information and suggestions"
                    ),
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information, not prior knowledge and not using own knowledge alone. "
                        "answer the query.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    ),
                ),
            ]
            text_qa_template = ChatPromptTemplate(message_templates=chat_text_qa_msgs)

            chat_engine = index.as_chat_engine(
                text_qa_template=text_qa_template,
                # refine_template=refine_template,
                chat_mode="context",
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
                    "You are an AI assistant specialized in providing information from the uploaded document. "
                    "Please consider all content of the document to find the answer of the user query"
                    "Please ensure that your responses are derived only from the content of the document."
                    "If the information is not found in the document, please indicate that explicitly."
                    "Provide the exact source from where the response is extracted."
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
        st.error("Please upload a file")

if __name__ == '__main__':
    main()
