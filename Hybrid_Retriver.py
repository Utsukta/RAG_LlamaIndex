import os
import tempfile
import qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser,SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.query_engine import RetrieverQueryEngine

import nest_asyncio
nest_asyncio.apply()

from llama_index.core.retrievers import BaseRetriever

#Custom Retriver Implementation
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        

    def _retrieve(self, query, **kwargs):
        print("HybridRetriever _retrieve method called") 
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        print('a')
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes



# Function to reset pipeline status
def reset_pipeline_generated():
    st.session_state['pipeline_generated'] = False
    st.session_state.pop('messages', None)

# Function to handle file upload
def upload_file():
    file = st.sidebar.file_uploader('Upload your document', on_change=reset_pipeline_generated)
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
    Settings.llm = Ollama(model="llama3", request_timeout=400.0)
    Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
    Settings.text_splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model)
    # splitter = SemanticSplitterNodeParser(embed_model=Settings.embed_model)
  
  
    st.set_page_config(page_title="QuickChat")
    st.title("💬 QuickChat")
    st.caption('📝 Chat with your document')

    file = upload_file()
    
    splitter=SentenceSplitter(chunk_size=1024)
   
    
    
    if file and st.sidebar.button("Generate RAG Pipeline"):
        with st.spinner("Generating RAG Pipeline..."):
            nodes=splitter.get_nodes_from_documents(documents=file)
           
            vector_store = initialize_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            storage_context.docstore.add_documents(nodes)
            
            index = VectorStoreIndex.from_documents(
                documents=file,
                storage_context=storage_context,
                show_progress=True,
                transformations=[Settings.text_splitter]
            )

            vector_retriever = index.as_retriever(similarity_top_k=10)

            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

            index.as_retriever(similarity_top_k=5)
            
            hybrid_retriever=HybridRetriever(vector_retriever,bm25_retriever)
            

            index.storage_context.persist(persist_dir="Hyrbid_dir")

            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                llm=Settings.llm
            )
            # retrieved_nodes = hybrid_retriever.retrieve('Explain Donatio Mortis Causa.')
            # print("Here after calling retriever method")
            # print(retrieved_nodes)

            # hyde_transform = HyDEQueryTransform(include_original=True)
            # hyde_query_engine = TransformQueryEngine(query_engine, hyde_transform)

            st.session_state['hybrid_retriever'] = hybrid_retriever
            st.session_state['Retriever_query_engine'] =query_engine
            st.session_state['pipeline_generated'] = True

    if st.session_state.get('pipeline_generated', False):
        if 'messages' not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Enter your query", key='query'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            st.chat_message('user').write(prompt)

            if 'Retriever_query_engine' in st.session_state:
                system_prompt = (
                    "You are an AI assistant specialized in providing information from the uploaded document. "
                    "Please consider all content of the document to find the answer of the user query"
                    "Please ensure that your responses are derived only from the content of the document."
                    "If the information is not found in the document, please indicate that explicitly."
                )
                query_with_prompt = f"{system_prompt}\nUser query: {prompt}"


                retrieved_nodes = st.session_state['hybrid_retriever'].retrieve(query_with_prompt)
                print("Here after calling retriever method")
                print(retrieved_nodes)
                
                print('Retriever_query_engine')
                query_engine = st.session_state['Retriever_query_engine']
                print(query_engine)

                response = query_engine.query(query_with_prompt)
                
                msg = response.response

                st.session_state.messages.append({'role': 'assistant', 'content': msg})
                st.chat_message("assistant").write(msg)
            else:
                st.error("Query engine is not initialized")
    elif file is None:
        st.error("Please upload a file")

if __name__ == '__main__':
    main()
