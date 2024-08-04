import os,tempfile,qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext,Settings,KnowledgeGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from neo4j import GraphDatabase
from llama_index.core.graph_stores import SimpleGraphStore
from pyvis.network import Network
from IPython.display import display,HTML

class App:
   def reset_pipeline_generated(self):
      if 'pipeline_generated' in st.session_state:
        st.session_state['pipeline_generated'] = False

      if 'messages' in st.session_state:
         del st.session_state['messages']

   #Here, we upload the file
   def upload_file(self):
        file = st.sidebar.file_uploader('Upload your document',on_change=self.reset_pipeline_generated)
        if file is not None:
           file_path=self.save_uploaded_file(file)

           if file_path:
              loaded_file=SimpleDirectoryReader(input_files=[file_path]).load_data()
              return loaded_file
        return None
    
   #Here, we save the uploaded file
   def save_uploaded_file(self,uploaded_file):
        try:
          with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
              tmp_file.write(uploaded_file.getvalue())
              return tmp_file.name
        except Exception as e:
           st.error(f"Error saving file: {e}")
           return None
        
   #Next, we chunk the file
   def node_parser(self):
      Settings.text_splitter=SentenceSplitter(chunk_size=1024,chunk_overlap=20)
      return Settings.text_splitter
      
   #Here, we define our LLM model to use
   def llm_model(self):
      #For Ollama
      Settings.llm=Ollama(model="llama3",request_timeout=120.0)
      return Settings.llm
       
   #Here, we select the embedding model
   def embedding_model(self):
      with st.spinner("please wait") as status:
         Settings.embed_model=OllamaEmbedding(model_name="snowflake-arctic-embed")
         st.session_state['embed_model'] = Settings.embed_model
      return Settings.embed_model
      
   #Here, we define response synthesis method
   def response_synthesis_method(self):
      response_mode="tree_summarize"
      return response_mode
   
   #Here, we make a vector database
   def vector_store(self):
      client=qdrant_client.QdrantClient(location=":memory:")
      vector_store=QdrantVectorStore(client=client,collection_name="sampledata")
      return vector_store
   

   #Now, creating the rag pipeline
   def generate_rag_pipeline(self,file,llm,embed_model,node_parser,response_mode,vector_store):
      graph_store = SimpleGraphStore()
      storage_context = StorageContext.from_defaults(graph_store=graph_store)
      index = KnowledgeGraphIndex.from_documents(
         documents=file,
         max_triplets_per_chunk=2,
         storage_context=storage_context,
         embed_model=embed_model,
         include_embeddings=True,
      )
            
      query_engine = index.as_query_engine(
         include_text=False, 
         response_mode=response_mode,
         embedding_mode="hybid",
         similarity_top_k=5
      )
      graph=index.get_networkx_graph()
      net=Network(notebook=True,cdn_resources="in_line",directed=True)
      net.from_nx(graph)
      net.show("graph.html")
      

      return query_engine

   def main(self):
      st.set_page_config(page_title="QuickChat")
      st.title("💬 QuickChat")
      st.caption('📝 Chat with your document')

      file=self.upload_file()
      node_parser=self.node_parser()
        
      llm=self.llm_model()
      embed_model=self.embedding_model()
      vectore_store=self.vector_store()
      response_mode=self.response_synthesis_method()
      
      #Generate RAG Pipeline Button  
      if file is not None:
         if st.sidebar.button("Generate RAG Pipeline"):
            with st.spinner():
               query_engine=self.generate_rag_pipeline(file, llm, embed_model,node_parser,response_mode,vectore_store)
               st.session_state['query_engine'] = query_engine
               st.session_state['pipeline_generated'] = True
               st.success("RAG Pipeline Generated Successfully!")

         elif file is None:
            st.error("Please upload a file")

      #After Generating the RAG pipeline
      if st.session_state.get('pipeline_generated',False):
        if 'messages' not in st.session_state:
            st.session_state["messages"]=[{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
           st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Enter your query", key='query'):
            st.session_state.messages.append({'role':'user','content':prompt})
            st.chat_message('user').write(prompt)
            st.markdown("")
            if 'query_engine' in st.session_state:
               system_prompt = (
   "You are an AI assistant specialized in providing information from the uploaded document. "
   "Please ensure that your responses are strictly derived from the content of the document. "
   "If the information is not found in the document, please indicate that explicitly."
)              
               query_with_prompt=f"{system_prompt}\nUser query:{prompt}"
               Message=st.session_state['query_engine'].query(query_with_prompt)
               msg=Message.response
               st.session_state.messages.append({'role':'assistant','content':msg})
               st.chat_message("assistant").write(msg)   
            else:
               st.error("Query engine is not initialised")

if __name__=='__main__':
    a=App()
    a.main()

