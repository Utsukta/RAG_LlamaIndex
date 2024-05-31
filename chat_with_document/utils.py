import tempfile,os,qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext,VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore


def upload_file(file):
        # file = st.sidebar.file_uploader('Upload your document',on_change=self.reset_pipeline_generated)
        if file is not None:
           file_path=save_uploaded_file(file)

           if file_path:
              loaded_file=SimpleDirectoryReader(input_files=[file_path]).load_data()
              return loaded_file
        return None
    
   #Here, we save the uploaded file
def save_uploaded_file(uploaded_file):
        try:
          with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
              tmp_file.write(uploaded_file.getvalue())
              return tmp_file.name
        except Exception as e:
           return e
        
   #Next, we chunk the file
def node_parser():
      Settings.text_splitter=SentenceSplitter(chunk_size=1024,chunk_overlap=20)
      return Settings.text_splitter

def response_synthesis_method():
      response_mode="refine"
      return response_mode
      
def vector_store():
      client=qdrant_client.QdrantClient(location=":memory:")
      vector_store=QdrantVectorStore(client=client,collection_name="sampledata",enable_hybrid=True,)
      return vector_store



#Now, creating the rag pipeline
def generate_rag_pipeline(file,node_parser,response_mode,vector_store):

      if vector_store is not None:
        #Set storage context if vector store is not None
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
      else:
        storage_context=None
   
      #Create the vector index
      vector_index=VectorStoreIndex.from_documents(
                                       documents=file,
                                       node_parser=node_parser,
                                       storage_context=storage_context,
                                       show_progress=True
                                    )
      if storage_context:
        vector_index.storage_context.persist(persist_dir="dir")

      #Create the query engine
      query_engine=vector_index.as_query_engine(response_mode=response_mode,
                                             
                                                verbose=True,
                                                similarity_top_k=5,
                                                sparse_top_k=15,
                                                vector_store_query_mode="hybrid")
      return query_engine