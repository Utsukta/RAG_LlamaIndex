import streamlit as st
from chat_with_document.utils import *
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class ChatwithDocument:

    def reset_pipeline_generated(self):
      if 'pipeline_generated' in st.session_state:
        st.session_state['pipeline_generated'] = False

      if 'messages' in st.session_state:
         del st.session_state['messages']

    def main(self):
      st.set_page_config(page_title="QuickChat")
      st.title("💬 QuickChat")
      st.caption('📝 Chat with your document')

      Settings.llm=Ollama(model="llama3",request_timeout=120.0)
      with st.spinner("please wait") as status:
        Settings.embed_model=OllamaEmbedding(model_name="snowflake-arctic-embed")


      file = st.sidebar.file_uploader('Upload your document',on_change=self.reset_pipeline_generated)
      uploaded_file=upload_file(file)
      vectore_store=vector_store()
      response_mode=response_synthesis_method()
      Node_parser=node_parser()

      if file is not None:
         if st.sidebar.button("Generate RAG Pipeline"):
            with st.spinner():
               query_engine=generate_rag_pipeline(uploaded_file,Node_parser,response_mode,vectore_store)
               st.session_state['query_engine'] = query_engine
               st.session_state['pipeline_generated'] = True
               st.success("RAG Pipeline Generated Successfully!")


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
            #    source_nodes=Message.source_nodes
            #    st.markdown(source_nodes)
               # keywords=Message.excerpt_keywords
               # st.markdown(keywords)
               st.session_state.messages.append({'role':'assistant','content':msg})
               st.chat_message("assistant").write(msg)   
            else:
               st.error("Query engine is not initialised")


chatwithdoc=ChatwithDocument()
chatwithdoc.main()
    
        
    

    
