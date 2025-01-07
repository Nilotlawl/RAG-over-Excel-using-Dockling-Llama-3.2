import os
import requests
import gc
import tempfile
import uuid
import pandas as pd
import chardet

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

import streamlit as st
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

@st.cache_resource
def load_llm():
    try:
        llm = Ollama(
            model="llama3.2",  # This is correct
            request_timeout=300.0,
            temperature=0.1,
            num_ctx=4096,
            base_url="http://localhost:11434",
            additional_kwargs={
                "num_predict": 2048,
                "top_k": 40,
                "top_p": 0.9,
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.error("Please ensure Ollama is running locally with 'llama3.2' model downloaded")
        return None


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_file(file, file_type):
    st.markdown("### File Preview")
    if file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file)
    else:  # CSV
        # Detect encoding
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Reset file pointer and read with detected encoding
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding=encoding)
        except:
            # Fallback to latin-1 if detection fails
            file.seek(0)
            df = pd.read_csv(file, encoding='latin-1')
    
    st.dataframe(df)


with st.sidebar:
    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.xlsx` or `.csv` file", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        reader = DoclingReader()
                        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                        if file_ext == '.csv':
                            # Handle CSV files
                            with open(file_path, 'rb') as f:
                                raw_data = f.read()
                                result = chardet.detect(raw_data)
                                encoding = result['encoding']
                            
                            try:
                                df = pd.read_csv(file_path, encoding=encoding)
                            except:
                                df = pd.read_csv(file_path, encoding='latin-1')
                            
                            # Save as temporary Excel file for Docling
                            temp_excel = os.path.join(temp_dir, 'converted.xlsx')
                            df.to_excel(temp_excel, index=False)
                            temp_dir = os.path.dirname(temp_excel)
                    
                    if os.path.exists(temp_dir):
                            reader = DoclingReader()
                            loader = SimpleDirectoryReader(
                                input_dir=temp_dir,
                                file_extractor={".xlsx": reader},
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # setup llm & embedding model
                    llm = load_llm()
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-large-en-v1.5",
                        trust_remote_code=True,
                        device="cuda"  # Enable GPU usage
                    )
                    # Creating an index over loaded data
                    Settings.embed_model = embed_model
                    node_parser = MarkdownNodeParser()
                    index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)

                    # Create the query engine, where we use a cohere reranker on the fetched nodes
                    Settings.llm = llm
                    query_engine = index.as_query_engine(streaming=True)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # Inform the user that the file is processed and Display the PDF uploaded
                st.success("Ready to Chat!")
                display_file(uploaded_file, uploaded_file.name.split('.')[-1].lower())
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"RAG over Excel using Dockling 🐥 &  Llama-3.2")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    if "file_cache" not in st.session_state or not st.session_state.file_cache:
        st.error("Please upload a file first!")
        st.stop()
        
    try:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                streaming_response = query_engine.query(prompt)
                
                for chunk in streaming_response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                message_placeholder.error(error_message)
                full_response = error_message

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

try:
    response = requests.get("http://localhost:11434")
    print("Ollama service is running")
    
    # Test the model
    from llama_index.llms.ollama import Ollama
    llm = Ollama(model="llama3.2")
    response = llm.complete("Hello!")
    print("Model response:", response)
except Exception as e:
    print("Error:", e)